"""Plays the scheduled games — process-parallel, with a sequential fallback.

The tournament fans its games across a :class:`concurrent.futures.
ProcessPoolExecutor`, mirroring the proven self-play collector
(:mod:`wingspan.training.mp_collect`): module-level picklable worker functions,
one torch thread per worker so the pool scales across cores, and results
streamed back via :func:`futures.as_completed` (so completion order — but not
any reported statistic — is nondeterministic). Each worker caches the model nets
it loads, so a competitor's checkpoint is read from disk at most once per
process.

:func:`play_tournament_game` is the pure per-game unit shared by the worker path
and the in-process path (``in_process=True``), so a test can exercise the real
game logic without paying the spawn cost.
"""

from __future__ import annotations

import logging
import os
import random
import typing
from concurrent import futures

import pydantic
import torch

from wingspan import engine
from wingspan.tournament import models, participants, results, schedule
from wingspan.training import collect

# Fired with each finished game as it lands (live Elo / dashboard updates).
type ResultCallback = typing.Callable[[models.GameResult], None]
# Polled as games complete; once true, pending games are cancelled.
type StopCheck = typing.Callable[[], bool]

# Salt that reseeds a random competitor per game, so a (deal, orientation)
# reproduces the same random play across processes and reruns — kept distinct
# from the deal seed itself.
_RANDOM_SALT = 0x6D2B79F5

# Leave a couple of cores for the main thread and the live dashboard render.
_RESERVED_CORES = 2
# Past this the per-worker spawn + import-torch cost rarely beats the gain.
_MAX_WORKERS = 16


class _WorkerRoster(pydantic.BaseModel):
    """The competitor roster a worker needs to build agents on demand. Shipped
    once to each worker as pool ``initargs``; frozen so it is immutable work."""

    model_config = pydantic.ConfigDict(frozen=True)

    specs: list[models.ParticipantSpec]
    device: str


def run_tournament(
    cfg: models.TournamentConfig,
    on_result: ResultCallback | None = None,
    should_stop: StopCheck | None = None,
    *,
    in_process: bool = False,
) -> models.TournamentReport:
    """Play the whole round-robin and return the aggregated report.

    ``on_result`` fires from the calling thread as each game finishes (so it may
    update shared dashboard state under the caller's lock); ``should_stop`` is
    polled to cancel the remainder early. ``in_process`` runs every game in this
    process (no pool) — the path tests use to stay fast.
    """
    tasks = schedule.build_schedule(cfg.participants, cfg.games_per_pair, cfg.base_seed)
    device = torch.device(cfg.device)
    if in_process:
        collected = _run_in_process(cfg, tasks, device, on_result, should_stop)
    else:
        collected = _run_parallel(cfg, tasks, on_result, should_stop)
    return results.aggregate(cfg, collected)


def play_tournament_game(
    specs_by_id: dict[str, models.ParticipantSpec],
    model_agents: dict[str, engine.Agent],
    task: models.GameTask,
    device: torch.device,
) -> models.GameResult:
    """Play one scheduled game and read out competitor A's result.

    ``model_agents`` caches loaded model nets across games by competitor id;
    random competitors are rebuilt (reseeded) per game so the mirror's two
    orientations still differ. Seats are assigned by ``task.a_seat``; the deal's
    own first player is read from the engine into ``a_was_start_player``.
    """
    agent_a = _agent_for(specs_by_id, model_agents, task.player_a_id, task, device)
    agent_b = _agent_for(specs_by_id, model_agents, task.player_b_id, task, device)
    seats: tuple[engine.Agent, engine.Agent] = (
        (agent_a, agent_b) if task.a_seat == 0 else (agent_b, agent_a)
    )
    game = collect.new_engine(task.deal_seed)
    engine.Engine.play_one_game(game.state, seats)

    score0 = game.state.players[0].final_score or 0
    score1 = game.state.players[1].final_score or 0
    a_score, b_score = (score0, score1) if task.a_seat == 0 else (score1, score0)
    return models.GameResult(
        round_index=task.round_index,
        pair_index=task.pair_index,
        orientation=task.orientation,
        player_a_id=task.player_a_id,
        player_b_id=task.player_b_id,
        a_score=a_score,
        b_score=b_score,
        a_was_start_player=game.state.start_player == task.a_seat,
    )


###### PRIVATE #######


def _run_in_process(
    cfg: models.TournamentConfig,
    tasks: typing.Sequence[models.GameTask],
    device: torch.device,
    on_result: ResultCallback | None,
    should_stop: StopCheck | None,
) -> list[models.GameResult]:
    """Play every game in this process (no pool) — the test/headless path."""
    specs_by_id = {spec.id: spec for spec in cfg.participants}
    model_agents: dict[str, engine.Agent] = {}
    collected: list[models.GameResult] = []
    for task in tasks:
        if should_stop is not None and should_stop():
            break
        result = play_tournament_game(specs_by_id, model_agents, task, device)
        collected.append(result)
        if on_result is not None:
            on_result(result)
    return collected


def _run_parallel(
    cfg: models.TournamentConfig,
    tasks: typing.Sequence[models.GameTask],
    on_result: ResultCallback | None,
    should_stop: StopCheck | None,
) -> list[models.GameResult]:
    """Fan the games across a persistent worker pool, streaming completions."""
    roster = _WorkerRoster(specs=list(cfg.participants), device=cfg.device)
    pool = futures.ProcessPoolExecutor(
        max_workers=_default_worker_count(len(tasks)),
        initializer=_worker_init,
        initargs=(roster,),
    )
    collected: list[models.GameResult] = []
    try:
        pending = {pool.submit(_worker_play, task): task for task in tasks}
        for future in futures.as_completed(pending):
            collected.append(future.result())
            if on_result is not None:
                on_result(collected[-1])
            if should_stop is not None and should_stop():
                for other in pending:
                    other.cancel()
                break
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return collected


def _agent_for(
    specs_by_id: dict[str, models.ParticipantSpec],
    model_agents: dict[str, engine.Agent],
    participant_id: str,
    task: models.GameTask,
    device: torch.device,
) -> engine.Agent:
    """The agent for one competitor in one game — a cached model net, or a fresh
    per-game-seeded random agent."""
    spec = specs_by_id[participant_id]
    if spec.kind is models.ParticipantKind.RANDOM:
        seed = task.deal_seed ^ _RANDOM_SALT ^ int(task.orientation)
        return participants.load_player(spec, device, random.Random(seed))
    cached = model_agents.get(participant_id)
    if cached is None:
        cached = participants.load_player(spec, device, random.Random(0))
        model_agents[participant_id] = cached
    return cached


def _default_worker_count(n_tasks: int) -> int:
    """Workers default to (cores − reserved), capped, never more than the games."""
    cores = os.cpu_count() or 4
    usable = max(1, min(cores - _RESERVED_CORES, _MAX_WORKERS))
    return max(1, min(usable, n_tasks))


# ---------------------------------------------------------------------------
# Worker-process state: one roster of competitor specs and a per-process cache of
# the model agents this worker has loaded, populated by ``_worker_init``.

_worker_specs: dict[str, models.ParticipantSpec] = {}
_worker_model_agents: dict[str, engine.Agent] = {}
_worker_device: torch.device | None = None


def _worker_init(roster: _WorkerRoster) -> None:
    """Pin torch to one thread, silence stray logging, and stash the roster so
    each game just loads (and caches) the agents it needs."""
    global _worker_specs, _worker_model_agents, _worker_device
    torch.set_num_threads(1)
    logging.getLogger().addHandler(logging.NullHandler())
    _worker_specs = {spec.id: spec for spec in roster.specs}
    _worker_model_agents = {}
    _worker_device = torch.device(roster.device)


def _worker_play(task: models.GameTask) -> models.GameResult:
    """Play one scheduled game under this worker's cached roster + nets."""
    assert _worker_device is not None, "worker not initialized"
    return play_tournament_game(
        _worker_specs, _worker_model_agents, task, _worker_device
    )
