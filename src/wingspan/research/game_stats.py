"""The model-performance game-stats study (``docs/RESEARCH.md``, "Model
performance stats").

Plays N games of greedy self-play — one trained net on every seat, its setup
net resolving the opening keep — and records, per game and per seat, the score
after every turn (two conventions), the main action chosen on every turn, the
winner / tie, and the final score with its six-way breakdown. Capture is the
reusable :class:`~wingspan.instrumentation.handlers.game_stats.GameStatsCapture`
handler; no game log or event recorder is attached, so a game costs only its
inference.

Work is planned as shards of consecutive games
(``constants.GAME_STATS_GAMES_PER_SHARD``). :func:`study_shard` is a pure
function of the spec, the shard, and the nets — each game is seeded by
``spec.game_seed(index)`` alone — so the in-process path and the process pool
produce byte-identical CSVs, and the pool streams shard results to disk in
shard order.
"""

from __future__ import annotations

import pathlib
import time
import typing

import pydantic
import torch

from wingspan import engine, model
from wingspan.instrumentation.handlers import game_stats
from wingspan.players import loaders
from wingspan.research import constants, models, shard_runner
from wingspan.training import artifacts, collect, policy
from wingspan.training import setup_net as setup_net_module


class GameNets(pydantic.BaseModel):
    """Everything a worker loads once from the run directory: the policy net,
    the optional separate setup net, and the regime the run was trained under
    (seat count and the three engine-regime flags)."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    policy_net: model.PolicyValueNet
    setup_net: setup_net_module.SetupNet | None
    device: torch.device
    num_players: int
    split_setup_bonus: bool
    split_setup_food: bool
    combine_gain_food: bool


def load_game_nets(checkpoint_dir: pathlib.Path, device: torch.device) -> GameNets:
    """Load the run's ``last.pt`` policy net, its setup net, and its regime.

    Raises ``FileNotFoundError`` when the run has neither a separate setup net
    nor a policy net that embeds setup (the study needs the setup keep resolved
    by the model, not at random)."""
    policy_net, run_config = loaders.load_policy_net(
        checkpoint_dir / artifacts.LAST_CKPT, device
    )
    setup_net = loaders.load_setup_net(checkpoint_dir, device)
    if setup_net is None and not policy_net.include_setup:
        raise FileNotFoundError(
            f"no setup model in {checkpoint_dir}: the game-stats study resolves "
            "the opening keep with the model (setup.pt + run_config_*.json)"
        )
    return GameNets(
        policy_net=policy_net,
        setup_net=setup_net,
        device=device,
        num_players=run_config.num_players,
        split_setup_bonus=run_config.split_setup_bonus_active,
        split_setup_food=run_config.split_setup_food_active,
        combine_gain_food=run_config.engine.combine_gain_food,
    )


def run_game_stats_study(
    spec: models.GameStatsSpec,
    checkpoint_dir: pathlib.Path,
    device: torch.device,
    out_path: pathlib.Path,
    *,
    progress_stream: typing.TextIO | None = None,
) -> models.GameStatsSummary:
    """Run the whole study against the run in ``checkpoint_dir`` and write the
    raw per-seat table to ``out_path`` (rows stream to disk shard by shard).

    When ``progress_stream`` is given, a throttled progress line with
    throughput and ETA is written to it as shards complete; the default is
    silent. Raises before the output file is created when the run cannot be
    loaded."""
    started = time.monotonic()
    nets = load_game_nets(checkpoint_dir, device)
    rows = shard_runner.stream_study(
        spec=spec,
        shards=plan_shards(spec),
        nets=nets,
        load_nets=load_game_nets,
        study_shard=study_shard,
        checkpoint_dir=checkpoint_dir,
        device=device,
        out_path=out_path,
        columns=models.game_stats_csv_columns(),
        to_row=models.GameSeatRecord.to_csv_row,
        progress_stream=progress_stream,
    )
    return models.GameStatsSummary(
        out_path=str(out_path),
        games=spec.games,
        rows=rows,
        workers=spec.workers,
        elapsed_seconds=time.monotonic() - started,
    )


def plan_shards(
    spec: models.GameStatsSpec, games_per_shard: int | None = None
) -> list[models.GameStatsShard]:
    """Split the study's games into consecutive shards of at most
    ``games_per_shard`` games (default ``constants.GAME_STATS_GAMES_PER_SHARD``)."""
    size = (
        constants.GAME_STATS_GAMES_PER_SHARD
        if games_per_shard is None
        else games_per_shard
    )
    return [
        models.GameStatsShard(
            first_game_index=first, num_games=min(size, spec.games - first)
        )
        for first in range(0, spec.games, size)
    ]


def study_shard(
    spec: models.GameStatsSpec, shard: models.GameStatsShard, nets: GameNets
) -> list[models.GameSeatRecord]:
    """Play every game of ``shard`` and return its per-seat records in game
    order, then seat order."""
    records: list[models.GameSeatRecord] = []
    for game_index in shard.game_indices:
        records.extend(play_game(spec, game_index, nets))
    return records


def play_game(
    spec: models.GameStatsSpec, game_index: int, nets: GameNets
) -> list[models.GameSeatRecord]:
    """Play one seeded greedy self-play game and return one record per seat."""
    seed = spec.game_seed(game_index)
    eng = collect.new_engine(seed, nets.num_players)
    agent = policy.greedy_agent(nets.policy_net, nets.device, setup_net=nets.setup_net)
    capture = game_stats.GameStatsCapture()
    engine.Engine.play_one_game(
        eng.state,
        [agent] * nets.num_players,
        instrumentation=capture.instrumentation(),
        verbose_turn_log=False,
        split_setup_bonus=nets.split_setup_bonus,
        split_setup_food=nets.split_setup_food,
        combine_gain_food=nets.combine_gain_food,
    )
    return [
        models.GameSeatRecord(
            game_index=game_index,
            seed=seed,
            seat=seat.seat,
            won=seat.won,
            tie=capture.tie,
            final_score=seat.final_score,
            birds=seat.bird_pts,
            bonus=seat.bonus_pts,
            eggs=seat.eggs,
            tucked=seat.tucked,
            cached=seat.cached,
            goals=seat.round_goal,
            score_proj=seat.score_proj,
            score_raw=seat.score_raw,
            actions=seat.actions,
        )
        for seat in capture.seats
    ]
