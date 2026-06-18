"""Batched self-play collection.

Playing games one at a time means one batch-of-one forward pass per decision —
~130 per game, each a tiny CPU matmul whose Python/dispatch overhead dwarfs the
actual compute. This module instead plays many games *concurrently* — one OS
thread per game, each running the ordinary synchronous
:class:`wingspan.engine.Engine` — and funnels every game's policy query through
a single :class:`_BatchInferenceServer`. The server blocks each requesting game
until all currently-live games are waiting at a decision, then runs **one**
padded, masked forward pass over the whole set and hands each game back its own
probability row. Sampling itself still happens in the game thread off the
game's own RNG, so trajectories stay per-game reproducible and match the
single-decision sampling rule (:func:`policy.sample_index_from_probs`).

This collapses the ~130-per-game forward calls into ~130 *total* per wave
(one per decision round, shared across games), which on CPU is ~1.2x faster
end-to-end at the default 64-game iteration. The forward pass was only ~20%
of collection, though — per-candidate encoding (``net.encode_choices`` via the
server, GIL-bound and run per game thread) is the larger remaining cost and the
next target. See the ``training-throughput-bottleneck`` analysis.

Only the forward pass is shared; engine mutation, encoding, and sampling are
per-game. The network is read-only during collection and is touched by exactly
one thread (the server), so no torch state is shared concurrently.

The public entry point is :func:`collect_games`; it returns the same
:class:`collect.GameRecord` objects the sequential collector produced, in seed
order, and fires ``on_game_done`` as each game finishes so the dashboard can
advance mid-iteration.

Selection: ``loop._collect`` now routes **CUDA** collection here (one shared GPU
forward beats one model copy per process); the CPU path moved to the
process-parallel ``mp_collect`` once threads proved GIL-bound. The ~1.2x figure
above is from when this *was* the CPU path — it documents the shared-forward
technique, not the current CPU route. See ``training/COLLECTORS.md``.
"""

from __future__ import annotations

import random
import threading
import typing

import numpy as np
import torch
import torch.nn.functional as F

from wingspan import agents, decisions, encode, engine, model
from wingspan.training import collect, policy, steps, timestamps

if typing.TYPE_CHECKING:
    from wingspan import state

# Distinct salt for the bootstrap phase's random opponent, kept separate from
# the policy-sampling stream (mirrors ``mp_collect._OPPONENT_RNG_SALT``).
_OPPONENT_RNG_SALT = 0x85EBCA6B

# Distinct RNG stream for in-game sampling, kept separate from the per-game
# board-shuffle seed so the two never share a sequence. Xed into the seed.
_SAMPLE_RNG_SALT = 0x9E3779B9

# Upper bound on threads (and therefore batch size) in flight at once. Games
# beyond this run in later waves. Bounds OS-thread count for large
# ``games_per_iter`` while keeping the default 64-game iteration a single wave.
_MAX_CONCURRENT_GAMES = 64


def collect_games(
    net: model.PolicyValueNet,
    device: torch.device,
    seeds: typing.Sequence[int],
    on_game_done: typing.Callable[[collect.GameRecord], None] | None = None,
    should_stop: typing.Callable[[], bool] | None = None,
    max_concurrent: int = _MAX_CONCURRENT_GAMES,
    vs_random: bool = False,
) -> list[collect.GameRecord]:
    """Play ``len(seeds)`` games concurrently with batched inference — self-play,
    or (when ``vs_random``) the net at seat 0 against the random agent.

    Games run in waves of at most ``max_concurrent`` threads; within a wave
    every game's policy forward pass is batched together (in the ``vs_random``
    phase only the net's seat-0 queries route through the server — seat 1 picks
    randomly off-server). Returns the finished records in ``seeds`` order.
    ``on_game_done`` fires once per game as it completes (from a worker thread,
    serialized), and ``should_stop`` is polled between waves so a stop request
    halts before launching more games (games already in flight finish)."""
    server = _BatchInferenceServer(net, device)
    server.start()
    results: list[collect.GameRecord | None] = [None] * len(seeds)
    done_lock = threading.Lock()
    try:
        wave = max(1, min(max_concurrent, len(seeds)))
        for start in range(0, len(seeds), wave):
            if should_stop is not None and should_stop():
                break
            batch = list(range(start, min(start + wave, len(seeds))))
            _run_wave(
                net,
                device,
                server,
                seeds,
                batch,
                results,
                on_game_done,
                done_lock,
                vs_random,
            )
    finally:
        server.stop()
    return [record for record in results if record is not None]


###### PRIVATE #######


class _InferenceRequest:
    """One game's pending policy query plus the slot the server fills.

    Not a Pydantic record: it is a synchronization primitive (a future) — a
    game thread parks on ``done`` until the server writes ``probs`` and sets it.
    """

    __slots__ = ("state_vec", "choice_feats", "family_idx", "probs", "value", "done")

    def __init__(
        self, state_vec: np.ndarray, choice_feats: np.ndarray, family_idx: int
    ):
        self.state_vec = state_vec
        self.choice_feats = choice_feats
        self.family_idx = family_idx
        self.probs: np.ndarray = _EMPTY_PROBS
        self.value: float = 0.0
        self.done = threading.Event()


_EMPTY_PROBS = np.zeros((0,), dtype=np.float32)


class _BatchInferenceServer:
    """Coordinates one batched forward pass across all concurrently-waiting games.

    A game calls :meth:`infer` and blocks. The server thread fires a batch the
    moment every live game (tracked by :meth:`register` / :meth:`unregister`)
    has a request queued, so the batch is as large as the number of games still
    in flight and shrinks naturally as games finish.
    """

    def __init__(self, net: model.PolicyValueNet, device: torch.device):
        self._net = net
        self._device = device
        self._cond = threading.Condition()
        self._pending: list[_InferenceRequest] = []
        self._active = 0
        self._stopping = False
        self._thread = threading.Thread(
            target=self._serve, name="wingspan-batch-infer", daemon=True
        )

    @property
    def spec(self) -> encode.EncodingSpec:
        """The served net's encoding spec, so workers encode states/choices at the
        shape the net was built for (the setup axis is config-driven)."""
        return self._net.spec

    def encode_state(
        self,
        game_state: "state.GameState",
        decision: decisions.Decision[typing.Any],
    ) -> np.ndarray:
        """Featurize a state for the served net — delegated to the net itself so
        an era-pinned (compat) net's frozen encoder is used, never the live
        module functions paired with a spec by hand."""
        return self._net.encode_state(game_state, decision)

    def encode_choices(
        self,
        decision: decisions.Decision[typing.Any],
        game_state: "state.GameState",
    ) -> np.ndarray:
        """Featurize a decision's choices for the served net (see
        :meth:`encode_state`)."""
        return self._net.encode_choices(decision, game_state)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        """Signal the server thread to exit and join it. Callers must have
        joined every game thread first, so no requests are outstanding."""
        with self._cond:
            self._stopping = True
            self._cond.notify_all()
        self._thread.join()

    def register(self) -> None:
        with self._cond:
            self._active += 1

    def unregister(self) -> None:
        # A finished game lowers the bar for the fire condition, so the server
        # must re-check: the remaining games may now all be waiting.
        with self._cond:
            self._active -= 1
            self._cond.notify_all()

    def infer(
        self, state_vec: np.ndarray, choice_feats: np.ndarray, family_idx: int
    ) -> tuple[np.ndarray, float]:
        """Submit one decision and block until the batched forward fills it.

        Returns ``(probs, value)`` where ``probs`` is the per-candidate softmax
        distribution and ``value`` is the critic scalar (normalized-return units)
        — both filled by :func:`_fill_probs_batch` before the event is set."""
        request = _InferenceRequest(state_vec, choice_feats, family_idx)
        with self._cond:
            self._pending.append(request)
            # Only wake the server once the batch is actually complete (every
            # live game parked), not on every submit — avoids N-1 spurious
            # wakeups per round.
            if len(self._pending) >= self._active:
                self._cond.notify_all()
        request.done.wait()
        return request.probs, request.value

    def _serve(self) -> None:
        while True:
            with self._cond:
                # Fire only when every live game is parked on a request (so the
                # batch covers all in-flight games), never on an empty set.
                while not self._stopping and (
                    self._active == 0 or len(self._pending) < self._active
                ):
                    self._cond.wait()
                if self._stopping and not self._pending:
                    return
                batch = self._pending
                self._pending = []
            _fill_probs_batch(self._net, self._device, batch)
            for request in batch:
                request.done.set()


def _run_wave(
    net: model.PolicyValueNet,
    device: torch.device,
    server: _BatchInferenceServer,
    seeds: typing.Sequence[int],
    indices: list[int],
    results: list[collect.GameRecord | None],
    on_game_done: typing.Callable[[collect.GameRecord], None] | None,
    done_lock: threading.Lock,
    vs_random: bool,
) -> None:
    """Play one wave of games (one thread each) to completion."""

    def run_one(slot: int) -> None:
        server.register()
        try:
            record = _play_one_game(net, device, server, seeds[slot], vs_random)
        finally:
            server.unregister()
        results[slot] = record
        if on_game_done is not None:
            with done_lock:
                on_game_done(record)

    threads = [
        threading.Thread(target=run_one, args=(slot,), name=f"wingspan-game-{slot}")
        for slot in indices
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def _play_one_game(
    net: model.PolicyValueNet,
    device: torch.device,
    server: _BatchInferenceServer,
    seed: int,
    vs_random: bool,
) -> collect.GameRecord:
    """Play a single game whose policy queries route through the shared batch
    server. Mirrors :func:`collect.play_game` exactly apart from the batched
    inference path: self-play by default, or — when ``vs_random`` — the net's
    recording agent at seat 0 against an off-server random agent at seat 1."""
    eng = collect.new_engine(seed)
    recorded: list[steps.Step] = []
    sample_rng = random.Random(seed ^ _SAMPLE_RNG_SALT)
    net_agent = _batched_recording_agent(server, sample_rng, recorded)
    agent_a, agent_b = (
        (net_agent, agents.random_agent(random.Random(seed ^ _OPPONENT_RNG_SALT)))
        if vs_random
        else (net_agent, net_agent)
    )
    engine.Engine.play_one_game(eng.state, (agent_a, agent_b))
    timestamps.finalize_timestamps(recorded)

    breakdowns = (
        collect.player_breakdown(eng.state.players[0]),
        collect.player_breakdown(eng.state.players[1]),
    )
    score_0, score_1 = breakdowns[0].total, breakdowns[1].total
    winner = 0 if score_0 > score_1 else (1 if score_1 > score_0 else -1)
    return collect.GameRecord(
        steps=recorded,
        breakdowns=breakdowns,
        winner=winner,
        seed=seed,
        final_timestamp=timestamps.final_timestamp(eng.state.turn_counter),
    )


def _batched_recording_agent(
    server: _BatchInferenceServer,
    rng: random.Random,
    record_into: list[steps.Step],
) -> engine.Agent:
    """An agent that routes every multi-option decision's forward pass through
    ``server`` and records the chosen step — the batched twin of
    ``collect._recording_agent``."""

    def agent[C: decisions.Choice](
        eng: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        if len(decision.choices) == 1:
            return decision.choices[0]
        if not server.spec.include_setup and decisions.is_setup_decision(decision):
            return decisions.random_choice(decision, eng.state.rng)
        family_idx = decisions.family_index_for(type(decision))
        state_vec = server.encode_state(eng.state, decision)
        choice_feats = server.encode_choices(decision, eng.state)
        n_choices = choice_feats.shape[0]
        probs, value = server.infer(state_vec, choice_feats, family_idx)
        chosen_idx = policy.sample_index_from_probs(probs, n_choices, rng)
        record_into.append(
            steps.Step(
                state=state_vec,
                choices=choice_feats,
                chosen_idx=chosen_idx,
                player_id=decision.player_id,
                family_idx=family_idx,
                margin_before=collect.running_margin(eng.state, decision.player_id),
                score_before=collect.running_own_score(eng.state, decision.player_id),
                timestamp=timestamps.provisional_timestamp(
                    decision, eng.state.turn_counter
                ),
                behavior_logp=policy.behavior_logp(probs, chosen_idx, n_choices),
                value_pred=value,
            )
        )
        return decision.choices[chosen_idx]

    return agent


def _fill_probs_batch(
    net: model.PolicyValueNet,
    device: torch.device,
    batch: list[_InferenceRequest],
) -> None:
    """Run one padded, masked forward pass over ``batch`` and write each
    request's softmax row (trimmed to its real choice count) into ``.probs``.

    Padding rows are masked to zero probability by the model's own -inf masking,
    so trimming to ``[:k]`` recovers exactly the single-decision softmax."""
    batch_size = len(batch)
    max_k = max(request.choice_feats.shape[0] for request in batch)
    state_dim = batch[0].state_vec.shape[0]
    choice_dim = batch[0].choice_feats.shape[1]

    states = np.zeros((batch_size, state_dim), dtype=np.float32)
    choices = np.zeros((batch_size, max_k, choice_dim), dtype=np.float32)
    mask = np.zeros((batch_size, max_k), dtype=np.float32)
    families = np.zeros((batch_size,), dtype=np.int64)
    for i, request in enumerate(batch):
        k = request.choice_feats.shape[0]
        states[i] = request.state_vec
        choices[i, :k] = request.choice_feats
        mask[i, :k] = 1.0
        families[i] = request.family_idx

    with torch.no_grad():
        logits, value_t = net(
            torch.tensor(states, dtype=torch.float32, device=device),
            torch.tensor(choices, dtype=torch.float32, device=device),
            torch.tensor(mask, dtype=torch.float32, device=device),
            torch.tensor(families, dtype=torch.long, device=device),
        )
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        values = value_t.cpu().numpy()

    for i, request in enumerate(batch):
        k = request.choice_feats.shape[0]
        request.probs = probs[i, :k]
        request.value = float(values[i])
