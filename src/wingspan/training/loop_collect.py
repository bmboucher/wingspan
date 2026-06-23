# pyright: reportPrivateUsage=false
# (accesses TrainingLoop's private fields — deliberate intra-package coupling)
"""Collection helpers for ``TrainingLoop``.

Free functions whose first argument is a ``TrainingLoop`` instance orchestrate
batched self-play collection — fanning across worker processes on CPU, running
the in-process batched-inference path on CUDA, and handling the
setup-model-aware variants for both devices.
"""

from __future__ import annotations

import random
import typing

from wingspan import agents, setup_model
from wingspan.training import (
    batched_collect,
    collect,
    loop_metrics,
    mp_collect,
    runstate,
)

if typing.TYPE_CHECKING:
    from wingspan.training import loop

# Salt for the sequential (non-CPU) setup-collection path's random opponent,
# matching the role ``mp_collect._OPPONENT_RNG_SALT`` plays on the CPU path.
_SEQ_SETUP_OPPONENT_SALT = 0x85EBCA6B


def collect_games(
    training_loop: "loop.TrainingLoop",
    iteration: int,
    setup_enabled: bool,
    dagger_active: bool = False,
) -> list[collect.GameRecord]:
    """Play ``games_per_iter`` games with batched inference, updating the
    live state as each game finishes so the dashboard advances mid-iteration.

    Games run concurrently and complete out of order; the per-game callback
    runs under ``training_loop.lock`` so the shared state stays consistent.
    In the bootstrap phase the games are net-vs-random rather than self-play.

    When the setup model is enabled (``setup_enabled``), setups are chosen by
    the setup net via the setup-aware collection path.

    ``dagger_active`` signals the DAgger clone phase: each decision is labeled
    with the frozen expert's soft policy distribution.  Only supported on CPU
    (the DAgger validator enforces ``device='cpu'``).
    """
    vs_random = (
        training_loop.state.training_phase == runstate.TrainingPhase.RANDOM_OPPONENT
    )
    if setup_enabled:
        return collect_with_setup(training_loop, iteration, vs_random, dagger_active)
    seeds = [
        training_loop.config.misc.seed * 1_000_000 + iteration * 10_000 + game_idx
        for game_idx in range(training_loop.config.run.games_per_iter)
    ]
    # CPU collection is GIL-bound under threads, so it fans across worker
    # processes; CUDA collection keeps the in-process batched-inference path
    # (one shared GPU forward beats one model copy per process).
    if training_loop.device.type == "cpu":
        return collect_multiprocess(training_loop, seeds, vs_random, dagger_active)
    return batched_collect.collect_games(
        training_loop.net,
        training_loop.device,
        seeds,
        on_game_done=lambda rec: record_collected_game(training_loop, rec),
        should_stop=training_loop._stop.is_set,
        vs_random=vs_random,
    )


def collect_with_setup(
    training_loop: "loop.TrainingLoop",
    iteration: int,
    vs_random: bool,
    dagger_active: bool = False,
) -> list[collect.GameRecord]:
    """Collect games whose setups are chosen by the setup net.

    CPU fans across the worker pool (as ordinary collection does); the non-CPU
    path runs the games sequentially in-process (the batched CUDA collector
    does not implement the setup path — training is CPU-anyway).
    """
    specs = collect.build_setup_specs(training_loop.config, iteration)
    if training_loop.device.type == "cpu":
        return ensure_collector(training_loop).collect_games_with_setup(
            training_loop.net,
            training_loop._setup_net,
            training_loop.device,
            specs,
            on_game_done=lambda rec: record_collected_game(training_loop, rec),
            should_stop=training_loop._stop.is_set,
            vs_random=vs_random,
            dagger_active=dagger_active,
        )
    return collect_with_setup_sequential(training_loop, specs, vs_random)


def collect_with_setup_sequential(
    training_loop: "loop.TrainingLoop",
    specs: list[collect.SetupGameSpec],
    vs_random: bool,
) -> list[collect.GameRecord]:
    """In-process setup collection (the non-CPU fallback)."""
    generator = setup_model.RandomSetupGenerator(
        hand_combos=training_loop.config.training.setup.hand_combos,
        food_sets=training_loop.config.training.setup.food_sets,
        split_food=training_loop.config.split_setup_food_active,
    )
    records: list[collect.GameRecord] = []
    for spec in specs:
        if training_loop._stop.is_set():
            break
        opponent = (
            agents.random_agent(
                random.Random(spec.continuation_seed ^ _SEQ_SETUP_OPPONENT_SALT)
            )
            if vs_random
            else None
        )
        record = collect.play_game_with_setup(
            training_loop.net,
            training_loop.device,
            spec,
            generator,
            training_loop._setup_net,
            training_loop.config.training.setup.policy_temperature,
            opponent,
            split_setup_bonus=training_loop.config.split_setup_bonus_active,
            split_setup_food=training_loop.config.split_setup_food_active,
            setup_greedy=training_loop.config.training.setup.policy_greedy,
        )
        records.append(record)
        record_collected_game(training_loop, record)
    return records


def collect_multiprocess(
    training_loop: "loop.TrainingLoop",
    seeds: list[int],
    vs_random: bool,
    dagger_active: bool = False,
) -> list[collect.GameRecord]:
    """Collect across worker processes; the pool is built on first use and
    reused across iterations (closed in ``run``'s teardown)."""
    return ensure_collector(training_loop).collect_games(
        training_loop.net,
        training_loop.device,
        seeds,
        on_game_done=lambda rec: record_collected_game(training_loop, rec),
        should_stop=training_loop._stop.is_set,
        vs_random=vs_random,
        dagger_active=dagger_active,
    )


def ensure_collector(
    training_loop: "loop.TrainingLoop",
) -> mp_collect.ProcessCollector:
    """The shared worker pool, built on first use and reused for both
    collection and evaluation across iterations."""
    if training_loop._collector is None:
        training_loop._collector = mp_collect.ProcessCollector(training_loop.config)
    return training_loop._collector


def record_collected_game(
    training_loop: "loop.TrainingLoop", record: collect.GameRecord
) -> None:
    """Fold one finished self-play game into the live dashboard state."""
    with training_loop.lock:
        training_loop.state.record_game(
            record.breakdowns,
            len(record.steps),
            loop_metrics.family_counts(record),
            record.winner,
        )
        training_loop.state.game_in_iter += 1
