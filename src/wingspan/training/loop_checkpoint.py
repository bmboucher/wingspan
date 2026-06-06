# pyright: reportPrivateUsage=false
# (accesses TrainingLoop's private fields — deliberate intra-package coupling)
"""Checkpointing, seeding, and run-finish helpers for ``TrainingLoop``.

Free functions whose first argument is a ``TrainingLoop`` instance handle the
end-of-iteration commit, the main/best/setup checkpoint writes, game-history
appending, and run teardown.  Atomic I/O primitives (``atomic_save``,
``atomic_write_text``), seeding (``seed_everything``), and git metadata
(``git_sha``) live here as parameter-free helpers used across several of the
``loop_*.py`` sibling modules.
"""

from __future__ import annotations

import os
import pathlib
import random
import subprocess
import time
import typing

import numpy as np
import torch

from wingspan import version
from wingspan.training import (
    artifacts,
    collect,
    learner,
    loop_eval,
    loop_metrics,
    loop_setup,
    metrics,
    runstate,
)

if typing.TYPE_CHECKING:
    from wingspan.training import loop


def commit_iteration(
    training_loop: "loop.TrainingLoop",
    iter_metrics: metrics.IterationMetrics,
    stats: learner.UpdateStats,
    eval_result: metrics.EvalResult | None,
    records: list[collect.GameRecord],
) -> None:
    """Close out one training iteration: update state, graduate/advance the
    opponent, write the checkpoint, and update per-phase timing counters."""
    # Capture the phase before graduation fires so timing lands in the
    # right bucket (graduation mutates training_phase inside this call).
    iter_phase_for_timing: runstate.TrainingPhase
    with training_loop.lock:
        iter_phase_for_timing = training_loop.state.training_phase
        training_loop.state.last_iter = iter_metrics
        training_loop.state.history.append(iter_metrics)
        cap = training_loop.config.history_len
        if len(training_loop.state.history) > cap:
            del training_loop.state.history[: len(training_loop.state.history) - cap]
        if not np.isfinite(stats.loss):
            training_loop.state.push_event(
                runstate.EventKind.ALARM,
                f"non-finite loss at iter {iter_metrics.iteration}",
            )
        if eval_result is not None:
            training_loop.state.push_event(
                runstate.EventKind.EVAL,
                f"EVAL {eval_result.n_games} games in "
                f"{iter_metrics.eval_seconds:.1f}s · "
                f"{eval_result.win_rate * 100:.1f}% ±{eval_result.ci95 * 100:.1f}% "
                f"vs {loop_eval.opponent_label(training_loop)} · "
                f"margin {eval_result.mean_margin:+.1f}",
            )

    # Graduate out of the bootstrap phase (freezes self·gen1) or advance the
    # frozen opponent before checkpointing, so ``last.pt`` records the new
    # generation / phase alongside the matching ``opponent.pt`` snapshot.
    loop_eval.maybe_graduate_from_random_phase(training_loop)
    loop_eval.maybe_advance_opponent(training_loop, eval_result)

    with training_loop.lock:
        training_loop.state.phase = runstate.Phase.CHECKPOINTING
    checkpoint(training_loop, iter_metrics, eval_result, records)

    # Update per-phase timing counters for the time-to-target estimate.
    iter_secs = time.monotonic() - training_loop.state.iter_start_monotonic
    with training_loop.lock:
        if iter_phase_for_timing == runstate.TrainingPhase.RANDOM_OPPONENT:
            training_loop.state.random_phase_iter_count += 1
            training_loop.state.random_phase_seconds += iter_secs
        else:
            training_loop.state.self_play_iter_count += 1
            training_loop.state.self_play_seconds += iter_secs


def checkpoint(
    training_loop: "loop.TrainingLoop",
    iter_metrics: metrics.IterationMetrics,
    eval_result: metrics.EvalResult | None,
    records: list[collect.GameRecord],
) -> None:
    """Write ``last.pt`` (and ``best.pt`` when improved), log the metrics row,
    append game history, and save the setup checkpoint."""
    training_loop._ckpt_dir.mkdir(parents=True, exist_ok=True)
    with training_loop.lock:
        # "best" is per opponent-generation: the eval that triggers an
        # advancement belongs to the old opponent (its generation no longer
        # matches), so it is not credited as the new generation's best.
        improved = (
            eval_result is not None
            and eval_result.opponent_generation
            == training_loop.state.opponent_generation
            and (
                training_loop.state.best_win_rate is None
                or eval_result.win_rate > training_loop.state.best_win_rate
            )
        )
        prev_best = training_loop.state.best_win_rate
        if improved and eval_result is not None:
            training_loop.state.best_win_rate = eval_result.win_rate
        # Snapshot the resumable progress so a later run picks up exactly here.
        progress = training_loop.state.to_progress()
    payload: dict[str, object] = {
        "config": training_loop.config.model_dump(),
        "model": training_loop.net.state_dict(),
        "optimizer": training_loop.optimizer.state_dict(),
        "metrics": iter_metrics.model_dump(),
        "progress": progress.model_dump(),
        "git_sha": git_sha(),
        "version": version.MODEL_VERSION,
    }
    atomic_save(payload, training_loop._ckpt_dir / artifacts.LAST_CKPT)
    if improved and eval_result is not None:
        atomic_save(payload, training_loop._ckpt_dir / artifacts.BEST_CKPT)
    # The setup net resumes from its own checkpoint, written alongside last.pt.
    loop_setup.save_setup_checkpoint(training_loop)

    with training_loop.lock:
        if improved and eval_result is not None:
            prev_txt = (
                f" > prev {prev_best * 100:.1f}%" if prev_best is not None else ""
            )
            training_loop.state.push_event(
                runstate.EventKind.BEST,
                f"new {artifacts.BEST_CKPT} "
                f"(eval {eval_result.win_rate * 100:.1f}%{prev_txt})",
            )

    with open(
        training_loop._ckpt_dir / artifacts.METRICS_LOG, "a", encoding="utf-8"
    ) as handle:
        handle.write(iter_metrics.model_dump_json() + "\n")

    # Per-game history is appended after ``last.pt`` (written above) so a
    # crash between the two only ever loses this iteration's rows rather than
    # duplicating them on the resume that re-plays the un-checkpointed cycle.
    append_game_history(
        training_loop,
        loop_metrics.build_game_outcomes(records, iter_metrics.iteration),
    )


def append_game_history(
    training_loop: "loop.TrainingLoop", outcomes: list[metrics.GameOutcome]
) -> None:
    """Append one ``games.jsonl`` line per finished game (a single buffered
    write per iteration — ~256 lines every few seconds — so the per-game log
    never becomes a throughput drag)."""
    if not outcomes:
        return
    rows = "".join(outcome.model_dump_json() + "\n" for outcome in outcomes)
    with open(
        training_loop._ckpt_dir / artifacts.GAMES_LOG, "a", encoding="utf-8"
    ) as handle:
        handle.write(rows)


def finish(training_loop: "loop.TrainingLoop", phase: runstate.Phase) -> None:
    """Transition the run into its terminal phase and record the stop time."""
    with training_loop.lock:
        training_loop.state.phase = phase
        training_loop.state.stopped_monotonic = time.monotonic()
        training_loop.state.push_event(
            runstate.EventKind.INFO,
            (
                "run stopped by user"
                if phase is runstate.Phase.STOPPED
                else "run complete"
            ),
        )


###### PRIVATE #######

#### Seeding ####


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and torch (TRAINING.md §5 reproducibility)."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    # torch's seeding stubs are typed with unknown parameters; suppress the
    # stub-gap report narrowly rather than leaving the seed unset (TRAINING.md §5).
    torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # pyright: ignore[reportUnknownMemberType]


#### Checkpoint I/O ####


def atomic_save(payload: dict[str, object], path: pathlib.Path) -> None:
    """Write a checkpoint to a temp file then ``os.replace`` it into place so a
    crash mid-write never corrupts the destination (TRAINING.md §5.2)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def atomic_write_text(text: str, path: pathlib.Path) -> None:
    """Write text to a temp file then ``os.replace`` it into place, so a crash
    mid-write never leaves a partial JSON sidecar."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def git_sha() -> str | None:
    """Best-effort short git SHA of the working tree (None if unavailable)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        return result.stdout.strip() or None if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None
