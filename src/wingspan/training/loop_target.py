# pyright: reportPrivateUsage=false
# (accesses TrainingLoop's private fields — deliberate intra-package coupling)
"""Target-milestone helpers for ``TrainingLoop``.

Free functions whose first argument is a ``TrainingLoop`` instance execute
the full target-milestone sequence: a final checkpoint, a large fixed-model
self-play eval, and a pause for the dashboard's [C]ontinue / [E]nd keypress
(or a headless "end" on the cloud runner).
"""

from __future__ import annotations

import typing

from wingspan.training import artifacts, evaluate, loop_checkpoint, runstate

if typing.TYPE_CHECKING:
    from wingspan.training import loop


def handle_target_if_reached(
    training_loop: "loop.TrainingLoop", iteration: int
) -> bool:
    """Check whether the target milestone was reached after ``iteration``.

    If so, run the full target sequence (final checkpoint → large eval →
    pause for user input) and return ``True`` iff the user chose "end".
    Returns ``False`` immediately when no target is set or the target has
    not been reached yet.
    """
    with training_loop.lock:
        target = training_loop.state.target_iterations
    if target <= 0 or (iteration + 1) < target:
        return False
    handle_target_reached(training_loop, iteration)
    with training_loop.lock:
        return training_loop.state.user_target_choice == "end"


def handle_target_reached(training_loop: "loop.TrainingLoop", iteration: int) -> None:
    """Execute the target-milestone sequence: checkpoint → eval → pause."""
    # Step 1: save the final milestone checkpoint (same payload as last.pt).
    final_name = artifacts.final_ckpt_name(iteration + 1)
    training_loop._ckpt_dir.mkdir(parents=True, exist_ok=True)
    with training_loop.lock:
        training_loop.state.push_event(
            runstate.EventKind.CHECKPOINT,
            f"target {training_loop.state.target_iterations:,} reached "
            f"→ saving {final_name}",
        )
        progress = training_loop.state.to_progress()
    payload: dict[str, object] = {
        "config": training_loop.config.model_dump(),
        "model": training_loop.net.state_dict(),
        "optimizer": training_loop.optimizer.state_dict(),
        "progress": progress.model_dump(),
        "git_sha": loop_checkpoint.git_sha(),
    }
    loop_checkpoint.atomic_save(payload, training_loop._ckpt_dir / final_name)
    with training_loop.lock:
        training_loop.state.push_event(
            runstate.EventKind.CHECKPOINT, f"saved {final_name}"
        )

    # Step 2: run the large fixed-model self-play eval.
    n_eval = training_loop.config.effective_target_eval_games
    with training_loop.lock:
        training_loop.state.phase = runstate.Phase.FINAL_EVALUATING
        training_loop.state.final_eval_progress = (0, n_eval)

    def _on_progress(done: int, total: int) -> None:
        with training_loop.lock:
            training_loop.state.final_eval_progress = (done, total)

    final_stats = evaluate.run_final_self_play_eval(
        training_loop.net,
        training_loop.device,
        n_games=n_eval,
        seed=training_loop.config.seed + iteration * 1000,
        at_iteration=iteration + 1,
        on_progress=_on_progress,
        split_setup_bonus=training_loop.config.split_setup_bonus_active,
        split_setup_food=training_loop.config.split_setup_food_active,
    )

    # Persist the final-eval result beside ``final_<n>.pt`` so it is a
    # durable artifact (the cloud runner uploads it to its own S3 object)
    # rather than a dashboard-only readout.
    eval_name = artifacts.final_eval_name(iteration + 1)
    loop_checkpoint.atomic_write_text(
        final_stats.model_dump_json(), training_loop._ckpt_dir / eval_name
    )
    with training_loop.lock:
        training_loop.state.push_event(
            runstate.EventKind.CHECKPOINT, f"saved {eval_name}"
        )

    # Step 3: pin the eval stats. The dashboard pauses for [C]ontinue / [E]nd
    # input; the headless runner instead records an "end" choice so the run
    # finalizes and exits at this milestone.
    with training_loop.lock:
        training_loop.state.pinned_stats = final_stats
        if training_loop._pause_at_target:
            training_loop.state.phase = runstate.Phase.PAUSED_AT_TARGET
            training_loop.state.user_target_choice = None
        else:
            training_loop.state.user_target_choice = "end"
        training_loop.state.push_event(
            runstate.EventKind.EVAL,
            f"final eval {n_eval} games · "
            f"avg {final_stats.avg_breakdown.total:.1f} pts · "
            f"margin {final_stats.mean_margin:.1f} pts",
        )

    # Step 4: the dashboard blocks until a [C]ontinue / [E]nd keypress, then
    # resumes or ends. The headless runner has already chosen "end", so it
    # returns straight to the run loop, which sees the choice and stops.
    if not training_loop._pause_at_target:
        return
    training_loop._target_reached_event.wait()
    training_loop._target_reached_event.clear()

    # Step 5: if continuing, clear pinned stats so EWMA picks up.
    resume_after_target(training_loop)


def resume_after_target(training_loop: "loop.TrainingLoop") -> None:
    """Clear the pinned stats and resume COLLECTING after a target milestone.

    Separated from :func:`handle_target_reached` so pyright does not
    flow-narrow ``user_target_choice`` to ``None`` across the threading
    boundary where it is explicitly reset before waiting on the event.
    """
    with training_loop.lock:
        if training_loop.state.user_target_choice == "continue":
            training_loop.state.pinned_stats = None
            training_loop.state.phase = runstate.Phase.COLLECTING
