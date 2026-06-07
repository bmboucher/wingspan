# pyright: reportPrivateUsage=false
# (accesses TrainingLoop's private fields — deliberate intra-package coupling)
"""Evaluation, opponent advancement, and clone/save/load helpers for ``TrainingLoop``.

Free functions whose first argument is a ``TrainingLoop`` instance run the
periodic paired-game eval against the reference opponent, manage the opponent
lifecycle (graduation from the random phase, win-rate / time-cap advancement),
and clone/persist/restore the frozen reference net.
"""

from __future__ import annotations

import time
import typing

import torch

from wingspan import model, version
from wingspan.training import (
    artifacts,
    evaluate,
    loop_checkpoint,
    loop_collect,
    metrics,
    runstate,
)

if typing.TYPE_CHECKING:
    from wingspan.training import loop


def maybe_evaluate(
    training_loop: "loop.TrainingLoop", iteration: int
) -> tuple[metrics.EvalResult | None, float]:
    """Run the periodic paired-game eval; return ``(result, elapsed_seconds)``.

    The bootstrap phase reads strength from the collection win-rate vs random,
    so the separate eval block is paused until it graduates.  Returns
    ``(None, 0.0)`` in all skip cases.
    """
    if training_loop.state.training_phase == runstate.TrainingPhase.RANDOM_OPPONENT:
        return None, 0.0
    if (
        training_loop.config.eval_every <= 0
        or iteration % training_loop.config.eval_every != 0
    ):
        return None, 0.0
    with training_loop.lock:
        training_loop.state.phase = runstate.Phase.EVALUATING
        training_loop.state.eval_game_in_iter = 0
        training_loop.state.eval_games_in_iter = 2 * training_loop.config.eval_pairs
    start = time.monotonic()
    eval_seed = training_loop.config.seed * 7919 + iteration * 101 + 1
    # CPU eval fans across the same worker pool collection uses; CUDA keeps
    # the in-process sequential path (one shared GPU beats a model per
    # process). Both paths run identical per-game logic, so results match.
    if training_loop.device.type == "cpu":
        result = loop_collect.ensure_collector(training_loop).evaluate_games(
            training_loop.net,
            training_loop._opponent_net,
            training_loop.device,
            training_loop.config.eval_pairs,
            eval_seed,
            opponent_generation=training_loop.state.opponent_generation,
            on_progress=lambda done, total: record_eval_progress(
                training_loop, done, total
            ),
        )
    else:
        result = evaluate.evaluate_vs_opponent(
            training_loop.net,
            training_loop._opponent_net,
            training_loop.device,
            training_loop.config.eval_pairs,
            eval_seed,
            opponent_generation=training_loop.state.opponent_generation,
            on_progress=lambda done, total: record_eval_progress(
                training_loop, done, total
            ),
            split_setup_bonus=training_loop.config.split_setup_bonus_active,
            split_setup_food=training_loop.config.split_setup_food_active,
        )
    return result, time.monotonic() - start


def record_eval_progress(
    training_loop: "loop.TrainingLoop", games_done: int, total_games: int
) -> None:
    """Publish held-out eval progress so the header bar tracks eval games."""
    with training_loop.lock:
        training_loop.state.eval_game_in_iter = games_done
        training_loop.state.eval_games_in_iter = total_games


def opponent_label(training_loop: "loop.TrainingLoop") -> str:
    """A short name for the current reference opponent (for event messages)."""
    gen = training_loop.state.opponent_generation
    return "random" if gen == 0 else f"self·gen{gen}"


def maybe_graduate_from_random_phase(training_loop: "loop.TrainingLoop") -> None:
    """Leave the random-opponent bootstrap phase once the smoothed collection
    win-rate clears ``config.random_phase_win_rate``.

    Freezes the current policy as the first self-play opponent (self·gen1),
    switches collection to self-play, and resumes evaluation against it.
    A no-op outside the bootstrap phase, so it is safe to call every iteration.
    """
    if training_loop.state.training_phase != runstate.TrainingPhase.RANDOM_OPPONENT:
        return
    threshold = training_loop.config.random_phase_win_rate
    with training_loop.lock:
        ewma = training_loop.state.collection_win_rate_ewma()
        if ewma is None or ewma < threshold:
            return

    frozen = clone_net(training_loop)
    save_opponent(training_loop, frozen, generation=1)
    with training_loop.lock:
        training_loop._opponent_net = frozen
        training_loop.state.training_phase = runstate.TrainingPhase.SELF_PLAY
        training_loop.state.opponent_generation = 1
        training_loop.state.opponent_since_iteration = training_loop.state.iteration
        training_loop.state.opponent_change_iterations.append(
            training_loop.state.iteration
        )
        training_loop.state.best_win_rate = None  # best is per-opponent-generation
        training_loop.state.push_event(
            runstate.EventKind.BEST,
            f"graduated random phase → self·gen1 "
            f"(collection win-rate {ewma * 100:.0f}%) · self-play + eval resume",
        )


def maybe_advance_opponent(
    training_loop: "loop.TrainingLoop", eval_result: metrics.EvalResult | None
) -> None:
    """Advance the frozen reference opponent when either trigger fires.

    - *Win-rate trigger*: the EWMA win-rate against the current opponent
      clears ``config.opponent_reset_win_rate``.
    - *Time trigger*: more than ``config.opponent_max_iterations`` iterations
      have elapsed since the current opponent was set (0 disables the cap).

    Only active in the SELF_PLAY phase.
    """
    if training_loop.state.training_phase != runstate.TrainingPhase.SELF_PLAY:
        return

    threshold = training_loop.config.opponent_reset_win_rate
    max_iters = training_loop.config.opponent_max_iterations

    # Evaluate both triggers under the lock so iteration / EWMA are consistent.
    ewma_snap: metrics.EvalEwma | None
    iters_since: int
    new_generation: int
    with training_loop.lock:
        ewma_snap = training_loop.state.eval_ewma()
        iters_since = (
            training_loop.state.iteration - training_loop.state.opponent_since_iteration
        )
        win_rate_fires = (
            eval_result is not None
            and threshold > 0.0
            and ewma_snap is not None
            and ewma_snap.win_rate >= threshold
        )
        time_fires = max_iters > 0 and iters_since >= max_iters
        if not win_rate_fires and not time_fires:
            return
        new_generation = training_loop.state.opponent_generation + 1

    frozen = clone_net(training_loop)
    save_opponent(training_loop, frozen, new_generation)
    with training_loop.lock:
        training_loop._opponent_net = frozen
        training_loop.state.opponent_generation = new_generation
        training_loop.state.opponent_since_iteration = training_loop.state.iteration
        training_loop.state.opponent_change_iterations.append(
            training_loop.state.iteration
        )
        training_loop.state.best_win_rate = None  # best is per-opponent-generation
        if win_rate_fires and ewma_snap is not None:
            reason = (
                f"beat {prev_opponent_label(new_generation)} "
                f"{ewma_snap.win_rate * 100:.0f}%"
            )
        else:
            reason = f"stalled for {iters_since} iters"
        training_loop.state.push_event(
            runstate.EventKind.BEST,
            f"opponent advanced → self·gen{new_generation} "
            f"({reason}) · win-rate reset",
        )


def prev_opponent_label(new_generation: int) -> str:
    """Short name for the opponent that was just beaten (for event messages)."""
    return "random" if new_generation == 1 else f"self·gen{new_generation - 1}"


def clone_net(training_loop: "loop.TrainingLoop") -> model.PolicyValueNet:
    """An independent, eval-mode copy of the current policy network."""
    clone = model.PolicyValueNet(
        state_dim=training_loop.config.state_dim,
        choice_dim=training_loop.config.choice_dim,
        arch=training_loop.config.arch,
        spec=training_loop.config.encoding_spec,
    ).to(training_loop.device)
    clone.load_state_dict(training_loop.net.state_dict())
    clone.eval()
    return clone


def save_opponent(
    training_loop: "loop.TrainingLoop",
    opponent: model.PolicyValueNet,
    generation: int,
) -> None:
    """Persist the frozen opponent so a resumed run keeps the same reference."""
    training_loop._ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "config": training_loop.config.model_dump(),
        "model": opponent.state_dict(),
        "opponent_generation": generation,
        "git_sha": loop_checkpoint.git_sha(),
        "version": version.MODEL_VERSION,
    }
    loop_checkpoint.atomic_save(
        payload, training_loop._ckpt_dir / artifacts.OPPONENT_CKPT
    )


def load_opponent(training_loop: "loop.TrainingLoop") -> None:
    """Restore the frozen opponent from ``opponent.pt`` on resume.

    If it is missing or unreadable, fall back to the random agent (generation
    0) so the run stays consistent rather than evaluating against nothing.
    """
    path = training_loop._ckpt_dir / artifacts.OPPONENT_CKPT
    try:
        payload = typing.cast(
            "dict[str, typing.Any]",
            torch.load(path, map_location=training_loop.device, weights_only=False),
        )
        opponent = clone_net(training_loop)
        opponent.load_state_dict(payload["model"])
        opponent.eval()
    except Exception:  # noqa: BLE001 — a missing/corrupt opponent resets to random
        training_loop.state.opponent_generation = 0
        training_loop.state.push_event(
            runstate.EventKind.ALARM,
            f"could not read {artifacts.OPPONENT_CKPT} — opponent reset to random",
        )
        return
    training_loop._opponent_net = opponent
