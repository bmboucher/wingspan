"""The compact monitoring snapshot of a run — the one small object the monitor reads.

``RunStatus`` is a deliberately tiny projection of the live
:class:`~wingspan.training.runstate.RunState`: just the values a roster view
needs (progress toward the target, total games, average score, win-rate vs the
current challenger, ETA) plus a heartbeat. The cloud runner refreshes it
frequently and uploads it as ``status.json``, so the monitor can read every run
at a glance without torch-loading any checkpoint.
"""

from __future__ import annotations

import datetime

import pydantic

from wingspan.training import metrics, runstate


class RunStatus(pydantic.BaseModel):
    """A small, self-contained snapshot of one run's progress and health."""

    run_name: str
    phase: str  # runstate.Phase value (collecting / evaluating / done / ...)
    training_phase: str  # runstate.TrainingPhase value (random_opponent / self_play)
    iteration: int  # 0-based index of the current/last iteration
    completed_iterations: int  # iterations fully finished (0 before the first)
    target_iterations: int
    max_iterations: int
    pct_complete: float  # 0..100 toward the target (or max_iterations) milestone
    total_games: int
    total_decisions: int
    avg_score: float  # mean final score per player-game since the run started
    win_rate: float | None  # current strength vs the challenger (EWMA; 0..1)
    win_rate_ci95: float | None  # 95% half-width of the most recent eval, if any
    opponent_generation: int
    opponent_label: str  # "random" | "self·genN"
    best_win_rate: float | None
    games_per_sec: float | None  # collection throughput of the last iteration
    elapsed_seconds: float
    eta_seconds: float | None  # estimated time to the target, if a rate is known
    git_sha: str | None
    error: str | None  # the traceback head when the run crashed, else None
    finished: bool  # phase is terminal (done / stopped / error)
    final_eval: metrics.FinalEvalStats | None  # the large fixed-model eval, once landed
    started_at: str  # ISO-8601 (UTC) of this session's start
    updated_at: str  # ISO-8601 (UTC) heartbeat — staleness => not in-flight
    status_interval_seconds: (
        float  # the refresh cadence (lets the monitor judge freshness)
    )


def build_status(
    state: runstate.RunState,
    *,
    run_name: str,
    started_at: str,
    status_interval_seconds: float,
    git_sha: str | None,
) -> RunStatus:
    """Project the live :class:`runstate.RunState` into a :class:`RunStatus`.

    Reuses the run-state's own reader-side derivations (``avg_total_score``,
    ``time_remaining_seconds``, ``eval_ewma``, ...) so the snapshot matches what
    the dashboard would show. Call it under the loop's lock — every access here
    is a cheap pure computation; the (slow) upload happens outside the lock.
    """
    completed = state.iteration + 1 if state.last_iter is not None else 0
    win_rate, win_rate_ci95 = _strength(state)
    return RunStatus(
        run_name=run_name,
        phase=state.phase.value,
        training_phase=state.training_phase.value,
        iteration=state.iteration,
        completed_iterations=completed,
        target_iterations=state.target_iterations,
        max_iterations=state.config.max_iterations,
        pct_complete=_pct_complete(completed, state),
        total_games=state.total_games,
        total_decisions=state.total_decisions,
        avg_score=state.avg_total_score(),
        win_rate=win_rate,
        win_rate_ci95=win_rate_ci95,
        opponent_generation=state.opponent_generation,
        opponent_label=_opponent_label(state.opponent_generation),
        best_win_rate=state.best_win_rate,
        games_per_sec=(
            state.last_iter.games_per_sec if state.last_iter is not None else None
        ),
        elapsed_seconds=state.elapsed(),
        eta_seconds=state.time_remaining_seconds(),
        git_sha=git_sha,
        error=state.error,
        finished=state.phase.is_terminal,
        final_eval=state.pinned_stats,
        started_at=started_at,
        updated_at=datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        status_interval_seconds=status_interval_seconds,
    )


###### PRIVATE #######


def _opponent_label(generation: int) -> str:
    """The short challenger name (mirrors ``loop.TrainingLoop._opponent_label``)."""
    return "random" if generation == 0 else f"self·gen{generation}"


def _pct_complete(completed: int, state: runstate.RunState) -> float:
    """Percent toward the target milestone, falling back to ``max_iterations``."""
    denom = state.target_iterations or state.config.max_iterations
    if denom <= 0:
        return 0.0
    return min(100.0, 100.0 * completed / denom)


def _strength(state: runstate.RunState) -> tuple[float | None, float | None]:
    """Current win-rate vs the challenger and the latest eval's 95% CI half-width.

    Prefers the smoothed eval EWMA against the current opponent; during the
    random-opponent bootstrap (no eval blocks yet) falls back to the smoothed
    collection win-rate vs random. The CI comes from the most recent eval played
    against the current opponent generation, or None when none has landed.
    """
    ewma = state.eval_ewma()
    if ewma is not None:
        win_rate: float | None = ewma.win_rate
    elif state.training_phase is runstate.TrainingPhase.RANDOM_OPPONENT:
        win_rate = state.collection_win_rate_ewma()
    else:
        win_rate = None
    latest = _latest_eval(state)
    return win_rate, (latest.ci95 if latest is not None else None)


def _latest_eval(state: runstate.RunState) -> metrics.EvalResult | None:
    """The most recent eval in history against the current opponent generation."""
    for item in reversed(state.history):
        if (
            item.eval is not None
            and item.eval.opponent_generation == state.opponent_generation
        ):
            return item.eval
    return None
