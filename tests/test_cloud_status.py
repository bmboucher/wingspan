"""Tests for the compact ``RunStatus`` snapshot the monitor reads.

``build_status`` projects the live ``RunState`` into the small object uploaded as
``status.json``; these confirm the progress/opponent derivations and that a
finished run carries its final-eval result.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

pytest.importorskip("torch")
pytest.importorskip("boto3")

from wingspan.cloud import status
from wingspan.training import config, metrics, runstate


def _iteration_metrics(
    iteration: int, games_per_sec: float
) -> metrics.IterationMetrics:
    """A minimal completed-iteration row (only the fields without defaults)."""
    return metrics.IterationMetrics(
        iteration=iteration,
        total_games=0,
        games_this_iter=16,
        loss=0.0,
        policy_loss=0.0,
        value_loss=0.0,
        entropy=0.0,
        grad_norm=0.0,
        advantage_mean=0.0,
        advantage_std=0.0,
        avg_self_score=0.0,
        avg_margin=0.0,
        avg_breakdown=metrics.ScoreBreakdown(),
        avg_decisions=0.0,
        avg_winner_breakdown=metrics.ScoreBreakdown(),
        avg_abs_margin=0.0,
        margin_std=0.0,
        abs_margin_std=0.0,
        decisions_std=0.0,
        family_counts=metrics.FamilyCounts(),
        collect_seconds=1.0,
        update_seconds=0.1,
        eval_seconds=0.0,
        games_per_sec=games_per_sec,
    )


def test_build_status_progress_and_opponent() -> None:
    cfg = config.TrainConfig(games_per_iter=16, max_iterations=10, target_iterations=10)
    state = runstate.new_run_state(cfg)
    state.iteration = 4
    state.last_iter = _iteration_metrics(4, games_per_sec=12.5)
    state.total_games = 80
    state.total_decisions = 1000
    state.opponent_generation = 2

    snapshot = status.build_status(
        state,
        run_name="r",
        started_at="2026-01-01T00:00:00+00:00",
        status_interval_seconds=30.0,
        git_sha=None,
    )
    assert snapshot.completed_iterations == 5  # iteration index 4 -> 5 finished
    assert snapshot.pct_complete == 50.0
    assert snapshot.opponent_label == "self·gen2"
    assert snapshot.games_per_sec == 12.5
    assert snapshot.total_games == 80
    assert snapshot.finished is False
    assert snapshot.win_rate is None  # no eval blocks recorded yet


def test_build_status_finished_carries_final_eval() -> None:
    cfg = config.TrainConfig(games_per_iter=16, max_iterations=4, target_iterations=4)
    state = runstate.new_run_state(cfg)
    state.phase = runstate.Phase.DONE
    state.pinned_stats = metrics.FinalEvalStats(
        n_games=40,
        avg_breakdown=metrics.ScoreBreakdown(birds=10.0),
        avg_winner_breakdown=metrics.ScoreBreakdown(birds=12.0),
        decisions_per_game=130.0,
        mean_margin=3.5,
        self_play_win_rate=0.5,
        at_iteration=4,
    )
    snapshot = status.build_status(
        state,
        run_name="r",
        started_at="t",
        status_interval_seconds=30.0,
        git_sha="abc123",
    )
    assert snapshot.finished is True
    assert snapshot.final_eval is not None
    assert snapshot.final_eval.n_games == 40
    assert snapshot.git_sha == "abc123"
