"""Tests for the training/evaluate module: summarize_eval, evaluate_vs_opponent,
and run_final_self_play_eval (which exercises _counting_greedy_agent internally)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from wingspan import model
from wingspan.training import evaluate


def test_summarize_eval_empty_margins_returns_zero_result():
    """An empty game list hits the early-exit path and returns a zero EvalResult."""
    result = evaluate.summarize_eval([], opponent_generation=7)
    assert result.n_games == 0
    assert result.win_rate == 0.0
    assert result.ci95 == 0.0
    assert result.mean_margin == 0.0
    assert result.opponent_generation == 7


def test_summarize_eval_computes_correct_statistics():
    """Win (positive margin) = 1 pt, tie = 0.5, loss = 0; CI ≥ 0; mean margin is exact."""
    # margins=[10, -5, 0]: wins = 1 + 0 + 0.5 = 1.5, win_rate = 0.5
    result = evaluate.summarize_eval([10, -5, 0], opponent_generation=0)
    assert result.n_games == 3
    assert abs(result.win_rate - 0.5) < 1e-6
    assert result.ci95 >= 0.0
    assert abs(result.mean_margin - (10 - 5 + 0) / 3) < 1e-6


def test_evaluate_vs_opponent_calls_on_progress_after_each_game():
    """on_progress(games_done, total) is invoked after every game; the returned
    EvalResult carries valid statistics for 1 pair (2 mirrored games)."""
    net = model.PolicyValueNet()
    device = torch.device("cpu")
    progress_calls: list[tuple[int, int]] = []

    result = evaluate.evaluate_vs_opponent(
        net,
        opponent_net=None,  # random-agent opponent
        device=device,
        n_pairs=1,
        seed=0,
        on_progress=lambda done, total: progress_calls.append((done, total)),
    )

    assert result.n_games == 2
    assert 0.0 <= result.win_rate <= 1.0
    assert result.ci95 >= 0.0
    # Called exactly once per game: (1, 2) after game 1, (2, 2) after game 2.
    assert progress_calls == [(1, 2), (2, 2)]


def test_run_final_self_play_eval_returns_valid_stats():
    """run_final_self_play_eval runs model-vs-itself and returns a populated
    FinalEvalStats; on_progress is called once per game."""
    net = model.PolicyValueNet()
    device = torch.device("cpu")
    progress_calls: list[tuple[int, int]] = []

    stats = evaluate.run_final_self_play_eval(
        net,
        device=device,
        n_games=2,
        seed=0,
        at_iteration=5,
        on_progress=lambda done, total: progress_calls.append((done, total)),
    )

    assert stats.n_games == 2
    assert stats.at_iteration == 5
    assert 0.0 <= stats.self_play_win_rate <= 1.0
    assert stats.decisions_per_game >= 0.0
    assert stats.mean_margin >= 0.0
    # avg_breakdown total is a non-negative score average
    assert stats.avg_breakdown.total >= 0.0
    # one call per game (pair 0, flip 0 and flip 1)
    assert len(progress_calls) == 2
    assert progress_calls[-1] == (2, 2)
