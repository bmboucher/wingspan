"""Tests for the TRAINING IMPROVEMENT convergence charts.

Covers the pure series + window math in ``wingspan.training.convergence`` (full
WIN RATE range, the pinned FINAL SCORE / MARGIN window, EWMA series with the
per-generation reset, challenger markers) and the rendered behaviour of the
reworked panel (title swap, the dual-axis chart, the EVAL box challenger
readout, and a render at scale with markers).
"""

from __future__ import annotations

import io
import os
import pathlib
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

pytest.importorskip("torch")
pytest.importorskip("rich")

import rich.console as rich_console

from wingspan import decisions
from wingspan.training import config, convergence, dashboard, metrics, runstate


def _sample_iteration(breakdown: metrics.ScoreBreakdown) -> metrics.IterationMetrics:
    family = metrics.FamilyCounts()
    for index in range(140):
        family.bump(index % len(decisions.ALL_DECISION_FAMILIES))
    return metrics.IterationMetrics(
        iteration=0,
        total_games=2,
        games_this_iter=2,
        loss=1.0,
        policy_loss=0.5,
        value_loss=0.3,
        entropy=0.6,
        grad_norm=1.5,
        advantage_mean=0.0,
        advantage_std=1.0,
        avg_self_score=60.0,
        avg_margin=0.0,
        avg_breakdown=breakdown,
        avg_decisions=140.0,
        family_counts=family,
        collect_seconds=1.0,
        update_seconds=0.5,
        eval_seconds=1.0,
        games_per_sec=2.0,
    )


def _iter(
    iteration: int,
    score: float,
    *,
    win: float | None = None,
    margin: float = 0.0,
    generation: int = 0,
) -> metrics.IterationMetrics:
    """An ``IterationMetrics`` for the chart helpers: ``score`` is the avg
    self-play final score; an eval block is attached only when ``win`` is given
    (tagged with ``generation`` so the EWMA reset can be exercised)."""
    base = _sample_iteration(metrics.ScoreBreakdown(birds=score)).model_copy(
        update={"iteration": iteration, "avg_self_score": score, "eval": None}
    )
    if win is not None:
        base = base.model_copy(
            update={
                "eval": metrics.EvalResult(
                    n_games=4,
                    win_rate=win,
                    ci95=0.05,
                    mean_margin=margin,
                    opponent_generation=generation,
                )
            }
        )
    return base


def _render(
    state: runstate.RunState, width: int = 120, height: int = 44, colorize: bool = True
) -> str:
    buffer = io.StringIO()
    term = rich_console.Console(
        file=buffer,
        width=width,
        height=height,
        force_terminal=True,
        color_system="truecolor" if colorize else None,
    )
    root = dashboard.build_layout()
    dashboard.render(root, state, frame=1)
    term.print(root)
    return buffer.getvalue()


def _state(tmp_path: pathlib.Path) -> runstate.RunState:
    """A run state whose checkpoint dir is an empty tmp dir, so the charts read
    no on-disk history and fall back to the in-memory ``state.history``."""
    cfg = config.TrainConfig(device="cpu", checkpoint_dir=str(tmp_path / "ckpt"))
    return runstate.new_run_state(cfg)


# ---------------------------------------------------------------------------
# Pure series + window math


def test_score_margin_window_pins_left_edge_to_hundred():
    it_lo, it_hi = convergence.score_margin_window([_iter(0, 50.0), _iter(2345, 60.0)])
    assert it_lo == 300  # floor((2345 - 2000 + 1) / 100) * 100
    assert it_hi - it_lo == convergence.SCORE_MARGIN_WINDOW == 2000
    # A short run pins the left edge at 0 and leaves a gap on the right.
    short_lo, short_hi = convergence.score_margin_window(
        [_iter(0, 50.0), _iter(50, 60.0)]
    )
    assert short_lo == 0 and short_hi == 2000


def test_full_range_spans_whole_history():
    assert convergence.full_range(
        [_iter(0, 50.0, win=0.5), _iter(900, 60.0, win=0.9)]
    ) == (0, 900)


def test_marker_columns_maps_change_iterations():
    cols = 50
    columns = convergence.marker_columns([0, 50, 100], 0, 100, cols)
    assert columns == {0, round(0.5 * (cols - 1)), cols - 1}
    # Out-of-range upgrades are dropped.
    assert convergence.marker_columns([200], 0, 100, cols) == set()


def test_winrate_ewma_resets_on_opponent_advance():
    history = [
        _iter(0, 50.0, win=0.5, generation=0),
        _iter(1, 50.0, win=0.7, generation=0),
        _iter(2, 50.0, win=0.4, generation=1),
    ]
    points = convergence.winrate_ewma_points(history, alpha=0.5)
    assert points[0] == (0, 50.0)
    assert points[1] == (1, 60.0)  # 0.5*70 + 0.5*50
    assert points[2] == (2, 40.0)  # generation changed -> reset to raw


def test_margin_ewma_resets_on_opponent_advance():
    history = [
        _iter(0, 50.0, win=0.5, margin=10.0, generation=0),
        _iter(1, 50.0, win=0.6, margin=20.0, generation=0),
        _iter(2, 50.0, win=0.5, margin=4.0, generation=1),
    ]
    points = convergence.margin_ewma_points(history, alpha=0.5)
    assert points[0] == (0, 10.0)
    assert points[1] == (1, 15.0)  # 0.5*20 + 0.5*10
    assert points[2] == (2, 4.0)  # reset on advance


def test_score_ewma_smooths_every_iteration():
    points = convergence.score_ewma_points(
        [_iter(0, 60.0), _iter(1, 40.0), _iter(2, 40.0)], alpha=0.5
    )
    assert [it for it, _ in points] == [0, 1, 2]
    assert points[0][1] == 60.0
    assert points[1][1] == 50.0  # 0.5*40 + 0.5*60


def test_opponent_change_iterations_round_trip():
    state = runstate.new_run_state(config.TrainConfig(device="cpu"))
    state.opponent_change_iterations.extend([120, 340])
    restored = runstate.new_run_state(config.TrainConfig(device="cpu"))
    restored.restore_progress(state.to_progress())
    assert restored.opponent_change_iterations == [120, 340]


# ---------------------------------------------------------------------------
# Rendered behaviour


def test_titles_swap_to_final_score_and_margin(tmp_path: pathlib.Path):
    state = _state(tmp_path)
    state.history.append(_iter(0, 60.0, win=0.83))
    state.last_iter = state.history[-1]
    plain = _render(state, colorize=False)
    assert "WIN RATE" in plain
    assert "FINAL SCORE" in plain and "MARGIN" in plain
    assert "AVG POINTS" not in plain  # the old title is gone


def test_eval_box_shows_random_challenger(tmp_path: pathlib.Path):
    # Generation 0 (still vs the random agent): the challenger row reads
    # "random", via the public render path (no private access).
    state = _state(tmp_path)
    for iteration in range(0, 200, 50):
        state.history.append(_iter(iteration, 55.0, win=0.6, margin=2.0, generation=0))
    state.last_iter = state.history[-1]
    plain = _render(state, colorize=False)
    assert "challenger" in plain
    assert "random" in plain


def test_eval_box_shows_frozen_challenger(tmp_path: pathlib.Path):
    state = _state(tmp_path)
    for iteration in range(0, 200, 50):
        state.history.append(_iter(iteration, 55.0, win=0.9, margin=5.0, generation=2))
    state.opponent_generation = 2
    state.opponent_since_iteration = 120
    state.iteration = 175
    state.last_iter = state.history[-1]
    plain = _render(state, colorize=False)
    assert "challenger" in plain
    assert "gen2 @ 0120" in plain
    assert "since adv" in plain


def test_renders_long_history_with_markers(tmp_path: pathlib.Path):
    # A render at scale exercises the full-range win-rate markers, the pinned
    # 2000-window, and the dual margin axis without raising.
    state = _state(tmp_path)
    for iteration in range(0, 2400, 100):
        generation = iteration // 800
        win = 0.5 + 0.05 * ((iteration // 100) % 9)
        state.history.append(
            _iter(
                iteration,
                50.0 + iteration / 100.0,
                win=win,
                margin=5.0,
                generation=generation,
            )
        )
    state.opponent_change_iterations.extend([800, 1600])
    state.opponent_generation = 2
    state.opponent_since_iteration = 1600
    state.iteration = 2300
    state.last_iter = state.history[-1]
    assert len(_render(state)) > 1000
    plain = _render(state, colorize=False)
    assert "FINAL SCORE" in plain and "WIN RATE" in plain
