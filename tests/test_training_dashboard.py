"""Tests for the training + live-dashboard pipeline (``wingspan.training``).

* Metric value-objects (``ScoreBreakdown`` / ``FamilyCounts``) arithmetic.
* Chart primitives (sparkline, eighth-block bar, human counts, braille canvas).
* The dashboard renders without error for an empty *and* a populated state, at
  a wide width (eval inset docked) and a narrow one (inset drops to a strip).
* The SYSTEM band: percentage math, a live host sample, and the CPU/RAM gauges
  rendering.
* One real end-to-end training iteration (collect -> length-bucketed update ->
  paired eval -> checkpoint) runs to completion and writes resumable artifacts.
* Resuming a second run from ``last.pt`` continues the counters and charts.
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
from wingspan.training import (
    charts,
    config,
    dashboard,
    loop,
    metrics,
    runstate,
    sysmon,
)


def test_score_breakdown_arithmetic():
    breakdown = metrics.ScoreBreakdown(
        birds=1, eggs=2, food=3, tucked=4, rounds=5, bonus=6
    )
    assert breakdown.total == 21
    doubled = breakdown + breakdown
    assert doubled.birds == 2 and doubled.total == 42
    halved = breakdown.scaled(0.5)
    assert halved.eggs == 1.0
    assert [name for name, _ in breakdown.components()] == list(
        metrics.SCORE_COMPONENTS
    )


def test_family_counts_align_to_families():
    counts = metrics.FamilyCounts()
    counts.bump(0)
    counts.bump(0)
    counts.bump(3)
    assert counts.total() == 3
    assert (counts + counts).total() == 6
    assert len(counts.items()) == len(decisions.ALL_DECISION_FAMILIES)


def test_chart_helpers():
    assert charts.human_count(842) == "842"
    assert charts.human_count(2600) == "2.6k"
    assert charts.human_count(2_560_000) == "2.56M"
    assert charts.eighth_bar(0.0, 10, min_tick=True) == ""
    assert charts.eighth_bar(0.001, 10, min_tick=True) == "▏"
    assert len(charts.sparkline([1.0, 2.0, 3.0], 3)) == 3

    canvas = charts.BrailleCanvas(4, 3, 1)
    canvas.set_dot(0, 0, 0)
    char, owner = canvas.cell(0, 0)
    assert owner == 0 and char != " "


def _render(
    state: runstate.RunState,
    width: int = 120,
    height: int = 44,
    colorize: bool = True,
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
        eval=metrics.EvalResult(n_games=4, win_rate=0.8, ci95=0.05, mean_margin=20.0),
    )


def test_dashboard_renders_empty_state():
    empty = runstate.new_run_state(config.TrainConfig(device="cpu"))
    # The colored render must produce a substantial frame without raising; the
    # per-character gradient splits the wordmark across ANSI runs, so content is
    # asserted against a plain (uncolored) render instead.
    assert len(_render(empty)) > 1000
    assert "WINGSPAN" in _render(empty, colorize=False)


@pytest.mark.parametrize("width", [128, 84])
def test_dashboard_renders_populated_state(width: int):
    state = runstate.new_run_state(config.TrainConfig(device="cpu"))
    breakdown = metrics.ScoreBreakdown(
        birds=27, eggs=14, food=9, tucked=6, rounds=7, bonus=5
    )
    family = metrics.FamilyCounts()
    for index in range(140):
        family.bump(index % len(decisions.ALL_DECISION_FAMILIES))
    state.record_game((breakdown, breakdown), 140, family)
    iteration = _sample_iteration(breakdown)
    state.history.append(iteration)
    state.last_iter = iteration
    state.best_win_rate = 0.8
    state.push_event(runstate.EventKind.BEST, "new best.pt")

    assert len(_render(state, width=width)) > 1000  # colored path, no crash
    plain = _render(state, width=width, colorize=False)
    assert "WINGSPAN" in plain
    assert "macro_action" in plain  # the histogram rendered its family rows


def test_system_stats_percentages():
    stats = metrics.SystemStats(
        cpu_percent=50.0, ram_used_gb=8.0, ram_total_gb=32.0, proc_rss_gb=1.0
    )
    assert stats.ram_percent == 25.0
    # A zero total (e.g. the degraded fallback sample) never divides by zero.
    empty = metrics.SystemStats(
        cpu_percent=0.0, ram_used_gb=0.0, ram_total_gb=0.0, proc_rss_gb=0.0
    )
    assert empty.ram_percent == 0.0


def test_system_monitor_sample():
    stats = sysmon.SystemMonitor().sample()
    assert 0.0 <= stats.cpu_percent <= 100.0
    assert stats.ram_total_gb > 0.0
    assert stats.proc_rss_gb >= 0.0


def test_dashboard_system_band():
    state = runstate.new_run_state(config.TrainConfig(device="cpu"))
    state.system = metrics.SystemStats(
        cpu_percent=58.9, ram_used_gb=27.0, ram_total_gb=68.6, proc_rss_gb=1.3
    )
    plain = _render(state, colorize=False)
    assert "SYSTEM" in plain and "CPU" in plain and "RAM" in plain
    # The GPU line was removed entirely.
    for absent in ("GPU", "VRAM", "CUDA"):
        assert absent not in plain
    assert len(_render(state)) > 1000  # colored path renders without error


def test_training_loop_one_iteration(tmp_path: pathlib.Path):
    cfg = config.TrainConfig(
        device="cpu",
        games_per_iter=2,
        max_iterations=1,
        eval_every=1,
        eval_games=1,
        hidden=32,
        checkpoint_dir=str(tmp_path),
    )
    training = loop.TrainingLoop(cfg)
    training.run()  # synchronous (no worker thread) for a deterministic test

    state = training.state
    assert state.phase is runstate.Phase.DONE
    assert state.total_games == 2
    assert state.last_iter is not None
    assert state.last_iter.eval is not None
    assert state.cum_family.total() == state.total_decisions
    assert state.system is not None  # the monitor thread took at least one sample

    assert (tmp_path / "last.pt").exists()
    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "metrics.jsonl").exists()

    # The dashboard renders the real post-run state without error.
    assert len(_render(state)) > 1000
    assert "macro_action" in _render(state, colorize=False)


def test_training_loop_resumes_from_checkpoint(tmp_path: pathlib.Path):
    cfg = config.TrainConfig(
        device="cpu",
        games_per_iter=2,
        max_iterations=1,
        eval_every=1,
        eval_games=1,
        hidden=32,
        checkpoint_dir=str(tmp_path),
    )
    first = loop.TrainingLoop(cfg)
    first.run()
    games = first.state.total_games
    last_iter = first.state.iteration
    best = first.state.best_win_rate
    assert (tmp_path / "last.pt").exists()

    # A fresh loop on the same dir restores progress instead of starting at zero.
    resumed = loop.TrainingLoop(cfg)
    assert resumed.state.total_games == games
    assert resumed.state.iteration == last_iter
    assert resumed.state.best_win_rate == best
    assert resumed.state.history  # convergence chart history carried over

    resumed.run()  # one more iteration continues the counts from the checkpoint
    assert resumed.state.total_games == games + cfg.games_per_iter
    assert resumed.state.iteration == last_iter + 1

    # --no-resume ignores the checkpoint and starts fresh.
    fresh = loop.TrainingLoop(cfg.model_copy(update={"resume": False}))
    assert fresh.state.total_games == 0
