"""Tests for the training + live-dashboard pipeline (``wingspan.training``).

* Metric value-objects (``ScoreBreakdown`` / ``FamilyCounts``) arithmetic.
* Chart primitives (sparkline, eighth-block bar, human counts, braille canvas).
* The dashboard renders without error for an empty *and* a populated state, at
  a wide width (eval inset docked) and a narrow one (inset drops to a strip).
* The header CPU/RAM gauges: percentage math, a live host sample, and the gauges
  rendering on the progress row (the separate SYSTEM band was folded away).
* One real end-to-end training iteration (collect -> length-bucketed update ->
  paired eval -> checkpoint) runs to completion and writes resumable artifacts.
* Resuming a second run from ``last.pt`` continues the counters and charts.
"""

from __future__ import annotations

import io
import json
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


def test_family_histogram_keeps_total_when_short():
    # The total-decisions footer is reserved: when the panel is too short to
    # hold every family row the bars clip from the bottom, but the total stays.
    counts = metrics.FamilyCounts()
    for index in range(len(decisions.ALL_DECISION_FAMILIES)):
        counts.bump(index)
    total = counts.total()
    buffer = io.StringIO()
    term = rich_console.Console(file=buffer, width=60, height=8, color_system=None)
    term.print(charts.FamilyHistogram(counts, total_decisions=total))
    assert f"{total:,} total decisions" in buffer.getvalue()


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
    state.record_game((breakdown, breakdown), 140, family, winner=0)
    iteration = _sample_iteration(breakdown)
    state.history.append(iteration)
    state.last_iter = iteration
    state.best_win_rate = 0.8
    state.push_event(runstate.EventKind.BEST, "new best.pt")

    assert len(_render(state, width=width)) > 1000  # colored path, no crash
    plain = _render(state, width=width, colorize=False)
    assert "WINGSPAN" in plain
    assert "main_action" in plain  # the histogram rendered its family rows


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


def test_dashboard_header_gauges():
    state = runstate.new_run_state(config.TrainConfig(device="cpu"))
    state.system = metrics.SystemStats(
        cpu_percent=58.9, ram_used_gb=27.0, ram_total_gb=68.6, proc_rss_gb=1.3
    )
    plain = _render(state, colorize=False)
    # CPU + RAM gauges now ride on the header progress row, not a separate band.
    assert "CPU" in plain and "RAM" in plain
    assert "SYSTEM" not in plain  # the separate SYSTEM band was folded into the header
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
        eval_games=2,
        trunk_layers=(32, 32),
        choice_layers=(32, 32),
        checkpoint_dir=str(tmp_path),
        # Exercise the self-play + eval path directly (the random-opponent
        # bootstrap phase pauses eval; that path is covered separately).
        initial_vs_random=False,
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

    # The four required run artifacts are all left behind: the model descriptor,
    # this session's dated process record, and one game-history row per game.
    assert (tmp_path / "model_config.json").exists()
    assert len(list(tmp_path.glob("process_*.json"))) == 1
    game_rows = [
        line for line in (tmp_path / "games.jsonl").read_text().splitlines() if line
    ]
    assert len(game_rows) == state.total_games  # one record per game played
    outcomes = [
        metrics.GameOutcome.model_validate(json.loads(row)) for row in game_rows
    ]
    assert all(game.iteration == 0 and game.decisions > 0 for game in outcomes)
    assert outcomes[0].family_counts.total() == outcomes[0].decisions
    # The board-shuffle seed rides through the real collector to each row, so the
    # distinct per-game seeds make every record independently reproducible.
    assert len({game.seed for game in outcomes}) == len(outcomes)

    # The dashboard renders the real post-run state without error.
    assert len(_render(state)) > 1000
    assert "main_action" in _render(state, colorize=False)


def test_training_loop_resumes_from_checkpoint(tmp_path: pathlib.Path):
    cfg = config.TrainConfig(
        device="cpu",
        games_per_iter=2,
        max_iterations=1,
        eval_every=1,
        eval_games=2,
        trunk_layers=(32, 32),
        choice_layers=(32, 32),
        checkpoint_dir=str(tmp_path),
        # Resume continuity is tested on the self-play + eval regime; the
        # bootstrap phase has its own resume/graduation coverage.
        initial_vs_random=False,
    )
    first = loop.TrainingLoop(cfg)
    first.run()
    games = first.state.total_games
    last_iter = first.state.iteration
    best = first.state.best_win_rate
    elapsed = first.state.elapsed()
    assert (tmp_path / "last.pt").exists()

    # A fresh loop on the same dir restores progress instead of starting at zero.
    resumed = loop.TrainingLoop(cfg)
    assert resumed.state.total_games == games
    assert resumed.state.iteration == last_iter
    assert resumed.state.best_win_rate == best
    assert resumed.state.history  # convergence chart history carried over
    # The dashboard reopens with the prior clock and event log, not from scratch:
    # the T+ chronometer resumes from the checkpointed elapsed (saved mid-iteration,
    # so at most the first run's final elapsed), and the events ring carries the
    # first run's "run started" line plus the new "resumed" line on top of it.
    assert 0.0 < resumed.state.elapsed_offset <= elapsed
    assert any("run started" in line.text for line in resumed.state.events)
    assert any("resumed" in line.text for line in resumed.state.events)

    resumed.run()  # one more iteration continues the counts from the checkpoint
    assert resumed.state.total_games == games + cfg.games_per_iter
    assert resumed.state.iteration == last_iter + 1

    # The game-history log was appended across the resume (not truncated), and
    # each session dropped its own dated process record.
    game_rows = [
        line for line in (tmp_path / "games.jsonl").read_text().splitlines() if line
    ]
    assert len(game_rows) == games + cfg.games_per_iter
    assert len(list(tmp_path.glob("process_*.json"))) == 2

    # --no-resume ignores the checkpoint and starts fresh.
    fresh = loop.TrainingLoop(cfg.model_copy(update={"resume": False}))
    assert fresh.state.total_games == 0
    # A fresh start clears the prior run's game log and stale session records,
    # leaving only this startup's process file.
    assert (tmp_path / "games.jsonl").read_text() == ""
    assert len(list(tmp_path.glob("process_*.json"))) == 1


# ---------------------------------------------------------------------------
# Random-opponent bootstrap phase


def _bootstrap_iteration(
    collection_win_rate: float, margin: float = 0.0
) -> metrics.IterationMetrics:
    """A finished-iteration metrics row in the bootstrap shape: a collection
    win-rate vs random and no eval block."""
    breakdown = metrics.ScoreBreakdown(birds=20.0)
    return _sample_iteration(breakdown).model_copy(
        update={
            "collection_win_rate": collection_win_rate,
            "avg_margin": margin,
            "eval": None,
        }
    )


def _bootstrap_config(
    tmp_path: pathlib.Path, **overrides: object
) -> config.TrainConfig:
    base: dict[str, object] = {
        "device": "cpu",
        "games_per_iter": 2,
        "max_iterations": 1,
        "trunk_layers": (32, 32),
        "choice_layers": (32, 32),
        "checkpoint_dir": str(tmp_path),
        "initial_vs_random": True,
        "random_phase_win_rate": 0.5,
        "eval_ewma_alpha": 0.3,
    }
    base.update(overrides)
    return config.TrainConfig.model_validate(base)


def test_collection_win_rate_ewma():
    state = runstate.new_run_state(
        config.TrainConfig(device="cpu", eval_ewma_alpha=0.3)
    )
    assert state.collection_win_rate_ewma() is None  # nothing folded yet

    # Self-play iterations (collection_win_rate is None) are skipped; only the
    # bootstrap rows fold, in order, at alpha = 0.3.
    state.history.append(_bootstrap_iteration(0.4))
    state.history.append(_sample_iteration(metrics.ScoreBreakdown()))  # eval row
    state.history.append(_bootstrap_iteration(0.6))
    state.history.append(_bootstrap_iteration(0.8))

    expected = 0.4
    expected = 0.3 * 0.6 + 0.7 * expected
    expected = 0.3 * 0.8 + 0.7 * expected
    ewma = state.collection_win_rate_ewma()
    assert ewma is not None
    assert abs(ewma - expected) < 1e-9


def test_collection_margin_ewma():
    state = runstate.new_run_state(
        config.TrainConfig(device="cpu", eval_ewma_alpha=0.3)
    )
    assert state.collection_margin_ewma() is None  # nothing folded yet

    # The margin twin of the win-rate EWMA: self-play rows (no collection win-rate)
    # are skipped; only the bootstrap rows' ``avg_margin`` folds, in order.
    state.history.append(_bootstrap_iteration(0.4, margin=10.0))
    state.history.append(_sample_iteration(metrics.ScoreBreakdown()))  # self-play row
    state.history.append(_bootstrap_iteration(0.6, margin=20.0))
    state.history.append(_bootstrap_iteration(0.8, margin=30.0))

    expected = 10.0
    expected = 0.3 * 20.0 + 0.7 * expected
    expected = 0.3 * 30.0 + 0.7 * expected
    ewma = state.collection_margin_ewma()
    assert ewma is not None
    assert abs(ewma - expected) < 1e-9


def test_produce_ewma_resets_at_self_play_graduation():
    # IN-GAME PERFORMANCE folds only the current phase's rows, so the EWMA restarts
    # fresh at graduation instead of dragging the vs-random character forward.
    state = runstate.new_run_state(config.TrainConfig(device="cpu"))
    state.history.append(_bootstrap_iteration(0.9, margin=30.0))  # birds=20, margin 30
    self_play = _sample_iteration(metrics.ScoreBreakdown(eggs=12.0)).model_copy(
        update={"avg_margin": 0.0}
    )
    state.history.append(self_play)  # collection_win_rate is None -> a self-play row

    # A single current-phase row folds to itself exactly (no EWMA arithmetic).
    state.training_phase = runstate.TrainingPhase.SELF_PLAY
    self_play_stats = state.produce_stats()
    assert self_play_stats is not None
    assert self_play_stats.breakdown.eggs == 12.0
    assert self_play_stats.breakdown.birds == 0.0  # bootstrap row dropped
    assert self_play_stats.margin == 0.0

    state.training_phase = runstate.TrainingPhase.RANDOM_OPPONENT
    boot_stats = state.produce_stats()
    assert boot_stats is not None
    assert boot_stats.breakdown.birds == 20.0  # only the bootstrap row
    assert boot_stats.margin == 30.0


def test_training_phase_round_trips_through_progress():
    cfg = config.TrainConfig(device="cpu")
    # A checkpoint written before the phase existed defaults to SELF_PLAY.
    assert runstate.RunProgress().training_phase is runstate.TrainingPhase.SELF_PLAY

    state = runstate.new_run_state(cfg)
    state.training_phase = runstate.TrainingPhase.RANDOM_OPPONENT
    progress = state.to_progress()
    assert progress.training_phase is runstate.TrainingPhase.RANDOM_OPPONENT

    restored = runstate.new_run_state(cfg)
    restored.restore_progress(progress)
    assert restored.training_phase is runstate.TrainingPhase.RANDOM_OPPONENT


def test_training_loop_bootstrap_collects_vs_random(tmp_path: pathlib.Path):
    # A graduation bar of 1.0 keeps the single iteration in the bootstrap phase;
    # the assertions below hold whether or not it happens to graduate.
    training = loop.TrainingLoop(_bootstrap_config(tmp_path, random_phase_win_rate=1.0))
    # A fresh run opens in the bootstrap phase against the random agent (no
    # frozen opponent yet — generation 0).
    assert training.state.training_phase is runstate.TrainingPhase.RANDOM_OPPONENT
    assert training.state.opponent_generation == 0

    training.run()

    last = training.state.last_iter
    assert last is not None
    # Eval is paused in the bootstrap phase; strength is the collection win-rate.
    assert last.eval is None
    assert last.eval_seconds == 0.0
    assert last.collection_win_rate is not None
    assert 0.0 <= last.collection_win_rate <= 1.0
    # No eval ran, so no best.pt was written during the bootstrap phase.
    assert not (tmp_path / "best.pt").exists()


def test_bootstrap_graduates_to_self_play(tmp_path: pathlib.Path):
    training = loop.TrainingLoop(_bootstrap_config(tmp_path))
    assert training.state.training_phase is runstate.TrainingPhase.RANDOM_OPPONENT

    # Pre-load the win-rate history so the first real iteration's EWMA clears the
    # 0.5 bar regardless of how the untrained net actually fares against random
    # (EWMA ends at 0.3·win + 0.7·1.0 >= 0.7). Graduation then runs through the
    # ordinary commit path.
    for _ in range(5):
        training.state.history.append(_bootstrap_iteration(1.0))
    training.run()

    assert training.state.training_phase is runstate.TrainingPhase.SELF_PLAY
    assert training.state.opponent_generation == 1  # froze self·gen1
    assert (tmp_path / "opponent.pt").exists()  # and persisted for resume

    # A resumed run restores the graduated phase and reloads self·gen1.
    resumed = loop.TrainingLoop(_bootstrap_config(tmp_path))
    assert resumed.state.training_phase is runstate.TrainingPhase.SELF_PLAY
    assert resumed.state.opponent_generation == 1


def test_bootstrap_stays_below_graduation_threshold(tmp_path: pathlib.Path):
    training = loop.TrainingLoop(_bootstrap_config(tmp_path))

    # Pre-load a 0.0 win-rate history so the first real iteration's EWMA stays
    # below the 0.5 bar regardless of the net's actual result (EWMA ends at
    # 0.3·win + 0.7·0.0 <= 0.3): the run must remain in the bootstrap phase.
    for _ in range(5):
        training.state.history.append(_bootstrap_iteration(0.0))
    training.run()

    assert training.state.training_phase is runstate.TrainingPhase.RANDOM_OPPONENT
    assert training.state.opponent_generation == 0
    assert not (tmp_path / "opponent.pt").exists()
