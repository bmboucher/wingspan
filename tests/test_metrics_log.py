"""Tests for the cached ``metrics.jsonl`` reader (``wingspan.training.metrics_log``).

The convergence charts read the full on-disk iteration history through this
module, so it must parse every row, memoise the parse, and re-read when the file
grows — while tolerating a missing log and a truncated final line.
"""

from __future__ import annotations

import os
import pathlib
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

pytest.importorskip("torch")

from wingspan import decisions
from wingspan.training import artifacts, metrics, metrics_log


def _row(iteration: int, score: float) -> metrics.IterationMetrics:
    family = metrics.FamilyCounts()
    family.bump(iteration % len(decisions.ALL_DECISION_FAMILIES))
    return metrics.IterationMetrics(
        iteration=iteration,
        total_games=2,
        games_this_iter=2,
        loss=1.0,
        policy_loss=0.5,
        value_loss=0.3,
        entropy=0.6,
        grad_norm=1.5,
        advantage_mean=0.0,
        advantage_std=1.0,
        avg_self_score=score,
        avg_margin=0.0,
        avg_breakdown=metrics.ScoreBreakdown(birds=score),
        avg_decisions=140.0,
        avg_winner_breakdown=metrics.ScoreBreakdown(),
        avg_abs_margin=0.0,
        margin_std=0.0,
        abs_margin_std=0.0,
        decisions_std=0.0,
        family_counts=family,
        collect_seconds=1.0,
        update_seconds=0.5,
        eval_seconds=1.0,
        games_per_sec=2.0,
    )


def _write(path: pathlib.Path, rows: list[metrics.IterationMetrics]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.model_dump_json() + "\n")


def test_missing_log_returns_empty(tmp_path: pathlib.Path):
    assert metrics_log.read_iteration_history(str(tmp_path)) == []


def test_reads_all_rows(tmp_path: pathlib.Path):
    log = tmp_path / artifacts.METRICS_LOG
    _write(log, [_row(0, 50.0), _row(1, 55.0), _row(2, 60.0)])
    rows = metrics_log.read_iteration_history(str(tmp_path))
    assert [row.iteration for row in rows] == [0, 1, 2]
    assert rows[2].avg_self_score == 60.0


def test_unchanged_fingerprint_serves_cache(tmp_path: pathlib.Path):
    log = tmp_path / artifacts.METRICS_LOG
    _write(log, [_row(0, 50.0)])
    original = log.stat()
    first = metrics_log.read_iteration_history(str(tmp_path))
    assert [row.iteration for row in first] == [0]

    # Overwrite with a same-length payload (iteration 1 prints the same byte
    # length as iteration 0) and restore the exact size + mtime, so the cache
    # fingerprint is unchanged. A correct cache then serves the stale parse
    # rather than re-reading the new bytes.
    log.write_text(_row(1, 50.0).model_dump_json() + "\n", encoding="utf-8")
    os.utime(log, (original.st_atime, original.st_mtime))
    assert log.stat().st_size == original.st_size  # fingerprint unchanged

    cached = metrics_log.read_iteration_history(str(tmp_path))
    assert [row.iteration for row in cached] == [0]  # stale snapshot served


def test_growth_invalidates_cache(tmp_path: pathlib.Path):
    log = tmp_path / artifacts.METRICS_LOG
    _write(log, [_row(0, 50.0)])
    assert len(metrics_log.read_iteration_history(str(tmp_path))) == 1
    _write(log, [_row(1, 55.0)])
    grown = metrics_log.read_iteration_history(str(tmp_path))
    assert [row.iteration for row in grown] == [0, 1]


def test_truncated_final_line_is_skipped(tmp_path: pathlib.Path):
    log = tmp_path / artifacts.METRICS_LOG
    _write(log, [_row(0, 50.0)])
    with open(log, "a", encoding="utf-8") as handle:
        handle.write('{"iteration": 1, "not-valid')  # crash mid-append
    rows = metrics_log.read_iteration_history(str(tmp_path))
    assert [row.iteration for row in rows] == [0]
