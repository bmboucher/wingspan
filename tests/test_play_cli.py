"""Smoke tests for the unified ``wingspan play`` CLI entry point.

A random-vs-random series must run to completion and write its logs; a seat
spec that cannot be resolved must fail before any game runs, with a clean
message and exit code 1.
"""

from __future__ import annotations

import os
import pathlib
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest  # noqa: E402

from wingspan import cli  # noqa: E402


def test_random_vs_random_game_runs_and_writes_log(tmp_path: pathlib.Path):
    """One quick random-vs-random game exits 0 and writes a non-empty log."""
    log_path = tmp_path / "game.log"
    exit_code = cli.main_play(
        [
            "--p0",
            "random",
            "--p1",
            "random",
            "--seed",
            "123",
            "--quiet",
            "--log",
            str(log_path),
        ]
    )
    assert exit_code == 0
    assert log_path.read_text(encoding="utf-8").strip()


def test_multi_game_logs_get_index_suffixes(tmp_path: pathlib.Path):
    """With ``--games N`` each game's log lands at ``<path>.<game_idx>``."""
    log_path = tmp_path / "games.log"
    exit_code = cli.main_play(
        [
            "--p0",
            "random",
            "--p1",
            "random",
            "--seed",
            "7",
            "--games",
            "2",
            "--quiet",
            "--log",
            str(log_path),
        ]
    )
    assert exit_code == 0
    assert (tmp_path / "games.log.0").exists()
    assert (tmp_path / "games.log.1").exists()


def test_missing_checkpoint_fails_cleanly(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
):
    """A model seat whose checkpoint does not exist fails before any game runs,
    printing the loader's message to stderr and exiting 1 (the bare
    ``wingspan play`` path on a machine with no trained model)."""
    exit_code = cli.main_play(
        [
            "--p0",
            "last",
            "--p1",
            "random",
            "--checkpoint-dir",
            str(tmp_path / "nothing-here"),
            "--quiet",
        ]
    )
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Checkpoint not found" in captured.err
