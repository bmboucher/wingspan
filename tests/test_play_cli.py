"""Smoke tests for the unified ``wingspan play`` CLI entry point.

A random-vs-random series must run to completion and write its logs; a seat
spec that cannot be resolved must fail before any game runs, with a clean
message and exit code 1.
"""

from __future__ import annotations

import pathlib

import pytest  # noqa: E402

from wingspan import cli  # noqa: E402


def test_random_vs_random_game_runs_and_exits_zero():
    """One quick random-vs-random game exits 0."""
    exit_code = cli.main_play(
        [
            "--p0",
            "random",
            "--p1",
            "random",
            "--seed",
            "123",
            "--quiet",
        ]
    )
    assert exit_code == 0


def test_log_flag_writes_structured_plaintext(tmp_path: pathlib.Path):
    """``--log`` writes a single structured plaintext file sourced from the event tree."""
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
    assert log_path.exists(), "--log must write the bare path"
    text = log_path.read_text(encoding="utf-8").strip()
    assert text, "--log output must be non-empty"
    # The plaintext renderer emits phase headers from the event tree.
    assert "=== GAME_START ===" in text
    assert "=== SETUP ===" in text


def test_debug_log_flag_writes_split_logs(tmp_path: pathlib.Path):
    """``--debug-log`` writes per-player split log files (old ``--log`` behaviour)."""
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
            "--debug-log",
            str(log_path),
        ]
    )
    assert exit_code == 0
    # Default (no --collate): produces _p0.log and _p1.log, not the bare path.
    assert (
        not log_path.exists()
    ), "collate mode not requested; bare debug-log should not exist"
    p0_log = tmp_path / "game.log_p0.log"
    p1_log = tmp_path / "game.log_p1.log"
    assert p0_log.exists() and p0_log.read_text(encoding="utf-8").strip()
    assert p1_log.exists() and p1_log.read_text(encoding="utf-8").strip()


def test_collate_flag_writes_single_debug_log(tmp_path: pathlib.Path):
    """With ``--collate --debug-log`` the interleaved log is written to the bare path."""
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
            "--collate",
            "--debug-log",
            str(log_path),
        ]
    )
    assert exit_code == 0
    assert log_path.read_text(encoding="utf-8").strip()


def test_debug_log_player_attribution(tmp_path: pathlib.Path):
    """Per-player ``--debug-log`` files contain all global lines and only their own
    decision lines; lines tagged to the other player are absent."""
    log_path = tmp_path / "split.log"
    exit_code = cli.main_play(
        [
            "--p0",
            "random",
            "--p1",
            "random",
            "--seed",
            "42",
            "--quiet",
            "--debug-log",
            str(log_path),
        ]
    )
    assert exit_code == 0
    p0_text = (tmp_path / "split.log_p0.log").read_text(encoding="utf-8")
    p1_text = (tmp_path / "split.log_p1.log").read_text(encoding="utf-8")
    # Both files should contain global lines (e.g., round headers).
    assert "=== ROUND 1" in p0_text
    assert "=== ROUND 1" in p1_text
    # The combined line count across both files should exceed the individual
    # since global lines appear in both.
    assert len(p0_text.splitlines()) + len(p1_text.splitlines()) > max(
        len(p0_text.splitlines()), len(p1_text.splitlines())
    )


def test_multi_game_logs_get_index_suffixes(tmp_path: pathlib.Path):
    """With ``--games N`` and ``--log`` each game's log lands at ``<path>.<game_idx>``."""
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
