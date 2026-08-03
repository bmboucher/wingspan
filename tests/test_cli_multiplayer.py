"""Tests for N-player support in the ``wingspan play`` CLI (Stage 4).

Covers the ``--p2`` / ``--p3`` seat-spec args (contiguous fill, the
``--p3``-without-``--p2`` argparse-level error), :func:`players.resolve_num_players`
at the unit level, and that the 2-seat default path is unchanged end-to-end.
"""

from __future__ import annotations

import pathlib

import pytest

from wingspan import cli, players
from wingspan.training import config as train_config

###### Argument parsing ######


def test_two_seat_args_have_no_p2_p3_by_default():
    """The 2-seat default: --p0/--p1 only, --p2/--p3 both absent (None)."""
    args = cli._build_parser().parse_args(  # type: ignore[attr-defined]
        ["--p0", "random", "--p1", "random"]
    )
    assert args.p0 == "random"
    assert args.p1 == "random"
    assert args.p2 is None
    assert args.p3 is None


def test_three_seat_args_parse_p2():
    """--p2 is accepted alongside --p0/--p1, leaving --p3 absent."""
    args = cli._build_parser().parse_args(  # type: ignore[attr-defined]
        ["--p0", "random", "--p1", "random", "--p2", "human"]
    )
    assert args.p2 == "human"
    assert args.p3 is None


def test_four_seat_args_parse_p2_and_p3():
    """--p2 and --p3 together are accepted (4-seat table)."""
    args = cli._build_parser().parse_args(  # type: ignore[attr-defined]
        ["--p0", "random", "--p1", "random", "--p2", "random", "--p3", "random"]
    )
    assert args.p2 == "random"
    assert args.p3 == "random"


def test_p3_without_p2_is_argparse_level_error(
    capsys: pytest.CaptureFixture[str],
):
    """--p3 given without --p2 is rejected before any game runs — an
    argparse-level error (SystemExit), not a main_play() return code."""
    with pytest.raises(SystemExit):
        cli.main_play(
            [
                "--p0",
                "random",
                "--p1",
                "random",
                "--p3",
                "random",
                "--quiet",
            ]
        )
    captured = capsys.readouterr()
    assert "--p2" in captured.err


###### resolve_num_players (unit-level) ######


def _config_with_num_players(num_players: int) -> train_config.RunConfig:
    return train_config.RunConfig(
        architecture=train_config.ArchitectureConfig(num_players=num_players)
    )


def test_resolve_num_players_mismatch_raises():
    """A checkpoint trained at num_players=3 cannot be seated at a 2-seat table."""
    three_player_config = _config_with_num_players(3)
    with pytest.raises(ValueError, match="Seat 0"):
        players.resolve_num_players((three_player_config, None), 2)


def test_resolve_num_players_names_trained_count_and_table_size():
    """The error names both the checkpoint's trained count and the table size."""
    three_player_config = _config_with_num_players(3)
    with pytest.raises(ValueError, match="num_players=3") as excinfo:
        players.resolve_num_players((None, three_player_config), 4)
    assert "4 seats" in str(excinfo.value)


def test_resolve_num_players_matching_count_passes():
    """A checkpoint trained at the table's own seat count raises nothing."""
    three_player_config = _config_with_num_players(3)
    players.resolve_num_players((three_player_config, three_player_config), 3)


def test_resolve_num_players_config_free_seats_never_raise():
    """Human/random seats (None config) express no preference at any table size."""
    players.resolve_num_players((None, None, None, None), 4)


###### End-to-end (in-process) ######


def test_two_seat_default_path_still_exits_zero():
    """The pre-existing 2-seat default path is unaffected by the N-player CLI work."""
    exit_code = cli.main_play(
        ["--p0", "random", "--p1", "random", "--seed", "1", "--quiet"]
    )
    assert exit_code == 0


def test_three_seat_random_game_runs_and_exits_zero():
    """A 3-seat random-vs-random-vs-random game runs to completion."""
    exit_code = cli.main_play(
        [
            "--p0",
            "random",
            "--p1",
            "random",
            "--p2",
            "random",
            "--seed",
            "2",
            "--quiet",
        ]
    )
    assert exit_code == 0


def test_four_seat_random_game_runs_and_exits_zero():
    """A full 4-seat random game runs to completion (--p2 and --p3 both set)."""
    exit_code = cli.main_play(
        [
            "--p0",
            "random",
            "--p1",
            "random",
            "--p2",
            "random",
            "--p3",
            "random",
            "--seed",
            "3",
            "--quiet",
        ]
    )
    assert exit_code == 0


def test_three_seat_debug_log_writes_one_file_per_seat(tmp_path: pathlib.Path):
    """--debug-log at 3 seats writes _p0/_p1/_p2 split files (not just _p0/_p1)."""
    log_path = tmp_path / "game.log"
    exit_code = cli.main_play(
        [
            "--p0",
            "random",
            "--p1",
            "random",
            "--p2",
            "random",
            "--seed",
            "4",
            "--quiet",
            "--debug-log",
            str(log_path),
        ]
    )
    assert exit_code == 0
    for seat in range(3):
        seat_log = tmp_path / f"game.log_p{seat}.log"
        assert seat_log.exists() and seat_log.read_text(encoding="utf-8").strip()
    # No stray 4th-seat file for a 3-seat game.
    assert not (tmp_path / "game.log_p3.log").exists()


def test_three_seat_html_flag_renders(tmp_path: pathlib.Path):
    """--html at 3 seats writes a viewer file without error."""
    out_path = tmp_path / "game.html"
    exit_code = cli.main_play(
        [
            "--p0",
            "random",
            "--p1",
            "random",
            "--p2",
            "random",
            "--seed",
            "5",
            "--quiet",
            "--html",
            str(out_path),
            "--instrument-out",
            str(tmp_path),
        ]
    )
    assert exit_code == 0
    assert out_path.exists()
    assert out_path.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")
