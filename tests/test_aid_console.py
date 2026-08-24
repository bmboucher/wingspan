"""Tests for ``wingspan.aid.console`` -- ``LogEcho``'s player-attributed
color-coding and ``[Name]`` prefix stripping."""

from __future__ import annotations

import aid_helpers
from wingspan import state
from wingspan.aid import console
from wingspan.engine import core as engine_core

# Mirrors ``console._GREEN``/``console._RED``/``console._ANSI_RESET`` -- kept
# local so this test doesn't reach into console.py's private constants.
_GREEN = "\x1b[32m"
_RED = "\x1b[31m"
_ANSI_RESET = "\x1b[0m"


def _game_state_with_entries(entries: list[state.LogEntry]) -> state.GameState:
    """A real 2-player ``GameState`` (via ``Engine.create``) with the aid
    session's "You"/"Opponent" seat names and ``log_entries`` installed
    verbatim -- ``LogEcho`` only reads ``log_entries``, so ``log`` is left
    untouched."""
    eng, *_ = engine_core.Engine.create(seed=1)
    eng.state.players[0].name = "You"
    eng.state.players[1].name = "Opponent"
    eng.state.log_entries = list(entries)
    return eng.state


def test_own_seat_line_is_colored_green_and_prefix_stripped() -> None:
    con, transcript = aid_helpers.interactive_console([])
    echo = console.LogEcho(con)
    echo.game_state = _game_state_with_entries(
        [state.LogEntry(player_id=0, text="[You] plays Bushtit in Forest")]
    )

    echo.flush()

    assert transcript == [f"{_GREEN}plays Bushtit in Forest{_ANSI_RESET}"]


def test_opponent_seat_line_is_colored_red_and_prefix_stripped() -> None:
    con, transcript = aid_helpers.interactive_console([])
    echo = console.LogEcho(con)
    echo.game_state = _game_state_with_entries(
        [state.LogEntry(player_id=1, text="[Opponent] plays Anna's Hummingbird")]
    )

    echo.flush()

    assert transcript == [f"{_RED}plays Anna's Hummingbird{_ANSI_RESET}"]


def test_global_line_is_printed_plain_with_no_stripping_attempted() -> None:
    con, transcript = aid_helpers.interactive_console([])
    echo = console.LogEcho(con)
    echo.game_state = _game_state_with_entries(
        [state.LogEntry(player_id=None, text="=== ROUND 1 ===")]
    )

    echo.flush()

    assert transcript == ["=== ROUND 1 ==="]


def test_non_interactive_console_strips_prefix_but_emits_no_color() -> None:
    con, transcript = aid_helpers.scripted_console([])
    echo = console.LogEcho(con)
    echo.game_state = _game_state_with_entries(
        [
            state.LogEntry(player_id=0, text="[You] plays Bushtit in Forest"),
            state.LogEntry(player_id=1, text="[Opponent] plays Anna's Hummingbird"),
        ]
    )

    echo.flush()

    assert transcript == [
        "plays Bushtit in Forest",
        "plays Anna's Hummingbird",
    ]
    assert not any(_ANSI_RESET in line for line in transcript)


def test_line_without_the_expected_prefix_is_passed_through_unchanged() -> None:
    con, transcript = aid_helpers.scripted_console([])
    echo = console.LogEcho(con)
    echo.game_state = _game_state_with_entries(
        [state.LogEntry(player_id=0, text="skipping decision, only 1 choice")]
    )

    echo.flush()

    assert transcript == ["skipping decision, only 1 choice"]
