"""Tests for the tournament dashboard + picker rendering.

The live surfaces are interactive; these render them off-screen to a virtual
terminal so a style / rich-API regression raises here rather than mid-tournament.
Also exercises the live ``TournamentState`` writer/reader helpers the dashboard
reads (record_game, standings, push_event).
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import io
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import rich.console as rich_console

from wingspan.tournament import (
    config,
    dashboard,
    participants,
    picker,
    results,
    schedule,
)
from wingspan.tournament import state as state_module
from wingspan.training import runstate


def _render_dashboard(live: state_module.TournamentState, colorize: bool = True) -> str:
    buffer = io.StringIO()
    term = rich_console.Console(
        file=buffer,
        width=120,
        height=40,
        force_terminal=True,
        color_system="truecolor" if colorize else None,
    )
    root = dashboard.build_layout()
    dashboard.render(root, live)
    term.print(root)
    return buffer.getvalue()


def _cfg() -> config.TournamentConfig:
    specs = [
        participants.ParticipantSpec(
            id=f"p{index}",
            display_name=f"p{index}",
            kind=participants.ParticipantKind.RANDOM,
        )
        for index in range(3)
    ]
    return config.TournamentConfig(participants=specs, games_per_pair=4)


def test_dashboard_renders_empty_state() -> None:
    live = state_module.new_tournament_state(_cfg())
    assert len(_render_dashboard(live)) > 500
    assert "WINGSPAN" in _render_dashboard(live, colorize=False)


def test_dashboard_renders_populated_state() -> None:
    live = state_module.new_tournament_state(_cfg())
    live.record_game(
        results.GameResult(
            round_index=0,
            pair_index=0,
            orientation=schedule.Orientation.A_SEAT_0,
            player_a_id="p0",
            player_b_id="p1",
            a_score=30,
            b_score=22,
            a_was_start_player=True,
        )
    )
    live.push_event(runstate.EventKind.INFO, "p0 beat p1 by 8")

    rows = live.standings()
    assert rows[0].id == "p0" and rows[0].wins == 1  # winner leads on live Elo

    plain = _render_dashboard(live, colorize=False)
    assert "STANDINGS" in plain
    assert "p0" in plain
    assert "1/12" in plain  # games_done / total in the header + progress bar


def test_picker_renders() -> None:
    items = [
        picker._Item(
            spec=participants.ParticipantSpec(
                id="main",
                display_name="main",
                kind=participants.ParticipantKind.MODEL,
                checkpoint_dir="checkpoints",
            ),
            subtitle="iter 100 · best 80%",
        ),
        picker._Item(
            spec=participants.random_spec(), subtitle="uniform-random baseline"
        ),
    ]
    panel = picker._render(
        items,
        selected={0},
        cursor=1,
        base_dir="checkpoints",
        games_per_pair=32,
        warning="",
    )
    buffer = io.StringIO()
    term = rich_console.Console(
        file=buffer, width=100, height=24, force_terminal=True, color_system=None
    )
    term.print(panel)
    out = buffer.getvalue()
    assert "SELECT COMPETITORS" in out
    assert "[x]" in out and "[ ]" in out
    assert "random" in out
