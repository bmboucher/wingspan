"""Coverage for the HTML game-log feature: renderer, capture, and CLI wiring.

Three layers are exercised:

* the pure renderer in :mod:`wingspan.reporting.game_log_html` (primitives in,
  self-contained HTML out — no engine needed);
* the engine-aware capture path driven by the
  :class:`~wingspan.instrumentation.handlers.game_log_html.GameLogHtmlHandler`
  over a full random game (phase alignment, 3x5 grids, narration slicing,
  per-line seat attribution);
* the ``wingspan play --html`` CLI flag writing a file end-to-end.

Tests prepend ``src/`` to ``sys.path`` to match ``test_smoke.py``.
"""

from __future__ import annotations

import os
import pathlib
import random
import sys
import typing

import pydantic

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import agents, engine
from wingspan.instrumentation import config as instrumentation_config
from wingspan.instrumentation import events as instrumentation_events
from wingspan.reporting import game_log_capture, game_log_html

if typing.TYPE_CHECKING:
    from wingspan import decisions, state
    from wingspan.engine import core

_HEADER_PREFIX = "==="
_CAPTURE_EVENTS = [
    "game_start",
    "setup_applied",
    "round_start",
    "turn_start",
    "game_end",
]


class _PhaseRecorder(
    instrumentation_events.GameStartHandler,
    instrumentation_events.SetupAppliedHandler,
    instrumentation_events.RoundStartHandler,
    instrumentation_events.TurnStartHandler,
):
    """A test-only handler that snapshots a phase at each mid-game capture event.

    It mirrors the four mid-game events the production ``GameLogHtml`` handler
    subscribes to, appending each ``capture_phase`` result to a public list the
    assertions read directly. Game-end is intentionally omitted: the caller
    appends the final-scoring phase itself, exactly as the real ``game_end``
    callback would, so reading the records never depends on the production
    handler's protected accumulator."""

    captured: list[game_log_html.PhaseRecord] = pydantic.Field(
        default_factory=list["game_log_html.PhaseRecord"]
    )

    def game_start(self, *, engine: core.Engine) -> None:
        self._snap(engine, "Game start", "game_start", None)

    def setup_applied(
        self,
        *,
        engine: core.Engine,
        player: state.Player,
        choice: decisions.SetupChoice,
    ) -> None:
        self._snap(engine, f"Setup — {player.name}", "setup", player.id)

    def round_start(self, *, engine: core.Engine, round_num: int) -> None:
        self._snap(engine, f"Round {round_num + 1}", "round", None)

    def turn_start(self, *, engine: core.Engine, player: state.Player) -> None:
        self._snap(engine, f"{player.name} turn", "turn", player.id)

    def _snap(
        self, engine: core.Engine, title: str, kind: str, active: int | None
    ) -> None:
        self.captured.append(
            game_log_capture.capture_phase(
                engine,
                index=len(self.captured),
                title=title,
                kind=kind,
                active=active,
            )
        )


#### Renderer (no engine) ####


def _tiny_report() -> game_log_html.GameLogReport:
    """A minimal two-phase report exercising every model field."""
    bird = game_log_html.BirdCellInfo(
        name="American Robin",
        vp=5,
        nest="Cup",
        wingspan_cm=23,
        habitats="Forest/Grassland",
        food_cost="invertebrate",
        egg_limit=4,
        eggs=2,
        tucked=1,
        cached=0,
        power_color="brown",
        power_text="Draw 1 card.",
    )
    full_row = game_log_html.HabitatRow(
        label="Forest",
        cells=[
            game_log_html.BoardCell(bird=bird if i == 0 else None) for i in range(5)
        ],
    )
    empty_row = game_log_html.HabitatRow(
        label="Grassland",
        cells=[game_log_html.BoardCell(bird=None) for _ in range(5)],
    )
    panel = game_log_html.PlayerPanel(
        player_id=0,
        name="P0",
        action_cubes_left=8,
        rows=[full_row, empty_row, empty_row],
        hand=[
            game_log_html.BirdCellInfo(
                name="Mallard",
                vp=3,
                nest="Platform",
                wingspan_cm=58,
                habitats="Wetland",
                food_cost="grain",
                egg_limit=4,
                eggs=0,
                tucked=0,
                cached=0,
                power_color="white",
                power_text="",
            )
        ],
        food=[game_log_html.FoodCount(label="seed", count=2)],
        score=game_log_html.ScoreBreakdown(
            birds=5, eggs=2, tucked=1, cached=0, bonus=3, goals=0, total=11
        ),
        bonus_cards=[
            game_log_html.BonusCardInfo(
                name="Bird Bander", text="3 / 4 birds", vp_now=3
            )
        ],
    )
    phase = game_log_html.PhaseRecord(
        index=0,
        title="Game start",
        kind="game_start",
        round_idx=0,
        active_player_id=None,
        panels=[panel],
        tray=[bird, None],
        feeder_text="2 seed",
        round_goals=[
            game_log_html.RoundGoalInfo(
                round_num=1,
                description="most eggs",
                first_vp=4,
                second_vp=1,
                scored=False,
            )
        ],
        narration=[
            game_log_html.NarrationLine(player_id=None, text="game opens"),
            game_log_html.NarrationLine(player_id=0, text="P0 plays a bird"),
        ],
    )
    return game_log_html.GameLogReport(
        seed=7,
        matchup=("random", "random"),
        player_names=["P0", "P1"],
        final_scores=[11, 9],
        phases=[phase],
    )


def test_render_produces_self_contained_document():
    html = game_log_html.render_game_log_html(_tiny_report())
    # Single self-contained file: a doctype, the embedded data island, the
    # navigation controls, and the 5-column board grid the toggle reads.
    assert html.startswith("<!DOCTYPE html>")
    assert 'id="game-log-data"' in html
    assert 'id="view-toggle"' in html
    assert 'data-view="p0"' in html and 'data-view="p1"' in html
    assert "card-cell" in html
    assert "board-row" in html
    assert "American Robin" in html

    # Change 1: wingspan payload present and traits-line renderer shipped.
    assert '"wingspan_cm":23' in html
    assert "card-traits" in html

    # Change 2: status/egglist markup and helper source shipped.
    assert "card-status" in html
    assert "card-egglist" in html
    assert "' cached'" in html
    assert "' tucked'" in html

    # Change 3: fit-panel wiring present (the actual zoom is runtime JS only —
    # verify visually via `wingspan play --html`).
    assert "state-scaler" in html
    assert "fitStatePanel" in html


def test_embed_json_escapes_left_angle_bracket():
    report = _tiny_report()
    report.phases[0].narration.append(
        game_log_html.NarrationLine(player_id=0, text="if x < y then tuck")
    )
    html = game_log_html.render_game_log_html(report)
    # The payload must never carry a raw '<' that could break out of the
    # <script> data island; the escaped form is what lands in the document.
    payload_start = html.index('id="game-log-data"')
    payload = html[payload_start : html.index("</script>", payload_start)]
    assert "x < y" not in payload
    assert "x \\u003c y" in payload


def test_write_game_log_html_creates_file(tmp_path: pathlib.Path):
    out_path = tmp_path / "nested" / "game.html"
    game_log_html.write_game_log_html(_tiny_report(), out_path)
    assert out_path.exists()
    assert out_path.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")


#### Capture over a full game ####


def test_capture_phases_align_one_to_one_with_log_headers():
    eng, *_ = engine.Engine.create(seed=2024)
    rng = random.Random(2024)
    phases = _run_and_capture(eng, rng)
    header_count = sum(
        1 for entry in eng.state.log_entries if entry.text.startswith(_HEADER_PREFIX)
    )
    # One capture per === header means build_report's per-header segmentation
    # pairs every phase with its decision narration and leaves nothing over.
    assert len(phases) == header_count


def test_capture_kind_counts_match_game_shape():
    eng, *_ = engine.Engine.create(seed=99)
    rng = random.Random(99)
    phases = _run_and_capture(eng, rng)
    kinds = [phase.kind for phase in phases]
    assert kinds.count("game_start") == 1
    assert kinds.count("setup") == 2  # one per seat
    assert kinds.count("round") == 4  # four rounds
    assert kinds.count("game_end") == 1
    assert kinds.count("turn") == len(phases) - 8


def test_every_board_row_is_padded_to_five_columns():
    eng, *_ = engine.Engine.create(seed=11)
    rng = random.Random(11)
    phases = _run_and_capture(eng, rng)
    for phase in phases:
        assert len(phase.panels) == 2
        for panel in phase.panels:
            assert len(panel.rows) == 3  # three habitats
            for row in panel.rows:
                assert len(row.cells) == game_log_html.BOARD_COLUMNS


def test_turn_narration_drops_the_state_summary_block():
    eng, *_ = engine.Engine.create(seed=314)
    rng = random.Random(314)
    phases = _run_and_capture(eng, rng)
    report = game_log_capture.build_report(
        engine=eng, phases=phases, seed=314, matchup=("random", "random")
    )
    turn_phases = [phase for phase in report.phases if phase.kind == "turn"]
    assert turn_phases, "a full game has turns"
    first_turn = turn_phases[0]
    # The verbose board/hand/score summary the engine logs at turn start is
    # dropped; the narration opens at the action, not at a 'Board:' dump.
    assert first_turn.narration, "the turn still has decision narration"
    assert not any(line.text.startswith("Board") for line in first_turn.narration)


def test_narration_lines_carry_seat_attribution():
    eng, *_ = engine.Engine.create(seed=555)
    rng = random.Random(555)
    phases = _run_and_capture(eng, rng)
    report = game_log_capture.build_report(
        engine=eng, phases=phases, seed=555, matchup=None
    )
    all_lines = [line for phase in report.phases for line in phase.narration]
    seat_ids = {line.player_id for line in all_lines}
    # The filter rule (player_id is None or == seat) needs both global (None)
    # and per-seat lines to exist; every id is one of the legal three.
    assert seat_ids <= {None, 0, 1}
    assert 0 in seat_ids and 1 in seat_ids


def _run_and_capture(
    eng: engine.Engine, rng: random.Random
) -> list[game_log_html.PhaseRecord]:
    """Drive one game through a recording handler and return its phase records,
    with the final-scoring phase appended.

    The recorder subscribes to the four mid-game capture events; the final phase
    is captured directly afterwards, exactly as the production handler's
    ``game_end`` does, so the returned list aligns one-to-one with the log's
    ``=== ... ===`` headers."""
    recorder = _PhaseRecorder()
    cfg = instrumentation_config.InstrumentationConfig.model_validate(
        {
            "handlers": {"rec": recorder},
            "events": {event: ["rec"] for event in _CAPTURE_EVENTS[:-1]},
        }
    )
    instrumentation = cfg.build()
    instrumentation.open(
        instrumentation_config.RunContext(
            output_dir=pathlib.Path("."), run_name="t", seed=0
        )
    )
    engine.Engine.play_one_game(
        eng.state,
        (agents.random_agent(rng), agents.random_agent(rng)),
        instrumentation=instrumentation,
    )
    phases = list(recorder.captured)
    phases.append(
        game_log_capture.capture_phase(
            eng, index=len(phases), title="Final scoring", kind="game_end", active=None
        )
    )
    return phases


#### CLI flag ####


def test_cli_html_flag_writes_a_viewer(tmp_path: pathlib.Path):
    from wingspan import cli

    out_path = tmp_path / "game.html"
    code = cli.main_play(
        [
            "--p0",
            "random",
            "--p1",
            "random",
            "--seed",
            "3",
            "--quiet",
            "--html",
            str(out_path),
            "--instrument-out",
            str(tmp_path),
        ]
    )
    assert code == 0
    written = tmp_path / "game.html"
    assert written.exists()
    text = written.read_text(encoding="utf-8")
    assert text.startswith("<!DOCTYPE html>")
    assert 'id="game-log-data"' in text
