"""Coverage for the HTML game-log feature: renderer, capture, and CLI wiring.

Three layers are exercised:

* the pure renderer in :mod:`wingspan.reporting.game_log_html` (primitives in,
  self-contained HTML out — no engine needed);
* the engine-aware capture path driven by the
  :class:`~wingspan.instrumentation.handlers.game_log_html.GameLogHtmlHandler`
  over a full random game (phase alignment, 3x5 grids, log-item slicing,
  per-item seat attribution);
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

from wingspan import agents, cards, engine, state
from wingspan.instrumentation import config as instrumentation_config
from wingspan.instrumentation import events as instrumentation_events
from wingspan.reporting import game_log_capture, game_log_html

if typing.TYPE_CHECKING:
    from wingspan.engine import core

_HEADER_PREFIX = "==="
_CAPTURE_EVENTS = [
    "game_start",
    "setup_start",
    "round_start",
    "turn_start",
    "game_end",
]


class _PhaseRecorder(
    instrumentation_events.GameStartHandler,
    instrumentation_events.SetupStartHandler,
    instrumentation_events.RoundStartHandler,
    instrumentation_events.TurnStartHandler,
):
    """A test-only handler that snapshots a phase at each mid-game capture event.

    It mirrors the mid-game events the production ``GameLogHtml`` handler
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

    def setup_start(
        self,
        *,
        engine: core.Engine,
        player: state.Player,
        dealt_bonus: list[cards.BonusCard],
    ) -> None:
        # Mirror the production handler: create one combined setup phase per player.
        self.captured.append(
            game_log_capture.capture_setup_phase(
                engine,
                index=len(self.captured),
                title=f"{player.name} — Setup",
                active=player.id,
                dealt_bonus=dealt_bonus,
            )
        )

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
    """A minimal two-phase report exercising every model field including LogItems."""
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
                name="Bird Bander",
                condition="Birds with a band",
                text="3 / 4 birds",
                vp_now=3,
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
        log_items=[
            game_log_html.LogItem(kind="note", player_id=None, text="Game opens"),
            game_log_html.LogItem(
                kind="decision",
                player_id=0,
                text="Lay eggs",
                options=[
                    game_log_html.DecisionOption(
                        label="Lay eggs", prob=0.7, score=1.2, selected=True
                    ),
                    game_log_html.DecisionOption(
                        label="Gain food", prob=0.3, score=0.5, selected=False
                    ),
                ],
            ),
            game_log_html.LogItem(
                kind="forced",
                player_id=0,
                text="Draw from the deck",
                forced=True,
            ),
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


def test_render_decision_log_items_embedded():
    html = game_log_html.render_game_log_html(_tiny_report())
    # The JSON payload must include the new log_items structure.
    assert "log_items" in html
    assert "di-body" in html or "renderLog" in html  # JS still references the structure
    assert "Lay eggs" in html
    assert "Draw from the deck" in html


def test_embed_json_escapes_left_angle_bracket():
    report = _tiny_report()
    report.phases[0].log_items.append(
        game_log_html.LogItem(kind="note", player_id=0, text="if x < y then tuck")
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


#### Renderer — new setup features ####


def test_selected_bird_cell_renders_selected_class():
    """A BirdCellInfo with selected=True renders card-cell … selected in the HTML."""
    report = _tiny_report()
    phase = report.phases[0]
    # Replace hand with one selected bird and one non-selected bird.
    phase.panels[0].hand = [
        game_log_html.BirdCellInfo(
            name="Kept Bird",
            vp=2,
            nest="Cup",
            wingspan_cm=20,
            habitats="Forest",
            food_cost="seed",
            egg_limit=3,
            eggs=0,
            tucked=0,
            cached=0,
            power_color="brown",
            power_text="",
            selected=True,
        ),
        game_log_html.BirdCellInfo(
            name="Discarded Bird",
            vp=1,
            nest="Cup",
            wingspan_cm=18,
            habitats="Grassland",
            food_cost="",
            egg_limit=2,
            eggs=0,
            tucked=0,
            cached=0,
            power_color="white",
            power_text="",
            selected=False,
        ),
    ]
    html = game_log_html.render_game_log_html(report)
    # The payload embeds the selected flag; the JS reads it to add the CSS class.
    assert '"selected":true' in html
    # The JS source renders the selected class.
    assert "selCls" in html


def test_group_log_item_renders_nested_details():
    """A 'group' LogItem renders a collapsible parent with nested child items."""
    report = _tiny_report()
    child = game_log_html.LogItem(
        kind="decision",
        player_id=0,
        text="Discards seed",
        options=[game_log_html.DecisionOption(label="seed", prob=0.9, selected=True)],
    )
    group = game_log_html.LogItem(
        kind="group",
        player_id=0,
        text="Keeps seed, fish",
        children=[child],
    )
    report.phases[0].log_items = [group]
    html = game_log_html.render_game_log_html(report)
    # The payload carries the group item and its children.
    assert '"kind":"group"' in html
    assert "Keeps seed, fish" in html
    assert "Discards seed" in html
    # The JS renders groups with the group-body style.
    assert "di-group-body" in html
    assert "renderLogItem" in html


def test_setup_bonus_options_selected_class():
    """A selected setup bonus option carries the 'selected' class flag in payload."""
    report = _tiny_report()
    report.phases[0].setup_bonus_options = [
        game_log_html.BonusCardInfo(
            name="Cartographer",
            condition="Birds with a claw",
            text="3 / 4 / 5 birds",
            vp_now=0,
            pending=False,
            selected=True,
        ),
        game_log_html.BonusCardInfo(
            name="Bird Counter",
            condition="Birds in wetland",
            text="2 / 4 birds",
            vp_now=0,
            pending=True,
            selected=False,
        ),
    ]
    html = game_log_html.render_game_log_html(report)
    # The selected bonus flag lands in the payload; JS reads it for the CSS class.
    assert '"selected":true' in html
    assert "setup-opt" in html


#### Capture — finalize_setup_phase ####


def _make_phase_with_hand(
    active_id: int, hand_names: list[str]
) -> game_log_html.PhaseRecord:
    """A minimal PhaseRecord with the given hand for the active player."""
    hand = [
        game_log_html.BirdCellInfo(
            name=name,
            vp=1,
            nest="Cup",
            wingspan_cm=20,
            habitats="Forest",
            food_cost="",
            egg_limit=2,
            eggs=0,
            tucked=0,
            cached=0,
            power_color="white",
            power_text="",
        )
        for name in hand_names
    ]
    panel = game_log_html.PlayerPanel(
        player_id=active_id,
        name=f"P{active_id}",
        action_cubes_left=8,
        rows=[
            game_log_html.HabitatRow(
                label="Forest",
                cells=[game_log_html.BoardCell(bird=None)] * 5,
            )
        ]
        * 3,
        hand=hand,
        food=[],
        score=game_log_html.ScoreBreakdown(
            birds=0, eggs=0, tucked=0, cached=0, bonus=0, goals=0, total=0
        ),
        bonus_cards=[],
    )
    return game_log_html.PhaseRecord(
        index=0,
        title="P0 — Setup",
        kind="setup",
        round_idx=None,
        active_player_id=active_id,
        panels=[panel],
        tray=[],
        feeder_text="",
        round_goals=[],
        setup_bonus_options=[
            game_log_html.BonusCardInfo(
                name="Cartographer",
                condition="claw",
                text="3 / 4",
                vp_now=0,
                pending=True,
            ),
            game_log_html.BonusCardInfo(
                name="Bird Counter",
                condition="wetland",
                text="2 / 4",
                vp_now=0,
                pending=True,
            ),
        ],
    )


def test_finalize_setup_phase_highlights_kept_cards():
    """finalize_setup_phase sets selected=True on kept cards only."""
    phase = _make_phase_with_hand(0, ["Robin", "Mallard", "Egret"])
    capture = game_log_capture.SetupCaptureState(
        phase_index=0,
        kept_card_names={"Robin", "Egret"},
        kept_bonus_name="Cartographer",
    )
    game_log_capture.finalize_setup_phase(phase, capture)

    hand = phase.panels[0].hand
    assert hand[0].name == "Robin" and hand[0].selected
    assert hand[1].name == "Mallard" and not hand[1].selected
    assert hand[2].name == "Egret" and hand[2].selected


def test_finalize_setup_phase_marks_kept_bonus():
    """finalize_setup_phase sets selected=True and pending=False on the kept bonus."""
    phase = _make_phase_with_hand(0, ["Robin"])
    capture = game_log_capture.SetupCaptureState(
        phase_index=0,
        kept_card_names={"Robin"},
        kept_bonus_name="Cartographer",
    )
    game_log_capture.finalize_setup_phase(phase, capture)

    opts = phase.setup_bonus_options
    kept = next(bc for bc in opts if bc.name == "Cartographer")
    other = next(bc for bc in opts if bc.name == "Bird Counter")
    assert kept.selected and not kept.pending
    assert not other.selected and other.pending


def test_finalize_setup_phase_food_group_node():
    """finalize_setup_phase builds a food group node when food_items are present."""
    phase = _make_phase_with_hand(0, ["Robin"])
    child = game_log_html.LogItem(kind="decision", player_id=0, text="Discards fish")
    capture = game_log_capture.SetupCaptureState(
        phase_index=0,
        kept_card_names={"Robin"},
        kept_bonus_name=None,
        food_spent=["fish", "rodent"],
        food_items=[child],
    )
    game_log_capture.finalize_setup_phase(phase, capture)

    # A group node should appear in the log items (no keep_item or bonus_item set).
    assert len(phase.log_items) == 1
    group = phase.log_items[0]
    assert group.kind == "group"
    assert "Keeps" in group.text
    assert child in group.children


def test_finalize_setup_phase_assembles_all_three_nodes():
    """finalize_setup_phase produces [keep_item, food_group, bonus_item] when all present."""
    phase = _make_phase_with_hand(0, ["Robin"])
    keep_item = game_log_html.LogItem(kind="decision", player_id=0, text="Keeps Robin")
    bonus_item = game_log_html.LogItem(
        kind="decision", player_id=0, text="Keeps Cartographer"
    )
    child = game_log_html.LogItem(kind="decision", player_id=0, text="Discards fish")
    capture = game_log_capture.SetupCaptureState(
        phase_index=0,
        kept_card_names={"Robin"},
        kept_bonus_name="Cartographer",
        keep_item=keep_item,
        bonus_item=bonus_item,
        food_spent=["fish"],
        food_items=[child],
    )
    game_log_capture.finalize_setup_phase(phase, capture)

    assert len(phase.log_items) == 3
    assert phase.log_items[0] is keep_item
    assert phase.log_items[1].kind == "group"
    assert phase.log_items[2] is bonus_item


#### Capture — _merge_secondary_setup_segments ####


def _make_log_entries(texts: list[str]) -> list[state.LogEntry]:
    """Build a list of LogEntry objects from plain text strings."""
    return [state.LogEntry(text=text, player_id=None) for text in texts]


def test_merge_secondary_setup_segments_folds_bonus_card_header():
    """A CHOOSING BONUS CARD segment is folded into the preceding segment."""
    entries = _make_log_entries(
        [
            "=== SETUP: P0 CHOOSING BIRDS ===",
            "Dealt hand (5): ...",
            "[P0] SetupDecision ...",
            "=== SETUP: P0 CHOOSING BONUS CARD ===",
            "[P0] BirdPowerPickBonusCardDecision ...",
        ]
    )
    # Split into raw segments (two headers = two segments).
    from wingspan.reporting import game_log_capture as glc

    segments = glc._split_log_into_segments(entries)  # type: ignore[attr-defined]
    assert len(segments) == 2

    merged = glc._merge_secondary_setup_segments(segments)  # type: ignore[attr-defined]
    assert len(merged) == 1
    # Header from the primary segment, body entries from both.
    assert merged[0][0].text.startswith("=== SETUP: P0 CHOOSING BIRDS")
    texts = [entry.text for entry in merged[0]]
    assert any("CHOOSING BONUS CARD" not in text for text in texts[1:])
    assert any("BirdPowerPickBonusCardDecision" in text for text in texts)


def test_merge_secondary_setup_segments_no_op_for_combined_regime():
    """In combined regime (single header) the segments are returned unchanged."""
    entries = _make_log_entries(
        [
            "=== SETUP: P0 CHOOSING BIRDS, FOOD, AND BONUS CARD ===",
            "Dealt hand (5): ...",
        ]
    )
    from wingspan.reporting import game_log_capture as glc

    segments = glc._split_log_into_segments(entries)  # type: ignore[attr-defined]
    merged = glc._merge_secondary_setup_segments(segments)  # type: ignore[attr-defined]
    assert len(merged) == len(segments) == 1


#### Capture over a full game ####


def test_capture_phases_align_one_to_one_with_log_headers():
    eng, *_ = engine.Engine.create(seed=2024)
    rng = random.Random(2024)
    phases = _run_and_capture(eng, rng)
    # In the combined/random regime there are no secondary setup headers, so
    # raw count == merged count; phases should align 1:1 with merged segments.
    raw_headers = sum(
        1 for entry in eng.state.log_entries if entry.text.startswith(_HEADER_PREFIX)
    )
    assert len(phases) == raw_headers


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


def test_turn_log_drops_the_state_summary_block():
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
    # dropped; log items open at the action, not at a 'Board:' dump.
    all_texts = [item.text for item in first_turn.log_items]
    assert not any("Board" in text for text in all_texts)


def test_log_items_carry_seat_attribution():
    eng, *_ = engine.Engine.create(seed=555)
    rng = random.Random(555)
    phases = _run_and_capture(eng, rng)
    report = game_log_capture.build_report(
        engine=eng, phases=phases, seed=555, matchup=None
    )
    all_items = [item for phase in report.phases for item in phase.log_items]
    seat_ids = {item.player_id for item in all_items}
    # Both per-seat (0, 1) and global (None) items must appear; no other ids.
    assert seat_ids <= {None, 0, 1}
    # With two active seats there must be items attributed to each.
    assert 0 in seat_ids and 1 in seat_ids


def _run_and_capture(
    eng: engine.Engine, rng: random.Random
) -> list[game_log_html.PhaseRecord]:
    """Drive one game through a recording handler and return its phase records,
    with the final-scoring phase appended.

    The recorder subscribes to setup_start (one phase per player), round_start,
    and turn_start; the final phase is captured directly afterwards, exactly as
    the production handler's ``game_end`` does, so the returned list aligns
    one-to-one with the (merged) log's ``=== ... ===`` headers."""
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
