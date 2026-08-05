"""Smoke tests for the plaintext game-event tree renderer.

Verifies that ``render_plaintext`` produces a well-formed human-readable log
from a full random game.  Tests check structural invariants (phase headers, event
header patterns, decision / forced / effect prefixes) rather than exact text, so
they remain valid as the renderer's output evolves.
"""

from __future__ import annotations

import random
import re
import sys

# Add src/ to path as per test_smoke.py convention.
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "src"))


def _run_and_render(seed: int = 42) -> str:
    """Run one random game, build the event tree, and return the plaintext render."""
    from wingspan import agents
    from wingspan import engine as engine_mod
    from wingspan.gamelog import recorder as gamelog_recorder
    from wingspan.gamelog import render_text

    rec = gamelog_recorder.EventRecorder(
        probes=(None, None),
    )
    eng, *_ = engine_mod.Engine.create(seed=seed)
    rng = random.Random(seed)
    engine_mod.Engine.play_one_game(
        eng.state,
        (agents.random_agent(rng), agents.random_agent(rng)),
        event_recorder=rec,
    )
    return render_text.render_plaintext(rec.root)


def test_output_is_nonempty():
    """render_plaintext must return a non-empty string for any complete game."""
    text = _run_and_render()
    assert text.strip(), "render_plaintext returned an empty string"


def test_phase_headers_present():
    """All five phase kinds must appear as === KIND === headers."""
    text = _run_and_render()
    for kind in ("GAME_START", "SETUP", "ROUND", "TURN", "GAME_END"):
        assert f"=== {kind} ===" in text, f"Missing phase header for {kind!r}"


def test_decision_outcomes_present():
    """At least one decision outcome line (→ text) must appear."""
    text = _run_and_render()
    decision_lines = [line for line in text.splitlines() if "→ " in line]
    assert decision_lines, "No decision outcome lines (→) found in plaintext output"


def test_activate_brown_events_labeled():
    """Every bird crossed gets a header — powered ones tagged '(brown)', the
    rest explicitly marked as having no power."""
    text = _run_and_render()
    lines = text.splitlines()
    brown_lines = [line for line in lines if "(brown):" in line]
    assert brown_lines, "No brown-power event headers ('(brown):') found"
    powerless_lines = [line for line in lines if "— no brown power]" in line]
    assert powerless_lines, (
        "No headers for birds without a brown power — every bird crossed in an "
        "activated row must get its own line, powered or not"
    )


def test_main_action_events_name_their_habitat():
    """The main-action header must name the row the cube was spent on."""
    text = _run_and_render()
    main_lines = [
        line for line in text.splitlines() if line.lstrip().startswith("[Main action:")
    ]
    assert main_lines, "No [Main action: ...] event lines found"
    habitats = {"Forest", "Grassland", "Wetland"}
    assert any(
        any(habitat in line for habitat in habitats) for line in main_lines
    ), f"No main-action header names a habitat; got {main_lines[:3]}"


def test_event_headers_summarize_their_effects():
    """An event's header must report what its ledger says happened."""
    text = _run_and_render()
    # A habitat gain-food activation folds its recorded food gains into a
    # 'Gains ...' header rather than repeating the raw effect rows.
    assert any(
        line.lstrip().startswith("[Gains ") for line in text.splitlines()
    ), "No effect-derived '[Gains ...]' header found"


def test_deck_draws_name_the_card_drawn():
    """A draw from the deck must name the card in both the header and a row.

    The draw decision itself only says 'from the deck' — the identity of the
    card is hidden information the ledger is the sole record of."""
    lines = _run_and_render().splitlines()
    deck_draws = [
        (idx, match.group(1))
        for idx, line in enumerate(lines)
        if (match := re.search(r"· draw_card\(card=(.+?), source=deck\)", line))
    ]
    assert deck_draws, "No deck draws recorded in the rendered log"
    for idx, card in deck_draws:
        headers = [line for line in lines[:idx] if line.lstrip().startswith("[")]
        assert headers, f"deck draw of {card!r} sits under no event header"
        assert card in headers[-1], (
            "The header above a deck draw must name the drawn card; "
            f"{card!r} missing from {headers[-1]!r}"
        )


def test_setup_events_show_kept_cards():
    """Setup events must show kept cards in their label when cards were kept."""
    text = _run_and_render()
    setup_lines = [
        line for line in text.splitlines() if line.lstrip().startswith("[Setup")
    ]
    assert setup_lines, "No [Setup...] event lines found"
    # At least one setup event should include 'kept:' (some cards were selected).
    kept_lines = [line for line in setup_lines if "kept:" in line]
    assert kept_lines, "No setup event shows kept cards — expected 'kept: ...' in label"


def test_round_goal_events_labeled():
    """Round goal events must appear as [Round N goal — ...] with seat data."""
    text = _run_and_render()
    goal_lines = [
        line for line in text.splitlines() if "goal —" in line.lstrip().lstrip("[")
    ]
    assert goal_lines, "No round goal event lines found"
    # Exactly four rounds → at least four goal events.
    assert (
        len(goal_lines) >= 4
    ), f"Expected at least 4 round goal events, found {len(goal_lines)}"


def test_final_scoring_event_labeled():
    """A Final scoring event must appear in the GAME_END phase."""
    text = _run_and_render()
    final_lines = [
        line for line in text.splitlines() if line.lstrip().startswith("[Final scoring")
    ]
    assert final_lines, "No [Final scoring ...] event line found"


def test_forced_moves_use_bang_prefix():
    """Forced (single-choice) sub-events must render with the ! prefix."""
    # Run several seeds to increase the chance of hitting a forced move.
    forced_found = False
    for seed in (42, 99, 123, 777, 2024):
        text = _run_and_render(seed=seed)
        if any("  ! " in line for line in text.splitlines()):
            forced_found = True
            break
    assert forced_found, (
        "No forced sub-event lines (  ! ) found across 5 random games — "
        "expected at least one forced single-choice move"
    )


def test_nesting_depth_increases_for_children():
    """Child events (e.g. a white power under a bird play) must be indented
    more than their parent."""
    checked = False
    for seed in (42, 99, 123, 777):
        lines = _run_and_render(seed=seed).splitlines()
        for idx, line in enumerate(lines):
            if "(white):" not in line and "(white) —" not in line:
                continue
            leading = len(line) - len(line.lstrip())
            parent_found = any(
                (len(prev_line) - len(prev_line.lstrip())) < leading
                and prev_line.lstrip().startswith("[Plays ")
                for prev_line in lines[:idx]
            )
            assert parent_found, "White power not indented under its bird play"
            checked = True
            break
        if checked:
            break
    assert checked, "No white-power event found across 4 random games"
