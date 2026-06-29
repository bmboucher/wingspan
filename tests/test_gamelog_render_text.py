"""Smoke tests for the plaintext game-event tree renderer.

Verifies that ``render_plaintext`` produces a well-formed human-readable log
from a full random game.  Tests check structural invariants (phase headers, event
label patterns, decision / forced / note prefixes) rather than exact text, so
they remain valid as the renderer's output evolves.
"""

from __future__ import annotations

import random
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
        seat_configs=(None, None),
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
    """Brown-power slot events must appear as [Brown: ...] or [——: ...]."""
    text = _run_and_render()
    # At least one brown-power block must exist after any bird rows are activated.
    brown_lines = [
        line
        for line in text.splitlines()
        if line.lstrip().startswith("[Brown:") or line.lstrip().startswith("[——:")
    ]
    assert brown_lines, "No ActivateBrownEvent lines ([Brown:] or [——:]) found"


def test_activate_base_events_labeled():
    """Habitat base-activation events must appear as [Activate ... (...)]."""
    text = _run_and_render()
    base_lines = [
        line for line in text.splitlines() if line.lstrip().startswith("[Activate ")
    ]
    assert base_lines, "No ActivateBaseEvent lines ([Activate ...]) found"


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
    """Child events (e.g. WhitePowerEvent under PlayBirdEvent) must be indented
    more than their parent."""
    # White power events appear as [White power: ...] and are children of PlayBirdEvent.
    text = _run_and_render(seed=42)
    for seed in (42, 99, 123, 777):
        text = _run_and_render(seed=seed)
        lines = text.splitlines()
        for idx, line in enumerate(lines):
            if "[White power:" in line:
                leading = len(line) - len(line.lstrip())
                # The parent PlayBirdEvent must have less indentation.
                parent_found = any(
                    (len(prev_line) - len(prev_line.lstrip())) < leading
                    and "[PlayBirdEvent]" in prev_line
                    for prev_line in lines[:idx]
                )
                assert (
                    parent_found
                ), "WhitePowerEvent not properly indented under PlayBirdEvent"
                break
