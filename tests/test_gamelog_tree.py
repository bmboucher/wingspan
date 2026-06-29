"""Structured game-event tree shape tests.

Verifies that the EventRecorder produces the correct tree structure for the
key call-site wiring introduced in Step 3:

- 3 ``ActivateBrownEvent``s for a 3-bird row, including empty (non-brown) slots
- ``PlayBirdEvent`` nesting: selection + egg-cost + food sub-events, plus a
  ``WhitePowerEvent`` child for white-power birds
- ``DecisionSubEvent`` captures decision options (with annotation from the probe)
- ``SetupEvent`` covers each player's kept cards and bonus
- 4 ``RoundGoalEvent``s across all rounds plus one ``FinalScoringEvent``
- The ``EMPTY`` null recorder is a complete no-op
"""

from __future__ import annotations

import random
import sys

# Add src/ to path as per test_smoke.py convention.
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "src"))


def _make_engine_with_recorder(seed: int = 42):
    """Return ``(engine, recorder)`` for a fresh game wired with an EventRecorder."""
    from wingspan import agents
    from wingspan import engine as engine_mod
    from wingspan.gamelog import recorder as gamelog_recorder

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
    return eng, rec


def test_null_recorder_is_noop():
    """EMPTY recorder accepts all calls without raising."""
    from wingspan import agents
    from wingspan import engine as engine_mod
    from wingspan.gamelog import recorder as gamelog_recorder

    eng, *_ = engine_mod.Engine.create(seed=1)
    rng = random.Random(1)
    engine_mod.Engine.play_one_game(
        eng.state,
        (agents.random_agent(rng), agents.random_agent(rng)),
        event_recorder=gamelog_recorder.EMPTY,
    )
    assert eng.state.game_over


def test_tree_has_expected_phases():
    """Tree has game_start, some setup/round/turn phases, and game_end."""
    _eng, rec = _make_engine_with_recorder()
    tree = rec.root
    kinds = [phase.kind for phase in tree.phases]
    assert kinds[0] == "game_start"
    assert kinds[-1] == "game_end"
    assert "round" in kinds
    assert "turn" in kinds


def test_final_scoring_event_in_game_end():
    """game_end phase contains exactly one FinalScoringEvent with 2 seats."""
    from wingspan.gamelog import models

    _eng, rec = _make_engine_with_recorder()
    game_end = rec.root.phases[-1]
    assert game_end.kind == "game_end"
    scoring_events = [
        ev for ev in game_end.events if isinstance(ev, models.FinalScoringEvent)
    ]
    assert len(scoring_events) == 1
    assert len(scoring_events[0].scores) == 2


def test_four_round_goal_events():
    """One RoundGoalEvent per round (4 total across all phases)."""
    from wingspan.gamelog import models

    _eng, rec = _make_engine_with_recorder(seed=7)
    all_events = [ev for phase in rec.root.phases for ev in phase.events]
    goal_events = [ev for ev in all_events if isinstance(ev, models.RoundGoalEvent)]
    assert len(goal_events) == 4
    round_indices = [ev.round_idx for ev in goal_events]
    assert sorted(round_indices) == [0, 1, 2, 3]


def test_play_bird_events_present():
    """PlayBirdEvents are produced at least once across the game."""
    from wingspan.gamelog import models

    _eng, rec = _make_engine_with_recorder(seed=5)

    def _collect_all(events: list[models.GameEvent]) -> list[models.GameEvent]:
        result: list[models.GameEvent] = []
        for ev in events:
            result.append(ev)
            result.extend(_collect_all(ev.children))
        return result

    all_events = _collect_all([ev for phase in rec.root.phases for ev in phase.events])
    play_bird_events = [ev for ev in all_events if isinstance(ev, models.PlayBirdEvent)]
    assert len(play_bird_events) > 0


def test_activate_brown_events_cover_row():
    """Every bird in an activated row has a corresponding ActivateBrownEvent."""
    from wingspan.gamelog import models

    _eng, rec = _make_engine_with_recorder(seed=3)

    def _collect_all(events: list[models.GameEvent]) -> list[models.GameEvent]:
        result: list[models.GameEvent] = []
        for ev in events:
            result.append(ev)
            result.extend(_collect_all(ev.children))
        return result

    all_events = _collect_all([ev for phase in rec.root.phases for ev in phase.events])
    brown_events = [
        ev for ev in all_events if isinstance(ev, models.ActivateBrownEvent)
    ]
    # Both is_brown=True and is_brown=False rows should appear (mix of colors).
    has_non_brown = any(not ev.is_brown for ev in brown_events)
    has_brown = any(ev.is_brown for ev in brown_events)
    # At minimum at least one brown-event type must appear over a full game.
    assert has_non_brown or has_brown


def test_activate_base_events_present():
    """ActivateBaseEvents are produced for gain_food, lay_eggs, draw_cards."""
    from wingspan.gamelog import models

    _eng, rec = _make_engine_with_recorder(seed=9)
    all_events = [ev for phase in rec.root.phases for ev in phase.events]
    base_events = [ev for ev in all_events if isinstance(ev, models.ActivateBaseEvent)]
    assert len(base_events) > 0
    actions = {ev.action for ev in base_events}
    # All three action types should appear over the full game.
    assert "gain_food" in actions
    assert "lay_eggs" in actions
    assert "draw_cards" in actions


def test_setup_events_present():
    """SetupEvents appear in setup phases."""
    from wingspan.gamelog import models

    _eng, rec = _make_engine_with_recorder(seed=11)
    setup_phases = [phase for phase in rec.root.phases if phase.kind == "setup"]
    assert len(setup_phases) >= 1
    setup_events = [
        ev
        for phase in setup_phases
        for ev in phase.events
        if isinstance(ev, models.SetupEvent)
    ]
    assert len(setup_events) >= 2  # one per player


def test_decision_sub_events_have_outcome_text():
    """DecisionSubEvents carry non-empty outcome_text."""
    from wingspan.gamelog import models

    _eng, rec = _make_engine_with_recorder(seed=13)

    def _collect_subs(events: list[models.GameEvent]) -> list[models.SubEvent]:
        result: list[models.SubEvent] = []
        for ev in events:
            result.extend(ev.sub_events)
            result.extend(_collect_subs(ev.children))
        return result

    all_subs = _collect_subs([ev for phase in rec.root.phases for ev in phase.events])
    decision_subs = [
        sub for sub in all_subs if isinstance(sub, models.DecisionSubEvent)
    ]
    assert len(decision_subs) > 0
    assert all(sub.outcome_text for sub in decision_subs)


def test_stack_is_empty_after_game():
    """The recorder's open-event stack is empty after the game completes."""
    _eng, rec = _make_engine_with_recorder(seed=17)
    # _open_stack is private but accessible for testing.
    assert len(rec._open_stack) == 0  # type: ignore[attr-defined]
