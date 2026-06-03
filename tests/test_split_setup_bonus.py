"""Tests for the ``split_setup_bonus`` regime.

When the opening bonus pick is deferred, the engine resolves it through the
in-game ``CHOOSE_BONUS`` head (``BirdPowerPickBonusCardDecision``) over the
already-kept cards/food rather than baking it into the setup keep. The deferral
signal is purely data-driven: a ``SetupChoice`` whose ``bonus_card is None``
while bonus cards were dealt.

These tests use only the public engine surface. A *setup-time* bonus pick is
identifiable without reaching into private methods: it is asked over the
pre-loaded round-1 opening (``round_idx == 0`` with the deciding player still
holding a full ``ROUND_CUBES[0]`` action cubes), whereas any mid-game bonus
draft is asked of the active player *after* a cube has been spent — so the cube
count cleanly separates the two.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import cards, decisions, engine, setup_model, state  # noqa: E402
from wingspan.training import collect  # noqa: E402

# One recorded bonus-card decision: (player_id, deciding player's action cubes,
# round index) captured at ask time so setup picks are told apart from mid-game.
type _BonusEvent = tuple[int, int, int]


def _keep_no_cards(bonus_card: cards.BonusCard | None) -> setup_model.SetupCandidate:
    """Keep zero cards (so all five foods are retained) with the given bonus —
    a minimal legal keep that isolates the bonus-deferral behaviour."""
    return setup_model.SetupCandidate(
        kept_cards=(),
        kept_foods=tuple(cards.ALL_FOODS),
        bonus_card=bonus_card,
    )


def _recording_agent(
    bonus_events: list[_BonusEvent],
    setup_decisions: list[decisions.SetupDecision],
) -> engine.Agent:
    """A first-choice agent that logs every setup decision it is offered and the
    context of every bonus-card decision (so setup vs mid-game picks are told
    apart). The chosen choice is captured *before* any ``isinstance`` narrowing so
    its type stays the decision's ``Choice`` parameter."""

    def agent[C: decisions.Choice](
        eng: engine.Engine, decision: decisions.Decision[C]
    ) -> C:
        chosen = decision.choices[0]
        if isinstance(decision, decisions.SetupDecision):
            setup_decisions.append(decision)
        elif isinstance(decision, decisions.BirdPowerPickBonusCardDecision):
            deciding = eng.state.players[decision.player_id]
            bonus_events.append(
                (decision.player_id, deciding.action_cubes_left, eng.state.round_idx)
            )
        return chosen

    return agent


def _setup_time_picks(bonus_events: list[_BonusEvent]) -> list[_BonusEvent]:
    """The bonus picks made during setup — round 0 with a full opening cube count
    (a cube is spent before any in-game bonus power can fire)."""
    return [
        event
        for event in bonus_events
        if event[1] == state.ROUND_CUBES[0] and event[2] == 0
    ]


def test_fixed_setup_defers_bonus_to_in_game_pick():
    """Setup-model path: bonus-free keeps (the split regime) drive one in-game
    ``CHOOSE_BONUS`` pick per seat over the pre-loaded round-1 opening."""
    bonus_events: list[_BonusEvent] = []
    setup_decisions: list[decisions.SetupDecision] = []
    agent = _recording_agent(bonus_events, setup_decisions)

    def chooser(
        _engine: engine.Engine,
        dealt: tuple[tuple[list[cards.Bird], list[cards.BonusCard]], ...],
    ) -> list[setup_model.SetupCandidate]:
        return [_keep_no_cards(None) for _ in dealt]

    eng = collect.new_engine(7)
    engine.Engine.play_one_game_with_setups(eng.state, (agent, agent), chooser)

    # The chooser bypasses the SetupDecision entirely, and exactly one deferred
    # bonus pick is made per seat at the opening (full cubes, round 0).
    assert setup_decisions == []
    setup_picks = _setup_time_picks(bonus_events)
    assert {player_id for player_id, _cubes, _round in setup_picks} == {0, 1}
    assert len(setup_picks) == 2
    # Each seat ends up owning a bonus card (the deferred pick was applied).
    assert all(player.bonus_cards for player in eng.state.players)


def test_play_one_game_split_offers_bonus_free_setup_and_defers():
    """Agent-asked path (eval / manual): ``split_setup_bonus=True`` strips the
    bonus axis from the ``SetupDecision`` and defers it to an in-game pick."""
    bonus_events: list[_BonusEvent] = []
    setup_decisions: list[decisions.SetupDecision] = []
    agent = _recording_agent(bonus_events, setup_decisions)

    eng = collect.new_engine(11)
    engine.Engine.play_one_game(eng.state, (agent, agent), split_setup_bonus=True)

    assert len(setup_decisions) == 2  # one per seat
    for decision in setup_decisions:
        assert len(decision.choices) == 252  # the bonus axis (×2) is dropped
        assert all(choice.bonus_card is None for choice in decision.choices)
    # One deferred bonus pick per seat, at the opening.
    setup_picks = _setup_time_picks(bonus_events)
    assert {player_id for player_id, _cubes, _round in setup_picks} == {0, 1}


def test_default_setup_keeps_combined_bonus_and_does_not_defer():
    """Regression guard: with the flag off the ``SetupDecision`` carries the full
    bonus axis and no setup-time ``CHOOSE_BONUS`` pick fires."""
    bonus_events: list[_BonusEvent] = []
    setup_decisions: list[decisions.SetupDecision] = []
    agent = _recording_agent(bonus_events, setup_decisions)

    eng = collect.new_engine(11)
    engine.Engine.play_one_game(eng.state, (agent, agent))

    assert len(setup_decisions) == 2
    for decision in setup_decisions:
        assert len(decision.choices) == 504  # combined keep includes the bonus
        assert any(choice.bonus_card is not None for choice in decision.choices)
    # The bonus was applied inside the setup keep, so nothing is deferred.
    assert _setup_time_picks(bonus_events) == []
