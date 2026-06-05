"""Tests for the power-granted extra play.

An extra play is optional, so each credit with a legal play opens with an
``AcceptExchangeDecision`` (take the play or forfeit the credit, scored by the
``SKIP_OPTIONAL`` head). On accept, the play is offered as the same
``PlayBirdDecision`` menu the main action's ``PLAY_BIRD`` branch uses — one
``(bird, habitat)`` ``PlayBirdChoice`` per legal pair, scored by the
``PLAY_BIRD`` head — with the costs resolving as further follow-ups. No habitat
actions are offered (an extra play can only play a bird), and there is no
separate action-type pick (that happens only for the turn's main action).
"""

from __future__ import annotations

import os
import random
import sys
import typing

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import cards, decisions, engine, state
from wingspan.engine import actions


def _two_habitat_single_food_bird(birds: list[cards.Bird]) -> cards.Bird:
    """A non-WHITE bird playable in exactly two habitats whose cost is one
    specific food token — so a one-of-that-food stash affords it and the
    extra-play menu holds two ``PlayBirdChoice``s (one per habitat). Two options
    make the menu a genuine fork the engine actually presents: a single-option
    menu is forced and would be auto-resolved by ``Engine.ask`` without ever
    reaching the agent. Non-WHITE so playing it fires no when-played power."""
    for bird in birds:
        cost = bird.food_cost
        if (
            len(bird.habitats) == 2
            and bird.color != cards.PowerColor.WHITE
            and cost.wild == 0
            and cost.total == 1
        ):
            return bird
    raise AssertionError("no single-food two-habitat non-white bird in catalog")


def test_extra_play_offered_as_play_menu_and_plays_the_bird():
    birds, bonuses, goals = cards.load_all()
    gs = state.new_game(random.Random(0), birds, bonuses, goals)
    eng = engine.Engine(gs)
    gs.current_player = 0
    player = gs.me()

    bird = _two_habitat_single_food_bird(birds)
    food = next(f for f in cards.ALL_FOODS if bird.food_cost.specific_of(f) == 1)
    player.hand = [bird]
    for any_food in cards.ALL_FOODS:
        player.food[any_food] = 0
    player.food[food] = 1
    # Empty board -> the first bird in a row costs 0 eggs, so the play is
    # affordable with just the one food token.
    for habitat in cards.ALL_HABITATS:
        player.board[habitat] = []

    gs.turn_extra_plays = [None]

    sink: list[decisions.Decision[typing.Any]] = []

    def agent[C: decisions.Choice](
        _eng: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        sink.append(decision)
        # Accept the extra-play offer; otherwise pick the first bird play.
        return typing.cast(
            C,
            next(
                c
                for c in decision.choices
                if isinstance(c, (decisions.PayCostChoice, decisions.PlayBirdChoice))
            ),
        )

    eng.agents = [agent, agent]
    actions.consume_extra_plays(eng, player, agent)

    # The optional extra play opens with the take-it-or-leave-it accept, whose
    # only term is the +1 bird play; it routes to the skip-optional head.
    accepts = [d for d in sink if isinstance(d, decisions.AcceptExchangeDecision)]
    assert len(accepts) == 1, "expected exactly one extra-play accept decision"
    accept_option = next(
        c for c in accepts[0].choices if isinstance(c, decisions.PayCostChoice)
    )
    assert accept_option.gained_play_count == 1
    assert (
        decisions.family_for(decisions.AcceptExchangeDecision)
        == decisions.DecisionFamily.SKIP_OPTIONAL
    )

    extra = [d for d in sink if isinstance(d, decisions.PlayBirdDecision)]
    assert len(extra) == 1, "expected exactly one extra-play decision"
    # The menu is Decision[PlayBirdChoice] — every option is a bird play, and no
    # habitat actions are offered for an extra play.
    assert extra[0].choices
    # It routes to the play-bird head.
    assert (
        decisions.family_for(decisions.PlayBirdDecision)
        == decisions.DecisionFamily.PLAY_BIRD
    )
    # The bird actually moved from hand to board, paying its food. The payment
    # itself never reached the agent: with one food token there is exactly one
    # legal payment, so the PayBirdFoodDecision was forced and auto-resolved.
    assert bird not in player.hand
    assert any(pb.bird is bird for row in player.board.values() for pb in row)
    assert player.food[food] == 0
    assert not any(isinstance(d, decisions.PayBirdFoodDecision) for d in sink)


def test_extra_play_wasted_when_no_legal_play():
    """With no playable bird the credit is wasted and the agent is never asked
    for a play (not even the accept — there is nothing to accept)."""
    birds, bonuses, goals = cards.load_all()
    gs = state.new_game(random.Random(1), birds, bonuses, goals)
    eng = engine.Engine(gs)
    gs.current_player = 0
    player = gs.me()
    player.hand = []  # nothing to play
    gs.turn_extra_plays = [None]

    def agent[C: decisions.Choice](
        _eng: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:  # pragma: no cover - must not be consulted
        raise AssertionError("no extra-play decision should be offered with empty hand")

    eng.agents = [agent, agent]
    actions.consume_extra_plays(eng, player, agent)
    assert gs.turn_extra_plays == []


def test_extra_play_can_be_declined():
    """Answering the accept with ``SkipChoice`` forfeits the credit: the bird
    stays in hand, no food is spent, and no play menu is ever offered."""
    birds, bonuses, goals = cards.load_all()
    gs = state.new_game(random.Random(2), birds, bonuses, goals)
    eng = engine.Engine(gs)
    gs.current_player = 0
    player = gs.me()

    bird = _two_habitat_single_food_bird(birds)
    food = next(f for f in cards.ALL_FOODS if bird.food_cost.specific_of(f) == 1)
    player.hand = [bird]
    for any_food in cards.ALL_FOODS:
        player.food[any_food] = 0
    player.food[food] = 1
    for habitat in cards.ALL_HABITATS:
        player.board[habitat] = []

    gs.turn_extra_plays = [None]

    sink: list[decisions.Decision[typing.Any]] = []

    def agent[C: decisions.Choice](
        _eng: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        sink.append(decision)
        return typing.cast(
            C,
            next(c for c in decision.choices if isinstance(c, decisions.SkipChoice)),
        )

    eng.agents = [agent, agent]
    actions.consume_extra_plays(eng, player, agent)

    assert [type(d) for d in sink] == [decisions.AcceptExchangeDecision]
    assert bird in player.hand
    assert player.food[food] == 1
    assert gs.turn_extra_plays == []


def test_habitat_restricted_credit_only_offers_matching_habitat_birds():
    """A credit with a habitat restriction (e.g. FOREST) must only offer birds
    whose legal habitats include that one — birds only in WETLAND or GRASSLAND
    must not appear in the play menu (gap #9)."""
    birds, bonuses, goals = cards.load_all()
    gs = state.new_game(random.Random(3), birds, bonuses, goals)
    eng = engine.Engine(gs)
    gs.current_player = 0
    player = gs.me()

    # Find a bird playable ONLY in wetland (not forest).
    wetland_only_bird = next(
        bird
        for bird in birds
        if bird.habitats == (cards.Habitat.WETLAND,)
        and bird.color != cards.PowerColor.WHITE
        and bird.food_cost.total <= 1
    )
    # Find TWO distinct birds playable in forest. Two are required so the
    # PlayBirdDecision has >= 2 choices; Engine.ask auto-resolves single-choice
    # menus without calling the agent, which would leave captured_plays empty.
    forest_birds: list[cards.Bird] = []
    for bird in birds:
        if (
            cards.Habitat.FOREST in bird.habitats
            and bird.color != cards.PowerColor.WHITE
            and bird.food_cost.total <= 1
            and bird is not wetland_only_bird
        ):
            forest_birds.append(bird)
            if len(forest_birds) == 2:
                break
    assert len(forest_birds) == 2, "need >= 2 affordable forest birds in catalog"
    forest_bird, forest_bird_2 = forest_birds

    player.hand = [wetland_only_bird, forest_bird, forest_bird_2]
    for any_food in cards.ALL_FOODS:
        player.food[any_food] = 1
    for habitat in cards.ALL_HABITATS:
        player.board[habitat] = []

    # Forest-restricted credit.
    gs.turn_extra_plays = [cards.Habitat.FOREST]

    captured_plays: list[decisions.PlayBirdDecision] = []

    def agent[C: decisions.Choice](
        _eng: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        if isinstance(decision, decisions.PlayBirdDecision):
            captured_plays.append(decision)
        for choice in decision.choices:
            if not isinstance(choice, (decisions.SkipChoice,)):
                return typing.cast(C, choice)
        return typing.cast(C, decision.choices[0])

    eng.agents = [agent, agent]
    actions.consume_extra_plays(eng, player, agent)

    assert len(captured_plays) == 1
    offered_birds = [c.bird for c in captured_plays[0].choices]
    assert forest_bird in offered_birds, "forest bird must be offered for forest credit"
    assert (
        forest_bird_2 in offered_birds
    ), "second forest bird must also be offered for forest credit"
    assert (
        wetland_only_bird not in offered_birds
    ), "wetland-only bird must NOT be offered for a forest-restricted credit"
