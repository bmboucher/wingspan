"""Tests for the habitat action reward tracks and their one-step resource trade.

The printed player mat awards a growing base reward as a habitat row fills, and
puts a trade arrow on every other action space (so the cube lands on one only
when the row holds an odd number of birds). The trade is a single exchange, not
a repeatable loop. These tests pin the exact reward table the rules specify and
verify the engine offers the conversion exactly once, only on a trade space.

Reference table (forest, from the rules):

    0 birds -> gain 1 food
    1 bird  -> gain 1 food or discard 1 card to gain 2 food
    2 birds -> gain 2 food
    3 birds -> gain 2 food or discard 1 card to gain 3 food
    4 birds -> gain 3 food
    5 birds -> gain 3 food or discard 1 card to gain 4 food

Wetland mirrors forest (draw cards, discard an egg); grassland mirrors it but
starts at 2 eggs and spends a food.
"""

from __future__ import annotations

import os
import random
import sys
import typing

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import cards, decisions, engine, state  # noqa: E402
from wingspan.engine import actions  # noqa: E402

# Expected base reward keyed on the number of birds already in the row (0..5),
# transcribed straight from the printed mat.
_FOREST_FOOD = {0: 1, 1: 1, 2: 2, 3: 2, 4: 3, 5: 3}
_GRASSLAND_EGGS = {0: 2, 1: 2, 2: 3, 3: 3, 4: 4, 5: 4}
_WETLAND_CARDS = {0: 1, 1: 1, 2: 2, 3: 2, 4: 3, 5: 3}
# The trade arrow is reached on odd-indexed slots only.
_TRADE_AVAILABLE = {0: False, 1: True, 2: False, 3: True, 4: False, 5: True}


def _make_engine() -> engine.Engine:
    birds, bonuses, goals = cards.load_all()
    rng = random.Random(0)
    gs = state.new_game(rng, birds, bonuses, goals)
    return engine.Engine(gs)


def _non_brown_bird() -> cards.Bird:
    """A bird whose row-power activation is a no-op, so filling a row with it
    leaves the action's reward/convert logic as the only thing under test."""
    birds, _, _ = cards.load_all()
    return next(bird for bird in birds if bird.color != cards.PowerColor.BROWN)


def _roomy_bird() -> cards.Bird:
    """The bird with the most egg capacity, so a grassland row of one still has
    an open slot after the base lay."""
    birds, _, _ = cards.load_all()
    return max(birds, key=lambda bird: bird.egg_limit)


def _fill_row(
    board: state.Board, habitat: cards.Habitat, count: int, bird: cards.Bird
) -> None:
    board[habitat] = [state.PlayedBird(bird=bird) for _ in range(count)]


def _accepting_agent(
    sink: list[decisions.Decision[typing.Any]],
) -> engine.Agent:
    """Records every decision and always takes the first non-skip option, so it
    accepts every offered trade. Against a single-shot conversion it still only
    ever sees one convert decision; against a (buggy) repeating loop it would
    accept until it ran out of resources."""

    def agent[C: decisions.Choice](
        _eng: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        sink.append(decision)
        for choice in decision.choices:
            if not isinstance(choice, decisions.SkipChoice):
                return choice
        return decision.choices[0]

    return agent


# ---------------------------------------------------------------------------
# Base reward track + trade availability (pure Board logic)


def test_forest_food_track_matches_rules():
    bird = _non_brown_bird()
    for count, expected in _FOREST_FOOD.items():
        board = state.Board()
        _fill_row(board, cards.Habitat.FOREST, count, bird)
        assert board.gain_food_count() == expected, f"{count} birds"
        assert (
            board.action_offers_convert(cards.Habitat.FOREST) == _TRADE_AVAILABLE[count]
        )


def test_grassland_egg_track_matches_rules():
    bird = _non_brown_bird()
    for count, expected in _GRASSLAND_EGGS.items():
        board = state.Board()
        _fill_row(board, cards.Habitat.GRASSLAND, count, bird)
        assert board.lay_eggs_count() == expected, f"{count} birds"
        assert (
            board.action_offers_convert(cards.Habitat.GRASSLAND)
            == _TRADE_AVAILABLE[count]
        )


def test_wetland_card_track_matches_rules():
    bird = _non_brown_bird()
    for count, expected in _WETLAND_CARDS.items():
        board = state.Board()
        _fill_row(board, cards.Habitat.WETLAND, count, bird)
        assert board.draw_cards_count() == expected, f"{count} birds"
        assert (
            board.action_offers_convert(cards.Habitat.WETLAND)
            == _TRADE_AVAILABLE[count]
        )


# ---------------------------------------------------------------------------
# The engine offers the trade once, only on a trade space


def _commit_decisions_for_gain_food(
    n_birds: int, hand_size: int
) -> list[decisions.AcceptExchangeDecision]:
    """Run a Gain Food action and return all AcceptExchangeDecisions raised."""
    eng = _make_engine()
    eng.state.current_player = 0
    player = eng.state.players[0]
    _fill_row(player.board, cards.Habitat.FOREST, n_birds, _non_brown_bird())
    player.hand = [_non_brown_bird() for _ in range(hand_size)]
    for food in cards.Food:
        eng.state.birdfeeder.counts[food] = 5
    sink: list[decisions.Decision[typing.Any]] = []
    actions.do_gain_food(eng, _accepting_agent(sink))
    return [
        decision
        for decision in sink
        if isinstance(decision, decisions.AcceptExchangeDecision)
    ]


def test_gain_food_offers_no_trade_on_even_slot():
    assert _commit_decisions_for_gain_food(n_birds=2, hand_size=3) == []


def test_gain_food_offers_trade_exactly_once_on_odd_slot():
    # Three cards in hand: a repeating loop would convert all three. The single
    # exchange must offer the trade (commit step) exactly once.
    convs = _commit_decisions_for_gain_food(n_birds=1, hand_size=3)
    assert len(convs) == 1


def test_gain_food_skips_trade_on_odd_slot_with_empty_hand():
    assert _commit_decisions_for_gain_food(n_birds=1, hand_size=0) == []


def test_gain_food_commit_then_discard_removes_card_from_hand():
    """Committing to the trade then discarding a card removes it from hand, adds
    it to the discard pile, and the player gains one extra food.

    Note: ``engine.ask`` auto-resolves single-option decisions without consulting
    the agent (the forced-move optimisation). With exactly one card in hand the
    ``DiscardBirdForFoodDecision`` has one choice and is resolved that way; we
    verify the side effects rather than agent-call counts.
    """
    eng = _make_engine()
    eng.state.current_player = 0
    player = eng.state.players[0]
    _fill_row(player.board, cards.Habitat.FOREST, 1, _non_brown_bird())
    the_bird = _non_brown_bird()
    player.hand = [the_bird]
    for food in cards.Food:
        eng.state.birdfeeder.counts[food] = 5

    def agent[C: decisions.Choice](
        _eng: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        # At the commit step: accept the trade.
        if isinstance(decision, decisions.AcceptExchangeDecision):
            commit = next(
                ch for ch in decision.choices if isinstance(ch, decisions.PayCostChoice)
            )
            return typing.cast(C, commit)
        # For all other decisions: first choice.
        return decision.choices[0]

    food_before = player.food.total()
    actions.do_gain_food(eng, agent)

    assert player.hand == []
    assert the_bird in eng.state.bird_discard
    assert player.food.total() > food_before


def test_gain_food_skip_at_commit_leaves_hand_unchanged():
    """Skipping at the commit step leaves the hand intact and raises no
    DiscardBirdForFoodDecision."""
    eng = _make_engine()
    eng.state.current_player = 0
    player = eng.state.players[0]
    _fill_row(player.board, cards.Habitat.FOREST, 1, _non_brown_bird())
    player.hand = [_non_brown_bird(), _non_brown_bird()]
    for food in cards.Food:
        eng.state.birdfeeder.counts[food] = 5

    discard_decisions: list[decisions.DiscardBirdForFoodDecision] = []

    def agent[C: decisions.Choice](
        _eng: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        if isinstance(decision, decisions.AcceptExchangeDecision):
            skip = next(
                ch for ch in decision.choices if isinstance(ch, decisions.SkipChoice)
            )
            return typing.cast(C, skip)
        if isinstance(decision, decisions.DiscardBirdForFoodDecision):
            discard_decisions.append(decision)
            return typing.cast(C, decision.choices[0])
        return decision.choices[0]

    hand_before = list(player.hand)
    actions.do_gain_food(eng, agent)

    assert discard_decisions == []
    assert player.hand == hand_before


def test_lay_eggs_offers_trade_exactly_once_on_odd_slot():
    eng = _make_engine()
    eng.state.current_player = 0
    player = eng.state.players[0]
    _fill_row(player.board, cards.Habitat.GRASSLAND, 1, _roomy_bird())
    for food in cards.Food:
        player.food[food] = 3
    sink: list[decisions.Decision[typing.Any]] = []
    actions.do_lay_eggs(eng, _accepting_agent(sink))
    convs = [
        decision
        for decision in sink
        if isinstance(decision, decisions.LayExtraEggsDecision)
    ]
    assert len(convs) == 1


def test_lay_eggs_offers_no_trade_on_even_slot():
    eng = _make_engine()
    eng.state.current_player = 0
    player = eng.state.players[0]
    _fill_row(player.board, cards.Habitat.GRASSLAND, 2, _roomy_bird())
    for food in cards.Food:
        player.food[food] = 3
    sink: list[decisions.Decision[typing.Any]] = []
    actions.do_lay_eggs(eng, _accepting_agent(sink))
    assert [
        decision
        for decision in sink
        if isinstance(decision, decisions.LayExtraEggsDecision)
    ] == []


def test_draw_cards_offers_trade_exactly_once_on_odd_slot():
    eng = _make_engine()
    eng.state.current_player = 0
    player = eng.state.players[0]
    _fill_row(player.board, cards.Habitat.WETLAND, 1, _roomy_bird())
    # An egg to spend on the trade.
    player.board[cards.Habitat.WETLAND][0].eggs = 1
    sink: list[decisions.Decision[typing.Any]] = []
    actions.do_draw_cards(eng, _accepting_agent(sink))
    convs = [
        decision
        for decision in sink
        if isinstance(decision, decisions.AcceptExchangeDecision)
    ]
    assert len(convs) == 1


def test_draw_cards_offers_no_trade_on_even_slot():
    eng = _make_engine()
    eng.state.current_player = 0
    player = eng.state.players[0]
    _fill_row(player.board, cards.Habitat.WETLAND, 2, _roomy_bird())
    player.board[cards.Habitat.WETLAND][0].eggs = 1
    sink: list[decisions.Decision[typing.Any]] = []
    actions.do_draw_cards(eng, _accepting_agent(sink))
    assert [
        decision
        for decision in sink
        if isinstance(decision, decisions.AcceptExchangeDecision)
    ] == []


# ---------------------------------------------------------------------------
# Display: the action line shows the trade only on a trade space


def test_board_render_shows_trade_only_on_trade_space():
    from wingspan.agents import display

    bird = _non_brown_bird()
    # One bird in the forest -> trade space; two in the grassland -> not.
    eng = _make_engine()
    player = eng.state.players[0]
    _fill_row(player.board, cards.Habitat.FOREST, 1, bird)
    _fill_row(player.board, cards.Habitat.GRASSLAND, 2, bird)
    rendered = display.format_board(eng.state, player)
    # Forest (1 bird) is a trade space; grassland (2 birds) is not.
    assert "+1 food / -1 🃏 -> +2 food" in rendered
    assert "+3 🥚 / -1 food" not in rendered
    assert "+3 🥚" in rendered
