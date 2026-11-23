# pyright: reportPrivateUsage=false
# (tests read the package-private layout offsets to pin individual stripes)
"""Tests for the shared accept-row ledger featurizer (v1.6).

``TuckActivateChoice`` and ``PayCostChoice`` are both ``ExchangeLedger``s and
featurize through one function, so identical ledgers must encode to identical
rows. Also pins the typed-food ``becomes_playable`` (audit S1) and the combined
``bonus_delta`` of an egg-laying accept row (audit S2).
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from wingspan import cards, decisions, encode, state
from wingspan.encode import choice_encode, layout
from wingspan.engine import scoring

_BIRDS, _BONUSES, _GOALS = cards.load_all()
_BONUS_BY_NAME = {bonus_card.name: bonus_card for bonus_card in _BONUSES}
_EXCHANGE = slice(layout._OFF_EXCHANGE, layout._OFF_EXCHANGE + layout._EXCHANGE_DIM)
_BONUS_DELTA = slice(
    layout._OFF_BONUS_DELTA, layout._OFF_BONUS_DELTA + layout._BONUS_DELTA_DIM
)
_GAIN_FOOD = slice(layout._OFF_GAIN_FOOD, layout._OFF_GAIN_FOOD + cards.N_FOODS)


def _game() -> state.GameState:
    return state.new_game(random.Random(3), _BIRDS, _BONUSES, _GOALS)


def _tuck_row(
    game_state: state.GameState, ledger: decisions.ExchangeLedger
) -> np.ndarray:
    """The encoded yes row of an ``ActivateTuckDecision`` carrying ``ledger``."""
    decision = decisions.ActivateTuckDecision(
        player_id=0,
        prompt="tuck?",
        choices=[
            decisions.TuckActivateChoice(label="tuck", **ledger.model_dump()),
            decisions.SkipChoice(label="skip"),
        ],
    )
    return encode.encode_choices(decision, game_state)[0]


def _pay_row(
    game_state: state.GameState, ledger: decisions.ExchangeLedger
) -> np.ndarray:
    """The encoded accept row of an ``AcceptExchangeDecision`` carrying ``ledger``."""
    decision = decisions.AcceptExchangeDecision(
        player_id=0,
        prompt="accept?",
        choices=[
            decisions.PayCostChoice(label="pay", **ledger.model_dump()),
            decisions.SkipChoice(label="skip"),
        ],
    )
    return encode.encode_choices(decision, game_state)[0]


def _single_food_bird() -> tuple[cards.Bird, cards.Food]:
    """A bird whose whole cost is exactly one unit of one specific food."""
    for bird in _BIRDS:
        cost = bird.food_cost
        if cost.total == 1 and cost.wild == 0 and not cost.is_or_cost:
            food = next(food for food in cards.ALL_FOODS if cost.specific_of(food) == 1)
            return bird, food
    raise AssertionError("no single-food-cost bird in the catalog")


def _feeder_without(food: cards.Food) -> state.Birdfeeder:
    """A feeder showing exactly one die, of a food other than ``food``."""
    other = next(candidate for candidate in cards.ALL_FOODS if candidate != food)
    counts = [0] * cards.N_FOODS
    counts[cards.food_index(other)] = 1
    return state.Birdfeeder(counts=state.FoodPool(counts=counts), choice_dice=0)


_TUCK = decisions.ExchangeLedger(paid_card_count=1, gained_tuck_count=1)

# ---------------------------------------------------------------------------
# Featurizer parity


def test_tuck_and_pay_rows_with_the_same_ledger_encode_identically():
    ledgers = [
        _TUCK,
        _TUCK.model_copy(update={"gained_card_count": 1}),
        _TUCK.model_copy(update={"gained_egg_count": 1}),
        _TUCK.model_copy(
            update={"gained_food_count": 1, "gained_food": cards.Food.FRUIT}
        ),
    ]
    for ledger in ledgers:
        game_state = _game()
        np.testing.assert_array_equal(
            _tuck_row(game_state, ledger), _pay_row(game_state, ledger)
        )


def test_tuck_row_carries_the_exchange_terms_and_hand_shrink():
    game_state = _game()
    game_state.players[0].bonus_cards = [_BONUS_BY_NAME["Visionary Leader"]]
    row = _tuck_row(game_state, _TUCK)

    scale = layout._EXCHANGE_SCALE
    assert row[layout._OFF_EXCHANGE + layout._EXCHANGE_CARDS_TO_DISCARD] == 1 / scale
    assert row[layout._OFF_EXCHANGE + layout._EXCHANGE_CARDS_TO_TUCK] == 1 / scale
    # The hand shrinks by the tucked card: the hand-counting bonus card moves.
    assert row[layout._OFF_BONUS_DELTA + layout._BONUS_DELTA_QUAL] > 0


def test_skip_row_of_the_tuck_gate_stays_bare():
    game_state = _game()
    game_state.players[0].bonus_cards = [_BONUS_BY_NAME["Visionary Leader"]]
    decision = decisions.ActivateTuckDecision(
        player_id=0,
        prompt="x",
        choices=[
            decisions.TuckActivateChoice(label="t", **_TUCK.model_dump()),
            decisions.SkipChoice(label="skip"),
        ],
    )
    skip = encode.encode_choices(decision, game_state)[1]
    assert not skip[_EXCHANGE].any()
    assert not skip[_BONUS_DELTA].any()
    assert skip[layout._OFF_SPECIAL + layout._SPECIAL_IS_SKIP] == 1.0


def test_draw_payoff_cancels_the_hand_shrink():
    """tuck 1 + draw 1 leaves the hand size unchanged: no hand bonus delta."""
    game_state = _game()
    game_state.players[0].bonus_cards = [_BONUS_BY_NAME["Visionary Leader"]]
    row = _tuck_row(game_state, _TUCK.model_copy(update={"gained_card_count": 1}))
    assert not row[_BONUS_DELTA].any()


# ---------------------------------------------------------------------------
# S1: typed gained food


def _single_food_state() -> tuple[state.GameState, cards.Bird, cards.Food]:
    bird, food = _single_food_bird()
    game_state = _game()
    player = game_state.players[0]
    player.hand = [bird]
    player.food = state.FoodPool()
    game_state.birdfeeder = _feeder_without(food)
    return game_state, bird, food


def test_typed_gained_food_prices_becomes_playable_at_the_exact_food():
    game_state, bird, food = _single_food_state()
    index = layout.CHOICE_BECOMES_PLAYABLE_OFFSET + cards.bird_index(bird)

    typed = _pay_row(
        game_state, decisions.ExchangeLedger(gained_food_count=1, gained_food=food)
    )
    untyped = _pay_row(game_state, decisions.ExchangeLedger(gained_food_count=1))

    assert typed[index] == 1.0
    # Untyped keeps the any-feeder-die assumption: the feeder lacks the food.
    assert untyped[index] == 0.0


def test_typed_gained_food_fills_the_gain_food_stripe():
    game_state, _bird, food = _single_food_state()
    typed = _pay_row(
        game_state, decisions.ExchangeLedger(gained_food_count=1, gained_food=food)
    )
    untyped = _pay_row(game_state, decisions.ExchangeLedger(gained_food_count=1))

    expected = np.zeros(cards.N_FOODS, dtype=np.float32)
    expected[cards.food_index(food)] = 1.0
    np.testing.assert_array_equal(typed[_GAIN_FOOD], expected)
    assert not untyped[_GAIN_FOOD].any()


# ---------------------------------------------------------------------------
# S2: combined bonus delta


def _egg_state() -> state.GameState:
    """Player 0 holds Oologist + Visionary Leader and has an eggless bird."""
    game_state = _game()
    player = game_state.players[0]
    player.bonus_cards = [
        _BONUS_BY_NAME["Oologist"],
        _BONUS_BY_NAME["Visionary Leader"],
    ]
    bird = next(candidate for candidate in _BIRDS if candidate.egg_limit >= 1)
    player.board[cards.Habitat.FOREST] = [state.PlayedBird(bird=bird)]
    return game_state


def _qual(row: np.ndarray) -> float:
    return float(row[layout._OFF_BONUS_DELTA + layout._BONUS_DELTA_QUAL])


def test_egg_gain_prices_egg_bonus_cards_into_bonus_delta():
    game_state = _egg_state()
    egg_only = _pay_row(game_state, decisions.ExchangeLedger(gained_egg_count=1))
    # Oologist qualifies one more bird once an egg lands.
    assert _qual(egg_only) == pytest.approx(1 / layout._BONUS_COUNT_SCALE)


def test_bonus_delta_combines_hand_flow_and_egg_best_case():
    game_state = _egg_state()
    hand_only = _pay_row(game_state, _TUCK)
    egg_only = _pay_row(game_state, decisions.ExchangeLedger(gained_egg_count=1))
    both = _pay_row(game_state, _TUCK.model_copy(update={"gained_egg_count": 1}))

    assert _qual(hand_only) > 0 and _qual(egg_only) > 0
    # Different bonus cards: the combined row sums the two, it does not let the
    # later fill overwrite the earlier one.
    assert _qual(both) == _qual(hand_only) + _qual(egg_only)
    np.testing.assert_allclose(
        both[_BONUS_DELTA], hand_only[_BONUS_DELTA] + egg_only[_BONUS_DELTA]
    )


def test_egg_bonus_best_case_matches_the_scoring_primitive():
    game_state = _egg_state()
    player = game_state.players[0]
    qual, _stepped, _linear = choice_encode._bonus_delta_best_case_for_eggs(player, 1)
    assert qual == 1
    assert (
        scoring.bonus_best_case_count_delta_for_eggs(
            _BONUS_BY_NAME["Oologist"], player, 1
        )
        == 1
    )
