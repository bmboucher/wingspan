"""Tests for the food-payment helpers (cost_meets / enumerate_payments).

Covers the three Wingspan payment rules these helpers implement:

* a specific food slot is satisfied 1-for-1 by a matching food,
* a specific food slot can be substituted by 2 of any other food (2-for-1),
* a wild slot takes 1 food of any type.

Both helpers are strict: ``cost_meets`` only accepts an *exact* payment
(no over- or under-pay), and ``enumerate_payments`` only returns payments
that ``cost_meets`` would accept. They operate on the vector types
(:class:`cards.BirdCost` and :class:`state.FoodPool`) — the test helpers
below build those from concise ``{Food: count}`` literals."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import cards, state
from wingspan.engine import helpers

SEED = cards.Food.SEED
FRUIT = cards.Food.FRUIT
FISH = cards.Food.FISH
INV = cards.Food.INVERTEBRATE
RODENT = cards.Food.RODENT


def _pool(counts: dict[cards.Food, int] | None = None) -> state.FoodPool:
    return state.FoodPool.from_dict(counts or {})


def _cost(
    specific: dict[cards.Food, int] | None = None, wild: int = 0
) -> cards.BirdCost:
    return cards.BirdCost.from_specific(specific or {}, wild)


def _payment_multisets(
    payments: list[state.FoodPool],
) -> set[frozenset[tuple[cards.Food, int]]]:
    return {
        frozenset((food, count) for food, count in payment.items() if count > 0)
        for payment in payments
    }


# --- cost_meets -----------------------------------------------------------


def test_cost_meets_exact_specific_only():
    assert helpers.cost_meets(_cost({SEED: 2}), _pool({SEED: 2}))


def test_cost_meets_rejects_overpay_on_specific():
    assert not helpers.cost_meets(_cost({SEED: 2}), _pool({SEED: 3}))


def test_cost_meets_rejects_underpay():
    assert not helpers.cost_meets(_cost({SEED: 2}), _pool({SEED: 1}))


def test_cost_meets_wild_one_for_one():
    assert helpers.cost_meets(_cost({SEED: 1}, wild=1), _pool({SEED: 1, FRUIT: 1}))
    assert helpers.cost_meets(_cost({SEED: 1}, wild=1), _pool({SEED: 2}))


def test_cost_meets_two_for_one_substitution():
    # Specific 1 seed; pay 2 fruit instead.
    assert helpers.cost_meets(_cost({SEED: 1}), _pool({FRUIT: 2}))


def test_cost_meets_two_for_one_with_partial_specific():
    # Specific 2 seeds; pay 1 seed + 2 fruit (sub for second seed).
    assert helpers.cost_meets(_cost({SEED: 2}), _pool({SEED: 1, FRUIT: 2}))
    # 1 seed + 1 fruit underpays.
    assert not helpers.cost_meets(_cost({SEED: 2}), _pool({SEED: 1, FRUIT: 1}))


def test_cost_meets_substitution_plus_wild():
    # 1 seed specific + 1 wild; pay 3 fruit (2 sub for seed + 1 for wild).
    assert helpers.cost_meets(_cost({SEED: 1}, wild=1), _pool({FRUIT: 3}))


def test_cost_meets_free_cost():
    assert helpers.cost_meets(_cost(), _pool())
    assert not helpers.cost_meets(_cost(), _pool({SEED: 1}))


def test_cost_meets_pure_wild():
    assert helpers.cost_meets(_cost(wild=2), _pool({SEED: 1, FRUIT: 1}))
    assert not helpers.cost_meets(_cost(wild=2), _pool({SEED: 3}))


# --- enumerate_payments: requested coverage cases ------------------------


def test_case_1_specific_unaffordable():
    """cost = 1 INV, available = 0 INV → no legal payment."""
    assert helpers.enumerate_payments(_pool(), _cost({INV: 1})) == []


def test_case_2_specific_exact_supply():
    """cost = 1 INV, available = 1 INV → pay the INV."""
    payments = helpers.enumerate_payments(_pool({INV: 1}), _cost({INV: 1}))
    assert _payment_multisets(payments) == {frozenset({(INV, 1)})}


def test_case_3_specific_with_surplus():
    """cost = 1 INV, available = 2 INV → one option ({INV: 1}). 2-for-1
    substitution requires a different food, so paying 2 INV for the 1-INV
    slot is not a legal alternative."""
    payments = helpers.enumerate_payments(_pool({INV: 2}), _cost({INV: 1}))
    assert _payment_multisets(payments) == {frozenset({(INV, 1)})}


def test_case_4_substitution_only_option():
    """cost = 1 INV, available = 2 SEED → substitute 2 seed for the INV."""
    payments = helpers.enumerate_payments(_pool({SEED: 2}), _cost({INV: 1}))
    assert _payment_multisets(payments) == {frozenset({(SEED, 2)})}


def test_case_5_three_substitution_combinations():
    """cost = 1 INV, available = 2 SEED + 2 FRUIT → 3 ways to pay 2 foods:
    seed+seed, seed+fruit, fruit+fruit."""
    payments = helpers.enumerate_payments(
        _pool({SEED: 2, FRUIT: 2}),
        _cost({INV: 1}),
    )
    assert _payment_multisets(payments) == {
        frozenset({(SEED, 2)}),
        frozenset({(SEED, 1), (FRUIT, 1)}),
        frozenset({(FRUIT, 2)}),
    }


def test_case_6_wild_picks_any_one_food():
    """cost = 1 wild, available = 1 of each food → 5 single-food payments."""
    available = _pool({food: 1 for food in cards.ALL_FOODS})
    payments = helpers.enumerate_payments(available, _cost(wild=1))
    assert _payment_multisets(payments) == {
        frozenset({(food, 1)}) for food in cards.ALL_FOODS
    }


def test_case_7_partial_match_plus_substitution():
    """cost = 1 INV + 1 SEED, available = 1 INV + 3 FISH → pay the INV
    directly and substitute 2 FISH for the SEED."""
    payments = helpers.enumerate_payments(
        _pool({INV: 1, FISH: 3}),
        _cost({INV: 1, SEED: 1}),
    )
    assert _payment_multisets(payments) == {frozenset({(INV, 1), (FISH, 2)})}


def test_case_8_two_wild_picks_two_distinct_foods():
    """cost = 2 wild, available = 1 of each food → C(5,2) = 10 distinct pairs."""
    available = _pool({food: 1 for food in cards.ALL_FOODS})
    payments = helpers.enumerate_payments(available, _cost(wild=2))
    expected = {
        frozenset({(food_a, 1), (food_b, 1)})
        for i, food_a in enumerate(cards.ALL_FOODS)
        for food_b in cards.ALL_FOODS[i + 1 :]
    }
    assert _payment_multisets(payments) == expected
    assert len(payments) == 10


# --- additional invariants -----------------------------------------------


def test_enumerate_payments_specific_only_when_exact_supply():
    payments = helpers.enumerate_payments(_pool({SEED: 2}), _cost({SEED: 2}))
    assert _payment_multisets(payments) == {frozenset({(SEED, 2)})}


def test_enumerate_payments_keep_or_substitute_with_surplus():
    """When the player owns the matching food but also has 2 of another,
    both ``pay it directly`` and ``substitute 2-for-1`` are legal options."""
    payments = helpers.enumerate_payments(
        _pool({SEED: 1, FRUIT: 2}),
        _cost({SEED: 1}),
    )
    assert _payment_multisets(payments) == {
        frozenset({(SEED, 1)}),
        frozenset({(FRUIT, 2)}),
    }


def test_enumerate_payments_no_overpay():
    payments = helpers.enumerate_payments(
        _pool({food: 5 for food in cards.ALL_FOODS}),
        _cost(),
    )
    assert _payment_multisets(payments) == {frozenset()}


def test_enumerate_payments_all_results_satisfy_cost_meets():
    food = _pool({SEED: 2, FRUIT: 2, FISH: 1})
    cost = _cost({SEED: 1}, wild=1)
    payments = helpers.enumerate_payments(food, cost)
    assert payments, "expected at least one legal payment"
    for pay in payments:
        assert helpers.cost_meets(cost, pay), pay.as_dict()
