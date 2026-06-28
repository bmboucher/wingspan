"""Tests for the food-payment helpers (cost_meets / enumerate_payments).

Covers the Wingspan payment rules these helpers implement:

**AND-cost birds:**
* a specific food slot is satisfied 1-for-1 by a matching food,
* a specific food slot can be substituted by 2 of any other food (2-for-1),
* a wild slot takes 1 food of any type.

**OR-cost birds (is_or_cost=True):**
* counts is an accepted-food mask; pay exactly 1 token of any accepted type,
* or pay exactly 2 tokens of non-accepted food(s) as a 2-for-1 substitution.

Both helpers are strict: ``cost_meets`` only accepts an *exact* payment
(no over- or under-pay), and ``enumerate_payments`` only returns payments
that ``cost_meets`` would accept. They operate on the vector types
(:class:`cards.BirdCost` and :class:`state.FoodPool`) — the test helpers
below build those from concise ``{Food: count}`` literals."""

from __future__ import annotations

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


# =============================================================================
# OR-cost birds
# =============================================================================


def _or_cost(accepted: dict[cards.Food, int]) -> cards.BirdCost:
    """Build an OR-cost from an accepted-food mask."""
    return cards.BirdCost.from_specific(accepted, is_or_cost=True)


# --- cost_meets: OR costs -------------------------------------------------


def test_or_cost_meets_direct_listed() -> None:
    """Paying 1 accepted food is valid."""
    cost = _or_cost({INV: 1, SEED: 1})
    assert helpers.cost_meets(cost, _pool({INV: 1}))


def test_or_cost_meets_other_listed() -> None:
    """Either accepted food satisfies the cost."""
    cost = _or_cost({INV: 1, SEED: 1})
    assert helpers.cost_meets(cost, _pool({SEED: 1}))


def test_or_cost_meets_rejects_unlisted_single() -> None:
    """1 non-accepted food does not pay an OR cost."""
    cost = _or_cost({INV: 1, SEED: 1})
    assert not helpers.cost_meets(cost, _pool({FISH: 1}))


def test_or_cost_meets_sub_two_same_unlisted() -> None:
    """2-for-1: 2 of the same non-accepted food is a valid substitution."""
    cost = _or_cost({INV: 1, SEED: 1})
    assert helpers.cost_meets(cost, _pool({FISH: 2}))


def test_or_cost_meets_sub_two_mixed_unlisted() -> None:
    """2-for-1: 1 each of two different non-accepted foods is valid."""
    cost = _or_cost({INV: 1, SEED: 1})
    assert helpers.cost_meets(cost, _pool({FISH: 1, FRUIT: 1}))


def test_or_cost_meets_rejects_overpay_listed_plus_any() -> None:
    """1 listed + 1 anything is overpay — rejected."""
    cost = _or_cost({INV: 1, SEED: 1})
    assert not helpers.cost_meets(cost, _pool({INV: 1, FISH: 1}))
    assert not helpers.cost_meets(cost, _pool({INV: 1, SEED: 1}))


def test_or_cost_meets_rejects_empty_supply() -> None:
    cost = _or_cost({INV: 1, SEED: 1})
    assert not helpers.cost_meets(cost, _pool())


def test_or_cost_meets_three_accepted_types() -> None:
    """Any single accepted type from a wider OR mask is valid."""
    cost = _or_cost({INV: 1, SEED: 1, FRUIT: 1})
    assert helpers.cost_meets(cost, _pool({INV: 1}))
    assert helpers.cost_meets(cost, _pool({SEED: 1}))
    assert helpers.cost_meets(cost, _pool({FRUIT: 1}))


# --- enumerate_payments: OR costs ----------------------------------------


def test_or_cost_pay_one_listed() -> None:
    cost = _or_cost({INV: 1, SEED: 1})
    payments = helpers.enumerate_payments(_pool({INV: 1}), cost)
    assert _payment_multisets(payments) == {frozenset({(INV, 1)})}


def test_or_cost_pay_other_listed() -> None:
    cost = _or_cost({INV: 1, SEED: 1})
    payments = helpers.enumerate_payments(_pool({SEED: 1}), cost)
    assert _payment_multisets(payments) == {frozenset({(SEED, 1)})}


def test_or_cost_cannot_pay_unlisted_single() -> None:
    cost = _or_cost({INV: 1, SEED: 1})
    payments = helpers.enumerate_payments(_pool({FISH: 1}), cost)
    assert payments == []


def test_or_cost_sub_two_same_unlisted() -> None:
    cost = _or_cost({INV: 1, SEED: 1})
    payments = helpers.enumerate_payments(_pool({FISH: 2}), cost)
    assert _payment_multisets(payments) == {frozenset({(FISH, 2)})}


def test_or_cost_sub_two_mixed_unlisted() -> None:
    cost = _or_cost({INV: 1, SEED: 1})
    payments = helpers.enumerate_payments(_pool({FISH: 1, FRUIT: 1}), cost)
    assert _payment_multisets(payments) == {frozenset({(FISH, 1), (FRUIT, 1)})}


def test_or_cost_empty_supply_gives_no_options() -> None:
    cost = _or_cost({INV: 1, SEED: 1})
    assert helpers.enumerate_payments(_pool(), cost) == []


def test_or_cost_direct_options_all_accepted_types() -> None:
    """With 1 of each accepted type available, each appears as an option."""
    cost = _or_cost({INV: 1, SEED: 1, FRUIT: 1})
    payments = helpers.enumerate_payments(_pool({INV: 1, SEED: 1, FRUIT: 1}), cost)
    direct = {frozenset({(INV, 1)}), frozenset({(SEED, 1)}), frozenset({(FRUIT, 1)})}
    assert direct.issubset(_payment_multisets(payments))


def test_or_cost_ample_supply_many_options() -> None:
    """With plenty of all food types: 3 direct options + substitution combos."""
    cost = _or_cost({INV: 1, SEED: 1})
    pool = _pool({INV: 2, SEED: 2, FISH: 2, FRUIT: 2, RODENT: 2})
    payments = helpers.enumerate_payments(pool, cost)
    multisets = _payment_multisets(payments)
    # Direct matches.
    assert frozenset({(INV, 1)}) in multisets
    assert frozenset({(SEED, 1)}) in multisets
    # 2-for-1 substitutions with non-accepted foods.
    assert frozenset({(FISH, 2)}) in multisets
    assert frozenset({(FRUIT, 2)}) in multisets
    assert frozenset({(RODENT, 2)}) in multisets
    assert frozenset({(FISH, 1), (FRUIT, 1)}) in multisets


# --- any_payment_exists: OR costs ----------------------------------------


def test_or_cost_any_payment_exists_listed() -> None:
    cost = _or_cost({INV: 1, SEED: 1})
    assert helpers.any_payment_exists(_pool({INV: 1}), cost)


def test_or_cost_any_payment_exists_unlisted_single() -> None:
    cost = _or_cost({INV: 1, SEED: 1})
    assert not helpers.any_payment_exists(_pool({FISH: 1}), cost)


def test_or_cost_any_payment_exists_unlisted_pair() -> None:
    cost = _or_cost({INV: 1, SEED: 1})
    assert helpers.any_payment_exists(_pool({FISH: 2}), cost)


def test_or_cost_any_payment_exists_mixed_pair() -> None:
    cost = _or_cost({INV: 1, SEED: 1})
    assert helpers.any_payment_exists(_pool({FISH: 1, FRUIT: 1}), cost)


def test_or_cost_any_payment_exists_empty() -> None:
    cost = _or_cost({INV: 1, SEED: 1})
    assert not helpers.any_payment_exists(_pool(), cost)


# --- OR vs AND with same counts differ -----------------------------------


def test_or_vs_and_same_counts_differ() -> None:
    """Same counts vector but different is_or_cost produces different payment sets."""
    and_cost = _cost({INV: 1, SEED: 1})
    or_cost = _or_cost({INV: 1, SEED: 1})
    pool = _pool({INV: 1, SEED: 1})

    and_payments = _payment_multisets(helpers.enumerate_payments(pool, and_cost))
    or_payments = _payment_multisets(helpers.enumerate_payments(pool, or_cost))

    # AND cost requires both: {INV:1, SEED:1}.
    assert frozenset({(INV, 1), (SEED, 1)}) in and_payments
    # OR cost: pay either one.
    assert frozenset({(INV, 1)}) in or_payments
    assert frozenset({(SEED, 1)}) in or_payments
    # They are not the same set.
    assert and_payments != or_payments


# --- invariant: all OR enumerated payments pass cost_meets ---------------


def test_or_cost_all_results_satisfy_cost_meets() -> None:
    cost = _or_cost({INV: 1, SEED: 1, FRUIT: 1})
    pool = _pool({INV: 2, SEED: 2, FISH: 2, FRUIT: 2, RODENT: 2})
    payments = helpers.enumerate_payments(pool, cost)
    assert payments, "expected at least one legal payment"
    for payment in payments:
        assert helpers.cost_meets(cost, payment), payment.as_dict()
