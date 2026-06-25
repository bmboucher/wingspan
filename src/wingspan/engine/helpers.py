"""Pure helper functions used by the engine.

The helpers here implement Wingspan's food-payment rules for playing a bird.

**AND-cost birds** (``BirdCost.is_or_cost == False``):

* a specific food slot is satisfied 1-for-1 by a matching food,
* a specific food slot can also be satisfied by 2 of any other food
  (the 2-for-1 substitution),
* a wild slot takes 1 of any food.

**OR-cost birds** (``BirdCost.is_or_cost == True``): ``counts`` is an
*accepted-food mask* and the player pays exactly 1 token of any listed type,
or exactly 2 tokens of any non-listed type(s) as a 2-for-1 substitution.

Both operate on the vector types — :class:`cards.BirdCost` (6-tuple:
5 specific + 1 wild) and :class:`state.FoodPool` (5-vector of food amounts)
— so callers don't shuttle through ``dict[Food, int]``."""

from __future__ import annotations

import typing

from wingspan import cards, state


def cost_meets(cost: cards.BirdCost, payment: state.FoodPool) -> bool:
    """Return ``True`` iff ``payment`` is *exactly* enough to pay ``cost``.

    For AND costs: lets the payment cover any unfulfilled specific slot via
    2-for-1 substitution, and each wild slot via 1-of-any; strict equality
    so the player neither overpays nor underpays.

    For OR costs: accepts exactly 1 token of any accepted food type, or
    exactly 2 tokens of non-accepted food(s) as a 2-for-1 substitution."""
    if cost.is_or_cost:
        return _or_cost_meets(cost, payment.counts)
    return _cost_meets_counts(cost, payment.counts)


def enumerate_payments(
    available: state.FoodPool,
    cost: cards.BirdCost,
) -> list[state.FoodPool]:
    """Enumerate every legal payment the player could make for a bird.

    Returns one :class:`state.FoodPool` per distinct food-multiset payment.
    Every returned payment satisfies :func:`cost_meets` for the inputs."""
    if cost.is_or_cost:
        return _enumerate_or_payments(available, cost)
    return _enumerate_and_payments(available, cost)


def any_payment_exists(available: state.FoodPool, cost: cards.BirdCost) -> bool:
    """Whether ``available`` can pay ``cost`` *any* legal way.

    The early-exit twin of :func:`enumerate_payments`. Used by
    :func:`wingspan.engine.actions.any_playable_bird_play` to gate the
    ``PLAY_BIRD`` main-action option without materializing the whole menu."""
    if cost.is_or_cost:
        return _any_or_payment_exists(available, cost)
    return _any_and_payment_exists(available, cost)


###### PRIVATE — AND-cost helpers ######


def _cost_meets_counts(cost: cards.BirdCost, paid: typing.Sequence[int]) -> bool:
    """``cost_meets`` on a raw per-food count vector, skipping the
    ``FoodPool`` wrapper. ``_enumerate_and_payments`` calls this in its inner
    loop (millions of times per self-play game) so it can test a candidate
    without constructing a pydantic model for the ones that don't pay."""
    extra = 0
    unfulfilled = 0
    for i in range(cards.N_FOODS):
        need = cost.counts[i]
        matched = min(need, paid[i])
        unfulfilled += need - matched
        extra += paid[i] - matched
    return extra == 2 * unfulfilled + cost.wild


def _enumerate_and_payments(
    available: state.FoodPool,
    cost: cards.BirdCost,
) -> list[state.FoodPool]:
    """Enumerate payments for an AND-cost bird via recursive enumeration.

    Considers all combinations of 1-for-1 matching, 2-for-1 substitution
    for specific food slots, and 1-of-any for wild slots — including the
    choice to *substitute* an available specific food rather than match
    it directly (e.g. keep a seed in supply and pay 2 fruit for the seed
    slot instead)."""
    cost_vec = cost.counts[: cards.N_FOODS]
    avail = available.counts
    wild = cost.wild
    cost_total = sum(cost_vec)
    # Tightest upper bound: substitute every specific slot (2x) + wild (1x).
    max_total = 2 * cost_total + wild
    # Tightest lower bound: match every specific slot (1x) + wild (1x).
    min_total = cost_total + wild

    payments: list[state.FoodPool] = []
    counts = [0] * cards.N_FOODS

    def rec(idx: int, total: int) -> None:
        if total > max_total:
            return
        if idx == cards.N_FOODS:
            if total < min_total:
                return
            # Test the raw count vector first; only the candidates that
            # actually pay get wrapped in a (validated) ``FoodPool``. The
            # counts are already a valid length-N vector, so ``model_construct``
            # skips redundant validation on this hot path.
            if _cost_meets_counts(cost, counts):
                payments.append(state.FoodPool.model_construct(counts=list(counts)))
            return
        for k in range(0, avail[idx] + 1):
            counts[idx] = k
            rec(idx + 1, total + k)
        counts[idx] = 0

    rec(0, 0)
    return payments


def _any_and_payment_exists(available: state.FoodPool, cost: cards.BirdCost) -> bool:
    """Early-exit twin of :func:`_enumerate_and_payments`."""
    cost_vec = cost.counts[: cards.N_FOODS]
    avail = available.counts
    wild = cost.wild
    cost_total = sum(cost_vec)
    max_total = 2 * cost_total + wild
    min_total = cost_total + wild
    counts = [0] * cards.N_FOODS

    def rec(idx: int, total: int) -> bool:
        if total > max_total:
            return False
        if idx == cards.N_FOODS:
            return total >= min_total and _cost_meets_counts(cost, counts)
        for k in range(0, avail[idx] + 1):
            counts[idx] = k
            if rec(idx + 1, total + k):
                counts[idx] = 0
                return True
        counts[idx] = 0
        return False

    return rec(0, 0)


###### PRIVATE — OR-cost helpers ######


def _or_cost_meets(cost: cards.BirdCost, paid: typing.Sequence[int]) -> bool:
    """Validate a payment against an OR-cost bird.

    Valid iff exactly 1 token of an accepted food type was paid, or exactly
    2 tokens of non-accepted food type(s) were paid (2-for-1 substitution)."""
    total_paid = sum(paid)
    matching_paid = sum(paid[i] for i in range(cards.N_FOODS) if cost.counts[i] > 0)
    non_matching_paid = total_paid - matching_paid
    return (total_paid == 1 and matching_paid == 1) or (
        total_paid == 2 and non_matching_paid == 2
    )


def _enumerate_or_payments(
    available: state.FoodPool,
    cost: cards.BirdCost,
) -> list[state.FoodPool]:
    """Enumerate payments for an OR-cost bird.

    Option A: pay 1 token of any accepted food type (direct match).
    Option B: pay 2 tokens of non-accepted food type(s) as a 2-for-1 sub
    (same type twice, or one each of two different non-accepted types)."""
    avail = available.counts
    listed = [i for i in range(cards.N_FOODS) if cost.counts[i] > 0]
    non_listed = [i for i in range(cards.N_FOODS) if cost.counts[i] == 0]
    payments: list[state.FoodPool] = []

    # Option A: 1 of any accepted food type.
    for food_idx in listed:
        if avail[food_idx] >= 1:
            vec = [0] * cards.N_FOODS
            vec[food_idx] = 1
            payments.append(state.FoodPool.model_construct(counts=vec))

    # Option B: 2-for-1 substitution with non-accepted foods.
    for outer_idx, outer in enumerate(non_listed):
        if avail[outer] >= 2:
            vec = [0] * cards.N_FOODS
            vec[outer] = 2
            payments.append(state.FoodPool.model_construct(counts=vec))
        for inner in non_listed[outer_idx + 1 :]:
            if avail[outer] >= 1 and avail[inner] >= 1:
                vec = [0] * cards.N_FOODS
                vec[outer] = 1
                vec[inner] = 1
                payments.append(state.FoodPool.model_construct(counts=vec))

    return payments


def _any_or_payment_exists(available: state.FoodPool, cost: cards.BirdCost) -> bool:
    """Early-exit check for OR-cost affordability."""
    avail = available.counts

    # Direct match: any accepted food available?
    if any(cost.counts[i] > 0 and avail[i] >= 1 for i in range(cards.N_FOODS)):
        return True

    # 2-for-1 sub: at least 2 non-accepted food tokens available?
    total_non_listed = sum(
        avail[i] for i in range(cards.N_FOODS) if cost.counts[i] == 0
    )
    return total_non_listed >= 2
