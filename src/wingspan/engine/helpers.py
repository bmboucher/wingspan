"""Pure helper functions used by the engine.

The two helpers here implement Wingspan's food-payment rules for playing a
bird:

* a specific food slot is satisfied 1-for-1 by a matching food,
* a specific food slot can also be satisfied by 2 of any other food
  (the 2-for-1 substitution),
* a wild slot takes 1 of any food.

Both operate on the vector types — :class:`cards.BirdCost` (6-tuple:
5 specific + 1 wild) and :class:`state.FoodPool` (5-vector of food amounts)
— so callers don't shuttle through ``dict[Food, int]``."""

from __future__ import annotations

import typing

from wingspan import cards, state


def cost_meets(cost: cards.BirdCost, payment: state.FoodPool) -> bool:
    """Return ``True`` iff ``payment`` is *exactly* enough to pay ``cost``.

    Lets the payment cover any unfulfilled specific slot via 2-for-1
    substitution, and each wild slot via 1-of-any. The check is strict
    equality so the player neither overpays nor underpays::

        extra == 2 * unfulfilled_specific + wild

    where ``extra`` is the number of paid foods beyond the part that
    matches the bird's specific cost."""
    return _cost_meets_counts(cost, payment.counts)


def _cost_meets_counts(cost: cards.BirdCost, paid: typing.Sequence[int]) -> bool:
    """``cost_meets`` on a raw per-food count vector, skipping the
    ``FoodPool`` wrapper. ``enumerate_payments`` calls this in its inner loop
    (millions of times per self-play game) so it can test a candidate without
    constructing a pydantic model for the ones that don't pay."""
    extra = 0
    unfulfilled = 0
    for i in range(cards.N_FOODS):
        need = cost.counts[i]
        matched = min(need, paid[i])
        unfulfilled += need - matched
        extra += paid[i] - matched
    return extra == 2 * unfulfilled + cost.wild


def enumerate_payments(
    available: state.FoodPool,
    cost: cards.BirdCost,
) -> list[state.FoodPool]:
    """Enumerate every legal payment the player could make for a bird.

    Returns one :class:`state.FoodPool` per distinct food-multiset payment.
    Every returned payment satisfies :func:`cost_meets` for the inputs.

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


def any_payment_exists(available: state.FoodPool, cost: cards.BirdCost) -> bool:
    """Whether ``available`` can pay ``cost`` *any* legal way.

    The early-exit twin of :func:`enumerate_payments`: it walks the same
    recursion but returns ``True`` the moment one satisfying payment is found
    and builds no ``FoodPool`` objects. Used by
    :func:`wingspan.engine.actions.any_playable_bird_play` to gate the
    ``PLAY_BIRD`` main-action option without materializing the whole menu."""
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
