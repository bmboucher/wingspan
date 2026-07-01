"""Playability predicates over a player's hand.

Pure functions over :class:`~wingspan.state.Player` — no engine state, no I/O.
Builds on :func:`~wingspan.engine.helpers.any_payment_exists` to classify which
birds in hand are playable now, which become playable with more eggs, and which
become newly playable after a food or egg gain, and which become newly
*un*playable after a food, egg, or slot loss. Used by the state / choice
encoders to fill the ``hand_playable_me``, ``hand_playable_eggs_me``,
``becomes_playable``, and ``becomes_unplayable`` multi-hot stripes.

Import this module **locally inside encoder functions** (``from wingspan.engine
import playability``) to keep :mod:`wingspan.encode` engine-free at import time —
the same pattern ``choice_encode`` uses for ``scoring``.
"""

from __future__ import annotations

import itertools

from wingspan import cards, state
from wingspan.engine import helpers

# ---------------------------------------------------------------------------
# Core playability predicate


def _bird_playable(
    player: state.Player,
    bird: cards.Bird,
    *,
    extra_food: state.FoodPool | None = None,
    extra_eggs: int = 0,
    ignore_eggs: bool = False,
) -> bool:
    """Whether ``player`` could play ``bird`` given optional counterfactual resources.

    Returns True when:
    * ``any_payment_exists`` for player's food (+ extra_food) against bird's cost, AND
    * at least one habitat in bird.habitats has an open slot AND either
      ``ignore_eggs`` is True (egg cost not checked) or
      ``total_eggs + extra_eggs >= next_egg_cost`` for that habitat.

    ``extra_food`` is never mutated. ``extra_eggs`` is added to ``player.total_eggs``
    for the check (counterfactual "what if I had N more eggs"). ``ignore_eggs`` drops
    the egg-cost gate entirely — used on the food-gain path where eggs are irrelevant
    to whether gaining the food makes the bird's cost payable."""
    food = player.food if extra_food is None else _pool_add(player.food, extra_food)
    if not helpers.any_payment_exists(food, bird.food_cost):
        return False
    if ignore_eggs:
        return any(player.can_play_in(habitat) for habitat in bird.habitats)
    total_eggs = max(0, player.total_eggs + extra_eggs)
    for habitat in bird.habitats:
        if player.can_play_in(habitat) and total_eggs >= player.board.next_egg_cost(
            habitat
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# Hand classification


def classify_hand_playability(
    player: state.Player,
) -> tuple[list[cards.Bird], list[cards.Bird]]:
    """Classify the player's hand into two disjoint groups.

    Returns ``(playable_now, playable_if_more_eggs)``:

    * ``playable_now`` — food affordable, open slot, egg cost met right now.
    * ``playable_if_more_eggs`` — food affordable AND has at least one open
      slot, but the egg cost is not yet met (one more egg would unlock it for at
      least one valid habitat). Birds with NO open slot are excluded; more eggs
      would never help them.
    """
    playable_now: list[cards.Bird] = []
    playable_if_eggs: list[cards.Bird] = []
    for bird in player.hand:
        if not helpers.any_payment_exists(player.food, bird.food_cost):
            continue
        # Check each habitat: track whether any slot is open + whether the egg
        # cost is the only remaining blocker.
        any_slot_open = False
        egg_blocked = False
        for habitat in bird.habitats:
            if not player.can_play_in(habitat):
                continue
            any_slot_open = True
            egg_cost = player.board.next_egg_cost(habitat)
            if player.total_eggs >= egg_cost:
                # Fully playable right now.
                playable_now.append(bird)
                egg_blocked = False
                break
            else:
                egg_blocked = True
        else:
            # Did not break (not fully playable now).
            if any_slot_open and egg_blocked:
                playable_if_eggs.append(bird)
    return playable_now, playable_if_eggs


# ---------------------------------------------------------------------------
# Counterfactual gain helpers


def newly_playable_after_food(
    player: state.Player,
    food: cards.Food,
    *,
    already_playable: list[cards.Bird],
    ignore_eggs: bool = True,
) -> list[cards.Bird]:
    """Hand birds not in ``already_playable`` that become playable after gaining one
    unit of ``food``.

    Builds a counterfactual food pool with +1 of ``food`` (never mutates
    ``player.food``) and tests each not-yet-playable hand bird. ``ignore_eggs``
    defaults to ``True`` so the food-gain path signals food-driven transitions
    without the egg-cost gate masking the signal (a bird that needs food AND eggs
    should still light up when the food requirement is newly met)."""
    already_set = set(id(bird) for bird in already_playable)
    extra = _single_food_pool(food)
    return [
        bird
        for bird in player.hand
        if id(bird) not in already_set
        and _bird_playable(player, bird, extra_food=extra, ignore_eggs=ignore_eggs)
    ]


def newly_playable_after_foods(
    player: state.Player,
    gained: state.FoodPool,
    *,
    already_playable: list[cards.Bird],
    ignore_eggs: bool = True,
) -> list[cards.Bird]:
    """Hand birds not in ``already_playable`` that become playable after gaining
    the whole ``gained`` multiset of foods at once.

    The multi-food generalization of :func:`newly_playable_after_food`, used by
    the ``combine_gain_food`` regime's ``FoodSubsetChoice`` rows: a bird that
    needs two different foods together lights up only when *both* are gained in
    the same subset. ``gained`` is the realized pool (single-face foods plus the
    choice dice already resolved to invertebrate / seed); it is never mutated.
    ``ignore_eggs`` carries the same food-gain semantics as
    :func:`newly_playable_after_food`."""
    already_set = set(id(bird) for bird in already_playable)
    return [
        bird
        for bird in player.hand
        if id(bird) not in already_set
        and _bird_playable(player, bird, extra_food=gained, ignore_eggs=ignore_eggs)
    ]


def min_food_to_unlock(
    player: state.Player,
    candidates: list[cards.Bird],
    *,
    ignore_eggs: bool = True,
) -> list[int]:
    """Per-food smallest count that would newly unlock a candidate bird.

    For each food in ``cards.ALL_FOODS`` order, the smallest ``n >= 1`` such that
    adding ``n`` tokens of that food to ``player.food`` makes at least one
    currently-unplayable ``candidates`` bird playable — an open matching habitat
    slot is required, egg cost is dropped under ``ignore_eggs``. ``0`` for a food
    when no candidate can be unlocked at all: either every candidate is already
    playable, or every unplayable one is slot-blocked (food never helps a bird
    with no open matching slot). ``candidates`` may be the hand or the face-up
    tray (scored as if those cards were in hand); ``player.food`` is never
    mutated. Affordability follows the full engine rule (1-for-1 matches, 2-for-1
    substitution, wild), so once any unplayable-open bird exists every food
    unlocks it at some finite amount — the stripe is then all-nonzero."""
    already = set(
        id(bird)
        for bird in candidates
        if _bird_playable(player, bird, ignore_eggs=ignore_eggs)
    )
    open_unplayable = [
        bird
        for bird in candidates
        if id(bird) not in already
        and any(player.can_play_in(habitat) for habitat in bird.habitats)
    ]
    if not open_unplayable:
        return [0] * cards.N_FOODS

    # A single food type covers its own specific slot 1-for-1 and any other slot
    # (or wild) via 2-for-1, so 2 * effective_total tokens always suffice as a cap;
    # affordability is monotonic in the pool, so the first hit is the minimum.
    cap = 2 * max(bird.food_cost.effective_total for bird in open_unplayable)
    result = [0] * cards.N_FOODS
    for food_idx, food in enumerate(cards.ALL_FOODS):
        for count in range(1, cap + 1):
            extra = _food_pool_of(food, count)
            if any(
                _bird_playable(player, bird, extra_food=extra, ignore_eggs=ignore_eggs)
                for bird in open_unplayable
            ):
                result[food_idx] = count
                break
    return result


def newly_playable_after_egg(
    player: state.Player,
    n_eggs: int = 1,
    *,
    already_playable: list[cards.Bird],
) -> list[cards.Bird]:
    """Hand birds not in ``already_playable`` that become playable after gaining
    ``n_eggs`` extra eggs."""
    already_set = set(id(bird) for bird in already_playable)
    return [
        bird
        for bird in player.hand
        if id(bird) not in already_set
        and _bird_playable(player, bird, extra_eggs=n_eggs)
    ]


def gainable_feeder_foods(birdfeeder: state.Birdfeeder) -> set[cards.Food]:
    """Food types a player could gain from the birdfeeder right now.

    Returns every food whose die face is showing, plus ``INVERTEBRATE`` and
    ``SEED`` when any choice die is present (those dice resolve to either)."""
    present: set[cards.Food] = set(birdfeeder.counts.types_with_positive())
    if birdfeeder.choice_dice > 0:
        present.add(cards.Food.INVERTEBRATE)
        present.add(cards.Food.SEED)
    return present


def newly_playable_after_feeder_food(
    player: state.Player,
    birdfeeder: state.Birdfeeder,
    *,
    already_playable: list[cards.Bird],
    ignore_eggs: bool = True,
) -> list[cards.Bird]:
    """The optimistic union of ``newly_playable_after_food`` over all foods currently
    gainable from the birdfeeder.

    Used for main-action ``GAIN_FOOD`` rows and ``PayCostChoice`` exchanges that
    give food from the feeder — the exact die is not yet committed, so we
    advertise the best possible outcome. ``ignore_eggs`` is forwarded to
    ``_bird_playable`` with the same semantics as in :func:`newly_playable_after_food`.
    """
    feeder_foods = gainable_feeder_foods(birdfeeder)
    if not feeder_foods:
        return []
    already_set = set(id(bird) for bird in already_playable)
    candidates = [bird for bird in player.hand if id(bird) not in already_set]
    newly: list[cards.Bird] = []
    seen_ids: set[int] = set()
    for food in feeder_foods:
        extra = _single_food_pool(food)
        for bird in candidates:
            if id(bird) not in seen_ids and _bird_playable(
                player, bird, extra_food=extra, ignore_eggs=ignore_eggs
            ):
                newly.append(bird)
                seen_ids.add(id(bird))
    return newly


# ---------------------------------------------------------------------------
# Counterfactual loss helpers


def newly_unplayable_after_egg_loss(
    player: state.Player,
    n_eggs: int,
    *,
    already_playable: list[cards.Bird],
) -> list[cards.Bird]:
    """Hand birds in ``already_playable`` that become egg-unaffordable after
    losing ``n_eggs`` eggs.

    Food affordability is not re-checked (the ``already_playable`` baseline
    confirms it); only the egg gate changes when eggs decrease."""
    if not already_playable or n_eggs == 0:
        return []
    return [
        bird
        for bird in already_playable
        if not _bird_playable(player, bird, extra_eggs=-n_eggs)
    ]


def newly_unplayable_after_food_removed(
    player: state.Player,
    removed: state.FoodPool,
    *,
    already_playable: list[cards.Bird],
) -> list[cards.Bird]:
    """Hand birds in ``already_playable`` that lose food-affordability when
    exactly ``removed`` tokens are spent from ``player.food``.

    ``removed`` is an exact multiset; used for ``FoodPaymentChoice`` and
    fully-specified single-food spends."""
    if not already_playable:
        return []
    food_after = _pool_subtract(player.food, removed)
    return [
        bird
        for bird in already_playable
        if not helpers.any_payment_exists(food_after, bird.food_cost)
    ]


def newly_unplayable_after_optimistic_food_loss(
    player: state.Player,
    n: int,
    *,
    already_playable: list[cards.Bird],
) -> list[cards.Bird]:
    """Hand birds in ``already_playable`` that become unplayable regardless of
    how ``n`` food tokens are chosen for removal.

    Optimistic semantics: a bird *survives* if at least one way to remove
    ``n`` tokens still leaves it food-affordable. We flag it only when every
    possible size-``n`` removal breaks affordability."""
    if not already_playable or n == 0:
        return []
    removal_options = _removal_multisets(player.food, n)
    if not removal_options:
        return []
    food_after_each = [_pool_subtract(player.food, r) for r in removal_options]
    return [
        bird
        for bird in already_playable
        if not any(
            helpers.any_payment_exists(f, bird.food_cost) for f in food_after_each
        )
    ]


def newly_unplayable_after_play(
    player: state.Player,
    played_bird: cards.Bird,
    habitat: cards.Habitat,
    *,
    already_playable: list[cards.Bird],
) -> list[cards.Bird]:
    """Hand birds in ``already_playable`` (excluding ``played_bird``) that
    lose playability after ``played_bird`` is placed into ``habitat``.

    Models the full play cost: −1 slot in ``habitat``, −``next_egg_cost``
    eggs, and −food payment (optimistic: a baseline bird survives if *some*
    legal payment of ``played_bird``'s cost still leaves it affordable)."""
    counterfactual_eggs = player.total_eggs - player.board.next_egg_cost(habitat)
    payments = helpers.enumerate_payments(player.food, played_bird.food_cost)
    food_after_payments = [_pool_subtract(player.food, payment) for payment in payments]
    result: list[cards.Bird] = []
    for bird in already_playable:
        if bird is played_bird:
            continue
        if not _bird_playable_after_play(
            player,
            bird,
            played_habitat=habitat,
            counterfactual_eggs=counterfactual_eggs,
            food_after_payments=food_after_payments,
        ):
            result.append(bird)
    return result


# ---------------------------------------------------------------------------
# Setup turn-1 predicates


def setup_playable_kept_cards(
    kept_cards: tuple[cards.Bird, ...],
) -> list[cards.Bird]:
    """Birds in ``kept_cards`` for which some keepable food set would pay their cost.

    At setup a player keeps ``(5 − bird_count)`` *distinct* food tokens — one of
    each of up to 5 types.  A kept bird is playable iff some
    ``(5 − bird_count)``-subset of the 5 food types pays its printed cost (food
    only, no habitat or egg check — the first bird in any habitat costs 0 eggs).

    Unlike :func:`setup_turn1_playable` this predicate is food-agnostic: it
    enumerates the ≤10 possible distinct-token keeps rather than requiring a
    concrete ``kept_foods`` tuple.  This makes it useful in the
    ``split_setup_food=True`` training regime where the food choice is deferred
    and ``candidate.kept_foods`` is empty.
    """
    keep_count = len(cards.ALL_FOODS) - len(kept_cards)
    if keep_count < 0:
        return []
    # Build one FoodPool per distinct-token keep (C(5, keep_count) ≤ 10).
    food_pools = [
        _foods_to_pool(combo)
        for combo in itertools.combinations(cards.ALL_FOODS, keep_count)
    ]
    return [
        bird
        for bird in kept_cards
        if any(helpers.any_payment_exists(pool, bird.food_cost) for pool in food_pools)
    ]


def setup_turn1_playable(
    kept_cards: tuple[cards.Bird, ...],
    kept_foods: tuple[cards.Food, ...],
) -> list[cards.Bird]:
    """Birds in ``kept_cards`` the player could play on turn 1, given their kept foods.

    On turn 1 the first bird in any habitat costs 0 eggs — every habitat row is
    empty so ``EGG_COSTS[0] == 0``. Food affordability uses ``any_payment_exists``
    over a pool of the ``kept_foods``, exactly 1 unit per food type.

    Any food in the tuple contributes 1 to the pool.
    """
    food_pool = _foods_to_pool(kept_foods)
    return [
        bird
        for bird in kept_cards
        if helpers.any_payment_exists(food_pool, bird.food_cost)
    ]


###### PRIVATE #######


def _pool_add(food_pool: state.FoodPool, extra: state.FoodPool) -> state.FoodPool:
    """Return a new FoodPool that is ``food_pool`` + ``extra`` without mutating either."""
    new_counts = [food_pool.counts[i] + extra.counts[i] for i in range(cards.N_FOODS)]
    return state.FoodPool.model_construct(counts=new_counts)


def _single_food_pool(food: cards.Food) -> state.FoodPool:
    """A FoodPool with exactly 1 of ``food`` and 0 of everything else."""
    return _food_pool_of(food, 1)


def _food_pool_of(food: cards.Food, count: int) -> state.FoodPool:
    """A FoodPool with ``count`` of ``food`` and 0 of everything else."""
    counts = [0] * cards.N_FOODS
    counts[cards.food_index(food)] = count
    return state.FoodPool.model_construct(counts=counts)


def _foods_to_pool(foods: tuple[cards.Food, ...]) -> state.FoodPool:
    """Convert a tuple of Food to a FoodPool with 1 of each food type present."""
    counts = [0] * cards.N_FOODS
    for food in foods:
        counts[cards.food_index(food)] = 1
    return state.FoodPool.model_construct(counts=counts)


def _pool_subtract(
    food_pool: state.FoodPool, removed: state.FoodPool
) -> state.FoodPool:
    """Return a new FoodPool that is ``food_pool`` minus ``removed``, clamped to 0."""
    new_counts = [
        max(0, food_pool.counts[i] - removed.counts[i]) for i in range(cards.N_FOODS)
    ]
    return state.FoodPool.model_construct(counts=new_counts)


def _removal_multisets(pool: state.FoodPool, n: int) -> list[state.FoodPool]:
    """All distinct multisets of exactly ``n`` tokens removable from ``pool``.

    Enumerates token-by-token across the 5 food types. Returns an empty list
    if the pool contains fewer than ``n`` total tokens."""
    result: list[state.FoodPool] = []
    _enumerate_removals(pool.counts, n, 0, [0] * cards.N_FOODS, result)
    return result


def _enumerate_removals(
    pool_counts: list[int],
    remaining: int,
    food_idx: int,
    current: list[int],
    result: list[state.FoodPool],
) -> None:
    """Recursively fill ``result`` with each distinct size-``remaining`` removal."""
    if remaining == 0:
        result.append(state.FoodPool.model_construct(counts=list(current)))
        return
    if food_idx >= cards.N_FOODS:
        return
    max_take = min(pool_counts[food_idx], remaining)
    for take in range(0, max_take + 1):
        current[food_idx] = take
        _enumerate_removals(
            pool_counts, remaining - take, food_idx + 1, current, result
        )
    current[food_idx] = 0


def _counterfactual_can_play_in(
    player: state.Player,
    played_habitat: cards.Habitat,
    target_habitat: cards.Habitat,
) -> bool:
    """Open slot in ``target_habitat`` after one bird is placed in ``played_habitat``."""
    n_birds = len(player.board[target_habitat])
    if target_habitat == played_habitat:
        n_birds += 1
    return n_birds < state.ROW_SLOTS


def _counterfactual_next_egg_cost(
    player: state.Player,
    played_habitat: cards.Habitat,
    target_habitat: cards.Habitat,
) -> int:
    """Egg cost to play into ``target_habitat`` after one bird lands in ``played_habitat``."""
    n_birds = len(player.board[target_habitat])
    if target_habitat == played_habitat:
        n_birds += 1
    if n_birds >= len(state.EGG_COSTS):
        return state.FULL_ROW_EGG_COST
    return state.EGG_COSTS[n_birds]


def _bird_playable_after_play(
    player: state.Player,
    bird: cards.Bird,
    *,
    played_habitat: cards.Habitat,
    counterfactual_eggs: int,
    food_after_payments: list[state.FoodPool],
) -> bool:
    """Whether ``bird`` remains playable on the counterfactual board after a
    play in ``played_habitat`` left ``counterfactual_eggs`` eggs and each
    element of ``food_after_payments`` as a possible remaining food supply.

    Returns True iff some habitat/payment combination keeps it playable."""
    for habitat in bird.habitats:
        if not _counterfactual_can_play_in(player, played_habitat, habitat):
            continue
        egg_needed = _counterfactual_next_egg_cost(player, played_habitat, habitat)
        if counterfactual_eggs < egg_needed:
            continue
        # Slot and eggs satisfied; check if any payment still affords this bird.
        if any(
            helpers.any_payment_exists(food, bird.food_cost)
            for food in food_after_payments
        ):
            return True
    return False
