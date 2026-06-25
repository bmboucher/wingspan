"""Playability predicates over a player's hand.

Pure functions over :class:`~wingspan.state.Player` — no engine state, no I/O.
Builds on :func:`~wingspan.engine.helpers.any_payment_exists` to classify which
birds in hand are playable now, which become playable with more eggs, and which
become newly playable after a food or egg gain. Used by the state / choice
encoders to fill the ``hand_playable_me``, ``hand_playable_eggs_me``, and
``becomes_playable`` multi-hot stripes.

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
) -> bool:
    """Whether ``player`` could play ``bird`` given optional counterfactual resources.

    Returns True when:
    * ``any_payment_exists`` for player's food (+ extra_food) against bird's cost, AND
    * at least one habitat in bird.habitats has an open slot AND
      ``total_eggs + extra_eggs >= next_egg_cost`` for that habitat.

    ``extra_food`` is never mutated. ``extra_eggs`` is added to ``player.total_eggs``
    for the check (counterfactual "what if I had N more eggs")."""
    food = player.food if extra_food is None else _pool_add(player.food, extra_food)
    if not helpers.any_payment_exists(food, bird.food_cost):
        return False
    total_eggs = player.total_eggs + extra_eggs
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
) -> list[cards.Bird]:
    """Hand birds not in ``already_playable`` that become playable after gaining one
    unit of ``food``.

    Builds a counterfactual food pool with +1 of ``food`` (never mutates
    ``player.food``) and tests each not-yet-playable hand bird."""
    already_set = set(id(bird) for bird in already_playable)
    extra = _single_food_pool(food)
    return [
        bird
        for bird in player.hand
        if id(bird) not in already_set
        and _bird_playable(player, bird, extra_food=extra)
    ]


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
) -> list[cards.Bird]:
    """The optimistic union of ``newly_playable_after_food`` over all foods currently
    gainable from the birdfeeder.

    Used for main-action ``GAIN_FOOD`` rows and ``PayCostChoice`` exchanges that
    give food from the feeder — the exact die is not yet committed, so we
    advertise the best possible outcome."""
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
                player, bird, extra_food=extra
            ):
                newly.append(bird)
                seen_ids.add(id(bird))
    return newly


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
    counts = [0] * cards.N_FOODS
    counts[cards.food_index(food)] = 1
    return state.FoodPool.model_construct(counts=counts)


def _foods_to_pool(foods: tuple[cards.Food, ...]) -> state.FoodPool:
    """Convert a tuple of Food to a FoodPool with 1 of each food type present."""
    counts = [0] * cards.N_FOODS
    for food in foods:
        counts[cards.food_index(food)] = 1
    return state.FoodPool.model_construct(counts=counts)
