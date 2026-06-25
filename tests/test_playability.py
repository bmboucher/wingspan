"""Tests for ``wingspan.engine.playability`` — hand-playability predicates.

Covers :func:`classify_hand_playability`, the ``newly_playable_after_*`` helpers,
:func:`gainable_feeder_foods`, and :func:`setup_turn1_playable`.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import cards, engine, state
from wingspan.engine import playability

# ---------------------------------------------------------------------------
# classify_hand_playability


def test_classify_hand_playability_empty_hand():
    """An empty hand produces two empty lists — nothing to classify."""
    eng, *_ = engine.Engine.create(seed=1)
    player = eng.state.players[0]
    player.hand = []
    playable, egg_blocked = playability.classify_hand_playability(player)
    assert playable == []
    assert egg_blocked == []


def test_classify_hand_playability_returns_two_lists():
    """The return value is always a 2-tuple of lists."""
    eng, *_ = engine.Engine.create(seed=2)
    player = eng.state.players[0]
    result = playability.classify_hand_playability(player)
    assert isinstance(result, tuple)
    assert len(result) == 2
    playable, egg_blocked = result
    assert isinstance(playable, list)
    assert isinstance(egg_blocked, list)


def test_classify_hand_playability_disjoint():
    """playable_now and playable_if_eggs are always disjoint."""
    eng, *_ = engine.Engine.create(seed=5)
    player = eng.state.players[0]
    playable, egg_blocked = playability.classify_hand_playability(player)
    overlap = set(id(bird) for bird in playable) & set(id(bird) for bird in egg_blocked)
    assert not overlap, f"Found {len(overlap)} birds in both classification groups"


def test_classify_hand_playability_only_hand_birds():
    """Every bird in the result lists must be one of the player's hand birds."""
    eng, *_ = engine.Engine.create(seed=6)
    player = eng.state.players[0]
    hand_ids = {id(bird) for bird in player.hand}
    playable, egg_blocked = playability.classify_hand_playability(player)
    for bird in playable:
        assert id(bird) in hand_ids
    for bird in egg_blocked:
        assert id(bird) in hand_ids


def test_classify_hand_playability_free_bird_in_empty_board():
    """A 0-cost bird on an empty board is in playable_now (0-egg first slot).

    After a fresh engine start, no birds are on the board — every habitat's
    first slot costs 0 eggs (EGG_COSTS[0] == 0), so any affordable bird fits."""
    eng, birds_list, *_ = engine.Engine.create(seed=3)
    player = eng.state.players[0]
    # Find a free-cost bird that has at least one open habitat.
    free_bird = next(
        (
            bird
            for bird in birds_list
            if bird.food_cost.total == 0
            and any(player.can_play_in(h) for h in bird.habitats)
        ),
        None,
    )
    if free_bird is None:
        pytest.skip("no free-cost bird with an open habitat in this catalog seed")
    player.hand = [free_bird]
    playable, _ = playability.classify_hand_playability(player)
    assert free_bird in playable


def test_classify_hand_playability_unaffordable_excluded():
    """A bird whose food cost cannot be met is excluded from both lists."""
    eng, birds_list, *_ = engine.Engine.create(seed=7)
    player = eng.state.players[0]
    # Find a bird with a food cost we cannot meet.
    costly = next(
        (bird for bird in birds_list if bird.food_cost.total > 0),
        None,
    )
    if costly is None:
        pytest.skip("all birds in catalog are free-cost")
    # Strip all food.
    player.food = state.FoodPool(counts=[0] * cards.N_FOODS)
    player.hand = [costly]
    playable, egg_blocked = playability.classify_hand_playability(player)
    assert costly not in playable
    assert costly not in egg_blocked


# ---------------------------------------------------------------------------
# newly_playable_after_food


def test_newly_playable_after_food_detects_transition():
    """A bird that needs exactly FISH and has an open slot becomes newly playable
    after gaining FISH when the player starts with no food."""
    eng, birds_list, *_ = engine.Engine.create(seed=10)
    player = eng.state.players[0]
    fish_idx = cards.food_index(cards.Food.FISH)

    # Find a 1-FISH-only bird with an open habitat.
    target = next(
        (
            bird
            for bird in birds_list
            if (
                bird.food_cost.counts[fish_idx] == 1
                and bird.food_cost.total == 1
                and any(player.can_play_in(h) for h in bird.habitats)
            )
        ),
        None,
    )
    if target is None:
        pytest.skip("no single-FISH bird with an open habitat in this catalog seed")

    player.food = state.FoodPool(counts=[0] * cards.N_FOODS)
    player.hand = [target]
    already: list[cards.Bird] = []
    newly = playability.newly_playable_after_food(
        player, cards.Food.FISH, already_playable=already
    )
    assert target in newly


def test_newly_playable_after_food_excludes_already_playable():
    """A bird already in ``already_playable`` is not included in the result."""
    eng, birds_list, *_ = engine.Engine.create(seed=11)
    player = eng.state.players[0]
    free_bird = next(
        (
            bird
            for bird in birds_list
            if bird.food_cost.total == 0
            and any(player.can_play_in(h) for h in bird.habitats)
        ),
        None,
    )
    if free_bird is None:
        pytest.skip("no free-cost bird with open habitat")
    player.hand = [free_bird]
    # Mark it as already playable so it should NOT appear in newly.
    already = [free_bird]
    newly = playability.newly_playable_after_food(
        player, cards.Food.FISH, already_playable=already
    )
    assert free_bird not in newly


def test_newly_playable_after_food_does_not_mutate_player_food():
    """The helper never mutates the player's food pool."""
    eng, birds_list, *_ = engine.Engine.create(seed=12)
    player = eng.state.players[0]
    fish_counts = list(player.food.counts)
    player.hand = birds_list[:3]
    playability.newly_playable_after_food(player, cards.Food.FISH, already_playable=[])
    assert list(player.food.counts) == fish_counts


# ---------------------------------------------------------------------------
# newly_playable_after_egg


def test_newly_playable_after_egg_excludes_already_playable():
    """A bird already in ``already_playable`` is excluded from the egg-gain result."""
    eng, birds_list, *_ = engine.Engine.create(seed=20)
    player = eng.state.players[0]
    free_bird = next(
        (
            b
            for b in birds_list
            if b.food_cost.total == 0 and any(player.can_play_in(h) for h in b.habitats)
        ),
        None,
    )
    if free_bird is None:
        pytest.skip("no free-cost bird with open habitat")
    player.hand = [free_bird]
    already = [free_bird]
    newly = playability.newly_playable_after_egg(
        player, n_eggs=1, already_playable=already
    )
    assert free_bird not in newly


def test_newly_playable_after_egg_result_is_subset_of_hand():
    """All birds returned are from the player's hand."""
    eng, *_ = engine.Engine.create(seed=21)
    player = eng.state.players[0]
    hand_ids = {id(bird) for bird in player.hand}
    newly = playability.newly_playable_after_egg(player, n_eggs=3, already_playable=[])
    for bird in newly:
        assert id(bird) in hand_ids


# ---------------------------------------------------------------------------
# gainable_feeder_foods


def test_gainable_feeder_foods_plain_dice():
    """Foods with at least one die showing that face are included."""
    eng, *_ = engine.Engine.create(seed=30)
    feeder = eng.state.birdfeeder
    # Clear and set exactly FISH dice.
    feeder.counts = state.FoodPool(counts=[0] * cards.N_FOODS)
    feeder.choice_dice = 0
    feeder.counts[cards.Food.FISH] = 2
    foods = playability.gainable_feeder_foods(feeder)
    assert cards.Food.FISH in foods
    assert cards.Food.FRUIT not in foods


def test_gainable_feeder_foods_choice_dice_add_invertebrate_and_seed():
    """Choice dice add INVERTEBRATE and SEED (the taker picks one)."""
    eng, *_ = engine.Engine.create(seed=31)
    feeder = eng.state.birdfeeder
    feeder.counts = state.FoodPool(counts=[0] * cards.N_FOODS)
    feeder.counts[cards.Food.FISH] = 1
    feeder.choice_dice = 2
    foods = playability.gainable_feeder_foods(feeder)
    assert cards.Food.INVERTEBRATE in foods
    assert cards.Food.SEED in foods
    assert cards.Food.FISH in foods


def test_gainable_feeder_foods_empty_feeder():
    """An empty feeder (no dice, no choice dice) yields an empty set."""
    eng, *_ = engine.Engine.create(seed=32)
    feeder = eng.state.birdfeeder
    feeder.counts = state.FoodPool(counts=[0] * cards.N_FOODS)
    feeder.choice_dice = 0
    assert playability.gainable_feeder_foods(feeder) == set()


# ---------------------------------------------------------------------------
# newly_playable_after_feeder_food


def test_newly_playable_after_feeder_food_empty_feeder_returns_empty():
    """No newly-playable birds when the feeder is empty."""
    eng, *_ = engine.Engine.create(seed=40)
    player = eng.state.players[0]
    feeder = eng.state.birdfeeder
    feeder.counts = state.FoodPool(counts=[0] * cards.N_FOODS)
    feeder.choice_dice = 0
    newly = playability.newly_playable_after_feeder_food(
        player, feeder, already_playable=[]
    )
    assert newly == []


def test_newly_playable_after_feeder_food_no_duplicates():
    """Even if multiple feeder foods unlock the same bird, it appears at most once."""
    eng, *_ = engine.Engine.create(seed=41)
    player = eng.state.players[0]
    feeder = eng.state.birdfeeder
    # Load feeder with all food types so many birds might unlock.
    feeder.counts = state.FoodPool(counts=[3] * cards.N_FOODS)
    feeder.choice_dice = 0
    feeder.choice_dice = 0
    player.food = state.FoodPool(counts=[0] * cards.N_FOODS)
    newly = playability.newly_playable_after_feeder_food(
        player, feeder, already_playable=[]
    )
    # No duplicates: every bird id appears at most once.
    seen_ids = [id(bird) for bird in newly]
    assert len(seen_ids) == len(set(seen_ids))


# ---------------------------------------------------------------------------
# setup_turn1_playable


def test_setup_turn1_playable_no_food_and_costly_bird():
    """A bird with nonzero food cost is not turn-1 playable when kept_foods is empty."""
    all_birds, *_ = cards.load_all()
    costly = next((b for b in all_birds if b.food_cost.total > 0), None)
    if costly is None:
        pytest.skip("no birds with nonzero food cost")
    playable = playability.setup_turn1_playable((costly,), ())
    assert costly not in playable


def test_setup_turn1_playable_free_bird_always_playable():
    """A bird with 0 food cost is always playable regardless of kept foods."""
    all_birds, *_ = cards.load_all()
    free = next((b for b in all_birds if b.food_cost.total == 0), None)
    if free is None:
        pytest.skip("no free-cost birds in catalog")
    playable = playability.setup_turn1_playable((free,), ())
    assert free in playable


def test_setup_turn1_playable_food_match():
    """A 1-FISH bird is turn-1 playable when FISH is in kept_foods."""
    all_birds, *_ = cards.load_all()
    fish_idx = cards.food_index(cards.Food.FISH)
    fish_bird = next(
        (
            b
            for b in all_birds
            if b.food_cost.counts[fish_idx] == 1 and b.food_cost.total == 1
        ),
        None,
    )
    if fish_bird is None:
        pytest.skip("no single-FISH bird in catalog")
    playable = playability.setup_turn1_playable((fish_bird,), (cards.Food.FISH,))
    assert fish_bird in playable


def test_setup_turn1_playable_food_mismatch():
    """A 1-FISH bird is NOT turn-1 playable when only FRUIT is kept."""
    all_birds, *_ = cards.load_all()
    fish_idx = cards.food_index(cards.Food.FISH)
    fish_bird = next(
        (
            b
            for b in all_birds
            if b.food_cost.counts[fish_idx] == 1 and b.food_cost.total == 1
        ),
        None,
    )
    if fish_bird is None:
        pytest.skip("no single-FISH bird in catalog")
    playable = playability.setup_turn1_playable((fish_bird,), (cards.Food.FRUIT,))
    assert fish_bird not in playable


def test_setup_turn1_playable_returns_subset_of_kept():
    """All returned birds are from the kept_cards tuple."""
    all_birds, *_ = cards.load_all()
    kept = tuple(all_birds[:5])
    kept_foods = (cards.Food.FISH, cards.Food.SEED, cards.Food.FRUIT)
    playable = playability.setup_turn1_playable(kept, kept_foods)
    assert all(bird in kept for bird in playable)


def test_setup_turn1_playable_empty_inputs():
    """Empty kept_cards always returns an empty list."""
    playable = playability.setup_turn1_playable((), (cards.Food.FISH,))
    assert playable == []


# ---------------------------------------------------------------------------
# setup_playable_kept_cards


def _make_bird(template: cards.Bird, cost: cards.BirdCost) -> cards.Bird:
    """Return a copy of ``template`` with its food_cost replaced by ``cost``."""
    return template.model_copy(update={"food_cost": cost})


def _base_bird() -> cards.Bird:
    """Any real bird to use as a template for minting test birds."""
    all_birds, *_ = cards.load_all()
    return all_birds[0]


def test_playable_kept_empty():
    """Empty kept_cards returns an empty list."""
    result = playability.setup_playable_kept_cards(())
    assert result == []


def test_playable_kept_example1_inv_seed_fruit():
    """Example 1: 1 inv+seed+fruit bird is playable (keep inv+seed+fruit+any 4th)."""
    template = _base_bird()
    bird = _make_bird(
        template,
        cards.BirdCost.from_specific(
            {cards.Food.INVERTEBRATE: 1, cards.Food.SEED: 1, cards.Food.FRUIT: 1}
        ),
    )
    result = playability.setup_playable_kept_cards((bird,))
    assert bird in result


def test_playable_kept_example2_inv_seed_seed():
    """Example 2: 1 inv+seed+seed bird is playable (2nd seed paid 2-for-1 from spare token).

    Budget K = 5 − 1 = 4.  inv+seed+seed needs inv + seed + 2 wild-for-seed = 4 total.
    Exactly meets the budget, so the bird is playable."""
    template = _base_bird()
    bird = _make_bird(
        template,
        cards.BirdCost.from_specific({cards.Food.INVERTEBRATE: 1, cards.Food.SEED: 2}),
    )
    result = playability.setup_playable_kept_cards((bird,))
    assert bird in result


def test_playable_kept_example3_two_birds_all_playable():
    """Example 3: inv+seed+fruit bird and 1-fish bird kept; both playable (K=3)."""
    template = _base_bird()
    isf_bird = _make_bird(
        template,
        cards.BirdCost.from_specific(
            {cards.Food.INVERTEBRATE: 1, cards.Food.SEED: 1, cards.Food.FRUIT: 1}
        ),
    )
    fish_bird = _make_bird(
        template,
        cards.BirdCost.from_specific({cards.Food.FISH: 1}),
    )
    result = playability.setup_playable_kept_cards((isf_bird, fish_bird))
    assert isf_bird in result
    assert fish_bird in result


def test_playable_kept_example4_two_birds_only_fish_playable():
    """Example 4: inv+seed+seed and 1-fish kept; only fish playable.

    K = 5 − 2 = 3 distinct tokens.  inv+seed+seed needs minimum 4 distinct-token
    slots (inv + seed + 2 wild sub), which exceeds 3 → not playable.
    fish needs 1 → playable."""
    template = _base_bird()
    iss_bird = _make_bird(
        template,
        cards.BirdCost.from_specific({cards.Food.INVERTEBRATE: 1, cards.Food.SEED: 2}),
    )
    fish_bird = _make_bird(
        template,
        cards.BirdCost.from_specific({cards.Food.FISH: 1}),
    )
    result = playability.setup_playable_kept_cards((iss_bird, fish_bird))
    assert iss_bird not in result
    assert fish_bird in result


def test_playable_kept_free_bird_at_zero_budget():
    """A free-cost bird is playable even at K=0 (all 5 bird slots filled).

    K = 5 − 5 = 0 means only the empty food-keep subset is enumerated.
    The empty pool can still pay a zero-cost bird."""
    template = _base_bird()
    free_bird = _make_bird(template, cards.BirdCost.from_specific({}))
    # 5 birds kept → budget K=0
    result = playability.setup_playable_kept_cards((free_bird,) * 5)
    assert free_bird in result


def test_playable_kept_result_is_subset_of_kept():
    """Every bird returned by setup_playable_kept_cards is in the input tuple."""
    all_birds, *_ = cards.load_all()
    kept = tuple(all_birds[:3])
    result = playability.setup_playable_kept_cards(kept)
    assert all(bird in kept for bird in result)


def test_playable_kept_wild_cost_bird():
    """A 1-wild-cost bird is playable (any single food token satisfies it)."""
    template = _base_bird()
    wild_bird = _make_bird(template, cards.BirdCost.from_specific({}, wild=1))
    result = playability.setup_playable_kept_cards((wild_bird,))
    assert wild_bird in result
