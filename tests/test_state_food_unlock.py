# pyright: reportPrivateUsage=false
# (reads the package-private ``layout._FOOD_UNLOCK_SCALE`` normalizer, like the
# sibling ``test_becomes_unplayable`` reads private layout constants)
"""Tests for the food-distance-to-playable state stripes (v1.4).

Covers ``engine.playability.min_food_to_unlock`` and the two state-encoding
stripes ``hand_food_unlock_me`` / ``tray_food_unlock_me`` populated by
``encode.encode_state``.
"""

from __future__ import annotations

import numpy as np

from wingspan import cards, decisions, encode, engine, state
from wingspan.encode import layout
from wingspan.engine import playability

# ---------------------------------------------------------------------------
# Helpers


def _base_bird() -> cards.Bird:
    """Any real bird to use as a template for minting cost-controlled test birds."""
    all_birds, *_ = cards.load_all()
    return all_birds[0]


def _make_bird(template: cards.Bird, cost: cards.BirdCost) -> cards.Bird:
    """A copy of ``template`` with its food_cost replaced by ``cost``."""
    return template.model_copy(update={"food_cost": cost})


def _inv_seed_bird(template: cards.Bird) -> cards.Bird:
    return _make_bird(
        template,
        cards.BirdCost.from_specific({cards.Food.INVERTEBRATE: 1, cards.Food.SEED: 1}),
    )


def _fish_bird(template: cards.Bird) -> cards.Bird:
    return _make_bird(template, cards.BirdCost.from_specific({cards.Food.FISH: 1}))


def _stripe(vec: np.ndarray, offset: int) -> np.ndarray:
    """The 5-wide food stripe at ``offset`` un-normalized back to raw counts."""
    raw = vec[offset : offset + encode.STATE_FOOD_UNLOCK_DIM]
    return np.round(raw * layout._FOOD_UNLOCK_SCALE).astype(int)


def _main_action_decision() -> decisions.MainActionDecision:
    return decisions.MainActionDecision(
        player_id=0,
        prompt="action",
        choices=[
            decisions.MainActionChoice(
                label="food", action=decisions.MainAction.GAIN_FOOD
            )
        ],
    )


# ---------------------------------------------------------------------------
# min_food_to_unlock semantics


def test_worked_example_inv_seed_and_fish_with_two_fruit() -> None:
    """hand = {1inv+1seed, 1fish}, food = 2 fruit -> [1, 1, 2, 2, 2].

    Foods in ``cards.ALL_FOODS`` order (inv, seed, fish, fruit, rodent). The
    1-fish bird is already payable (2 fruit -> 1 fish via 2-for-1) so it is
    excluded; the inv+seed bird drives the result: +1 inv or +1 seed completes it
    (the 2 fruit pay the other specific), while a non-matching food needs 2 tokens
    (two 2-for-1 subs)."""
    eng, *_ = engine.Engine.create(seed=7)
    player = eng.state.players[0]
    template = _base_bird()
    player.food = state.FoodPool.from_dict({cards.Food.FRUIT: 2})
    player.hand = [_inv_seed_bird(template), _fish_bird(template)]

    assert playability.min_food_to_unlock(player, player.hand) == [1, 1, 2, 2, 2]


def test_all_playable_hand_is_all_zeros() -> None:
    """A hand whose birds are all already playable yields all zeros — nothing to
    unlock."""
    eng, *_ = engine.Engine.create(seed=3)
    player = eng.state.players[0]
    free_bird = _make_bird(_base_bird(), cards.BirdCost.from_specific({}))
    player.hand = [free_bird]

    assert playability.min_food_to_unlock(player, player.hand) == [0, 0, 0, 0, 0]


def test_slot_blocked_unplayable_is_all_zeros() -> None:
    """An unplayable bird with no open matching slot never contributes — food can
    never unlock a bird that has nowhere to land."""
    eng, *_ = engine.Engine.create(seed=4)
    player = eng.state.players[0]
    template = _base_bird()
    fish_bird = _fish_bird(template)
    player.food = state.FoodPool(counts=[0] * cards.N_FOODS)
    # Fill every habitat the bird could use so no slot remains open.
    for habitat in fish_bird.habitats:
        while player.can_play_in(habitat):
            player.board[habitat].append(state.PlayedBird(bird=template))
    player.hand = [fish_bird]

    assert playability.min_food_to_unlock(player, player.hand) == [0, 0, 0, 0, 0]


def test_empty_candidates_is_all_zeros() -> None:
    """No candidate birds -> all zeros."""
    eng, *_ = engine.Engine.create(seed=5)
    player = eng.state.players[0]
    assert playability.min_food_to_unlock(player, []) == [0, 0, 0, 0, 0]


def test_does_not_mutate_player_food() -> None:
    """The counterfactual search never mutates the player's food pool."""
    eng, *_ = engine.Engine.create(seed=8)
    player = eng.state.players[0]
    template = _base_bird()
    player.food = state.FoodPool.from_dict({cards.Food.FRUIT: 2})
    before = list(player.food.counts)
    player.hand = [_inv_seed_bird(template)]
    playability.min_food_to_unlock(player, player.hand)
    assert list(player.food.counts) == before


# ---------------------------------------------------------------------------
# State-encoding stripes


def test_state_size_grew_and_stripes_are_contiguous_in_prefix() -> None:
    """The two stripes are contiguous and sit in the continuous prefix (before the
    card-index block), where the compat shim can slice them out."""
    assert encode.STATE_FOOD_UNLOCK_DIM == cards.N_FOODS
    assert (
        encode.STATE_TRAY_FOOD_UNLOCK_OFFSET
        == encode.STATE_HAND_FOOD_UNLOCK_OFFSET + encode.STATE_FOOD_UNLOCK_DIM
    )
    assert encode.STATE_HAND_FOOD_UNLOCK_OFFSET < encode.OFF_CARD_INDEX
    assert (
        encode.STATE_TRAY_FOOD_UNLOCK_OFFSET + encode.STATE_FOOD_UNLOCK_DIM
        <= encode.OFF_CARD_INDEX
    )


def test_hand_stripe_reads_min_food_to_unlock() -> None:
    """The ``hand_food_unlock_me`` stripe in ``encode_state`` reflects the worked
    example when read from the deciding player's POV."""
    eng, *_ = engine.Engine.create(seed=7)
    player = eng.state.players[0]
    template = _base_bird()
    player.food = state.FoodPool.from_dict({cards.Food.FRUIT: 2})
    player.hand = [_inv_seed_bird(template), _fish_bird(template)]

    vec = encode.encode_state(eng.state, _main_action_decision())
    stripe = _stripe(vec, encode.STATE_HAND_FOOD_UNLOCK_OFFSET)
    assert stripe.tolist() == [1, 1, 2, 2, 2]


def test_tray_stripe_reads_min_food_to_unlock() -> None:
    """The ``tray_food_unlock_me`` stripe scores the tray as if in hand; with an
    empty hand only the tray drives it."""
    eng, *_ = engine.Engine.create(seed=7)
    player = eng.state.players[0]
    template = _base_bird()
    player.food = state.FoodPool.from_dict({cards.Food.FRUIT: 2})
    player.hand = []
    eng.state.tray = [_inv_seed_bird(template), _fish_bird(template), None]

    vec = encode.encode_state(eng.state, _main_action_decision())
    assert _stripe(vec, encode.STATE_HAND_FOOD_UNLOCK_OFFSET).tolist() == [0] * 5
    stripe = _stripe(vec, encode.STATE_TRAY_FOOD_UNLOCK_OFFSET)
    assert stripe.tolist() == [1, 1, 2, 2, 2]
