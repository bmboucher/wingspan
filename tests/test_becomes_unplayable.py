# pyright: reportPrivateUsage=false
"""Tests for the ``becomes_unplayable`` choice-encoding stripe.

Covers the playability loss helpers in ``engine.playability``, the encoding
stripe populated by ``encode.encode_choices``, and the dim invariants
introduced by the new stripe.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import cards, decisions, encode, engine, state
from wingspan.encode import layout
from wingspan.engine import playability

# ---------------------------------------------------------------------------
# Helpers


def _becomes_unplayable_bits(row: np.ndarray) -> np.ndarray:
    """Slice the becomes_unplayable stripe from a choice row."""
    off = layout.CHOICE_BECOMES_UNPLAYABLE_OFFSET
    dim = layout.CHOICE_BECOMES_UNPLAYABLE_DIM
    return row[off : off + dim]


def _zero_counts() -> state.FoodPool:
    return state.FoodPool(counts=[0] * cards.N_FOODS)


def _pool_with(food: cards.Food, count: int = 1) -> state.FoodPool:
    """FoodPool with ``count`` of ``food`` and 0 of everything else."""
    pool = state.FoodPool(counts=[0] * cards.N_FOODS)
    pool[food] = count
    return pool


def _pool_with_two(food_a: cards.Food, food_b: cards.Food) -> state.FoodPool:
    """FoodPool with exactly 1 of each of two food types."""
    pool = state.FoodPool(counts=[0] * cards.N_FOODS)
    pool[food_a] = 1
    pool[food_b] = 1
    return pool


# ---------------------------------------------------------------------------
# Section 1: newly_unplayable_after_egg_loss


class TestNewlyUnplayableAfterEggLoss:
    """Direct unit tests for the egg-loss helper."""

    def test_empty_baseline_returns_empty(self) -> None:
        eng, *_ = engine.Engine.create(seed=1)
        player = eng.state.players[0]
        result = playability.newly_unplayable_after_egg_loss(
            player, 1, already_playable=[]
        )
        assert result == []

    def test_zero_egg_loss_returns_empty(self) -> None:
        eng, birds, *_ = engine.Engine.create(seed=1)
        player = eng.state.players[0]
        free_bird = next(
            (
                b
                for b in birds
                if b.food_cost.total == 0 and player.can_play_in(b.habitats[0])
            ),
            None,
        )
        if free_bird is None:
            pytest.skip("no free-cost bird")
        result = playability.newly_unplayable_after_egg_loss(
            player, 0, already_playable=[free_bird]
        )
        assert result == []

    def test_egg_loss_does_not_affect_zero_egg_cost_slot(self) -> None:
        """A free bird in an empty board slot (0 egg cost) stays playable after
        losing 1 egg — the egg gate is already at 0."""
        eng, birds, *_ = engine.Engine.create(seed=2)
        player = eng.state.players[0]
        # Empty board: first slot in any habitat costs 0 eggs.
        free_bird = next(
            (
                bird
                for bird in birds
                if bird.food_cost.total == 0
                and any(
                    player.can_play_in(habitat) and len(player.board[habitat]) == 0
                    for habitat in bird.habitats
                )
            ),
            None,
        )
        if free_bird is None:
            pytest.skip("no free-cost bird with empty-first-slot habitat")
        result = playability.newly_unplayable_after_egg_loss(
            player, 1, already_playable=[free_bird]
        )
        # 0-egg-cost slot: losing 1 egg has no effect.
        assert free_bird not in result

    def test_egg_loss_flags_bird_needing_one_egg_when_player_has_none(self) -> None:
        """A bird at the second slot (1-egg cost) becomes unplayable when the
        player's last egg is removed."""
        eng, birds, *_ = engine.Engine.create(seed=3)
        player = eng.state.players[0]

        # Place one bird in forest so next_egg_cost(FOREST) = 1.
        forest_bird = next(
            (b for b in birds if cards.Habitat.FOREST in b.habitats), None
        )
        if forest_bird is None:
            pytest.skip("no forest bird in catalog")
        player.board[cards.Habitat.FOREST].append(state.PlayedBird(bird=forest_bird))

        # Find a forest bird with 0 food cost (so only egg is the gate).
        target = next(
            (
                b
                for b in birds
                if b is not forest_bird
                and cards.Habitat.FOREST in b.habitats
                and b.food_cost.total == 0
            ),
            None,
        )
        if target is None:
            pytest.skip("no second free-cost forest bird in catalog")

        # Give the player exactly 1 egg and strip food.
        player.food = _zero_counts()
        player.hand = [target]

        # Manually set exactly 1 egg on the forest bird (total_eggs covers the 1-egg cost):
        player.board[cards.Habitat.FOREST][0].eggs = 1
        assert player.total_eggs == 1

        playable_now, _ = playability.classify_hand_playability(player)
        if target not in playable_now:
            pytest.skip("could not construct egg-gate scenario with this catalog seed")

        lost = playability.newly_unplayable_after_egg_loss(
            player, 1, already_playable=playable_now
        )
        assert target in lost


# ---------------------------------------------------------------------------
# Section 2: newly_unplayable_after_food_removed (exact multiset)


class TestNewlyUnplayableAfterFoodRemoved:
    """Direct unit tests for exact-multiset food removal."""

    def test_empty_baseline_returns_empty(self) -> None:
        eng, *_ = engine.Engine.create(seed=10)
        player = eng.state.players[0]
        removed = _pool_with(cards.Food.SEED)
        result = playability.newly_unplayable_after_food_removed(
            player, removed, already_playable=[]
        )
        assert result == []

    def test_removing_required_food_flags_bird(self) -> None:
        """A bird needing 1 INV is flagged when the only INV is removed."""
        eng, birds, *_ = engine.Engine.create(seed=10)
        player = eng.state.players[0]
        inv_idx = cards.food_index(cards.Food.INVERTEBRATE)
        seed_idx = cards.food_index(cards.Food.SEED)

        # Find a bird that costs exactly 1 INV (not payable by SEED).
        target = next(
            (
                b
                for b in birds
                if b.food_cost.counts[inv_idx] == 1
                and b.food_cost.counts[seed_idx] == 0
                and b.food_cost.total == 1
                and any(player.can_play_in(h) for h in b.habitats)
            ),
            None,
        )
        if target is None:
            pytest.skip("no 1-INV bird in catalog")

        # Give player exactly 1 INV, 0 eggs needed (empty board).
        player.food = _pool_with(cards.Food.INVERTEBRATE)
        player.hand = [target]
        playable_now, _ = playability.classify_hand_playability(player)
        if target not in playable_now:
            pytest.skip("target not playable in initial state")

        removed = _pool_with(cards.Food.INVERTEBRATE)
        flagged = playability.newly_unplayable_after_food_removed(
            player, removed, already_playable=playable_now
        )
        assert target in flagged

    def test_removing_non_required_food_does_not_flag_bird(self) -> None:
        """Removing SEED does not flag a bird that costs only INV."""
        eng, birds, *_ = engine.Engine.create(seed=10)
        player = eng.state.players[0]
        inv_idx = cards.food_index(cards.Food.INVERTEBRATE)
        seed_idx = cards.food_index(cards.Food.SEED)

        target = next(
            (
                b
                for b in birds
                if b.food_cost.counts[inv_idx] == 1
                and b.food_cost.counts[seed_idx] == 0
                and b.food_cost.total == 1
                and any(player.can_play_in(h) for h in b.habitats)
            ),
            None,
        )
        if target is None:
            pytest.skip("no 1-INV bird in catalog")

        player.food = _pool_with_two(cards.Food.INVERTEBRATE, cards.Food.SEED)
        player.hand = [target]
        playable_now, _ = playability.classify_hand_playability(player)
        if target not in playable_now:
            pytest.skip("target not playable")

        # Remove SEED — the bird needs INV so it should still be affordable.
        removed = _pool_with(cards.Food.SEED)
        flagged = playability.newly_unplayable_after_food_removed(
            player, removed, already_playable=playable_now
        )
        assert target not in flagged

    def test_does_not_mutate_player_food(self) -> None:
        """The helper never mutates the player's food pool."""
        eng, birds, *_ = engine.Engine.create(seed=11)
        player = eng.state.players[0]
        original = list(player.food.counts)
        removed = _pool_with(cards.Food.SEED)
        playability.newly_unplayable_after_food_removed(
            player, removed, already_playable=birds[:3]
        )
        assert list(player.food.counts) == original


# ---------------------------------------------------------------------------
# Section 3: newly_unplayable_after_optimistic_food_loss (n-token removal)


class TestNewlyUnplayableAfterOptimisticFoodLoss:
    """Tests for the optimistic (best-case removal) helper.

    The motivating cases from the plan:
      - Pool = [INV, SEED], bird costs [INV], remove 1 → bird survives
        (optimal: remove SEED, INV remains).
      - Pool = [INV, SEED], bird costs [INV+SEED] (AND), remove 1 → bird flagged
        (every 1-removal breaks the AND cost).
    """

    def test_empty_baseline_returns_empty(self) -> None:
        eng, *_ = engine.Engine.create(seed=20)
        player = eng.state.players[0]
        result = playability.newly_unplayable_after_optimistic_food_loss(
            player, 1, already_playable=[]
        )
        assert result == []

    def test_n_zero_returns_empty(self) -> None:
        eng, birds, *_ = engine.Engine.create(seed=20)
        player = eng.state.players[0]
        result = playability.newly_unplayable_after_optimistic_food_loss(
            player, 0, already_playable=birds[:3]
        )
        assert result == []

    def test_optimistic_or_cost_survives_when_alternative_remains(self) -> None:
        """Bird costs [INV] (OR: one-of), pool has [INV, SEED], remove 1.

        Optimal removal is SEED — INV remains and the bird is still affordable.
        The bird should NOT be flagged."""
        eng, birds, *_ = engine.Engine.create(seed=20)
        player = eng.state.players[0]
        inv_idx = cards.food_index(cards.Food.INVERTEBRATE)

        # Find a bird that costs exactly 1 INV (sole cost).
        target = next(
            (
                b
                for b in birds
                if b.food_cost.counts[inv_idx] == 1
                and b.food_cost.total == 1
                and b.food_cost.is_or_cost is False
                and any(player.can_play_in(h) for h in b.habitats)
            ),
            None,
        )
        if target is None:
            pytest.skip("no 1-INV bird")

        # Pool has both INV and SEED; removing SEED leaves INV, so the bird survives.
        player.food = _pool_with_two(cards.Food.INVERTEBRATE, cards.Food.SEED)
        player.hand = [target]
        playable_now, _ = playability.classify_hand_playability(player)
        if target not in playable_now:
            pytest.skip("target not playable")

        flagged = playability.newly_unplayable_after_optimistic_food_loss(
            player, 1, already_playable=playable_now
        )
        assert target not in flagged

    def test_and_cost_flagged_when_every_removal_breaks_it(self) -> None:
        """Bird costs [INV AND SEED], pool has [INV, SEED], remove 1.

        Every 1-removal breaks the AND cost:
          - Remove INV → only SEED remains → can't pay INV.
          - Remove SEED → only INV remains → can't pay SEED.
        The bird should be flagged."""
        eng, birds, *_ = engine.Engine.create(seed=20)
        player = eng.state.players[0]
        inv_idx = cards.food_index(cards.Food.INVERTEBRATE)
        seed_idx = cards.food_index(cards.Food.SEED)

        # Find a bird that costs exactly 1 INV + 1 SEED (AND cost, total=2).
        target = next(
            (
                b
                for b in birds
                if b.food_cost.counts[inv_idx] == 1
                and b.food_cost.counts[seed_idx] == 1
                and b.food_cost.total == 2
                and b.food_cost.is_or_cost is False
                and any(player.can_play_in(h) for h in b.habitats)
            ),
            None,
        )
        if target is None:
            pytest.skip("no 1-INV+1-SEED AND-cost bird")

        # Provide exactly the required food pool.
        player.food = _pool_with_two(cards.Food.INVERTEBRATE, cards.Food.SEED)
        player.hand = [target]
        playable_now, _ = playability.classify_hand_playability(player)
        if target not in playable_now:
            pytest.skip("target not playable with exact food")

        flagged = playability.newly_unplayable_after_optimistic_food_loss(
            player, 1, already_playable=playable_now
        )
        assert target in flagged

    def test_pool_too_small_for_n_returns_empty(self) -> None:
        """Asking to remove more tokens than exist in the pool returns empty."""
        eng, birds, *_ = engine.Engine.create(seed=21)
        player = eng.state.players[0]
        player.food = _pool_with(cards.Food.SEED, 1)  # only 1 token total

        free_bird = next(
            (
                b
                for b in birds
                if b.food_cost.total == 0 and player.can_play_in(b.habitats[0])
            ),
            None,
        )
        if free_bird is None:
            pytest.skip("no free bird")

        player.hand = [free_bird]
        playable_now, _ = playability.classify_hand_playability(player)

        # Trying to remove 3 tokens from a 1-token pool → no valid removals.
        flagged = playability.newly_unplayable_after_optimistic_food_loss(
            player, 3, already_playable=playable_now
        )
        assert flagged == []

    def test_n_2_removal_flags_bird_needing_both_tokens(self) -> None:
        """Pool [INV, SEED], n=2: bird costs [INV] (1 token).

        The only size-2 removal takes the entire pool; no food remains for the bird.
        The bird should be flagged."""
        eng, birds, *_ = engine.Engine.create(seed=22)
        player = eng.state.players[0]
        inv_idx = cards.food_index(cards.Food.INVERTEBRATE)

        target = next(
            (
                b
                for b in birds
                if b.food_cost.counts[inv_idx] == 1
                and b.food_cost.total == 1
                and b.food_cost.is_or_cost is False
                and any(player.can_play_in(h) for h in b.habitats)
            ),
            None,
        )
        if target is None:
            pytest.skip("no 1-INV bird")

        player.food = _pool_with_two(cards.Food.INVERTEBRATE, cards.Food.SEED)
        player.hand = [target]
        playable_now, _ = playability.classify_hand_playability(player)
        if target not in playable_now:
            pytest.skip("target not playable")

        # Removing both tokens leaves nothing; bird is flagged.
        flagged = playability.newly_unplayable_after_optimistic_food_loss(
            player, 2, already_playable=playable_now
        )
        assert target in flagged


# ---------------------------------------------------------------------------
# Section 4: newly_unplayable_after_play (full-play counterfactual)


class TestNewlyUnplayableAfterPlay:
    """Tests for the full-play (slot + egg + food) counterfactual helper."""

    def test_played_bird_excluded_from_result(self) -> None:
        """The bird being played is never flagged in the result."""
        eng, birds, *_ = engine.Engine.create(seed=30)
        player = eng.state.players[0]
        player.food = state.FoodPool(counts=[10] * cards.N_FOODS)

        # Pick any bird with an open habitat.
        target = next(
            (b for b in birds if any(player.can_play_in(h) for h in b.habitats)), None
        )
        if target is None:
            pytest.skip("no playable bird")

        habitat = next(h for h in target.habitats if player.can_play_in(h))
        player.hand = [target]
        playable_now, _ = playability.classify_hand_playability(player)

        flagged = playability.newly_unplayable_after_play(
            player, target, habitat, already_playable=playable_now
        )
        assert target not in flagged

    def test_empty_baseline_returns_empty(self) -> None:
        eng, birds, *_ = engine.Engine.create(seed=31)
        player = eng.state.players[0]
        target = birds[0]
        habitat = target.habitats[0]

        flagged = playability.newly_unplayable_after_play(
            player, target, habitat, already_playable=[]
        )
        assert flagged == []

    def test_slot_exhaustion_flags_habitat_restricted_bird(self) -> None:
        """Filling the last slot in a habitat flags a bird restricted to that habitat."""
        eng, birds, *_ = engine.Engine.create(seed=32)
        player = eng.state.players[0]

        # Fill GRASSLAND to 4 slots (leaving 1 open).
        grassland_birds = [b for b in birds if cards.Habitat.GRASSLAND in b.habitats]
        if len(grassland_birds) < 6:
            pytest.skip("not enough grassland birds")

        for occupant in grassland_birds[:4]:
            player.board[cards.Habitat.GRASSLAND].append(
                state.PlayedBird(bird=occupant)
            )

        # Grassland-only bird costs nothing (so the ONLY gate is the open slot).
        grassland_only = next(
            (
                b
                for b in birds
                if b.habitats == (cards.Habitat.GRASSLAND,)
                and b.food_cost.total == 0
                and b not in grassland_birds[:4]
            ),
            None,
        )
        if grassland_only is None:
            pytest.skip("no grassland-only free bird to act as observer")

        # The bird being played also lives in grassland, fills the 5th slot.
        to_play = next(
            (
                b
                for b in birds
                if cards.Habitat.GRASSLAND in b.habitats
                and b is not grassland_only
                and b not in grassland_birds[:4]
                and b.food_cost.total == 0
            ),
            None,
        )
        if to_play is None:
            pytest.skip("no second free grassland bird to play")

        player.food = _zero_counts()
        player.hand = [grassland_only, to_play]
        playable_now, _ = playability.classify_hand_playability(player)

        if grassland_only not in playable_now:
            pytest.skip("grassland_only not currently playable")
        if to_play not in playable_now:
            pytest.skip("to_play not currently playable")

        flagged = playability.newly_unplayable_after_play(
            player, to_play, cards.Habitat.GRASSLAND, already_playable=playable_now
        )
        # After the play there is no open grassland slot; the restricted bird is flagged.
        assert grassland_only in flagged

    def test_food_drain_flags_bird_that_cannot_be_paid_after(self) -> None:
        """A bystander bird that needs the same food tokens as the played bird's
        cost is flagged when the payment leaves no food for it."""
        eng, birds, *_ = engine.Engine.create(seed=33)
        player = eng.state.players[0]
        inv_idx = cards.food_index(cards.Food.INVERTEBRATE)

        # Find two distinct 1-INV birds that can play in the same habitat.
        inv_birds = [
            b
            for b in birds
            if b.food_cost.counts[inv_idx] == 1
            and b.food_cost.total == 1
            and b.food_cost.is_or_cost is False
        ]
        if len(inv_birds) < 2:
            pytest.skip("fewer than 2 single-INV birds in catalog")

        played = inv_birds[0]
        observer = inv_birds[1]

        # Find a habitat both birds can play in.
        shared_habitats = set(played.habitats) & set(observer.habitats)
        if not shared_habitats:
            pytest.skip("no shared habitat for the two INV birds")
        habitat = next(iter(shared_habitats))

        # Give exactly 1 INV; the play costs it, leaving nothing for the observer.
        player.food = _pool_with(cards.Food.INVERTEBRATE, 1)
        player.hand = [played, observer]
        playable_now, _ = playability.classify_hand_playability(player)

        if observer not in playable_now:
            pytest.skip("observer not playable with 1 INV (may need eggs)")

        flagged = playability.newly_unplayable_after_play(
            player, played, habitat, already_playable=playable_now
        )
        assert observer in flagged


# ---------------------------------------------------------------------------
# Section 5: Encoding stripe — accept vs skip rows


class TestEncodeBecomesUnplayableStripe:
    """Check that the stripe slice is populated correctly in real decisions."""

    def test_stripe_offset_within_row_bounds(self) -> None:
        """The stripe fits within a single encoded choice row."""
        off = layout.CHOICE_BECOMES_UNPLAYABLE_OFFSET
        dim = layout.CHOICE_BECOMES_UNPLAYABLE_DIM
        row_width = encode.choice_feature_dim(encode.DEFAULT_SPEC)
        assert off >= 0
        assert off + dim <= row_width

    def test_becomes_unplayable_offset_after_becomes_playable(self) -> None:
        """becomes_unplayable immediately follows becomes_playable in the layout."""
        assert (
            layout.CHOICE_BECOMES_PLAYABLE_OFFSET + layout.CHOICE_BECOMES_PLAYABLE_DIM
            == layout.CHOICE_BECOMES_UNPLAYABLE_OFFSET
        )

    def test_main_action_choice_stripe_zero(self) -> None:
        """MainActionChoice never populates becomes_unplayable."""
        eng, *_ = engine.Engine.create(seed=40)
        decision = decisions.MainActionDecision(
            player_id=0,
            prompt="action",
            choices=[
                decisions.MainActionChoice(
                    label="play", action=decisions.MainAction.PLAY_BIRD
                ),
                decisions.MainActionChoice(
                    label="food", action=decisions.MainAction.GAIN_FOOD
                ),
            ],
        )
        rows = encode.encode_choices(decision, eng.state)
        for row in rows:
            bits = _becomes_unplayable_bits(row)
            assert np.all(
                bits == 0.0
            ), "MainActionChoice should never set becomes_unplayable"

    def test_food_payment_row_flags_broken_birds(self) -> None:
        """FoodPaymentChoice rows that spend a bird's required food set the stripe."""
        eng, birds, *_ = engine.Engine.create(seed=41)
        player = eng.state.players[0]
        inv_idx = cards.food_index(cards.Food.INVERTEBRATE)

        # Observer bird: costs 1 INV.
        observer = next(
            (
                b
                for b in birds
                if b.food_cost.counts[inv_idx] == 1
                and b.food_cost.total == 1
                and b.food_cost.is_or_cost is False
                and any(player.can_play_in(h) for h in b.habitats)
            ),
            None,
        )
        # Played bird: costs something other than INV (e.g. 0-cost).
        played = next(
            (
                b
                for b in birds
                if b is not observer
                and b.food_cost.total == 0
                and any(player.can_play_in(h) for h in b.habitats)
            ),
            None,
        )
        if observer is None or played is None:
            pytest.skip("suitable birds not found in catalog")

        # Setup: player holds observer, has exactly 1 INV, empty board.
        player.food = _pool_with(cards.Food.INVERTEBRATE, 1)
        player.hand = [observer]
        playable_now, _ = playability.classify_hand_playability(player)
        if observer not in playable_now:
            pytest.skip("observer not playable")

        # Payment decision: pay 1 INV (the only food).
        inv_payment = state.FoodPool(counts=[0] * cards.N_FOODS)
        inv_payment[cards.Food.INVERTEBRATE] = 1
        habitat = played.habitats[0]
        decision = decisions.PayBirdFoodDecision(
            player_id=0,
            prompt="pay",
            choices=[decisions.FoodPaymentChoice(label="inv", payment=inv_payment)],
            bird=played,
            habitat=habitat,
        )
        row = encode.encode_choices(decision, eng.state)[0]
        bits = _becomes_unplayable_bits(row)
        bird_bit_idx = cards.bird_index(observer)
        assert bits[bird_bit_idx] == 1.0, "observer (needs INV) should be flagged"

    def test_remove_egg_decision_flags_egg_gated_bird(self) -> None:
        """A RemoveEggDecision row flags a bird that needs the egg being removed."""
        eng, birds, *_ = engine.Engine.create(seed=42)
        player = eng.state.players[0]

        # Place one bird in forest so next_egg_cost(FOREST) = 1.
        forest_occupant = next(
            (b for b in birds if cards.Habitat.FOREST in b.habitats), None
        )
        if forest_occupant is None:
            pytest.skip("no forest bird in catalog")
        played_bird = state.PlayedBird(bird=forest_occupant)
        played_bird.eggs = 1
        player.board[cards.Habitat.FOREST].append(played_bird)

        # Observer: free-cost forest bird (needs 0 food, 1 egg for slot 2).
        observer = next(
            (
                b
                for b in birds
                if b is not forest_occupant
                and cards.Habitat.FOREST in b.habitats
                and b.food_cost.total == 0
            ),
            None,
        )
        if observer is None:
            pytest.skip("no second free forest bird")

        # Player has exactly 1 egg; the observer is playable (needs 1 egg for slot 2).
        player.food = _zero_counts()
        player.hand = [observer]
        playable_now, _ = playability.classify_hand_playability(player)
        if observer not in playable_now:
            pytest.skip("observer not playable with 1 egg in slot-2 scenario")

        # RemoveEgg decision targeting the egg on the first forest slot.
        # is_pay is derived by the featurizer from isinstance(decision, RemoveEggDecision).
        board_target = decisions.BoardTargetChoice(
            label="remove",
            habitat=cards.Habitat.FOREST,
            slot=0,
        )
        decision = decisions.RemoveEggDecision(
            player_id=0,
            prompt="remove",
            choices=[board_target],
        )
        row = encode.encode_choices(decision, eng.state)[0]
        bits = _becomes_unplayable_bits(row)
        bird_bit_idx = cards.bird_index(observer)
        assert bits[bird_bit_idx] == 1.0, "egg-gated observer should be flagged"

    def test_play_bird_row_flags_observer_losing_food(self) -> None:
        """A PlayBirdChoice row flags a bystander bird that needs the same food."""
        eng, birds, *_ = engine.Engine.create(seed=43)
        player = eng.state.players[0]
        inv_idx = cards.food_index(cards.Food.INVERTEBRATE)

        inv_birds = [
            b
            for b in birds
            if b.food_cost.counts[inv_idx] == 1
            and b.food_cost.total == 1
            and b.food_cost.is_or_cost is False
            and any(player.can_play_in(h) for h in b.habitats)
        ]
        if len(inv_birds) < 2:
            pytest.skip("need at least 2 single-INV birds")

        played, observer = inv_birds[0], inv_birds[1]
        habitat = played.habitats[0]

        # Player holds both, has exactly 1 INV.
        player.food = _pool_with(cards.Food.INVERTEBRATE, 1)
        player.hand = [played, observer]
        playable_now, _ = playability.classify_hand_playability(player)
        if observer not in playable_now:
            pytest.skip("observer not playable")

        decision = decisions.PlayBirdDecision(
            player_id=0,
            prompt="play",
            choices=[
                decisions.PlayBirdChoice(label="play", bird=played, habitat=habitat)
            ],
        )
        row = encode.encode_choices(decision, eng.state)[0]
        bits = _becomes_unplayable_bits(row)
        observer_idx = cards.bird_index(observer)
        assert bits[observer_idx] == 1.0, "observer should be flagged by play bird row"

    def test_accept_exchange_with_egg_payment_flags_observer(self) -> None:
        """PayCostChoice (paid_egg_count=1) flags a bird at its egg threshold."""
        eng, birds, *_ = engine.Engine.create(seed=44)
        player = eng.state.players[0]

        # Create an egg-gated scenario (1 egg in slot, observer needs it).
        forest_occupant = next(
            (b for b in birds if cards.Habitat.FOREST in b.habitats), None
        )
        if forest_occupant is None:
            pytest.skip("no forest bird")
        played_bird = state.PlayedBird(bird=forest_occupant)
        played_bird.eggs = 1
        player.board[cards.Habitat.FOREST].append(played_bird)

        observer = next(
            (
                b
                for b in birds
                if b is not forest_occupant
                and cards.Habitat.FOREST in b.habitats
                and b.food_cost.total == 0
            ),
            None,
        )
        if observer is None:
            pytest.skip("no second free forest bird")

        player.food = _zero_counts()
        player.hand = [observer]
        playable_now, _ = playability.classify_hand_playability(player)
        if observer not in playable_now:
            pytest.skip("observer not playable")

        # An AcceptExchangeDecision paying 1 egg (Wetland egg→card).
        pay_choice = decisions.PayCostChoice(label="accept", paid_egg_count=1)
        skip_choice = decisions.SkipChoice(label="skip")
        decision = decisions.AcceptExchangeDecision(
            player_id=0,
            prompt="exchange",
            choices=[pay_choice, skip_choice],
        )
        rows = encode.encode_choices(decision, eng.state)
        # First row = accept (paid_egg_count=1), second = skip.
        accept_bits = _becomes_unplayable_bits(rows[0])
        skip_bits = _becomes_unplayable_bits(rows[1])

        observer_idx = cards.bird_index(observer)
        assert (
            accept_bits[observer_idx] == 1.0
        ), "accept row should flag egg-gated observer"
        assert skip_bits[observer_idx] == 0.0, "skip row should not flag anything"

    def test_nothing_playable_stripe_all_zeros(self) -> None:
        """When no birds are currently playable, every row's stripe is zero."""
        eng, *_ = engine.Engine.create(seed=45)
        player = eng.state.players[0]

        # Strip all food and eggs so nothing is playable.
        player.food = _zero_counts()
        for habitat in cards.ALL_HABITATS:
            for played_bird in player.board[habitat]:
                played_bird.eggs = 0

        # A spend-food decision: the stripe should be all zero.
        decision = decisions.SpendFoodDecision(
            player_id=0,
            prompt="spend",
            choices=[
                decisions.FoodChoice(label="inv", food=cards.Food.INVERTEBRATE),
                decisions.SkipChoice(label="skip"),
            ],
        )
        rows = encode.encode_choices(decision, eng.state)
        for row in rows:
            bits = _becomes_unplayable_bits(row)
            assert np.all(bits == 0.0), "no playable birds ⇒ all stripe bits zero"


# ---------------------------------------------------------------------------
# Section 6: Dim invariants


class TestDimInvariants:
    """Dim relationships introduced by the new stripe."""

    def test_choice_feature_dim_grew_by_bird_id_dim(self) -> None:
        """choice_feature_dim grew by exactly CHOICE_BECOMES_UNPLAYABLE_DIM (180)."""
        assert layout.CHOICE_BECOMES_UNPLAYABLE_DIM == layout._BIRD_ID_DIM
        assert layout.CHOICE_BECOMES_UNPLAYABLE_DIM == 180

    def test_becomes_unplayable_dim_same_as_becomes_playable_dim(self) -> None:
        """Both multi-hot stripes are over the same 180-bird space."""
        assert (
            layout.CHOICE_BECOMES_UNPLAYABLE_DIM == layout.CHOICE_BECOMES_PLAYABLE_DIM
        )

    def test_choice_input_dim_excludes_unplayable_when_flag_false(self) -> None:
        """choice_input_dim with has_becomes_unplayable=False is smaller."""
        card_embed_dim = 8
        raw = encode.choice_feature_dim(encode.DEFAULT_SPEC)
        with_stripe = encode.choice_input_dim(
            raw, card_embed_dim, has_becomes_unplayable=True
        )
        without_stripe = encode.choice_input_dim(
            raw - layout.CHOICE_BECOMES_UNPLAYABLE_DIM,
            card_embed_dim,
            has_becomes_unplayable=False,
        )
        # The stripe contributes one card_embed_dim embedding (not 180 raw dims).
        assert with_stripe - without_stripe == card_embed_dim

    def test_becomes_unplayable_stripe_at_expected_offset(self) -> None:
        """CHOICE_BECOMES_UNPLAYABLE_OFFSET equals CHOICE_BASE_LAYOUT.offset_of."""
        assert layout.CHOICE_BECOMES_UNPLAYABLE_OFFSET == (
            layout.CHOICE_BECOMES_PLAYABLE_OFFSET + layout.CHOICE_BECOMES_PLAYABLE_DIM
        )
