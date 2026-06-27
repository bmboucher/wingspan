"""Tests for the setup model's per-candidate feature encoder.

Cover the fixed layout (the eight blocks sum to ``SETUP_FEATURE_DIM``), the
exact index placement of each block, the per-round goal one-hots, and that the
birdfeeder stripe matches the live game state. The trailing candidate-pricing
blocks (kept-bonus value, per-goal kept affinity) have their own dedicated
file, ``test_setup_encode_pricing.py``.
"""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import cards, encode, state  # noqa: E402
from wingspan.setup_model import architecture as arch_module  # noqa: E402
from wingspan.setup_model import candidates  # noqa: E402
from wingspan.setup_model import encode as setup_encode  # noqa: E402


def test_feature_dim_is_sum_of_blocks():
    expected = (
        cards.n_birds()  # kept cards
        + cards.N_FOODS  # kept foods
        + cards.n_bonus_cards()  # kept bonus
        + state.TRAY_SIZE  # tray (positional integer indices)
        + (cards.N_FOODS + 1)  # birdfeeder faces + choice die
        + 4 * encode.MAX_GOAL_CATEGORIES  # four round goals
        + 4  # kept-bonus pricing (qual / stepped / linear / tray potential)
        + 4  # per-goal kept-card affinity
    )
    assert setup_encode.SETUP_FEATURE_DIM == expected
    assert setup_encode.SETUP_GOAL_DIM == encode.MAX_GOAL_CATEGORIES


def _deal(
    seed: int,
) -> tuple[list[cards.Bird], list[cards.BonusCard], setup_encode.SetupContext]:
    birds, bonuses, goals = cards.load_all()
    game_state = state.new_game(random.Random(seed), birds, bonuses, goals)
    return (
        list(birds[:5]),
        list(bonuses[:2]),
        setup_encode.SetupContext.from_state(game_state),
    )


def test_kept_card_and_bonus_indices_are_set():
    dealt_cards, dealt_bonus, context = _deal(7)
    candidate = candidates.SetupCandidate(
        kept_cards=(dealt_cards[0], dealt_cards[2]),
        kept_foods=(cards.Food.SEED, cards.Food.FISH, cards.Food.FRUIT),
        bonus_card=dealt_bonus[0],
    )
    vec = setup_encode.encode_setup_candidate(candidate, context)
    encoding = arch_module.SetupEncoding()
    assert vec.shape == (encoding.total_dim,)
    # Kept cards multi-hot has exactly the two kept birds set.
    assert vec[cards.bird_index(dealt_cards[0])] == 1.0
    assert vec[cards.bird_index(dealt_cards[2])] == 1.0
    assert vec[cards.bird_index(dealt_cards[1])] == 0.0
    # Kept-food block: exactly the three retained foods set.
    food_base = cards.n_birds()
    assert vec[food_base + cards.food_index(cards.Food.SEED)] == 1.0
    assert vec[food_base + cards.food_index(cards.Food.INVERTEBRATE)] == 0.0
    # Bonus one-hot: exactly the kept bonus set.
    bonus_base = cards.n_birds() + cards.N_FOODS
    assert vec[bonus_base + cards.bonus_index(dealt_bonus[0])] == 1.0


def test_tray_block_is_positional_card_indices():
    """The tray block carries one ``bird_index + 1`` integer per slot, in slot
    order, with 0 for an empty slot — matching the state vector's tray block."""
    birds, bonuses, goals = cards.load_all()
    game_state = state.new_game(random.Random(7), birds, bonuses, goals)
    game_state.tray = [birds[10], None, birds[30]]
    context = setup_encode.SetupContext.from_state(game_state)
    assert context.tray_birds == (birds[10], None, birds[30])

    candidate = candidates.enumerate_setup_candidates(
        list(birds[:5]), list(bonuses[:2])
    )[0]
    vec = setup_encode.encode_setup_candidate(candidate, context)
    tray_base = cards.n_birds() + cards.N_FOODS + cards.n_bonus_cards()
    assert vec[tray_base + 0] == cards.bird_index(birds[10]) + 1
    assert vec[tray_base + 1] == 0.0
    assert vec[tray_base + 2] == cards.bird_index(birds[30]) + 1


def test_birdfeeder_block_matches_state():
    seed = 11
    birds, bonuses, goals = cards.load_all()
    game_state = state.new_game(random.Random(seed), birds, bonuses, goals)
    context = setup_encode.SetupContext.from_state(game_state)
    candidate = candidates.enumerate_setup_candidates(
        list(birds[:5]), list(bonuses[:2])
    )[0]
    vec = setup_encode.encode_setup_candidate(candidate, context)
    feeder_base = (
        cards.n_birds() + cards.N_FOODS + cards.n_bonus_cards() + state.TRAY_SIZE
    )
    for offset, food in enumerate(cards.ALL_FOODS):
        assert vec[feeder_base + offset] == game_state.birdfeeder.counts[food]
    assert vec[feeder_base + cards.N_FOODS] == game_state.birdfeeder.choice_dice


def test_round_goal_one_hots_are_per_round():
    dealt_cards, dealt_bonus, context = _deal(13)
    candidate = candidates.enumerate_setup_candidates(dealt_cards, dealt_bonus)[0]
    vec = setup_encode.encode_setup_candidate(candidate, context)
    goals_base = (
        cards.n_birds()
        + cards.N_FOODS
        + cards.n_bonus_cards()
        + state.TRAY_SIZE
        + cards.N_FOODS
        + 1
    )
    for round_idx, category in enumerate(context.round_goal_categories):
        start = goals_base + round_idx * setup_encode.SETUP_GOAL_DIM
        stripe = vec[start : start + setup_encode.SETUP_GOAL_DIM]
        # Each round's stripe is a single one-hot at the goal's category index.
        assert stripe.sum() == 1.0
        assert stripe[encode.GOAL_CATEGORIES.index(category)] == 1.0


def test_playable_kept_cards_stripe_present_by_default():
    """The default encoding (flag on since v1.1) includes the playable stripe."""
    encoding = arch_module.SetupEncoding()
    assert encoding.include_playable_kept_cards
    assert encoding.total_dim == encoding.total_dim  # sanity: 488 with N=any


def test_no_flags_encoding_matches_base_constant():
    """Explicitly disabling all optional stripes gives the SETUP_FEATURE_DIM base."""
    enc_off = arch_module.SetupEncoding(include_playable_kept_cards=False)
    assert enc_off.total_dim == setup_encode.SETUP_FEATURE_DIM


def test_playable_kept_cards_stripe_grows_vector():
    """Enabling the flag adds exactly 180 dims to total_dim."""
    enc_off = arch_module.SetupEncoding(include_playable_kept_cards=False)
    enc_on = arch_module.SetupEncoding(include_playable_kept_cards=True)
    assert enc_on.total_dim == enc_off.total_dim + cards.n_birds()


def test_playable_kept_cards_stripe_marks_playable_birds():
    """With the flag on, the stripe is set for payable kept birds and clear for others.

    We mint an exact-cost fish bird (1 FISH) and a too-expensive bird (3 RODENT)
    and keep both.  With K=3 spare tokens, the fish bird is payable but the rodent
    bird is not (needs 3 of the same type, which costs 3 distinct tokens via the
    2-for-1 substitution rule — actually 1+2+2=5, let's use 4 rodent cost instead).
    """
    # Use two distinct template birds so bird_index returns different indices.
    dealt_cards, _dealt_bonus, context = _deal(7)
    fish_bird = dealt_cards[0].model_copy(
        update={"food_cost": cards.BirdCost.from_specific({cards.Food.FISH: 1})}
    )
    # 3-rodent cost: min payment = 1 rodent + 2*(2-for-1) = 5 tokens > K=3, not payable.
    expensive_bird = dealt_cards[1].model_copy(
        update={"food_cost": cards.BirdCost.from_specific({cards.Food.RODENT: 3})}
    )
    encoding = arch_module.SetupEncoding(include_playable_kept_cards=True)
    candidate = candidates.SetupCandidate(
        kept_cards=(fish_bird, expensive_bird),
        kept_foods=(),
        bonus_card=None,
    )
    vec = setup_encode.encode_setup_candidate(candidate, context, encoding)

    assert vec.shape == (encoding.total_dim,)
    off = encoding.off_playable_kept_cards
    fish_idx = cards.bird_index(fish_bird)
    expensive_idx = cards.bird_index(expensive_bird)
    assert fish_idx != expensive_idx, "templates must be different birds"
    assert vec[off + fish_idx] == 1.0, "fish bird should be marked playable"
    assert (
        vec[off + expensive_idx] == 0.0
    ), "3-rodent bird should not be marked playable"
