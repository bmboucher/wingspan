"""Tests for the random-setup generator.

Validate that every generated keep is legal for the deal: kept cards ⊆ dealt
hand, retained-food size == 5 − kept count, foods ⊆ ALL_FOODS, bonus ∈ dealt
bonus — for both the joint batch generator and the single-seat helper.
"""

from __future__ import annotations

import random

from wingspan import cards, state  # noqa: E402
from wingspan.setup_model import candidates  # noqa: E402
from wingspan.setup_model import generate  # noqa: E402
from wingspan.setup_model import encode as setup_encode  # noqa: E402

type SeatDeal = tuple[list[cards.Bird], list[cards.BonusCard]]


def _deal_both(
    seed: int,
) -> tuple[tuple[SeatDeal, SeatDeal], setup_encode.SetupContext]:
    birds, bonuses, goals = cards.load_all()
    game_state = state.new_game(random.Random(seed), birds, bonuses, goals)
    context = setup_encode.SetupContext.from_state(game_state)
    seat0: SeatDeal = (list(birds[:5]), list(bonuses[:2]))
    seat1: SeatDeal = (list(birds[5:10]), list(bonuses[2:4]))
    return (seat0, seat1), context


def _assert_legal(keep: object, seat: SeatDeal) -> None:
    dealt_cards, dealt_bonus = seat
    assert isinstance(keep, candidates.SetupCandidate)
    assert set(keep.kept_cards) <= set(dealt_cards)
    assert len(keep.kept_foods) == cards.N_FOODS - len(keep.kept_cards)
    assert set(keep.kept_foods) <= set(cards.ALL_FOODS)
    assert keep.bonus_card in dealt_bonus


def test_generated_setups_are_legal():
    dealt, context = _deal_both(21)
    generator = generate.RandomSetupGenerator(
        hand_combos=10, food_sets=3, tuples_per_batch=16
    )
    joint = generator.generate(random.Random(21), dealt, context)
    assert 0 < len(joint) <= 16
    for seat0_keep, seat1_keep in joint:
        _assert_legal(seat0_keep, dealt[0])
        _assert_legal(seat1_keep, dealt[1])


def test_generate_one_is_legal():
    dealt, context = _deal_both(22)
    generator = generate.RandomSetupGenerator(
        hand_combos=5, food_sets=3, tuples_per_batch=16
    )
    keep = generator.generate_one(random.Random(1), dealt[0], context)
    _assert_legal(keep, dealt[0])
