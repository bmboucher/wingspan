"""Behavior-preservation tests for the two aid-stage-1 seams in ``state.py``:
``Birdfeeder.roll_out_of_feeder`` (extracted from the dice-predator roll loop
and now shared by ``Birdfeeder.reroll``) and ``GameState.draw_bonus``
(extracted from the inline ``bonus_deck.pop()`` call sites).

The roll/reroll tests pin the rng call sequence against a hand-copied version
of the *old* inline algorithm, seeded identically, so a refactor that changed
the number or order of ``rng.randint`` calls would fail here even though the
statistical distribution looks the same.
"""

from __future__ import annotations

import random

from wingspan import cards, state

_SEEDS = (0, 1, 42, 12345, 999)


def _reference_roll(rng: random.Random, n: int) -> tuple[state.FoodPool, int]:
    """The pre-refactor roll loop, copied verbatim (same shape as the old
    dice-predator inline loop and the old ``Birdfeeder.reroll`` body)."""
    counts = state.FoodPool()
    choice = 0
    for _ in range(n):
        face = rng.randint(0, cards.N_FOODS)
        if face < cards.N_FOODS:
            counts[cards.ALL_FOODS[face]] += 1
        else:
            choice += 1
    return counts, choice


# ---------------------------------------------------------------------------
# roll_out_of_feeder: rng-stream equivalence


def test_roll_out_of_feeder_matches_reference_loop_for_various_dice_counts() -> None:
    for seed in _SEEDS:
        for n in (1, 2, 3, 5, 8):
            reference_rng = random.Random(seed)
            reference_counts, reference_choice = _reference_roll(reference_rng, n)

            seam_rng = random.Random(seed)
            feeder = state.Birdfeeder()
            seam_counts, seam_choice = feeder.roll_out_of_feeder(seam_rng, n)

            assert seam_counts.counts == reference_counts.counts, (seed, n)
            assert seam_choice == reference_choice, (seed, n)


def test_roll_out_of_feeder_consumes_the_rng_stream_identically() -> None:
    """Beyond matching output, the two rng objects must be left in the exact
    same state -- proof the seam makes the same number of ``randint`` calls,
    in the same order, as the code it replaced."""
    for seed in _SEEDS:
        reference_rng = random.Random(seed)
        _reference_roll(reference_rng, 5)

        seam_rng = random.Random(seed)
        state.Birdfeeder().roll_out_of_feeder(seam_rng, 5)

        assert reference_rng.random() == seam_rng.random(), seed


def test_roll_out_of_feeder_does_not_mutate_the_feeder() -> None:
    """The seam rolls dice that never enter the feeder -- the feeder's own
    dice must be untouched."""
    feeder = state.Birdfeeder()
    feeder.counts[cards.Food.FISH] = 2
    feeder.choice_dice = 1
    before_total = feeder.total()

    feeder.roll_out_of_feeder(random.Random(7), 3)

    assert feeder.total() == before_total
    assert feeder.counts[cards.Food.FISH] == 2
    assert feeder.choice_dice == 1


# ---------------------------------------------------------------------------
# reroll: still matches the old inline algorithm, still totals BIRDFEEDER_DICE


def test_reroll_matches_reference_implementation() -> None:
    for seed in _SEEDS:
        reference_rng = random.Random(seed)
        reference_counts, reference_choice = _reference_roll(
            reference_rng, state.BIRDFEEDER_DICE
        )

        seam_rng = random.Random(seed)
        feeder = state.Birdfeeder()
        feeder.reroll(seam_rng)

        assert feeder.counts.counts == reference_counts.counts, seed
        assert feeder.choice_dice == reference_choice, seed


def test_reroll_always_totals_birdfeeder_dice() -> None:
    feeder = state.Birdfeeder()
    for seed in _SEEDS:
        feeder.reroll(random.Random(seed))
        assert feeder.total() == state.BIRDFEEDER_DICE


# ---------------------------------------------------------------------------
# draw_bonus


def _new_game(seed: int = 0) -> state.GameState:
    birds, bonuses, goals = cards.load_all()
    return state.new_game(random.Random(seed), birds, bonuses, goals)


def test_draw_bonus_pops_from_the_end() -> None:
    gs = _new_game()
    expected_top = gs.bonus_deck[-1]
    expected_len = len(gs.bonus_deck) - 1

    drawn = gs.draw_bonus()

    assert drawn == expected_top
    assert len(gs.bonus_deck) == expected_len
    assert drawn not in gs.bonus_deck


def test_draw_bonus_sequential_draws_walk_backward_through_the_deck() -> None:
    gs = _new_game()
    expected_order = list(reversed(gs.bonus_deck))

    drawn_order = [gs.draw_bonus() for _ in range(len(expected_order))]

    assert drawn_order == expected_order
    assert gs.bonus_deck == []


def test_draw_bonus_returns_none_on_empty_without_raising() -> None:
    gs = _new_game()
    gs.bonus_deck = []

    assert gs.draw_bonus() is None
    assert gs.bonus_deck == []
