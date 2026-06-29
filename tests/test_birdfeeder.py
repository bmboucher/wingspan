"""The birdfeeder die has six faces — the five single foods plus an
*invertebrate-or-seed choice* face — each equally likely. So fish / fruit /
rodent each come up 1/6, while invertebrate and seed are each obtainable 1/3 of
the time (their own face 1/6 + the shared choice face 1/6). These tests pin the
roll odds and the choice semantics added by the die-fidelity fix.
"""

from __future__ import annotations

import random
import typing
from collections import Counter

import pytest

from wingspan import cards, decisions, engine, state  # noqa: E402
from wingspan.engine import actions  # noqa: E402


def _new_game(seed: int = 0) -> state.GameState:
    birds, bonuses, goals = cards.load_all()
    return state.new_game(random.Random(seed), birds, bonuses, goals)


def _empty_feeder(gs: state.GameState) -> state.Birdfeeder:
    """Zero out every die so a test can dictate the feeder's exact contents."""
    feeder = gs.birdfeeder
    feeder.counts.zero()
    feeder.choice_dice = 0
    return feeder


def test_reroll_yields_five_dice_over_six_equally_likely_faces() -> None:
    rng = random.Random(12345)
    rolls = 40000
    faces: Counter[str] = Counter()
    feeder = state.Birdfeeder()
    for _ in range(rolls):
        feeder.reroll(rng)
        assert feeder.total() == state.BIRDFEEDER_DICE  # five dice every reroll
        for food in cards.ALL_FOODS:
            faces[food.value] += feeder.counts[food]
        faces["choice"] += feeder.choice_dice
    total_dice = rolls * state.BIRDFEEDER_DICE
    # All six faces (the five foods + the choice face) are ~1/6 each. It follows
    # that invertebrate and seed are obtainable at ~1/3 (own face + choice face).
    for face in ("invertebrate", "seed", "fish", "fruit", "rodent", "choice"):
        freq = faces[face] / total_dice
        assert abs(freq - 1 / 6) < 0.01, (face, freq)


def test_choice_die_offers_invertebrate_or_seed_only() -> None:
    feeder = state.Birdfeeder()
    feeder.choice_dice = 1
    gainable = feeder.gainable_foods()
    assert cards.Food.INVERTEBRATE in gainable
    assert cards.Food.SEED in gainable
    assert cards.Food.FISH not in gainable


def test_take_choice_die_as_seed_consumes_it() -> None:
    feeder = state.Birdfeeder()
    feeder.choice_dice = 1
    feeder.take(cards.Food.SEED)
    assert feeder.choice_dice == 0
    assert feeder.is_empty()


def test_take_prefers_a_single_face_over_a_choice_die() -> None:
    feeder = state.Birdfeeder()
    feeder.counts[cards.Food.INVERTEBRATE] = 1
    feeder.choice_dice = 1
    feeder.take(cards.Food.INVERTEBRATE)
    assert feeder.counts[cards.Food.INVERTEBRATE] == 0  # single face spent first
    assert feeder.choice_dice == 1  # the choice die is left for later


def test_take_unavailable_food_raises() -> None:
    feeder = state.Birdfeeder()
    feeder.choice_dice = 1
    with pytest.raises(ValueError):
        feeder.take(cards.Food.FISH)  # a choice die can never yield fish


def test_gain_options_lists_plain_and_choice_die_separately() -> None:
    """When both a plain invertebrate die and a choice die show, gain_options
    offers taking invertebrate either way (and seed only from the choice die)."""
    feeder = state.Birdfeeder()
    feeder.counts[cards.Food.INVERTEBRATE] = 1
    feeder.choice_dice = 1
    options = feeder.gain_options()
    assert (cards.Food.INVERTEBRATE, False) in options  # the plain die
    assert (cards.Food.INVERTEBRATE, True) in options  # the choice die as inv
    assert (cards.Food.SEED, True) in options  # the choice die as seed
    assert (cards.Food.SEED, False) not in options  # no plain seed showing


def test_take_from_choice_die_spends_the_choice_die_not_a_plain_face() -> None:
    feeder = state.Birdfeeder()
    feeder.counts[cards.Food.INVERTEBRATE] = 1
    feeder.choice_dice = 1
    feeder.take(cards.Food.INVERTEBRATE, from_choice_die=True)
    assert feeder.choice_dice == 0  # the choice die was consumed
    assert feeder.counts[cards.Food.INVERTEBRATE] == 1  # the plain face is left


def test_take_from_choice_die_raises_without_one() -> None:
    feeder = state.Birdfeeder()
    feeder.counts[cards.Food.SEED] = 1
    with pytest.raises(ValueError):
        feeder.take(cards.Food.SEED, from_choice_die=True)


def test_total_and_distinct_faces_include_choice_dice() -> None:
    feeder = state.Birdfeeder()
    feeder.counts[cards.Food.FISH] = 2
    feeder.choice_dice = 1
    assert feeder.total() == 3
    assert feeder.distinct_faces() == 2  # the fish face + the choice face
    assert feeder.gainable_count(cards.Food.INVERTEBRATE) == 1  # via the choice die
    assert feeder.gainable_count(cards.Food.FISH) == 2


# ---------------------------------------------------------------------------
# Reset rules
#
# Rule 1: an empty feeder is rerolled automatically (never a player decision).
# Rule 2: when every die shows the same face, a player about to take food may
# reroll the whole feeder first — a single yes/no ResetBirdfeederDecision.


def _skip_any_reset[C: decisions.Choice](decision: decisions.Decision[C]) -> C:
    """Decline an offered reset; take the first option of anything else."""
    if isinstance(decision, decisions.ResetBirdfeederDecision):
        for choice in decision.choices:
            if isinstance(choice, decisions.SkipChoice):
                return typing.cast(C, choice)
    return typing.cast(C, decision.choices[0])


def test_gain_feeder_die_rerolls_when_the_take_empties_the_feeder() -> None:
    """Rule 1: taking the last die immediately refills the feeder."""
    gs = _new_game()
    feeder = _empty_feeder(gs)
    feeder.counts[cards.Food.FISH] = 1  # a single die
    eng = engine.Engine(gs)
    player = gs.players[0]
    before = player.food[cards.Food.FISH]

    actions.gain_feeder_die(eng, player, cards.Food.FISH)

    assert player.food[cards.Food.FISH] == before + 1
    assert feeder.total() == state.BIRDFEEDER_DICE  # auto-rerolled, never empty


def test_gain_feeder_die_leaves_a_nonempty_feeder_alone() -> None:
    """Rule 1 only fires on an emptying take; dice that remain are untouched."""
    gs = _new_game()
    feeder = _empty_feeder(gs)
    feeder.counts[cards.Food.FISH] = 2
    eng = engine.Engine(gs)
    player = gs.players[0]

    actions.gain_feeder_die(eng, player, cards.Food.FISH)

    assert feeder.counts[cards.Food.FISH] == 1
    assert feeder.total() == 1  # not rerolled


def test_offer_reset_auto_rerolls_an_empty_feeder() -> None:
    """Rule 1: an empty feeder is refilled with no player decision (the only
    decision that may surface is the optional reset, if the fresh roll happens
    to show a single face)."""
    gs = _new_game()
    feeder = _empty_feeder(gs)
    eng = engine.Engine(gs)
    player = gs.players[0]

    def agent[C: decisions.Choice](
        _eng: engine.Engine, decision: decisions.Decision[C]
    ) -> C:
        assert isinstance(decision, decisions.ResetBirdfeederDecision)
        return typing.cast(C, _skip_any_reset(decision))

    actions.offer_birdfeeder_reset(eng, agent, player)

    assert feeder.total() == state.BIRDFEEDER_DICE


def test_offer_reset_asks_on_a_single_food_face_and_respects_decline() -> None:
    """Rule 2: a feeder showing one food offers a reset; declining leaves it."""
    gs = _new_game()
    feeder = _empty_feeder(gs)
    feeder.counts[cards.Food.FISH] = 3  # all dice on one face
    eng = engine.Engine(gs)
    player = gs.players[0]
    asked = {"n": 0}

    def agent[C: decisions.Choice](
        _eng: engine.Engine, decision: decisions.Decision[C]
    ) -> C:
        asked["n"] += 1
        assert isinstance(decision, decisions.ResetBirdfeederDecision)
        return typing.cast(C, _skip_any_reset(decision))

    actions.offer_birdfeeder_reset(eng, agent, player)

    assert asked["n"] == 1
    assert feeder.counts[cards.Food.FISH] == 3  # declined: unchanged
    assert feeder.total() == 3


def test_offer_reset_asks_when_all_dice_show_the_choice_face() -> None:
    """Rule 2 trigger covers the invertebrate/seed choice face: all dice on it
    is still a single face."""
    gs = _new_game()
    feeder = _empty_feeder(gs)
    feeder.choice_dice = 5  # every die on the choice face
    eng = engine.Engine(gs)
    player = gs.players[0]
    asked = {"n": 0}

    def agent[C: decisions.Choice](
        _eng: engine.Engine, decision: decisions.Decision[C]
    ) -> C:
        asked["n"] += 1
        assert isinstance(decision, decisions.ResetBirdfeederDecision)
        return typing.cast(C, _skip_any_reset(decision))

    actions.offer_birdfeeder_reset(eng, agent, player)

    assert asked["n"] == 1
    assert feeder.choice_dice == 5  # declined: unchanged


def test_offer_reset_not_offered_on_mixed_faces() -> None:
    """Rule 2 does not fire while two or more faces are showing."""
    gs = _new_game()
    feeder = _empty_feeder(gs)
    feeder.counts[cards.Food.FISH] = 2
    feeder.counts[cards.Food.SEED] = 1
    eng = engine.Engine(gs)
    player = gs.players[0]

    def agent[C: decisions.Choice](  # pragma: no cover - must not be consulted
        _eng: engine.Engine, _decision: decisions.Decision[C]
    ) -> C:
        raise AssertionError("no reset should be offered with mixed faces")

    actions.offer_birdfeeder_reset(eng, agent, player)

    assert feeder.total() == 3  # untouched


def test_offer_reset_accept_rerolls_the_feeder() -> None:
    """Rule 2: accepting the reset rerolls all dice. A three-die single-face
    feeder becoming a full five-die roll proves the reroll happened."""
    gs = _new_game()
    feeder = _empty_feeder(gs)
    feeder.counts[cards.Food.FISH] = 3  # single face, three dice
    eng = engine.Engine(gs)
    player = gs.players[0]

    def agent[C: decisions.Choice](
        _eng: engine.Engine, decision: decisions.Decision[C]
    ) -> C:
        assert isinstance(decision, decisions.ResetBirdfeederDecision)
        for choice in decision.choices:
            if isinstance(choice, decisions.ResetBirdfeederChoice):
                return typing.cast(C, choice)
        raise AssertionError("reset choice not offered")

    actions.offer_birdfeeder_reset(eng, agent, player)

    assert feeder.total() == state.BIRDFEEDER_DICE  # rerolled (was 3 dice)


def test_main_gain_food_does_not_auto_reroll_a_declined_single_face() -> None:
    """Regression: the old engine auto-rerolled a single-face feeder at the end
    of a Gain Food action. The reroll is now a player choice — a player who
    declines leaves the leftover single face as-is."""
    gs = _new_game(seed=3)
    player = gs.players[0]
    gs.current_player = player.id
    player.hand = []  # empty hand suppresses the Forest trade-space convert
    # One non-BROWN Forest bird: the action pulls dice but fires no row power.
    forest_bird = next(
        bird
        for bird in gs.bird_deck
        if cards.Habitat.FOREST in bird.habitats
        and bird.color != cards.PowerColor.BROWN
    )
    player.board[cards.Habitat.FOREST].append(state.PlayedBird(bird=forest_bird))
    feeder = _empty_feeder(gs)
    feeder.counts[cards.Food.FISH] = 4  # single face, more than the dice pulled

    def agent[C: decisions.Choice](
        _eng: engine.Engine, decision: decisions.Decision[C]
    ) -> C:
        return _skip_any_reset(decision)

    eng = engine.Engine(gs, agents=[agent, agent])
    actions.do_gain_food(eng, agent)

    # The leftover dice are still all fish and fewer than a fresh roll, so the
    # single-face feeder was not automatically rerolled after the action.
    assert feeder.total() < state.BIRDFEEDER_DICE
    assert feeder.counts[cards.Food.FISH] == feeder.total()


# ---------------------------------------------------------------------------
# Combined-gain subset enumeration (combine_gain_food regime)
#
# ``Birdfeeder.subset_options(n)`` enumerates the multi-food subsets a player
# taking up to ``n`` dice may grab at once: a subset is offered when it reaches
# ``n`` (taken regardless of the leftover) or when it leaves a rerollable feeder
# (``distinct_faces() <= 1`` — a single face or empty), the partial-then-reroll
# branch.


def _subset_size(triple: tuple[state.FoodPool, int, int]) -> int:
    plain, choice_inv, choice_seed = triple
    return plain.total() + choice_inv + choice_seed


def test_subset_options_n1_matches_single_die_menu() -> None:
    """N==1 collapses to the current algorithm: one size-1 subset per
    gain_options entry, preserving the rigid-vs-choice-die distinction."""
    feeder = state.Birdfeeder()
    feeder.counts[cards.Food.FISH] = 1
    feeder.counts[cards.Food.INVERTEBRATE] = 1
    feeder.choice_dice = 1
    options = feeder.subset_options(1)

    assert all(_subset_size(opt) == 1 for opt in options)
    # One subset per single-die option (3 plain... wait: fish, inv + choice as inv/seed).
    assert len(options) == len(feeder.gain_options())
    has_plain_inv = any(
        plain[cards.Food.INVERTEBRATE] == 1 and ci == 0 and cs == 0
        for plain, ci, cs in options
    )
    has_choice_inv = any(ci == 1 for _, ci, _ in options)
    has_choice_seed = any(cs == 1 for _, _, cs in options)
    assert has_plain_inv and has_choice_inv and has_choice_seed


def test_subset_options_enumerated_triples_are_unique() -> None:
    """Count-based enumeration never emits the same (plain, inv, seed) twice."""
    feeder = state.Birdfeeder()
    feeder.counts[cards.Food.FISH] = 2
    feeder.counts[cards.Food.SEED] = 1
    feeder.choice_dice = 1
    signatures = [
        (tuple(plain.counts), ci, cs) for plain, ci, cs in feeder.subset_options(3)
    ]
    assert len(signatures) == len(set(signatures))


def test_subset_options_take_everything_when_n_exceeds_dice() -> None:
    """When n exceeds the dice on offer, the take-everything subset (empty
    leftover) must be present — using reset_available() alone would wrongly drop
    it (False on an empty feeder) and could yield an empty decision."""
    feeder = state.Birdfeeder()
    feeder.counts[cards.Food.FISH] = 2  # only two dice on offer
    options = feeder.subset_options(3)  # but three wanted

    assert options, "must offer at least the take-everything option"
    assert any(
        plain[cards.Food.FISH] == 2 and _subset_size((plain, ci, cs)) == 2
        for plain, ci, cs in options
    )


def test_subset_options_partial_requires_rerollable_leftover() -> None:
    """A subset smaller than n is offered only when the post-removal feeder shows
    at most one distinct face; a partial that leaves two+ faces is excluded
    (the player would take a larger subset of the same feeder instead)."""
    feeder = state.Birdfeeder()
    feeder.counts[cards.Food.FISH] = 2
    feeder.counts[cards.Food.SEED] = 2
    for plain, choice_inv, choice_seed in feeder.subset_options(3):
        if plain.total() + choice_inv + choice_seed < 3:
            leftover_faces = sum(
                1
                for food in (cards.Food.FISH, cards.Food.SEED)
                if feeder.counts[food] - plain[food] > 0
            )
            assert leftover_faces <= 1, plain.counts


def test_subset_options_size_n_offered_regardless_of_leftover() -> None:
    """Every size-n subset is a valid stopping point even when it leaves two
    distinct faces showing (the player is done; no reset needed)."""
    feeder = state.Birdfeeder()
    feeder.counts[cards.Food.FISH] = 2
    feeder.counts[cards.Food.SEED] = 2
    options = feeder.subset_options(2)
    # fish=1, seed=1 leaves both faces showing yet is a legal size-2 stop.
    assert any(
        plain[cards.Food.FISH] == 1 and plain[cards.Food.SEED] == 1
        for plain, _, _ in options
    )
