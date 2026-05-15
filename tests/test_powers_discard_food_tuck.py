"""Tests for the DISCARD_FOOD_TUCK_FROM_DECK power (5 birds in core).

The power text is "Discard 1 [food] to tuck 2 [card] from the deck behind this
bird." — the player optionally pays 1 food of a specific type, then 2 cards are
drawn from the bird deck and tucked behind this bird (their faces don't matter;
they count for tucked-card scoring).
"""
from __future__ import annotations

import random

from wingspan.actions import Choice, Decision, DecisionType
from wingspan.cards import (
    Effect, EffectKind, Food, PowerColor, load_all, parse_power,
)
from wingspan.game import Engine
from wingspan.state import PlayedBird, new_game


def test_parse_discard_food_tuck_from_deck():
    power = parse_power(
        PowerColor.BROWN,
        "Discard 1 [seed] to tuck 2 [card] from the deck behind this bird.",
    )
    kinds = [e.kind for e in power.effects]
    assert EffectKind.DISCARD_FOOD_TUCK_FROM_DECK in kinds
    eff = next(e for e in power.effects if e.kind == EffectKind.DISCARD_FOOD_TUCK_FROM_DECK)
    assert eff.food == Food.SEED
    assert eff.amount == 2


def test_all_5_target_birds_are_implemented():
    birds, _, _ = load_all()
    target_names = {
        "American White Pelican",
        "Black-Bellied Whistling-Duck",
        "Canada Goose",
        "Double-Crested Cormorant",
        "Sandhill Crane",
    }
    found = {b.name: b for b in birds if b.name in target_names}
    missing = target_names - set(found)
    assert not missing, f"birds not present in data: {missing}"
    for name, bird in found.items():
        kinds = [e.kind for e in bird.power.effects]
        assert EffectKind.DISCARD_FOOD_TUCK_FROM_DECK in kinds, (
            f"{name} parsed as {kinds}; raw_power_text={bird.raw_power_text!r}"
        )
        assert EffectKind.UNIMPLEMENTED not in kinds, name


def test_use_power_tucks_two_cards_and_spends_one_food():
    birds, bonuses, goals = load_all()
    rng = random.Random(0)
    state = new_game(rng, birds, bonuses, goals)
    eng = Engine(state)

    p = state.players[0]
    state.current_player = p.id

    canada_goose = next(b for b in birds if b.name == "Canada Goose")
    pb = PlayedBird(bird=canada_goose)

    # Setup: exactly 1 seed, baseline tucked = 0.
    for f in p.food:
        p.food[f] = 0
    p.food[Food.SEED] = 1
    pb.tucked_cards = 0

    # Snapshot deck/discard sizes so we can verify 2 cards moved to bird_discard.
    deck_before = len(state.bird_deck) + len(state.bird_discard)
    discard_before = len(state.bird_discard)

    eff = next(e for e in canada_goose.power.effects
               if e.kind == EffectKind.DISCARD_FOOD_TUCK_FROM_DECK)

    def use_power_agent(_engine: Engine, decision: Decision) -> Choice:
        assert decision.type == DecisionType.SKIP_OPTIONAL
        # First choice is "use power"
        return decision.choices[0]

    eng._apply_effect(use_power_agent, p, pb, canada_goose.habitats[0], eff, "activate")

    assert p.food[Food.SEED] == 0, "seed should be decremented by 1"
    assert pb.tucked_cards == 2, "two cards should be tucked behind the bird"
    assert len(state.bird_discard) == discard_before + 2, "two cards should have moved to bird_discard"
    # Total cards (deck + discard) is conserved; cards moved from deck to discard.
    assert len(state.bird_deck) + len(state.bird_discard) == deck_before


def test_skip_leaves_food_and_tucked_unchanged():
    birds, bonuses, goals = load_all()
    rng = random.Random(1)
    state = new_game(rng, birds, bonuses, goals)
    eng = Engine(state)

    p = state.players[0]
    state.current_player = p.id

    canada_goose = next(b for b in birds if b.name == "Canada Goose")
    pb = PlayedBird(bird=canada_goose)

    for f in p.food:
        p.food[f] = 0
    p.food[Food.SEED] = 1
    pb.tucked_cards = 0
    discard_before = len(state.bird_discard)

    eff = next(e for e in canada_goose.power.effects
               if e.kind == EffectKind.DISCARD_FOOD_TUCK_FROM_DECK)

    def skip_agent(_engine: Engine, decision: Decision) -> Choice:
        assert decision.type == DecisionType.SKIP_OPTIONAL
        # Second choice is "skip"
        return decision.choices[1]

    eng._apply_effect(skip_agent, p, pb, canada_goose.habitats[0], eff, "activate")

    assert p.food[Food.SEED] == 1, "seed should be unchanged when skipping"
    assert pb.tucked_cards == 0, "no cards should be tucked when skipping"
    assert len(state.bird_discard) == discard_before, "no cards should move to discard when skipping"


def test_no_food_skips_silently_without_decision():
    """If the player has 0 of the required food, the power should be a no-op
    and the engine should NOT raise a decision prompt."""
    birds, bonuses, goals = load_all()
    rng = random.Random(2)
    state = new_game(rng, birds, bonuses, goals)
    eng = Engine(state)

    p = state.players[0]
    state.current_player = p.id

    canada_goose = next(b for b in birds if b.name == "Canada Goose")
    pb = PlayedBird(bird=canada_goose)

    for f in p.food:
        p.food[f] = 0
    pb.tucked_cards = 0
    discard_before = len(state.bird_discard)

    eff = next(e for e in canada_goose.power.effects
               if e.kind == EffectKind.DISCARD_FOOD_TUCK_FROM_DECK)

    def boom_agent(_engine: Engine, decision: Decision) -> Choice:
        raise AssertionError(f"agent should not be asked; got {decision.type}")

    eng._apply_effect(boom_agent, p, pb, canada_goose.habitats[0], eff, "activate")

    assert pb.tucked_cards == 0
    assert len(state.bird_discard) == discard_before
