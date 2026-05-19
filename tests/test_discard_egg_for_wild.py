"""Tests for the DISCARD_EGG_FOR_WILD power (crows, ravens, night-heron)."""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import cards, decisions, engine, state
from wingspan.engine import powers


def _find_bird(birds, name):
    for b in birds:
        if b.name == name:
            return b
    raise AssertionError(f"bird {name!r} not found")


def _empty_engine(seed: int = 0) -> tuple[engine.Engine, list]:
    birds, bonuses, goals = cards.load_all()
    rng = random.Random(seed)
    gs = state.new_game(rng, birds, bonuses, goals)
    return engine.Engine(gs), birds


def test_all_five_birds_parse_to_discard_egg_for_wild():
    birds, _, _ = cards.load_all()
    expected = {
        "American Crow": 1,
        "Black-Crowned Night-Heron": 1,
        "Chihuahuan Raven": 2,
        "Common Raven": 2,
        "Fish Crow": 1,
    }
    seen = {}
    for b in birds:
        if b.name in expected:
            kinds = [e.kind for e in b.power.effects]
            assert (
                cards.EffectKind.DISCARD_EGG_FOR_WILD in kinds
            ), f"{b.name} did not parse to DISCARD_EGG_FOR_WILD: {kinds}"
            eff = next(
                e for e in b.power.effects if e.kind == cards.EffectKind.DISCARD_EGG_FOR_WILD
            )
            seen[b.name] = eff.amount
    assert seen == expected


def test_discard_egg_for_wild_decrements_egg_and_grants_food():
    eng, birds = _empty_engine(seed=1)
    bird = _find_bird(birds, "Common Raven")  # amount=2

    p = eng.state.me()
    raven = state.PlayedBird(bird=bird)
    other = _find_bird(birds, "American Crow")
    sibling = state.PlayedBird(bird=other, eggs=1)
    p.board[cards.Habitat.FOREST].extend([sibling, raven])

    for f in p.food:
        p.food[f] = 0
    for f in eng.state.food_supply:
        eng.state.food_supply[f] = 5

    script = iter(
        [
            (
                decisions.PlayBirdPickEggToPayDecision,
                lambda d: next(
                    c
                    for c in d.choices
                    if getattr(c, "habitat", None) == cards.Habitat.FOREST
                    and getattr(c, "slot", None) == 0
                ),
            ),
            (
                decisions.BirdPowerPickFoodDecision,
                lambda d: next(
                    c
                    for c in d.choices
                    if isinstance(c, decisions.FoodChoice) and c.food == cards.Food.SEED
                ),
            ),
            (
                decisions.BirdPowerPickFoodDecision,
                lambda d: next(
                    c
                    for c in d.choices
                    if isinstance(c, decisions.FoodChoice) and c.food == cards.Food.FRUIT
                ),
            ),
        ]
    )

    def agent(_engine, decision: decisions.Decision) -> decisions.Choice:
        want_cls, picker = next(script)
        assert isinstance(
            decision, want_cls
        ), f"unexpected decision type: {type(decision).__name__}"
        return picker(decision)

    powers.dispatch_power(eng, agent, p, raven, cards.Habitat.FOREST, trigger="activate")

    assert sibling.eggs == 0
    assert p.food[cards.Food.SEED] == 1
    assert p.food[cards.Food.FRUIT] == 1
    assert eng.state.food_supply[cards.Food.SEED] == 4
    assert eng.state.food_supply[cards.Food.FRUIT] == 4


def test_discard_egg_for_wild_skips_when_no_other_bird_has_an_egg():
    eng, birds = _empty_engine(seed=2)
    bird = _find_bird(birds, "American Crow")  # amount=1
    p = eng.state.me()

    crow = state.PlayedBird(bird=bird, eggs=3)  # even its own eggs should not count
    p.board[cards.Habitat.GRASSLAND].append(crow)

    before_food = p.food.as_dict()
    before_supply = eng.state.food_supply.as_dict()

    def agent(_engine, _decision: decisions.Decision) -> decisions.Choice:
        raise AssertionError("agent should not be asked anything when power is a no-op")

    powers.dispatch_power(eng, agent, p, crow, cards.Habitat.GRASSLAND, trigger="activate")

    assert crow.eggs == 3  # self eggs untouched
    assert p.food.as_dict() == before_food
    assert eng.state.food_supply.as_dict() == before_supply


def test_discard_egg_for_wild_can_be_skipped():
    eng, birds = _empty_engine(seed=3)
    bird = _find_bird(birds, "Fish Crow")  # amount=1
    p = eng.state.me()

    fishcrow = state.PlayedBird(bird=bird)
    sibling = state.PlayedBird(bird=_find_bird(birds, "Chihuahuan Raven"), eggs=2)
    p.board[cards.Habitat.WETLAND].extend([sibling, fishcrow])

    before_food = p.food.as_dict()
    before_supply = eng.state.food_supply.as_dict()

    def agent(_engine, decision: decisions.Decision) -> decisions.Choice:
        return next(c for c in decision.choices if c.label == "skip")

    powers.dispatch_power(eng, agent, p, fishcrow, cards.Habitat.WETLAND, trigger="activate")

    assert sibling.eggs == 2  # untouched
    assert p.food.as_dict() == before_food
    assert eng.state.food_supply.as_dict() == before_supply
