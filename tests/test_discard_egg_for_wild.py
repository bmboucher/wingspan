"""Tests for the DISCARD_EGG_FOR_WILD power (crows, ravens, night-heron)."""
from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import cards
from wingspan.actions import Choice, Decision, DecisionType
from wingspan.cards import EffectKind, Food, Habitat, PowerColor
from wingspan.game import Engine
from wingspan.state import PlayedBird, new_game


def _find_bird(birds, name):
    for b in birds:
        if b.name == name:
            return b
    raise AssertionError(f"bird {name!r} not found")


def _empty_engine(seed: int = 0) -> tuple[Engine, list]:
    birds, bonuses, goals = cards.load_all()
    rng = random.Random(seed)
    state = new_game(rng, birds, bonuses, goals)
    return Engine(state), birds


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
            assert EffectKind.DISCARD_EGG_FOR_WILD in kinds, (
                f"{b.name} did not parse to DISCARD_EGG_FOR_WILD: {kinds}"
            )
            eff = next(e for e in b.power.effects if e.kind == EffectKind.DISCARD_EGG_FOR_WILD)
            seen[b.name] = eff.amount
    assert seen == expected


def test_discard_egg_for_wild_decrements_egg_and_grants_food():
    eng, birds = _empty_engine(seed=1)
    bird = _find_bird(birds, "Common Raven")  # amount=2

    p = eng.state.me()
    raven = PlayedBird(bird=bird)
    other = _find_bird(birds, "American Crow")
    sibling = PlayedBird(bird=other, eggs=1)
    p.board[Habitat.FOREST].extend([sibling, raven])

    for f in p.food: p.food[f] = 0
    for f in eng.state.food_supply: eng.state.food_supply[f] = 5

    script = iter([
        ("PLAY_BIRD_PICK_EGG_TO_PAY", lambda d: next(c for c in d.choices if c.payload == (Habitat.FOREST, 0))),
        ("BIRD_POWER_PICK_FOOD", lambda d: next(c for c in d.choices if c.payload == Food.SEED)),
        ("BIRD_POWER_PICK_FOOD", lambda d: next(c for c in d.choices if c.payload == Food.FRUIT)),
    ])

    def agent(_engine, decision: Decision) -> Choice:
        want_type, picker = next(script)
        assert decision.type.name == want_type, f"unexpected decision type: {decision.type}"
        return picker(decision)

    eng._dispatch_power(agent, p, raven, Habitat.FOREST, trigger="activate")

    assert sibling.eggs == 0
    assert p.food[Food.SEED] == 1
    assert p.food[Food.FRUIT] == 1
    assert eng.state.food_supply[Food.SEED] == 4
    assert eng.state.food_supply[Food.FRUIT] == 4


def test_discard_egg_for_wild_skips_when_no_other_bird_has_an_egg():
    eng, birds = _empty_engine(seed=2)
    bird = _find_bird(birds, "American Crow")  # amount=1
    p = eng.state.me()

    crow = PlayedBird(bird=bird, eggs=3)  # even its own eggs should not count
    p.board[Habitat.GRASSLAND].append(crow)

    before_food = dict(p.food)
    before_supply = dict(eng.state.food_supply)

    def agent(_engine, _decision: Decision) -> Choice:
        raise AssertionError("agent should not be asked anything when power is a no-op")

    eng._dispatch_power(agent, p, crow, Habitat.GRASSLAND, trigger="activate")

    assert crow.eggs == 3  # self eggs untouched
    assert p.food == before_food
    assert eng.state.food_supply == before_supply


def test_discard_egg_for_wild_can_be_skipped():
    eng, birds = _empty_engine(seed=3)
    bird = _find_bird(birds, "Fish Crow")  # amount=1
    p = eng.state.me()

    fishcrow = PlayedBird(bird=bird)
    sibling = PlayedBird(bird=_find_bird(birds, "Chihuahuan Raven"), eggs=2)
    p.board[Habitat.WETLAND].extend([sibling, fishcrow])

    before_food = dict(p.food)
    before_supply = dict(eng.state.food_supply)

    def agent(_engine, decision: Decision) -> Choice:
        return next(c for c in decision.choices if c.label == "skip")

    eng._dispatch_power(agent, p, fishcrow, Habitat.WETLAND, trigger="activate")

    assert sibling.eggs == 2  # untouched
    assert p.food == before_food
    assert eng.state.food_supply == before_supply
