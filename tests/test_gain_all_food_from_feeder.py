"""Unit tests for the GAIN_ALL_FOOD_FROM_FEEDER bird power.

Covers the white "When played" powers on Bald Eagle ([fish]) and Northern
Flicker ([invertebrate]): "Gain all [food] that are in the birdfeeder."
"""
from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan.cards import EffectKind, Food, PowerColor, parse_power
from wingspan.game import Engine
from wingspan.state import PlayedBird, new_game


def _engine_with_bird(power_text: str) -> tuple[Engine, PlayedBird]:
    from wingspan import cards
    birds, bonuses, goals = cards.load_all()
    rng = random.Random(0)
    state = new_game(rng, birds, bonuses, goals)
    eng = Engine(state)
    template = next(b for b in birds if b.color == PowerColor.WHITE)
    bird = type(template)(**{
        **{f.name: getattr(template, f.name) for f in template.__dataclass_fields__.values()},
        "raw_power_text": power_text,
        "power": parse_power(PowerColor.WHITE, power_text),
    })
    return eng, PlayedBird(bird=bird)


def test_parser_recognises_gain_all_food_from_feeder():
    p = parse_power(PowerColor.WHITE, "Gain all [fish] that are in the birdfeeder.")
    assert len(p.effects) == 1
    eff = p.effects[0]
    assert eff.kind == EffectKind.GAIN_ALL_FOOD_FROM_FEEDER
    assert eff.food == Food.FISH


def test_transfers_all_matching_food_from_feeder():
    eng, pb = _engine_with_bird("Gain all [fish] that are in the birdfeeder.")
    # Stage a known birdfeeder state: 3 fish, 2 seed, 1 invertebrate.
    for f in eng.state.birdfeeder.counts:
        eng.state.birdfeeder.counts[f] = 0
    eng.state.birdfeeder.counts[Food.FISH] = 3
    eng.state.birdfeeder.counts[Food.SEED] = 2
    eng.state.birdfeeder.counts[Food.INVERTEBRATE] = 1
    player = eng.state.players[0]
    eng.state.current_player = 0
    fish_before = player.food.get(Food.FISH, 0)

    eng._dispatch_power(lambda *_: None, player, pb, pb.bird.habitats[0], "play")

    assert eng.state.birdfeeder.counts[Food.FISH] == 0
    assert eng.state.birdfeeder.counts[Food.SEED] == 2
    assert eng.state.birdfeeder.counts[Food.INVERTEBRATE] == 1
    assert player.food[Food.FISH] == fish_before + 3


def test_handles_empty_feeder_of_target_food():
    eng, pb = _engine_with_bird("Gain all [invertebrate] that are in the birdfeeder.")
    for f in eng.state.birdfeeder.counts:
        eng.state.birdfeeder.counts[f] = 0
    eng.state.birdfeeder.counts[Food.SEED] = 4  # nothing of target type
    player = eng.state.players[0]
    eng.state.current_player = 0
    inv_before = player.food.get(Food.INVERTEBRATE, 0)

    eng._dispatch_power(lambda *_: None, player, pb, pb.bird.habitats[0], "play")

    assert player.food[Food.INVERTEBRATE] == inv_before
    assert eng.state.birdfeeder.counts[Food.SEED] == 4


def test_bald_eagle_and_northern_flicker_parse_correctly():
    from wingspan import cards
    birds, _, _ = cards.load_all()
    by_name = {b.name: b for b in birds}
    eagle = by_name["Bald Eagle"]
    flicker = by_name["Northern Flicker"]
    eagle_kinds = [e.kind for e in eagle.power.effects]
    flicker_kinds = [e.kind for e in flicker.power.effects]
    assert EffectKind.GAIN_ALL_FOOD_FROM_FEEDER in eagle_kinds
    assert EffectKind.GAIN_ALL_FOOD_FROM_FEEDER in flicker_kinds
    eagle_eff = next(e for e in eagle.power.effects if e.kind == EffectKind.GAIN_ALL_FOOD_FROM_FEEDER)
    flicker_eff = next(e for e in flicker.power.effects if e.kind == EffectKind.GAIN_ALL_FOOD_FROM_FEEDER)
    assert eagle_eff.food == Food.FISH
    assert flicker_eff.food == Food.INVERTEBRATE
