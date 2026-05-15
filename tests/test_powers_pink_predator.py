"""Tests for the PINK ``ON_OTHER_PREDATOR_SUCCESS_GAIN_DIE`` reactive power
(Black Vulture, Black-Billed Magpie, Turkey Vulture in core).

Power text: "When another player's [predator] succeeds, gain 1 [die] from the
birdfeeder."
"""
from __future__ import annotations

import random

from wingspan.cards import EffectKind, Food, PowerColor, load_all, parse_power
from wingspan.game import Engine
from wingspan.state import PlayedBird, new_game


PINK_PREDATOR_REACTOR_NAMES = ("Black Vulture", "Black-Billed Magpie", "Turkey Vulture")


def test_parse_when_another_predator_succeeds_gain_die():
    power = parse_power(
        PowerColor.PINK,
        "When another player's [predator] succeeds, gain 1 [die] from the birdfeeder.",
    )
    assert len(power.effects) == 1
    eff = power.effects[0]
    assert eff.kind == EffectKind.ON_OTHER_PREDATOR_SUCCESS_GAIN_DIE
    assert eff.amount == 1


def test_three_target_birds_are_implemented():
    birds, _, _ = load_all()
    found = {b.name: b for b in birds if b.name in PINK_PREDATOR_REACTOR_NAMES}
    missing = set(PINK_PREDATOR_REACTOR_NAMES) - set(found)
    assert not missing, f"birds not present in data: {missing}"
    for name, bird in found.items():
        kinds = [e.kind for e in bird.power.effects]
        assert EffectKind.ON_OTHER_PREDATOR_SUCCESS_GAIN_DIE in kinds, (
            f"{name} parsed as {kinds}; raw_power_text={bird.raw_power_text!r}"
        )
        assert EffectKind.UNIMPLEMENTED not in kinds, name
        assert bird.color == PowerColor.PINK


def test_fire_pink_triggers_grants_die_to_other_player():
    """Black Vulture on p1's board reacts to p0's predator-success trigger."""
    birds, bonuses, goals = load_all()
    rng = random.Random(0)
    state = new_game(rng, birds, bonuses, goals)
    eng = Engine(state)

    p0, p1 = state.players

    # Deterministic birdfeeder: 1 fish only, so the only legal choice is fish.
    for f in state.birdfeeder.counts:
        state.birdfeeder.counts[f] = 0
    state.birdfeeder.counts[Food.FISH] = 1

    # Place a Black Vulture on p1's board.
    bv = next(b for b in birds if b.name == "Black Vulture")
    habitat = bv.habitats[0]
    pb = PlayedBird(bird=bv)
    p1.board[habitat].append(pb)

    p1.food[Food.FISH] = 0

    eng._fire_pink_triggers(actor=p0, trigger="predator_success", ctx={})

    assert p1.food[Food.FISH] == 1
    assert state.birdfeeder.counts[Food.FISH] == 0
    assert pb.pink_fired_round == state.round_idx


def test_pink_predator_success_once_per_round():
    birds, bonuses, goals = load_all()
    rng = random.Random(1)
    state = new_game(rng, birds, bonuses, goals)
    eng = Engine(state)

    p0, p1 = state.players

    for f in state.birdfeeder.counts:
        state.birdfeeder.counts[f] = 0
    state.birdfeeder.counts[Food.SEED] = 3

    bv = next(b for b in birds if b.name == "Turkey Vulture")
    pb = PlayedBird(bird=bv)
    p1.board[bv.habitats[0]].append(pb)
    p1.food[Food.SEED] = 0

    eng._fire_pink_triggers(actor=p0, trigger="predator_success", ctx={})
    assert p1.food[Food.SEED] == 1

    eng._fire_pink_triggers(actor=p0, trigger="predator_success", ctx={})
    assert p1.food[Food.SEED] == 1, "pink power fired twice in the same round"

    state.round_idx += 1
    eng._fire_pink_triggers(actor=p0, trigger="predator_success", ctx={})
    assert p1.food[Food.SEED] == 2


def test_pink_predator_success_skips_actor_own_board():
    birds, bonuses, goals = load_all()
    rng = random.Random(2)
    state = new_game(rng, birds, bonuses, goals)
    eng = Engine(state)

    p0, p1 = state.players

    for f in state.birdfeeder.counts:
        state.birdfeeder.counts[f] = 0
    state.birdfeeder.counts[Food.RODENT] = 1

    bv = next(b for b in birds if b.name == "Black-Billed Magpie")
    pb = PlayedBird(bird=bv)
    p0.board[bv.habitats[0]].append(pb)  # on the *actor*'s board
    p0.food[Food.RODENT] = 0

    eng._fire_pink_triggers(actor=p0, trigger="predator_success", ctx={})
    assert p0.food[Food.RODENT] == 0
    assert state.birdfeeder.counts[Food.RODENT] == 1


def test_pink_predator_success_handles_empty_birdfeeder():
    birds, bonuses, goals = load_all()
    rng = random.Random(3)
    state = new_game(rng, birds, bonuses, goals)
    eng = Engine(state)

    p0, p1 = state.players

    for f in state.birdfeeder.counts:
        state.birdfeeder.counts[f] = 0

    bv = next(b for b in birds if b.name == "Black Vulture")
    pb = PlayedBird(bird=bv)
    p1.board[bv.habitats[0]].append(pb)
    before = dict(p1.food)

    eng._fire_pink_triggers(actor=p0, trigger="predator_success", ctx={})
    assert p1.food == before
    assert pb.pink_fired_round == state.round_idx
