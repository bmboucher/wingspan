"""Tests for the ``ON_OTHER_LAY_EGGS_LAY_EGG`` PINK reactive power.

When another player takes the *lay eggs* action, a pink bird owned by an
opposing player lays 1 egg on one of that owner's own birds whose nest
type matches the printed filter (bowl / cavity / ground / platform).

Implemented in core by 5 birds:

- American Avocet (ground nest filter)
- Barrow's Goldeneye (cavity)
- Bronzed Cowbird (bowl)
- Brown-Headed Cowbird (bowl)
- Yellow-Billed Cuckoo (bowl)
"""
from __future__ import annotations

import sys
import os

# Make sure the in-tree package wins over any other installed copy. Other
# parallel worktrees may have pip-installed a different sibling — without
# this guard pytest can pick up *their* module.
_SRC = os.path.join(os.path.dirname(__file__), "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import random

import pytest

from wingspan.actions import MainAction
from wingspan.cards import EffectKind, Habitat, NestType, load_all
from wingspan.game import Engine
from wingspan.state import PlayedBird, new_game


TARGET_BIRDS = {
    "American Avocet": NestType.GROUND,
    "Barrow's Goldeneye": NestType.CAVITY,
    "Bronzed Cowbird": NestType.BOWL,
    "Brown-Headed Cowbird": NestType.BOWL,
    "Yellow-Billed Cuckoo": NestType.BOWL,
}


def test_all_five_pink_lay_eggs_birds_parsed():
    birds, _, _ = load_all()
    found = {b.name: b for b in birds if b.name in TARGET_BIRDS}
    missing = set(TARGET_BIRDS) - set(found)
    assert not missing, f"birds not present in data: {missing}"
    for name, expected_nest in TARGET_BIRDS.items():
        bird = found[name]
        kinds = [e.kind for e in bird.power.effects]
        assert EffectKind.ON_OTHER_LAY_EGGS_LAY_EGG in kinds, (
            f"{name} parsed as {kinds}; raw_power_text={bird.raw_power_text!r}"
        )
        assert EffectKind.UNIMPLEMENTED not in kinds, name
        # extra carries the nest filter
        eff = next(e for e in bird.power.effects
                   if e.kind == EffectKind.ON_OTHER_LAY_EGGS_LAY_EGG)
        assert eff.extra == (expected_nest,), (
            f"{name}: extra={eff.extra!r}, expected ({expected_nest!r},)"
        )
        assert eff.amount == 1


def _find_bird(birds, name: str):
    return next(b for b in birds if b.name == name)


def _find_carrier_with_nest(birds, nest: NestType, *, exclude=()):
    """Return any non-power-text bird with the given nest and a non-zero
    egg limit. Used to populate opponent boards as egg-laying targets."""
    for b in birds:
        if b.name in exclude:
            continue
        if b.nest != nest:
            continue
        if b.egg_limit <= 0:
            continue
        if any(e.kind != EffectKind.UNIMPLEMENTED for e in b.power.effects):
            # avoid carriers that would themselves react and complicate the test
            if b.power.color != b.power.color.NONE:
                # keep searching
                continue
        return b
    # fallback: any bird with the right nest type
    for b in birds:
        if b.nest == nest and b.egg_limit > 0:
            return b
    raise LookupError(f"no bird with nest {nest}")


def test_pink_lay_eggs_reaction_lays_an_egg_on_bowl_nest():
    birds, bonuses, goals = load_all()
    rng = random.Random(0)
    state = new_game(rng, birds, bonuses, goals)
    eng = Engine(state)

    # Player 1 (opponent of actor) holds the pink reactive bird (Bronzed
    # Cowbird, bowl-nest filter) plus a bowl-nest bird that can receive
    # the laid egg.
    cowbird = _find_bird(birds, "Bronzed Cowbird")
    p1 = state.players[1]
    p1.board[Habitat.GRASSLAND] = [PlayedBird(bird=cowbird)]

    # Find a bowl-nest carrier with room for an egg.
    bowl_carrier = _find_carrier_with_nest(birds, NestType.BOWL,
                                           exclude={"Bronzed Cowbird"})
    target_pb = PlayedBird(bird=bowl_carrier)
    assert target_pb.eggs == 0
    p1.board[Habitat.FOREST] = [target_pb]

    # Player 0 (actor) needs to be able to take the LAY_EGGS action.
    # They need a bird in grassland with eggs<limit so they have somewhere
    # to lay the resulting eggs themselves.
    actor_bird = _find_carrier_with_nest(birds, NestType.GROUND,
                                          exclude={"American Avocet"})
    state.players[0].board[Habitat.GRASSLAND] = [PlayedBird(bird=actor_bird)]
    state.current_player = 0

    # First-choice agent picks deterministically.
    def first_choice(_engine, decision):
        return decision.choices[0]

    eng._agents = (first_choice, first_choice)

    eng._do_lay_eggs(first_choice, Habitat.GRASSLAND)

    # The Bronzed Cowbird's reactive power should have laid 1 egg on the
    # bowl-nest carrier owned by the same player (p1).
    assert target_pb.eggs == 1, (
        f"expected bowl-carrier to gain 1 egg, got {target_pb.eggs}; "
        f"log:\n" + "\n".join(state.log[-20:])
    )

    # Once-per-round guard: the cowbird's pink_fired_round equals the
    # current round index.
    fired_pb = p1.board[Habitat.GRASSLAND][0]
    assert fired_pb.pink_fired_round == state.round_idx


def test_pink_lay_eggs_reaction_respects_nest_filter():
    """A pink reactor with a [bowl] filter should NOT lay on a [cavity]
    bird — the reaction is a no-op if no owner-side bird matches."""
    birds, bonuses, goals = load_all()
    rng = random.Random(1)
    state = new_game(rng, birds, bonuses, goals)
    eng = Engine(state)

    cowbird = _find_bird(birds, "Brown-Headed Cowbird")  # bowl filter
    p1 = state.players[1]
    p1.board[Habitat.GRASSLAND] = [PlayedBird(bird=cowbird)]

    # Only give the owner cavity-nest birds: nothing matches the filter.
    cavity_only = _find_carrier_with_nest(birds, NestType.CAVITY)
    p1.board[Habitat.FOREST] = [PlayedBird(bird=cavity_only)]

    actor_bird = _find_carrier_with_nest(birds, NestType.GROUND)
    state.players[0].board[Habitat.GRASSLAND] = [PlayedBird(bird=actor_bird)]
    state.current_player = 0

    def first_choice(_engine, decision):
        return decision.choices[0]

    eng._agents = (first_choice, first_choice)

    eng._do_lay_eggs(first_choice, Habitat.GRASSLAND)

    # No bowl bird to lay on; no eggs gained on owner side.
    assert p1.total_eggs == 0, (
        f"unexpected eggs on opponent: {p1.total_eggs}; log:\n"
        + "\n".join(state.log[-20:])
    )
    # But the once-per-round guard still trips — the reactor's owner had a
    # chance to fire and we don't want it firing again later this round.
    assert p1.board[Habitat.GRASSLAND][0].pink_fired_round == state.round_idx


def test_pink_lay_eggs_does_not_self_trigger():
    """If the *actor* is the one with the pink reactor, the reactor should
    NOT fire (it only triggers on *another* player's lay-eggs action)."""
    birds, bonuses, goals = load_all()
    rng = random.Random(2)
    state = new_game(rng, birds, bonuses, goals)
    eng = Engine(state)

    avocet = _find_bird(birds, "American Avocet")  # ground filter
    p0 = state.players[0]
    p0.board[Habitat.GRASSLAND] = [PlayedBird(bird=avocet)]

    ground_carrier = _find_carrier_with_nest(birds, NestType.GROUND,
                                              exclude={"American Avocet"})
    target_pb = PlayedBird(bird=ground_carrier)
    p0.board[Habitat.FOREST] = [target_pb]

    state.current_player = 0
    start_eggs = target_pb.eggs

    def first_choice(_engine, decision):
        return decision.choices[0]

    eng._agents = (first_choice, first_choice)
    eng._do_lay_eggs(first_choice, Habitat.GRASSLAND)

    # The avocet belongs to the *actor*, so it should never have fired its
    # reactive power. Any eggs on target_pb came from the actor's own
    # lay-eggs action, not from the reactor.
    assert p0.board[Habitat.GRASSLAND][0].pink_fired_round == -1


def test_pink_lay_eggs_once_per_round():
    """A second lay-eggs action in the same round must NOT re-fire the
    same pink reactor."""
    birds, bonuses, goals = load_all()
    rng = random.Random(3)
    state = new_game(rng, birds, bonuses, goals)
    eng = Engine(state)

    cowbird = _find_bird(birds, "Yellow-Billed Cuckoo")  # bowl filter
    p1 = state.players[1]
    p1.board[Habitat.GRASSLAND] = [PlayedBird(bird=cowbird)]

    bowl_carrier = _find_carrier_with_nest(birds, NestType.BOWL,
                                            exclude={"Yellow-Billed Cuckoo"})
    target_pb = PlayedBird(bird=bowl_carrier)
    p1.board[Habitat.FOREST] = [target_pb]

    state.players[0].board[Habitat.GRASSLAND] = [
        PlayedBird(bird=_find_carrier_with_nest(birds, NestType.GROUND))
    ]
    state.current_player = 0

    def first_choice(_engine, decision):
        return decision.choices[0]

    eng._agents = (first_choice, first_choice)

    eng._do_lay_eggs(first_choice, Habitat.GRASSLAND)
    eggs_after_first = target_pb.eggs

    eng._do_lay_eggs(first_choice, Habitat.GRASSLAND)
    eggs_after_second = target_pb.eggs

    # First call should have laid 1; the second triggers the same lay-eggs
    # trigger but the reactor's pink_fired_round guard blocks re-firing.
    assert eggs_after_first == 1
    assert eggs_after_second == eggs_after_first, (
        f"reactor fired twice in one round: {eggs_after_first} -> "
        f"{eggs_after_second}"
    )
