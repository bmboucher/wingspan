"""Unit tests for the LAY_EGG_ON_EACH_NEST power.

Covers Ash-Throated Flycatcher (cavity), Bobolink (ground),
Inca Dove (platform), and Say's Phoebe (bowl).
"""
from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan.cards import EffectKind, Habitat, NestType, load_all
from wingspan.game import make_engine
from wingspan.state import PlayedBird


def _find_bird(birds, name):
    for b in birds:
        if b.name == name:
            return b
    raise AssertionError(f"bird {name!r} not found in core set")


def _trigger_bird(name):
    eng, birds, _, _ = make_engine(seed=0)
    trigger = _find_bird(birds, name)
    # find filler birds for the test board
    candidates_by_nest: dict[NestType, list] = {}
    for b in birds:
        if b is trigger:
            continue
        if b.egg_limit <= 0:
            continue
        candidates_by_nest.setdefault(b.nest, []).append(b)

    p = eng.state.players[0]
    eng.state.current_player = 0

    # Need: 2 matching the target nest, 1 with a different nest, 1 already full.
    target_nest: NestType = trigger.power.effects[0].extra[0]
    match_pool = candidates_by_nest.get(target_nest, [])
    assert len(match_pool) >= 3, f"need at least 3 birds with {target_nest} nest"
    a, b_full, c = match_pool[:3]
    # pick a non-matching nest
    other = None
    for nest, pool in candidates_by_nest.items():
        if nest in (target_nest, NestType.STAR):
            continue
        if pool:
            other = pool[0]
            break
    assert other is not None

    p.board[Habitat.FOREST].append(PlayedBird(bird=a))
    p.board[Habitat.GRASSLAND].append(PlayedBird(bird=b_full, eggs=b_full.egg_limit))
    p.board[Habitat.WETLAND].append(PlayedBird(bird=c))
    p.board[Habitat.FOREST].append(PlayedBird(bird=other))

    trigger_pb = PlayedBird(bird=trigger)
    p.board[Habitat.FOREST].append(trigger_pb)

    # ensure parsed effect is what we expect
    eff = trigger.power.effects[0]
    assert eff.kind == EffectKind.LAY_EGG_ON_EACH_NEST
    assert eff.extra == (target_nest,)

    # dispatch power
    rng = random.Random(0)

    def dummy_agent(engine, decision):
        return decision.choices[0]

    eng._dispatch_power(dummy_agent, p, trigger_pb, Habitat.FOREST, "play")

    return p, target_nest, a, b_full, c, other, trigger_pb


def test_say_phoebe_lays_on_each_bowl_nest():
    p, _, a, b_full, c, other, _ = _trigger_bird("Say's Phoebe")
    assert a.nest == NestType.BOWL
    # bowls that had room each get +1 egg
    a_pb = next(pb for pb in p.board[Habitat.FOREST] if pb.bird is a)
    c_pb = next(pb for pb in p.board[Habitat.WETLAND] if pb.bird is c)
    assert a_pb.eggs == 1
    assert c_pb.eggs == 1
    # full one stays at limit, not exceeded
    full_pb = next(pb for pb in p.board[Habitat.GRASSLAND] if pb.bird is b_full)
    assert full_pb.eggs == b_full.egg_limit
    # non-matching nest gets nothing
    other_pb = next(pb for pb in p.board[Habitat.FOREST] if pb.bird is other)
    assert other_pb.eggs == 0


def test_ash_throated_flycatcher_lays_on_each_cavity_nest():
    p, target_nest, a, b_full, c, other, _ = _trigger_bird("Ash-Throated Flycatcher")
    assert target_nest == NestType.CAVITY
    a_pb = next(pb for pb in p.board[Habitat.FOREST] if pb.bird is a)
    c_pb = next(pb for pb in p.board[Habitat.WETLAND] if pb.bird is c)
    assert a_pb.eggs == 1
    assert c_pb.eggs == 1
    full_pb = next(pb for pb in p.board[Habitat.GRASSLAND] if pb.bird is b_full)
    assert full_pb.eggs == b_full.egg_limit


def test_bobolink_lays_on_each_ground_nest():
    p, target_nest, a, _, c, other, _ = _trigger_bird("Bobolink")
    assert target_nest == NestType.GROUND
    a_pb = next(pb for pb in p.board[Habitat.FOREST] if pb.bird is a)
    c_pb = next(pb for pb in p.board[Habitat.WETLAND] if pb.bird is c)
    assert a_pb.eggs == 1
    assert c_pb.eggs == 1
    other_pb = next(pb for pb in p.board[Habitat.FOREST] if pb.bird is other)
    assert other_pb.eggs == 0


def test_inca_dove_lays_on_each_platform_nest():
    p, target_nest, a, _, c, _, _ = _trigger_bird("Inca Dove")
    assert target_nest == NestType.PLATFORM
    a_pb = next(pb for pb in p.board[Habitat.FOREST] if pb.bird is a)
    c_pb = next(pb for pb in p.board[Habitat.WETLAND] if pb.bird is c)
    assert a_pb.eggs == 1
    assert c_pb.eggs == 1


def test_power_coverage_increased():
    birds, _, _ = load_all()
    impl = sum(
        1
        for b in birds
        if b.power.effects
        and not any(e.kind == EffectKind.UNIMPLEMENTED for e in b.power.effects)
    )
    # baseline was 100; this PR adds 4
    assert impl >= 104, f"expected >=104 implemented powers, got {impl}"
