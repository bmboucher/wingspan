"""Tests for the ALL_PLAYERS_LAY_EGG_ON_NEST bird power (3 birds in core).

Birds:
- Lazuli Bunting       -- bowl
- Pileated Woodpecker  -- cavity
- Western Meadowlark   -- ground

Power text: "All players lay 1 [egg] on any 1 [<nest>] bird. You may lay 1
[egg] on 1 additional [<nest>] bird."
"""
from __future__ import annotations

import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan.actions import Choice, Decision, DecisionType  # noqa: E402
from wingspan.cards import (  # noqa: E402
    EffectKind, NestType, PowerColor, load_all, parse_power,
)
from wingspan.game import Engine  # noqa: E402
from wingspan.state import PlayedBird, new_game  # noqa: E402


TARGET_BIRDS = {
    "Lazuli Bunting": NestType.BOWL,
    "Pileated Woodpecker": NestType.CAVITY,
    "Western Meadowlark": NestType.GROUND,
}


def _by_name(birds, name):
    for b in birds:
        if b.name == name:
            return b
    raise KeyError(name)


def test_parse_all_players_lay_egg_on_nest():
    """Each printed sentence variant should parse to the new effect kind."""
    for nest_word, expected in [("bowl", NestType.BOWL),
                                ("cavity", NestType.CAVITY),
                                ("ground", NestType.GROUND)]:
        text = (
            f"All players lay 1 [egg] on any 1 [{nest_word}] bird. "
            f"You may lay 1 [egg] on 1 additional [{nest_word}] bird."
        )
        power = parse_power(PowerColor.WHITE, text)
        kinds = [e.kind for e in power.effects]
        assert EffectKind.ALL_PLAYERS_LAY_EGG_ON_NEST in kinds, (
            f"failed to parse for nest={nest_word}: {kinds}")
        eff = next(e for e in power.effects if e.kind == EffectKind.ALL_PLAYERS_LAY_EGG_ON_NEST)
        assert eff.nest == expected
        assert eff.amount == 1  # optional second sentence present -> 1 extra for self
        assert EffectKind.UNIMPLEMENTED not in kinds

    # Variant without the optional second sentence: amount should be 0.
    text = "All players lay 1 [egg] on any 1 [bowl] bird."
    power = parse_power(PowerColor.WHITE, text)
    eff = next(e for e in power.effects if e.kind == EffectKind.ALL_PLAYERS_LAY_EGG_ON_NEST)
    assert eff.nest == NestType.BOWL
    assert eff.amount == 0


def test_all_three_target_birds_implemented():
    birds, _, _ = load_all()
    for name, expected_nest in TARGET_BIRDS.items():
        b = _by_name(birds, name)
        kinds = [e.kind for e in b.power.effects]
        assert EffectKind.UNIMPLEMENTED not in kinds, (
            f"{name} still UNIMPLEMENTED; raw={b.raw_power_text!r}")
        assert EffectKind.ALL_PLAYERS_LAY_EGG_ON_NEST in kinds, (
            f"{name} parsed as {kinds}; raw={b.raw_power_text!r}")
        eff = next(e for e in b.power.effects if e.kind == EffectKind.ALL_PLAYERS_LAY_EGG_ON_NEST)
        assert eff.nest == expected_nest
        assert eff.amount == 1


@pytest.mark.parametrize("bird_name,nest", list(TARGET_BIRDS.items()))
def test_power_every_player_lays_one_egg_on_matching_nest(bird_name, nest):
    """Give each player a matching-nest bird with room; expect each gets +1 egg."""
    birds, bonuses, goals = load_all()
    rng = random.Random(0)
    state = new_game(rng, birds, bonuses, goals)

    power_bird = _by_name(birds, bird_name)
    # Pick any bird with the right nest type and an egg_limit >= 2 so it has room.
    target = next(
        b for b in birds
        if b.nest == nest and b.egg_limit >= 2 and b.name != bird_name
    )

    # Each player gets one target-nest bird (empty) plus a non-matching bird so
    # we can confirm the egg lands on the matching bird, not the other.
    decoy = next(
        b for b in birds
        if b.nest != nest and b.nest != NestType.STAR and b.nest != NestType.NONE
        and b.egg_limit >= 1 and b.name != bird_name
    )
    pbs = []
    for q in state.players:
        habitat = target.habitats[0]
        decoy_habitat = decoy.habitats[0]
        pb_target = PlayedBird(bird=target)
        pb_decoy = PlayedBird(bird=decoy)
        # Place decoy in a different column slot if same habitat to avoid clobber.
        q.board[habitat].append(pb_target)
        if decoy_habitat == habitat:
            q.board[habitat].append(pb_decoy)
        else:
            q.board[decoy_habitat].append(pb_decoy)
        pbs.append((pb_target, pb_decoy))

    # The power bird is held off-board so its own (matching) nest doesn't
    # confuse choice resolution — we trigger the effect directly.
    pb_power = PlayedBird(bird=power_bird)

    # Scripted agent: prefer the `target` bird when present, otherwise pick
    # the first non-skip choice.
    target_label_substr = f"{target.name}@"

    def script_agent(_engine, decision: Decision) -> Choice:
        for c in decision.choices:
            if c.payload is not None and target_label_substr in c.label:
                return c
        for c in decision.choices:
            if c.payload is not None:
                return c
        return decision.choices[0]

    eng = Engine(state, agents=[script_agent, script_agent])
    state.current_player = 0

    eff = next(e for e in power_bird.power.effects if e.kind == EffectKind.ALL_PLAYERS_LAY_EGG_ON_NEST)
    eng._apply_effect(script_agent, state.players[0], pb_power,
                      power_bird.habitats[0], eff, trigger="play")

    # Each player's matching-nest bird should have an egg.
    for (pb_target, pb_decoy) in pbs:
        assert pb_target.eggs >= 1, (
            f"every player should lay >=1 egg on matching-nest bird; "
            f"target {pb_target.bird.name} has {pb_target.eggs}")
        assert pb_decoy.eggs == 0, (
            f"non-matching-nest bird {pb_decoy.bird.name} must not receive eggs")
    # The active player (P0) should have laid at least 2 eggs total
    # (1 mandatory + 1 optional bonus); the opponent should have laid exactly 1.
    p0_target, _ = pbs[0]
    p1_target, _ = pbs[1]
    assert p0_target.eggs >= 2, f"active player should lay 2 eggs (1+1 bonus); got {p0_target.eggs}"
    assert p1_target.eggs == 1, f"opponent should lay exactly 1 egg; got {p1_target.eggs}"


def test_power_skipped_when_no_matching_nest_bird():
    """If neither player has a matching-nest bird, the power is a silent no-op."""
    birds, bonuses, goals = load_all()
    rng = random.Random(1)
    state = new_game(rng, birds, bonuses, goals)

    power_bird = _by_name(birds, "Lazuli Bunting")  # bowl
    # Put only non-bowl birds on each player's board. Do NOT place the power
    # bird itself (which has a bowl nest) — that would make it eligible.
    non_bowl = next(b for b in birds if b.nest not in (NestType.BOWL, NestType.STAR, NestType.NONE)
                    and b.name != power_bird.name)
    for q in state.players:
        q.board[non_bowl.habitats[0]].append(PlayedBird(bird=non_bowl))

    pb_power = PlayedBird(bird=power_bird)  # off-board; we invoke directly

    def script_agent(_engine, decision: Decision) -> Choice:
        return decision.choices[0]

    eng = Engine(state, agents=[script_agent, script_agent])
    state.current_player = 0

    eff = next(e for e in power_bird.power.effects if e.kind == EffectKind.ALL_PLAYERS_LAY_EGG_ON_NEST)
    total_before = sum(pb.eggs for q in state.players for r in q.board.values() for pb in r)
    eng._apply_effect(script_agent, state.players[0], pb_power,
                      power_bird.habitats[0], eff, trigger="play")
    total_after = sum(pb.eggs for q in state.players for r in q.board.values() for pb in r)
    assert total_after == total_before, "no eggs should be laid when no eligible nests exist"


def test_egg_limit_respected():
    """A matching-nest bird already at egg_limit must not receive an egg."""
    birds, bonuses, goals = load_all()
    rng = random.Random(2)
    state = new_game(rng, birds, bonuses, goals)

    power_bird = _by_name(birds, "Pileated Woodpecker")  # cavity
    cavity = next(b for b in birds if b.nest == NestType.CAVITY and b.egg_limit >= 1
                  and b.name != power_bird.name)

    # P0 has the cavity bird full; P1 has a fresh cavity bird with room.
    state.players[0].board[cavity.habitats[0]].append(
        PlayedBird(bird=cavity, eggs=cavity.egg_limit)
    )
    state.players[1].board[cavity.habitats[0]].append(
        PlayedBird(bird=cavity, eggs=0)
    )
    pb_power = PlayedBird(bird=power_bird)
    state.players[0].board[power_bird.habitats[0]].append(pb_power)

    def script_agent(_engine, decision: Decision) -> Choice:
        for c in decision.choices:
            if c.payload is not None:
                return c
        return decision.choices[0]

    eng = Engine(state, agents=[script_agent, script_agent])
    state.current_player = 0
    eff = next(e for e in power_bird.power.effects if e.kind == EffectKind.ALL_PLAYERS_LAY_EGG_ON_NEST)
    eng._apply_effect(script_agent, state.players[0], pb_power,
                      power_bird.habitats[0], eff, trigger="play")

    # P0's only cavity bird was full -- no change.
    p0_cavity = state.players[0].board[cavity.habitats[0]][0]
    assert p0_cavity.eggs == cavity.egg_limit
    # P1's cavity bird received exactly 1 egg.
    p1_cavity = state.players[1].board[cavity.habitats[0]][0]
    assert p1_cavity.eggs == 1
