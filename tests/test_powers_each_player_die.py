"""Tests for the EACH_PLAYER_GAINS_DIE_CHOOSE_ORDER power.

Two birds carry this in the core set:
    Anna's Hummingbird
    Ruby-Throated Hummingbird

Power text: "Each player gains 1 [die] from the birdfeeder, starting with
the player of your choice."
"""
from __future__ import annotations

import random

from wingspan.actions import Choice, Decision, DecisionType
from wingspan.cards import (
    Effect, EffectKind, Food, PowerColor, load_all, parse_power,
)
from wingspan.game import Engine
from wingspan.state import PlayedBird, new_game


def test_parse_each_player_gains_die():
    power = parse_power(
        PowerColor.WHITE,
        "Each player gains 1 [die] from the birdfeeder, starting with the player of your choice.",
    )
    assert len(power.effects) == 1
    eff = power.effects[0]
    assert eff.kind == EffectKind.EACH_PLAYER_GAINS_DIE_CHOOSE_ORDER
    assert eff.amount == 1


def test_both_hummingbirds_are_implemented():
    birds, _, _ = load_all()
    target_names = {"Anna's Hummingbird", "Ruby-Throated Hummingbird"}
    found = {b.name: b for b in birds if b.name in target_names}
    missing = target_names - set(found)
    assert not missing, f"birds not present in data: {missing}"
    for name, bird in found.items():
        kinds = [e.kind for e in bird.power.effects]
        assert EffectKind.EACH_PLAYER_GAINS_DIE_CHOOSE_ORDER in kinds, (
            f"{name} parsed as {kinds}; raw_power_text={bird.raw_power_text!r}"
        )
        assert EffectKind.UNIMPLEMENTED not in kinds, name


def _setup_state(seed: int = 0):
    birds, bonuses, goals = load_all()
    rng = random.Random(seed)
    return new_game(rng, birds, bonuses, goals)


def test_each_player_gains_die_credits_both_players_in_chosen_order():
    """Active player picks P1 to start, so the dice are credited P1 then P0."""
    state = _setup_state(seed=0)

    # Make the feeder deterministic: 1 seed, 1 fruit, everything else 0.
    for f in state.birdfeeder.counts:
        state.birdfeeder.counts[f] = 0
    state.birdfeeder.counts[Food.SEED] = 1
    state.birdfeeder.counts[Food.FRUIT] = 1

    p0, p1 = state.players
    state.current_player = p0.id
    food_before_p0 = dict(p0.food)
    food_before_p1 = dict(p1.food)

    carrier = next(b for b in load_all()[0] if b.name == "Anna's Hummingbird")
    pb = PlayedBird(bird=carrier)
    eff = Effect(kind=EffectKind.EACH_PLAYER_GAINS_DIE_CHOOSE_ORDER, amount=1)

    # Scripted agents:
    #   - active player (p0) picks the starting player as P1
    #   - each player picks the first die available
    def agent_p0(_engine: Engine, decision: Decision) -> Choice:
        if decision.type == DecisionType.BIRD_POWER_PICK_STARTING_PLAYER:
            assert decision.player_id == p0.id
            for c in decision.choices:
                if c.payload == p1.id:
                    return c
            raise AssertionError("p1 not in choices")
        if decision.type == DecisionType.GAIN_FOOD_PICK_DIE:
            assert decision.player_id == p0.id
            return decision.choices[0]
        raise AssertionError(f"unexpected decision for p0: {decision.type}")

    def agent_p1(_engine: Engine, decision: Decision) -> Choice:
        assert decision.type == DecisionType.GAIN_FOOD_PICK_DIE
        assert decision.player_id == p1.id
        return decision.choices[0]

    eng = Engine(state, agents=[agent_p0, agent_p1])
    eng._apply_effect(agent_p0, p0, pb, carrier.habitats[0], eff, "play")

    # Total food gained: each player got exactly 1 die.
    assert sum(p0.food.values()) - sum(food_before_p0.values()) == 1
    assert sum(p1.food.values()) - sum(food_before_p1.values()) == 1
    assert state.birdfeeder.total() == 0


def test_each_player_gains_die_stops_when_feeder_empty():
    """Only 1 die is in the feeder, so only the starting player gets one."""
    state = _setup_state(seed=1)

    for f in state.birdfeeder.counts:
        state.birdfeeder.counts[f] = 0
    state.birdfeeder.counts[Food.SEED] = 1

    p0, p1 = state.players
    state.current_player = p0.id
    food_before_p0 = dict(p0.food)
    food_before_p1 = dict(p1.food)

    carrier = next(b for b in load_all()[0] if b.name == "Ruby-Throated Hummingbird")
    pb = PlayedBird(bird=carrier)
    eff = Effect(kind=EffectKind.EACH_PLAYER_GAINS_DIE_CHOOSE_ORDER, amount=1)

    def agent_p0(_engine: Engine, decision: Decision) -> Choice:
        if decision.type == DecisionType.BIRD_POWER_PICK_STARTING_PLAYER:
            for c in decision.choices:
                if c.payload == p0.id:
                    return c
            raise AssertionError("p0 not in choices")
        if decision.type == DecisionType.GAIN_FOOD_PICK_DIE:
            assert decision.player_id == p0.id
            return decision.choices[0]
        raise AssertionError(f"unexpected decision for p0: {decision.type}")

    def agent_p1(_engine: Engine, decision: Decision) -> Choice:
        raise AssertionError(f"p1 should not be asked when feeder empties: {decision.type}")

    eng = Engine(state, agents=[agent_p0, agent_p1])
    eng._apply_effect(agent_p0, p0, pb, carrier.habitats[0], eff, "play")

    assert sum(p0.food.values()) - sum(food_before_p0.values()) == 1
    assert sum(p1.food.values()) - sum(food_before_p1.values()) == 0
    assert state.birdfeeder.total() == 0


def test_each_player_gains_die_routes_decisions_to_correct_agent():
    """When active player chooses themselves to start, ensure each agent is
    queried for its own die pick (and only for its own)."""
    state = _setup_state(seed=2)

    for f in state.birdfeeder.counts:
        state.birdfeeder.counts[f] = 0
    state.birdfeeder.counts[Food.SEED] = 1
    state.birdfeeder.counts[Food.RODENT] = 1

    p0, p1 = state.players
    state.current_player = p0.id

    carrier = next(b for b in load_all()[0] if b.name == "Anna's Hummingbird")
    pb = PlayedBird(bird=carrier)
    eff = Effect(kind=EffectKind.EACH_PLAYER_GAINS_DIE_CHOOSE_ORDER, amount=1)

    queried: list[tuple[str, DecisionType, int]] = []

    def agent_p0(_engine: Engine, decision: Decision) -> Choice:
        queried.append(("p0", decision.type, decision.player_id))
        if decision.type == DecisionType.BIRD_POWER_PICK_STARTING_PLAYER:
            for c in decision.choices:
                if c.payload == p0.id:
                    return c
        assert decision.player_id == p0.id
        return decision.choices[0]

    def agent_p1(_engine: Engine, decision: Decision) -> Choice:
        queried.append(("p1", decision.type, decision.player_id))
        assert decision.player_id == p1.id
        return decision.choices[0]

    eng = Engine(state, agents=[agent_p0, agent_p1])
    eng._apply_effect(agent_p0, p0, pb, carrier.habitats[0], eff, "play")

    # Expected query log: p0 picks the starting player, p0 picks a die,
    # then p1 picks a die.
    kinds = [(who, dtype) for (who, dtype, _pid) in queried]
    assert kinds == [
        ("p0", DecisionType.BIRD_POWER_PICK_STARTING_PLAYER),
        ("p0", DecisionType.GAIN_FOOD_PICK_DIE),
        ("p1", DecisionType.GAIN_FOOD_PICK_DIE),
    ]
