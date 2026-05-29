"""Tests for the EACH_PLAYER_GAINS_DIE_CHOOSE_ORDER power.

Two birds carry this in the core set:
    Anna's Hummingbird
    Ruby-Throated Hummingbird

Power text: "Each player gains 1 [die] from the birdfeeder, starting with
the player of your choice."
"""

from __future__ import annotations

import random
import typing

from wingspan import cards, decisions, engine, state
from wingspan.engine import powers


def test_parse_each_player_gains_die():
    power = cards.parse_power(
        cards.PowerColor.WHITE,
        "Each player gains 1 [die] from the birdfeeder, starting with the player of your choice.",
    )
    assert len(power.effects) == 1
    eff = power.effects[0]
    assert eff.kind == cards.EffectKind.EACH_PLAYER_GAINS_DIE_CHOOSE_ORDER
    assert eff.amount == 1


def test_both_hummingbirds_are_implemented():
    birds, _, _ = cards.load_all()
    target_names = {"Anna's Hummingbird", "Ruby-Throated Hummingbird"}
    found = {bird.name: bird for bird in birds if bird.name in target_names}
    missing = target_names - set(found)
    assert not missing, f"birds not present in data: {missing}"
    for name, bird in found.items():
        kinds = [effect.kind for effect in bird.power.effects]
        assert (
            cards.EffectKind.EACH_PLAYER_GAINS_DIE_CHOOSE_ORDER in kinds
        ), f"{name} parsed as {kinds}; raw_power_text={bird.raw_power_text!r}"
        assert cards.EffectKind.UNIMPLEMENTED not in kinds, name


def _setup_state(seed: int = 0):
    birds, bonuses, goals = cards.load_all()
    rng = random.Random(seed)
    return state.new_game(rng, birds, bonuses, goals)


def test_each_player_gains_die_credits_both_players_in_chosen_order():
    """Active player picks P1 to start, so the dice are credited P1 then P0."""
    gs = _setup_state(seed=0)

    # Make the feeder deterministic: 1 seed, 1 fruit, everything else 0.
    for food in gs.birdfeeder.counts:
        gs.birdfeeder.counts[food] = 0
    gs.birdfeeder.counts[cards.Food.SEED] = 1
    gs.birdfeeder.counts[cards.Food.FRUIT] = 1

    p0, p1 = gs.players
    gs.current_player = p0.id
    food_before_p0 = p0.food.as_dict()
    food_before_p1 = p1.food.as_dict()

    carrier = next(
        bird for bird in cards.load_all()[0] if bird.name == "Anna's Hummingbird"
    )
    pb = state.PlayedBird(bird=carrier)
    eff = cards.Effect(
        kind=cards.EffectKind.EACH_PLAYER_GAINS_DIE_CHOOSE_ORDER, amount=1
    )

    # Scripted agents:
    #   - active player (p0) picks the starting player as P1
    #   - each player picks the first die available
    def agent_p0[C: decisions.Choice](
        _engine: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        if isinstance(decision, decisions.BirdPowerPickStartingPlayerDecision):
            assert decision.player_id == p0.id
            for choice in decision.choices:
                if choice.player_id == p1.id:
                    return typing.cast(C, choice)
            raise AssertionError("p1 not in choices")
        if isinstance(decision, decisions.GainFoodPickDieDecision):
            assert decision.player_id == p0.id
            return typing.cast(C, decision.choices[0])
        raise AssertionError(f"unexpected decision for p0: {type(decision).__name__}")

    def agent_p1[C: decisions.Choice](
        _engine: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        assert isinstance(decision, decisions.GainFoodPickDieDecision)
        assert decision.player_id == p1.id
        return typing.cast(C, decision.choices[0])

    eng = engine.Engine(gs, agents=[agent_p0, agent_p1])
    powers.apply_effect(eng, agent_p0, p0, pb, carrier.habitats[0], eff, "play")

    # Total food gained: each player got exactly 1 die.
    assert sum(p0.food.values()) - sum(food_before_p0.values()) == 1
    assert sum(p1.food.values()) - sum(food_before_p1.values()) == 1
    assert gs.birdfeeder.total() == 0


def test_each_player_gains_die_stops_when_feeder_empty():
    """Only 1 die is in the feeder, so only the starting player gets one."""
    gs = _setup_state(seed=1)

    for food in gs.birdfeeder.counts:
        gs.birdfeeder.counts[food] = 0
    gs.birdfeeder.counts[cards.Food.SEED] = 1

    p0, p1 = gs.players
    gs.current_player = p0.id
    food_before_p0 = p0.food.as_dict()
    food_before_p1 = p1.food.as_dict()

    carrier = next(
        bird for bird in cards.load_all()[0] if bird.name == "Ruby-Throated Hummingbird"
    )
    pb = state.PlayedBird(bird=carrier)
    eff = cards.Effect(
        kind=cards.EffectKind.EACH_PLAYER_GAINS_DIE_CHOOSE_ORDER, amount=1
    )

    def agent_p0[C: decisions.Choice](
        _engine: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        if isinstance(decision, decisions.BirdPowerPickStartingPlayerDecision):
            for choice in decision.choices:
                if choice.player_id == p0.id:
                    return typing.cast(C, choice)
            raise AssertionError("p0 not in choices")
        if isinstance(decision, decisions.GainFoodPickDieDecision):
            assert decision.player_id == p0.id
            return typing.cast(C, decision.choices[0])
        raise AssertionError(f"unexpected decision for p0: {type(decision).__name__}")

    def agent_p1[C: decisions.Choice](
        _engine: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        raise AssertionError(
            f"p1 should not be asked when feeder empties: {type(decision).__name__}"
        )

    eng = engine.Engine(gs, agents=[agent_p0, agent_p1])
    powers.apply_effect(eng, agent_p0, p0, pb, carrier.habitats[0], eff, "play")

    assert sum(p0.food.values()) - sum(food_before_p0.values()) == 1
    assert sum(p1.food.values()) - sum(food_before_p1.values()) == 0
    assert gs.birdfeeder.total() == 0


def test_each_player_gains_die_routes_decisions_to_correct_agent():
    """When active player chooses themselves to start, ensure each agent is
    queried for its own die pick (and only for its own)."""
    gs = _setup_state(seed=2)

    for food in gs.birdfeeder.counts:
        gs.birdfeeder.counts[food] = 0
    gs.birdfeeder.counts[cards.Food.SEED] = 1
    gs.birdfeeder.counts[cards.Food.RODENT] = 1

    p0, p1 = gs.players
    gs.current_player = p0.id

    carrier = next(
        bird for bird in cards.load_all()[0] if bird.name == "Anna's Hummingbird"
    )
    pb = state.PlayedBird(bird=carrier)
    eff = cards.Effect(
        kind=cards.EffectKind.EACH_PLAYER_GAINS_DIE_CHOOSE_ORDER, amount=1
    )

    queried: list[tuple[str, type, int]] = []

    def agent_p0[C: decisions.Choice](
        _engine: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        queried.append(("p0", type(decision), decision.player_id))
        if isinstance(decision, decisions.BirdPowerPickStartingPlayerDecision):
            for choice in decision.choices:
                if choice.player_id == p0.id:
                    return typing.cast(C, choice)
        assert decision.player_id == p0.id
        return typing.cast(C, decision.choices[0])

    def agent_p1[C: decisions.Choice](
        _engine: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        queried.append(("p1", type(decision), decision.player_id))
        assert decision.player_id == p1.id
        return decision.choices[0]

    eng = engine.Engine(gs, agents=[agent_p0, agent_p1])
    powers.apply_effect(eng, agent_p0, p0, pb, carrier.habitats[0], eff, "play")

    # Expected query log: p0 picks the starting player, p0 picks a die,
    # then p1 picks a die.
    kinds = [(who, dtype) for (who, dtype, _pid) in queried]
    assert kinds == [
        ("p0", decisions.BirdPowerPickStartingPlayerDecision),
        ("p0", decisions.GainFoodPickDieDecision),
        ("p1", decisions.GainFoodPickDieDecision),
    ]
