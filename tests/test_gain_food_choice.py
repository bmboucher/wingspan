"""Unit tests for GAIN_FOOD_FROM_FEEDER_CHOICE and GAIN_DIE_ANY effects.

These cover Indigo Bunting / Rose-Breasted Grosbeak / Western Tanager (food
disjunction) and American Redstart (any die face).
"""

from __future__ import annotations

import os
import random
import sys
import typing

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import cards, decisions, engine, state
from wingspan.engine import powers


def _make_engine() -> engine.Engine:
    birds, bonuses, goals = cards.load_all()
    rng = random.Random(0)
    gs = state.new_game(rng, birds, bonuses, goals)
    return engine.Engine(gs)


def _stage_played_bird(eng: engine.Engine, power_text: str) -> state.PlayedBird:
    """Attach a fresh PlayedBird carrying ``power_text`` to a template bird."""
    template = next(
        bird for bird in eng.state.bird_deck if bird.color == cards.PowerColor.BROWN
    )
    bird = template.model_copy(
        update={
            "raw_power_text": power_text,
            "power": cards.parse_power(cards.PowerColor.BROWN, power_text),
        }
    )
    return state.PlayedBird(bird=bird)


# ---------------------------------------------------------------------------
# Parser tests


def test_parser_recognises_food_choice():
    power = cards.parse_power(
        cards.PowerColor.BROWN,
        "Gain 1 [invertebrate] or [fruit] from the birdfeeder, if available.",
    )
    assert len(power.effects) == 1
    eff = power.effects[0]
    assert eff.kind == cards.EffectKind.GAIN_FOOD_FROM_FEEDER_CHOICE
    assert eff.amount == 1
    assert eff.food_a == cards.Food.INVERTEBRATE and eff.food_b == cards.Food.FRUIT


def test_parser_recognises_die_any():
    power = cards.parse_power(
        cards.PowerColor.BROWN, "Gain 1 [die] from the birdfeeder."
    )
    assert len(power.effects) == 1
    eff = power.effects[0]
    assert eff.kind == cards.EffectKind.GAIN_DIE_ANY
    assert eff.amount == 1


def test_parser_die_any_does_not_collide_with_food_birdfeeder():
    """A regular 'Gain 1 [seed] from the birdfeeder' must still be parsed as
    GAIN_FOOD_BIRDFEEDER, not as GAIN_DIE_ANY."""
    power = cards.parse_power(
        cards.PowerColor.BROWN, "Gain 1 [seed] from the birdfeeder."
    )
    kinds = [effect.kind for effect in power.effects]
    assert cards.EffectKind.GAIN_FOOD_BIRDFEEDER in kinds
    assert cards.EffectKind.GAIN_DIE_ANY not in kinds


def test_parser_choice_does_not_also_emit_birdfeeder():
    """Disjunction text must not also match the single-food pattern."""
    power = cards.parse_power(
        cards.PowerColor.BROWN,
        "Gain 1 [seed] or [fruit] from the birdfeeder, if available.",
    )
    kinds = [effect.kind for effect in power.effects]
    assert kinds == [cards.EffectKind.GAIN_FOOD_FROM_FEEDER_CHOICE]


# ---------------------------------------------------------------------------
# Engine dispatch tests


def test_food_choice_takes_only_available_food():
    eng = _make_engine()
    pb = _stage_played_bird(
        eng,
        "Gain 1 [invertebrate] or [fruit] from the birdfeeder, if available.",
    )
    player = eng.state.players[0]
    eng.state.current_player = 0
    # Birdfeeder: only fruit available.
    for food in cards.Food:
        eng.state.birdfeeder.counts[food] = 0
    eng.state.birdfeeder.counts[cards.Food.FRUIT] = 2
    food_before = player.food[cards.Food.FRUIT]

    def agent[C: decisions.Choice](
        _eng: engine.Engine,
        _d: decisions.Decision[C],
    ) -> C:  # pragma: no cover - not reached, only one option
        raise AssertionError("should auto-take the only available food")

    powers.dispatch_power(eng, agent, player, pb, pb.bird.habitats[0], "activate")
    assert player.food[cards.Food.FRUIT] == food_before + 1
    assert eng.state.birdfeeder.counts[cards.Food.FRUIT] == 1


def test_food_choice_asks_when_both_present():
    eng = _make_engine()
    pb = _stage_played_bird(
        eng,
        "Gain 1 [invertebrate] or [fruit] from the birdfeeder, if available.",
    )
    player = eng.state.players[0]
    eng.state.current_player = 0
    for food in cards.Food:
        eng.state.birdfeeder.counts[food] = 0
    eng.state.birdfeeder.counts[cards.Food.INVERTEBRATE] = 1
    eng.state.birdfeeder.counts[cards.Food.FRUIT] = 1
    inv_before = player.food[cards.Food.INVERTEBRATE]
    fruit_before = player.food[cards.Food.FRUIT]

    asked = {"n": 0}

    def agent[C: decisions.Choice](
        _eng: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        asked["n"] += 1
        assert isinstance(decision, decisions.BirdPowerPickFoodDecision)
        foods = {
            choice.food
            for choice in decision.choices
            if isinstance(choice, decisions.FoodChoice)
        }
        assert foods == {cards.Food.INVERTEBRATE, cards.Food.FRUIT}
        # Choose invertebrate.
        for choice in decision.choices:
            if (
                isinstance(choice, decisions.FoodChoice)
                and choice.food == cards.Food.INVERTEBRATE
            ):
                return typing.cast(C, choice)
        raise AssertionError("invertebrate not offered")

    powers.dispatch_power(eng, agent, player, pb, pb.bird.habitats[0], "activate")
    assert asked["n"] == 1
    assert player.food[cards.Food.INVERTEBRATE] == inv_before + 1
    assert player.food[cards.Food.FRUIT] == fruit_before
    assert eng.state.birdfeeder.counts[cards.Food.INVERTEBRATE] == 0
    assert eng.state.birdfeeder.counts[cards.Food.FRUIT] == 1


def test_food_choice_skips_when_neither_present():
    eng = _make_engine()
    pb = _stage_played_bird(
        eng,
        "Gain 1 [seed] or [fruit] from the birdfeeder, if available.",
    )
    player = eng.state.players[0]
    eng.state.current_player = 0
    for food in cards.Food:
        eng.state.birdfeeder.counts[food] = 0
    eng.state.birdfeeder.counts[cards.Food.INVERTEBRATE] = 3  # neither seed nor fruit
    snapshot = player.food.as_dict()

    def agent[C: decisions.Choice](
        _eng: engine.Engine,
        _d: decisions.Decision[C],
    ) -> C:  # pragma: no cover - must not be consulted
        raise AssertionError("agent should not be asked when nothing available")

    powers.dispatch_power(eng, agent, player, pb, pb.bird.habitats[0], "activate")
    assert player.food.as_dict() == snapshot
    assert eng.state.birdfeeder.counts[cards.Food.INVERTEBRATE] == 3


def test_die_any_picks_from_all_available_foods():
    eng = _make_engine()
    pb = _stage_played_bird(eng, "Gain 1 [die] from the birdfeeder.")
    player = eng.state.players[0]
    eng.state.current_player = 0
    for food in cards.Food:
        eng.state.birdfeeder.counts[food] = 0
    eng.state.birdfeeder.counts[cards.Food.SEED] = 2
    eng.state.birdfeeder.counts[cards.Food.RODENT] = 1
    seed_before = player.food[cards.Food.SEED]
    rodent_before = player.food[cards.Food.RODENT]

    def agent[C: decisions.Choice](
        _eng: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        assert isinstance(decision, decisions.BirdPowerPickFoodDecision)
        foods = {
            choice.food
            for choice in decision.choices
            if isinstance(choice, decisions.FoodChoice)
        }
        assert foods == {cards.Food.SEED, cards.Food.RODENT}
        for choice in decision.choices:
            if (
                isinstance(choice, decisions.FoodChoice)
                and choice.food == cards.Food.RODENT
            ):
                return typing.cast(C, choice)
        raise AssertionError("rodent not offered")

    powers.dispatch_power(eng, agent, player, pb, pb.bird.habitats[0], "activate")
    assert player.food[cards.Food.RODENT] == rodent_before + 1
    assert player.food[cards.Food.SEED] == seed_before
    assert eng.state.birdfeeder.counts[cards.Food.RODENT] == 0
    assert eng.state.birdfeeder.counts[cards.Food.SEED] == 2


def test_die_any_skips_when_feeder_empty():
    eng = _make_engine()
    pb = _stage_played_bird(eng, "Gain 1 [die] from the birdfeeder.")
    player = eng.state.players[0]
    eng.state.current_player = 0
    for food in cards.Food:
        eng.state.birdfeeder.counts[food] = 0
    snapshot = player.food.as_dict()

    def agent[C: decisions.Choice](
        _eng: engine.Engine,
        _d: decisions.Decision[C],
    ) -> C:  # pragma: no cover - must not be consulted
        raise AssertionError("agent should not be asked when feeder is empty")

    powers.dispatch_power(eng, agent, player, pb, pb.bird.habitats[0], "activate")
    assert player.food.as_dict() == snapshot


# ---------------------------------------------------------------------------
# Coverage of the four target birds


def test_target_birds_parse_to_expected_kinds():
    from wingspan import cards as cards_mod

    birds, _, _ = cards_mod.load_all()
    by_name = {bird.name: bird for bird in birds}
    for name in ("Indigo Bunting", "Rose-Breasted Grosbeak", "Western Tanager"):
        kinds = [effect.kind for effect in by_name[name].power.effects]
        assert kinds == [
            cards.EffectKind.GAIN_FOOD_FROM_FEEDER_CHOICE
        ], f"{name}: kinds={kinds}"
    redstart_kinds = [
        effect.kind for effect in by_name["American Redstart"].power.effects
    ]
    assert redstart_kinds == [
        cards.EffectKind.GAIN_DIE_ANY
    ], f"American Redstart: kinds={redstart_kinds}"
