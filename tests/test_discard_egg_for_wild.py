"""Tests for the DISCARD_EGG_FOR_WILD power (crows, ravens, night-heron)."""

from __future__ import annotations

import random
import typing

from wingspan import cards, decisions, engine, state
from wingspan.engine import powers


def _find_bird(birds: list[cards.Bird], name: str) -> cards.Bird:
    for bird in birds:
        if bird.name == name:
            return bird
    raise AssertionError(f"bird {name!r} not found")


def _empty_engine(seed: int = 0) -> tuple[engine.Engine, list[cards.Bird]]:
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
    for bird in birds:
        if bird.name in expected:
            kinds = [effect.kind for effect in bird.power.effects]
            assert (
                cards.EffectKind.DISCARD_EGG_FOR_WILD in kinds
            ), f"{bird.name} did not parse to DISCARD_EGG_FOR_WILD: {kinds}"
            eff = next(
                effect
                for effect in bird.power.effects
                if effect.kind == cards.EffectKind.DISCARD_EGG_FOR_WILD
            )
            seen[bird.name] = eff.amount
    assert seen == expected


def test_discard_egg_for_wild_decrements_egg_and_grants_food():
    eng, birds = _empty_engine(seed=1)
    bird = _find_bird(birds, "Common Raven")  # amount=2

    player = eng.state.me()
    raven = state.PlayedBird(bird=bird)
    other = _find_bird(birds, "American Crow")
    sibling = state.PlayedBird(bird=other, eggs=1)
    player.board[cards.Habitat.FOREST].extend([sibling, raven])

    for food in player.food:
        player.food[food] = 0

    # Scripted agent: accept the exchange, pay with FOREST slot-0 egg, then
    # take the two wild foods in order (SEED first, then FRUIT).
    food_order = iter([cards.Food.SEED, cards.Food.FRUIT])

    def agent[C: decisions.Choice](
        _engine: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        if isinstance(decision, decisions.AcceptExchangeDecision):
            # Accept the trade.
            return typing.cast(
                C,
                next(
                    choice
                    for choice in decision.choices
                    if isinstance(choice, decisions.PayCostChoice)
                ),
            )
        if isinstance(decision, decisions.RemoveEggDecision):
            return typing.cast(
                C,
                next(
                    choice
                    for choice in decision.choices
                    if isinstance(choice, decisions.BoardTargetChoice)
                    and choice.habitat == cards.Habitat.FOREST
                    and choice.slot == 0
                ),
            )
        if isinstance(decision, decisions.GainFoodDecision):
            want = next(food_order)
            return typing.cast(
                C,
                next(
                    choice
                    for choice in decision.choices
                    if isinstance(choice, decisions.FoodChoice) and choice.food == want
                ),
            )
        raise AssertionError(f"unexpected decision type: {type(decision).__name__}")

    powers.dispatch_power(
        eng, agent, player, raven, cards.Habitat.FOREST, trigger="activate"
    )

    assert sibling.eggs == 0
    assert player.food[cards.Food.SEED] == 1
    assert player.food[cards.Food.FRUIT] == 1


def test_discard_egg_for_wild_skips_when_no_other_bird_has_an_egg():
    eng, birds = _empty_engine(seed=2)
    bird = _find_bird(birds, "American Crow")  # amount=1
    player = eng.state.me()

    crow = state.PlayedBird(bird=bird, eggs=3)  # even its own eggs should not count
    player.board[cards.Habitat.GRASSLAND].append(crow)

    before_food = player.food.as_dict()

    def agent[C: decisions.Choice](
        _engine: engine.Engine,
        _decision: decisions.Decision[C],
    ) -> C:
        raise AssertionError("agent should not be asked anything when power is a no-op")

    powers.dispatch_power(
        eng, agent, player, crow, cards.Habitat.GRASSLAND, trigger="activate"
    )

    assert crow.eggs == 3  # self eggs untouched
    assert player.food.as_dict() == before_food


def test_discard_egg_for_wild_can_be_skipped():
    eng, birds = _empty_engine(seed=3)
    bird = _find_bird(birds, "Fish Crow")  # amount=1
    player = eng.state.me()

    fishcrow = state.PlayedBird(bird=bird)
    sibling = state.PlayedBird(bird=_find_bird(birds, "Chihuahuan Raven"), eggs=2)
    player.board[cards.Habitat.WETLAND].extend([sibling, fishcrow])

    before_food = player.food.as_dict()

    def agent[C: decisions.Choice](
        _engine: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        return next(choice for choice in decision.choices if choice.label == "skip")

    powers.dispatch_power(
        eng, agent, player, fishcrow, cards.Habitat.WETLAND, trigger="activate"
    )

    assert sibling.eggs == 2  # untouched
    assert player.food.as_dict() == before_food
