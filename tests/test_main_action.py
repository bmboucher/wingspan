"""Tests for the main-action decision composition.

Playing a bird is no longer a single ``play bird`` main action followed by
separate card / habitat / payment picks: each legal ``(bird, habitat, food
payment)`` is surfaced as its own ``PlayBirdChoice`` at the main-action stage,
alongside the three always-present habitat actions. Only the egg cost stays a
follow-up decision.

These drive a real game and capture every ``MainActionDecision`` the engine
offers, so the assertions cover the live turn loop rather than a private
helper.
"""

from __future__ import annotations

import os
import random
import sys
import typing

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import agents, decisions, engine  # noqa: E402
from wingspan.engine import core as engine_core  # noqa: E402
from wingspan.engine import helpers  # noqa: E402

_HABITAT_ACTIONS = {
    decisions.MainAction.GAIN_FOOD,
    decisions.MainAction.LAY_EGGS,
    decisions.MainAction.DRAW_CARDS,
}


def _recording_agent(
    rng: random.Random,
    sink: list[decisions.Decision[typing.Any]],
) -> engine_core.Agent:
    """A random agent that records every decision it is asked to resolve."""
    inner = agents.random_agent(rng)

    def agent[C: decisions.Choice](
        eng: engine_core.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        sink.append(decision)
        return inner(eng, decision)

    return agent


def _play_and_capture_main_decisions() -> list[decisions.MainActionDecision]:
    eng, *_ = engine.Engine.create(seed=123)
    rng = random.Random(123)
    sink: list[decisions.Decision[typing.Any]] = []
    engine.Engine.play_one_game(
        eng.state, (_recording_agent(rng, sink), agents.random_agent(rng))
    )
    return [
        decision
        for decision in sink
        if isinstance(decision, decisions.MainActionDecision)
    ]


def test_every_main_action_offers_exactly_three_habitat_actions():
    main_decisions = _play_and_capture_main_decisions()
    assert main_decisions, "expected the recorded player to take some turns"
    for decision in main_decisions:
        habitat = {
            choice.action
            for choice in decision.choices
            if isinstance(choice, decisions.MainActionChoice)
        }
        assert habitat == _HABITAT_ACTIONS


def test_playable_birds_appear_as_choices_at_main_action_stage():
    main_decisions = _play_and_capture_main_decisions()
    # Over a full game the player has playable birds on at least one turn, and
    # those must surface as PlayBirdChoices directly in the main-action menu.
    assert any(
        any(isinstance(choice, decisions.PlayBirdChoice) for choice in decision.choices)
        for decision in main_decisions
    ), "expected at least one main-action menu to list a playable bird"


def test_play_choices_carry_habitat_and_payment():
    main_decisions = _play_and_capture_main_decisions()
    play_choices = [
        choice
        for decision in main_decisions
        for choice in decision.choices
        if isinstance(choice, decisions.PlayBirdChoice)
    ]
    assert play_choices, "expected at least one PlayBirdChoice over the game"
    for choice in play_choices:
        # The habitat must be one the bird can live in, and the bundled payment
        # must be a legal (exact, allowing 2-for-1 substitution) cover of the
        # printed cost.
        assert choice.habitat in choice.bird.habitats
        assert helpers.cost_meets(choice.bird.food_cost, choice.payment)
