"""Tests for the main-action decision composition.

Playing a bird is a ``PLAY_BIRD`` main-action *type*, offered (alongside the
three always-present habitat actions) only when the player has a legal play.
Choosing *which* bird to play — in which habitat, for which payment — is a
separate follow-up ``PlayBirdDecision``; only the egg cost remains a further
follow-up.

These drive a real game and capture every ``MainActionDecision`` and
``PlayBirdDecision`` the engine offers, so the assertions cover the live turn
loop rather than a private helper.
"""

from __future__ import annotations

import random
import typing

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


def _play_and_capture(
    agent_factory: typing.Callable[
        [random.Random, list[decisions.Decision[typing.Any]]], engine_core.Agent
    ],
) -> list[decisions.Decision[typing.Any]]:
    """Play one seeded game with ``agent_factory`` in seat 0 (which records every
    decision it is asked) and a random opponent, returning the recorded sink."""
    eng, *_ = engine.Engine.create(seed=123)
    rng = random.Random(123)
    sink: list[decisions.Decision[typing.Any]] = []
    engine.Engine.play_one_game(
        eng.state, (agent_factory(rng, sink), agents.random_agent(rng))
    )
    return sink


def _forced_play_bird_choice(
    decision: decisions.Decision[typing.Any],
) -> decisions.MainActionChoice | None:
    """The ``PLAY_BIRD`` option if ``decision`` is a main-action pick offering it,
    else ``None``. Kept separate so the agent never narrows its own generic
    ``decision`` (which would break the random-fallthrough call)."""
    if isinstance(decision, decisions.MainActionDecision):
        for choice in decision.choices:
            if choice.action == decisions.MainAction.PLAY_BIRD:
                return choice
    return None


def _play_bird_preferring_agent(
    rng: random.Random,
    sink: list[decisions.Decision[typing.Any]],
) -> engine_core.Agent:
    """Records every decision and always takes the ``PLAY_BIRD`` main action when
    it is offered, so the follow-up ``PlayBirdDecision`` is guaranteed to fire
    over a full game; otherwise plays randomly."""
    inner = agents.random_agent(rng)

    def agent[C: decisions.Choice](
        eng: engine_core.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        sink.append(decision)
        forced = _forced_play_bird_choice(decision)
        if forced is not None:
            return typing.cast(C, forced)
        return inner(eng, decision)

    return agent


def test_main_action_always_offers_the_three_habitat_actions():
    sink = _play_and_capture(_recording_agent)
    main_decisions = [
        decision
        for decision in sink
        if isinstance(decision, decisions.MainActionDecision)
    ]
    assert main_decisions, "expected the recorded player to take some turns"
    for decision in main_decisions:
        actions_offered = {choice.action for choice in decision.choices}
        # The three habitat actions are always present; PLAY_BIRD is the only
        # extra, optional type.
        assert _HABITAT_ACTIONS <= actions_offered
        assert actions_offered <= _HABITAT_ACTIONS | {decisions.MainAction.PLAY_BIRD}


def test_play_bird_offered_as_a_main_action_type():
    sink = _play_and_capture(_recording_agent)
    main_decisions = [
        decision
        for decision in sink
        if isinstance(decision, decisions.MainActionDecision)
    ]
    # Over a full game the player can play a bird on at least one turn, so
    # PLAY_BIRD must be offered as a main-action *type* (no longer a pile of
    # PlayBirdChoices folded into the menu).
    assert any(
        decisions.MainAction.PLAY_BIRD in {choice.action for choice in decision.choices}
        for decision in main_decisions
    ), "expected PLAY_BIRD to be offered on at least one turn"


def test_choosing_play_bird_opens_a_play_menu_with_valid_plays():
    sink = _play_and_capture(_play_bird_preferring_agent)
    play_decisions = [
        decision
        for decision in sink
        if isinstance(decision, decisions.PlayBirdDecision)
    ]
    assert play_decisions, "expected choosing PLAY_BIRD to open a play menu"
    for decision in play_decisions:
        for choice in decision.choices:
            # The habitat must be one the bird can live in. The food payment is
            # no longer bundled into the candidate — it resolves as a follow-up
            # PayBirdFoodDecision, asserted in test_play_bird_payment.py.
            assert choice.habitat in choice.bird.habitats


def test_play_menu_followed_by_a_cost_meeting_payment_menu():
    """Every captured ``PayBirdFoodDecision`` pays for the play just picked:
    its bird/habitat context matches, and every offered payment multiset
    legally covers the bird's printed cost (1-for-1, 2-for-1 substitution,
    wild fills)."""
    sink = _play_and_capture(_play_bird_preferring_agent)
    pay_decisions = [
        decision
        for decision in sink
        if isinstance(decision, decisions.PayBirdFoodDecision)
    ]
    for decision in pay_decisions:
        assert decision.habitat in decision.bird.habitats
        for choice in decision.choices:
            assert helpers.cost_meets(decision.bird.food_cost, choice.payment)
