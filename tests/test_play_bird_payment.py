"""Tests for the play-bird cost follow-ups (``PayBirdFoodDecision``).

A committed play resolves its costs as follow-up decisions, eggs then food:
``RemoveEggDecision`` (one ask per egg, the ``PAY_EGG`` head) then
``PayBirdFoodDecision`` (one ask choosing among the legal payment multisets,
the ``SPEND_FOOD`` head). The payment decision is mandatory — no skip, the
commitment happened upstream — and is forced (auto-resolved by ``Engine.ask``
without consulting the agent) when exactly one payment is legal.
"""

from __future__ import annotations

import os
import random
import sys
import typing

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import cards, decisions, engine, state
from wingspan.engine import actions, helpers


def _single_food_bird(birds: list[cards.Bird]) -> cards.Bird:
    """A non-WHITE bird whose printed cost is exactly one specific food token,
    so the legal-payment set is easy to control: holding the token gives one
    payment (forced); adding two tokens of another food adds the 2-for-1
    substitute (a genuine fork). Non-WHITE so playing it fires no power."""
    for bird in birds:
        cost = bird.food_cost
        if bird.color != cards.PowerColor.WHITE and cost.wild == 0 and cost.total == 1:
            return bird
    raise AssertionError("no single-food non-white bird in catalog")


def _prepared_game(
    seed: int,
) -> tuple[engine.Engine, state.Player, cards.Bird, cards.Food, cards.Habitat]:
    """A game whose current player holds exactly one ``_single_food_bird`` and
    an empty board, ready for ``actions.do_play_bird`` with no egg cost."""
    birds, bonuses, goals = cards.load_all()
    gs = state.new_game(random.Random(seed), birds, bonuses, goals)
    eng = engine.Engine(gs)
    gs.current_player = 0
    player = gs.me()

    bird = _single_food_bird(birds)
    food = next(f for f in cards.ALL_FOODS if bird.food_cost.specific_of(f) == 1)
    player.hand = [bird]
    for any_food in cards.ALL_FOODS:
        player.food[any_food] = 0
    for habitat in cards.ALL_HABITATS:
        player.board[habitat] = []
    return eng, player, bird, food, bird.habitats[0]


def test_payment_follow_up_offers_every_legal_payment():
    """With two legal payments the follow-up reaches the agent carrying the
    committed play as context, offers only cost-meeting ``FoodPaymentChoice``s
    (no skip), and deducts exactly the chosen one."""
    eng, player, bird, food, habitat = _prepared_game(seed=0)
    other = next(f for f in cards.ALL_FOODS if f != food)
    player.food[food] = 1
    player.food[other] = 2  # 2-for-1 substitute for the printed token

    sink: list[decisions.Decision[typing.Any]] = []

    def agent[C: decisions.Choice](
        _eng: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        sink.append(decision)
        assert isinstance(decision, decisions.PayBirdFoodDecision)
        # Pay the printed token, keeping the two substitutes.
        return typing.cast(
            C,
            next(c for c in decision.choices if c.payment[food] == 1),
        )

    eng.agents = [agent, agent]
    actions.do_play_bird(eng, agent, bird, habitat)

    assert [type(d) for d in sink] == [decisions.PayBirdFoodDecision]
    pay_decision = sink[0]
    assert isinstance(pay_decision, decisions.PayBirdFoodDecision)
    # The committed play rides along as typed context.
    assert pay_decision.bird is bird
    assert pay_decision.habitat == habitat
    # Mandatory (no skip), one choice per legal payment, every payment legal.
    assert len(pay_decision.choices) == 2
    for choice in pay_decision.choices:
        assert isinstance(choice, decisions.FoodPaymentChoice)
        assert helpers.cost_meets(bird.food_cost, choice.payment)
    # The chosen payment (and only it) was deducted; the bird was placed.
    assert player.food[food] == 0
    assert player.food[other] == 2
    assert bird not in player.hand
    assert any(pb.bird is bird for row in player.board.values() for pb in row)


def test_egg_cost_resolves_before_food_payment():
    """The printed cost order is eggs then food: with both costs a genuine
    fork, the ``RemoveEggDecision`` reaches the agent before the
    ``PayBirdFoodDecision``."""
    eng, player, bird, food, habitat = _prepared_game(seed=1)
    other = next(f for f in cards.ALL_FOODS if f != food)
    player.food[food] = 1
    player.food[other] = 2  # keep the payment a fork too

    # One bird already in the target row -> the play costs 1 egg; eggs on two
    # different board birds make the removal a fork that reaches the agent.
    birds, _, _ = cards.load_all()
    filler_a = next(b for b in birds if b is not bird)
    filler_b = next(b for b in birds if b is not bird and b is not filler_a)
    occupant = state.PlayedBird(bird=filler_a)
    occupant.eggs = 1
    bystander = state.PlayedBird(bird=filler_b)
    bystander.eggs = 1
    player.board[habitat] = [occupant]
    other_habitat = next(h for h in cards.ALL_HABITATS if h != habitat)
    player.board[other_habitat] = [bystander]

    sink: list[decisions.Decision[typing.Any]] = []

    def agent[C: decisions.Choice](
        _eng: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        sink.append(decision)
        if isinstance(decision, decisions.RemoveEggDecision):
            return typing.cast(
                C,
                next(
                    c
                    for c in decision.choices
                    if isinstance(c, decisions.BoardTargetChoice)
                ),
            )
        assert isinstance(decision, decisions.PayBirdFoodDecision)
        return typing.cast(
            C,
            next(c for c in decision.choices if c.payment[food] == 1),
        )

    eng.agents = [agent, agent]
    actions.do_play_bird(eng, agent, bird, habitat)

    assert [type(d) for d in sink] == [
        decisions.RemoveEggDecision,
        decisions.PayBirdFoodDecision,
    ]
    assert occupant.eggs + bystander.eggs == 1  # exactly one egg was paid
    assert player.food[food] == 0


def test_single_payment_is_forced_and_auto_resolved():
    """With exactly one legal payment the ``PayBirdFoodDecision`` never reaches
    the agent — ``Engine.ask`` auto-resolves the forced move — yet the food is
    still deducted and the bird placed."""
    eng, player, bird, food, habitat = _prepared_game(seed=2)
    player.food[food] = 1  # exactly the printed token: one legal payment

    def agent[C: decisions.Choice](
        _eng: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:  # pragma: no cover - must not be consulted
        raise AssertionError(
            f"forced play should ask the agent nothing, got {type(decision).__name__}"
        )

    eng.agents = [agent, agent]
    actions.do_play_bird(eng, agent, bird, habitat)

    assert player.food[food] == 0
    assert bird not in player.hand
    assert any(pb.bird is bird for row in player.board.values() for pb in row)
