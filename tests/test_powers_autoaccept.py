"""Tests for the strictly-free exchange auto-accept infrastructure.

Two concerns are exercised here:

1.  ``offer_exchange_or_auto_accept`` in ``dispatch.py`` auto-accepts when the
    ``PayCostChoice`` ledger is strictly free (zero payment, zero opponent gain,
    positive own gain), logging the auto-accept without consulting the agent.

2.  The Burrowing-Owl-style veto (``ROLL_NOT_IN_FEEDER_CACHE``) still reaches the
    agent when an opponent has a ``PINK_PREDATOR_FEEDER`` bird, and the accept
    choice label now names the opponent food benefit so the model understands the
    tradeoff.
"""

from __future__ import annotations

import os
import random
import sys
import typing

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import cards, decisions, engine, state
from wingspan.engine import powers
from wingspan.engine.powers import dispatch

# ---------------------------------------------------------------------------
# Shared fixtures


def _new_game(seed: int = 0) -> state.GameState:
    birds, bonuses, goals = cards.load_all()
    return state.new_game(random.Random(seed), birds, bonuses, goals)


def _no_agent[C: decisions.Choice](
    _eng: engine.Engine,
    decision: decisions.Decision[C],
) -> C:
    """Raises on any consultation — used when the agent must not be queried."""
    raise AssertionError(
        f"agent should not be consulted (got {type(decision).__name__})"
    )


def _make_feeder_bird() -> cards.Bird:
    """Synthesise a pink bird with a ``PINK_PREDATOR_FEEDER`` power."""
    birds, _, _ = cards.load_all()
    text = (
        "When another player's [predator] succeeds, gain 1 [die] from the birdfeeder."
    )
    template = next(bird for bird in birds if bird.color == cards.PowerColor.PINK)
    return template.model_copy(
        update={
            "power": cards.parse_power(cards.PowerColor.PINK, text),
            "raw_power_text": text,
        }
    )


# ---------------------------------------------------------------------------
# Tests for _is_strictly_free


def test_is_strictly_free_free_gain():
    """A choice with only gained_food_count set is strictly free."""
    choice = decisions.PayCostChoice(label="gain 1 food", gained_food_count=1)
    assert dispatch.is_strictly_free(choice) is True


def test_is_strictly_free_free_cache():
    """A choice with only gained_cache_count set is strictly free."""
    choice = decisions.PayCostChoice(label="cache food", gained_cache_count=1)
    assert dispatch.is_strictly_free(choice) is True


def test_is_strictly_free_free_egg():
    """A choice with only gained_egg_count is strictly free."""
    choice = decisions.PayCostChoice(label="lay egg", gained_egg_count=1)
    assert dispatch.is_strictly_free(choice) is True


def test_is_not_strictly_free_has_payment_food():
    """Paying a named food makes the exchange NOT strictly free."""
    choice = decisions.PayCostChoice(
        label="discard seed for cache",
        paid_food=cards.Food.SEED,
        gained_cache_count=1,
    )
    assert dispatch.is_strictly_free(choice) is False


def test_is_not_strictly_free_has_opp_gain():
    """An opponent gaining food makes the exchange NOT strictly free."""
    choice = decisions.PayCostChoice(
        label="roll dice (opp gains)",
        gained_cache_count=1,
        opp_gained_food_count=1,
    )
    assert dispatch.is_strictly_free(choice) is False


def test_is_not_strictly_free_no_own_gain():
    """Zero gains on all fields means the exchange is NOT strictly free."""
    choice = decisions.PayCostChoice(label="nothing")
    assert dispatch.is_strictly_free(choice) is False


# ---------------------------------------------------------------------------
# Tests for offer_exchange_or_auto_accept


def test_offer_exchange_or_auto_accept_strictly_free_never_consults_agent():
    """When the ledger is strictly free, auto-accept does not call the agent."""
    gs = _new_game()
    eng = engine.Engine(gs, agents=[_no_agent, _no_agent])
    player = gs.players[0]

    free_choice = decisions.PayCostChoice(label="gain 1 food", gained_food_count=1)
    accepted = dispatch.offer_exchange_or_auto_accept(
        eng, _no_agent, player, "prompt", free_choice
    )

    assert accepted is True


def test_offer_exchange_or_auto_accept_strictly_free_logs_auto_accept():
    """Auto-accepted exchanges appear in the log as 'auto-accept: ...'."""
    gs = _new_game()
    eng = engine.Engine(gs, agents=[_no_agent, _no_agent])
    player = gs.players[0]

    free_choice = decisions.PayCostChoice(label="gain 1 food", gained_food_count=1)
    dispatch.offer_exchange_or_auto_accept(
        eng, _no_agent, player, "prompt", free_choice
    )

    auto_accept_lines = [line for line in gs.log if "auto-accept:" in line]
    assert auto_accept_lines, "no auto-accept log line found"
    assert "gain 1 food" in auto_accept_lines[0]
    assert "no cost" in auto_accept_lines[0]


def test_offer_exchange_or_auto_accept_with_opp_gain_consults_agent():
    """When the ledger has opponent gain, the agent IS consulted via the veto gate."""
    gs = _new_game()
    player = gs.players[0]

    decisions_seen: list[type] = []

    def recording_agent[C: decisions.Choice](
        _eng: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        decisions_seen.append(type(decision))
        # Accept the gate — return the first non-skip choice.
        for choice in decision.choices:
            if not isinstance(choice, decisions.SkipChoice):
                return choice
        return decision.choices[0]

    eng = engine.Engine(gs, agents=[recording_agent, recording_agent])

    non_free_choice = decisions.PayCostChoice(
        label="roll dice (opponent gains)",
        gained_cache_count=1,
        opp_gained_food_count=1,
    )
    accepted = dispatch.offer_exchange_or_auto_accept(
        eng, recording_agent, player, "prompt", non_free_choice
    )

    assert accepted is True
    assert decisions.AcceptExchangeDecision in decisions_seen


def test_offer_exchange_or_auto_accept_with_opp_gain_skip():
    """When the ledger has opponent gain and the agent skips, returns False."""
    gs = _new_game()
    player = gs.players[0]

    def skip_agent[C: decisions.Choice](
        _eng: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        for choice in decision.choices:
            if isinstance(choice, decisions.SkipChoice):
                return choice
        return decision.choices[0]

    eng = engine.Engine(gs, agents=[skip_agent, skip_agent])

    non_free_choice = decisions.PayCostChoice(
        label="roll dice (opponent gains)",
        gained_cache_count=1,
        opp_gained_food_count=1,
    )
    accepted = dispatch.offer_exchange_or_auto_accept(
        eng, skip_agent, player, "prompt", non_free_choice
    )

    assert accepted is False


# ---------------------------------------------------------------------------
# Tests for Burrowing Owl veto label transparency


def _make_roll_not_in_feeder_bird() -> cards.Bird:
    """Synthesise a bird with a ``ROLL_NOT_IN_FEEDER_CACHE`` effect on rodent."""
    birds, _, _ = cards.load_all()
    text = "Roll any [die] not in the birdfeeder. If you roll a [rodent], cache it on this bird."
    burrowing_owl = next((bird for bird in birds if bird.name == "Burrowing Owl"), None)
    if burrowing_owl is not None and any(
        eff.kind == cards.EffectKind.ROLL_NOT_IN_FEEDER_CACHE
        for eff in burrowing_owl.power.effects
    ):
        return burrowing_owl
    # Fall back to synthesising a bird with the power.
    template = next(bird for bird in birds if bird.color == cards.PowerColor.BROWN)
    return template.model_copy(
        update={
            "color": cards.PowerColor.BROWN,
            "raw_power_text": text,
            "power": cards.parse_power(cards.PowerColor.BROWN, text),
        }
    )


def test_roll_not_in_feeder_no_feeder_bird_no_gate():
    """Without opposing PINK_PREDATOR_FEEDER birds, the veto gate is NOT offered
    and the roll proceeds immediately."""
    gs = _new_game(seed=7)

    # Clear all opponent birds so there are no predator-feeder reactors.
    for habitat in cards.ALL_HABITATS:
        gs.players[1].board[habitat].clear()

    # Put all 7 dice outside the feeder (feeder empty → max dice out).
    gs.birdfeeder.counts = state.FoodPool()
    gs.birdfeeder.choice_dice = 0

    player = gs.players[0]
    bird = _make_roll_not_in_feeder_bird()
    eff = next(
        effect
        for effect in bird.power.effects
        if effect.kind == cards.EffectKind.ROLL_NOT_IN_FEEDER_CACHE
    )
    pb = state.PlayedBird(bird=bird)

    gate_seen: list[bool] = []

    def tracking_agent[C: decisions.Choice](
        _eng: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        if isinstance(decision, decisions.AcceptExchangeDecision):
            gate_seen.append(True)
            # Return first non-skip choice from the narrowed AcceptExchangeDecision.
            for choice in decision.choices:
                if not isinstance(choice, decisions.SkipChoice):
                    return typing.cast(C, choice)
            return typing.cast(C, decision.choices[0])
        # For all other decision types use the generic choices sequence.
        for choice in decision.choices:
            if not isinstance(choice, decisions.SkipChoice):
                return choice
        return decision.choices[0]

    eng = engine.Engine(gs, agents=[tracking_agent, tracking_agent])
    powers.apply_effect(eng, tracking_agent, player, pb, bird.habitats[0], eff, "play")

    assert not gate_seen, "veto gate should not be offered without feeder birds"


def test_roll_not_in_feeder_with_feeder_bird_gate_label_mentions_opponent():
    """When an opponent has a PINK_PREDATOR_FEEDER bird, the veto IS offered and
    the accept choice label mentions the opponent food benefit."""
    gs = _new_game(seed=7)

    # Give opponent a pink predator-feeder bird.
    feeder_bird = _make_feeder_bird()
    pb_feeder = state.PlayedBird(bird=feeder_bird)
    gs.players[1].board[feeder_bird.habitats[0]].append(pb_feeder)

    # Put all 7 dice outside the feeder.
    gs.birdfeeder.counts = state.FoodPool()
    gs.birdfeeder.choice_dice = 0

    player = gs.players[0]
    bird = _make_roll_not_in_feeder_bird()
    eff = next(
        effect
        for effect in bird.power.effects
        if effect.kind == cards.EffectKind.ROLL_NOT_IN_FEEDER_CACHE
    )
    pb = state.PlayedBird(bird=bird)

    accept_labels_seen: list[str] = []

    def recording_agent[C: decisions.Choice](
        _eng: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        if isinstance(decision, decisions.AcceptExchangeDecision):
            for choice in decision.choices:
                if isinstance(choice, decisions.PayCostChoice):
                    accept_labels_seen.append(choice.label)
            # Return first non-skip choice from the narrowed AcceptExchangeDecision.
            for choice in decision.choices:
                if not isinstance(choice, decisions.SkipChoice):
                    return typing.cast(C, choice)
            return typing.cast(C, decision.choices[0])
        # For all other decision types use the generic choices sequence.
        for choice in decision.choices:
            if not isinstance(choice, decisions.SkipChoice):
                return choice
        return decision.choices[0]

    eng = engine.Engine(gs, agents=[recording_agent, recording_agent])
    powers.apply_effect(eng, recording_agent, player, pb, bird.habitats[0], eff, "play")

    assert accept_labels_seen, "veto gate was not presented"
    label = accept_labels_seen[0]
    assert (
        "opponent may gain food" in label
    ), f"label does not mention opponent benefit: {label!r}"
