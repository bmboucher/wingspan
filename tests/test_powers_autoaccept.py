"""Tests for the Burrowing-Owl-style veto (``ROLL_NOT_IN_FEEDER_CACHE``).

The veto still reaches the agent when an opponent has a ``PINK_PREDATOR_FEEDER``
bird, and the accept choice label names the opponent food benefit so the model
understands the tradeoff.
"""

from __future__ import annotations

import random
import typing

from wingspan import cards, decisions, engine, state
from wingspan.engine import powers

# ---------------------------------------------------------------------------
# Shared fixtures


def _new_game(seed: int = 0) -> state.GameState:
    birds, bonuses, goals = cards.load_all()
    return state.new_game(random.Random(seed), birds, bonuses, goals)


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
