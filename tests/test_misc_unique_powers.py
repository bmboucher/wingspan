"""Unit tests for the five misc-unique bird powers.

Each test builds a minimal engine, patches a played bird with the target
power text, and invokes ``powers.dispatch_power`` directly so we can
assert engine behaviour for the new EffectKinds without driving a full
turn sequence.
"""

from __future__ import annotations

import os
import random
import sys
import typing

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import cards, decisions, engine, state
from wingspan.engine import powers


def _no_agent[C: decisions.Choice](
    _engine: engine.Engine,
    _decision: decisions.Decision[C],
) -> C:
    """An ``Agent``-typed stub for powers that resolve without a decision; it
    raises if a power unexpectedly consults it."""
    raise AssertionError(
        f"agent should not be consulted (got {type(_decision).__name__})"
    )


def _make_engine_with_bird(
    power_text: str,
    color: cards.PowerColor = cards.PowerColor.WHITE,
    agents: list[engine.Agent] | None = None,
) -> tuple[engine.Engine, state.PlayedBird]:
    """Build a near-empty engine and stage a played bird carrying ``power_text``."""
    birds, bonuses, goals = cards.load_all()
    rng = random.Random(0)
    gs = state.new_game(rng, birds, bonuses, goals)
    eng = engine.Engine(gs, agents=agents) if agents is not None else engine.Engine(gs)
    template = next(candidate for candidate in birds if candidate.color == color)
    bird = template.model_copy(
        update={
            "color": color,
            "raw_power_text": power_text,
            "power": cards.parse_power(color, power_text),
        }
    )
    pb = state.PlayedBird(bird=bird)
    return eng, pb


# ---------------------------------------------------------------------------
# Parser tests


def test_parser_draw_from_tray_all():
    power = cards.parse_power(
        cards.PowerColor.WHITE, "Draw the 3 face-up [card] in the bird tray."
    )
    assert [effect.kind for effect in power.effects] == [
        cards.EffectKind.DRAW_FROM_TRAY_ALL
    ]


def test_parser_trade_wild_food():
    power = cards.parse_power(
        cards.PowerColor.BROWN, "Trade 1 [wild] for any other type from the supply."
    )
    assert [effect.kind for effect in power.effects] == [
        cards.EffectKind.TRADE_WILD_FOOD
    ]


def test_parser_fewest_forest_gains_die():
    power = cards.parse_power(
        cards.PowerColor.BROWN,
        "Player(s) with the fewest birds in their [forest] gain 1 [die] from birdfeeder.",
    )
    assert [effect.kind for effect in power.effects] == [
        cards.EffectKind.FEWEST_FOREST_GAINS_DIE
    ]


def test_parser_play_additional_bird_here():
    power = cards.parse_power(
        cards.PowerColor.WHITE,
        "Play an additional bird in this bird's habitat. Pay its normal cost.",
    )
    assert [effect.kind for effect in power.effects] == [
        cards.EffectKind.PLAY_ADDITIONAL_BIRD_HERE
    ]


def test_parser_draw_n_plus_one_draft():
    power = cards.parse_power(
        cards.PowerColor.WHITE,
        "Draw [card] equal to the number of players +1. Starting with you and "
        "proceeding clockwise, each player selects 1 of those cards and places "
        "it in their hand. You keep the extra card.",
    )
    assert [effect.kind for effect in power.effects] == [
        cards.EffectKind.DRAW_N_PLUS_ONE_DRAFT
    ]


# ---------------------------------------------------------------------------
# Brant — DRAW_FROM_TRAY_ALL


def test_draw_from_tray_all_takes_all_three_and_refills():
    eng, pb = _make_engine_with_bird("Draw the 3 face-up [card] in the bird tray.")
    gs = eng.state
    gs.current_player = 0
    player = gs.me()
    player.hand = []
    # Force a deterministic tray of 3 known birds.
    original_tray_names = [bird.name for bird in gs.tray]
    deck_before = len(gs.bird_deck)
    powers.dispatch_power(eng, _no_agent, player, pb, cards.Habitat.WETLAND, "play")
    assert [bird.name for bird in player.hand] == original_tray_names
    assert len(gs.tray) == 3  # refilled
    # 3 cards moved from deck to tray to refill.
    assert len(gs.bird_deck) == deck_before - 3


# ---------------------------------------------------------------------------
# Green Heron — TRADE_WILD_FOOD


def test_trade_wild_food_swaps_one_food():
    eng, pb = _make_engine_with_bird(
        "Trade 1 [wild] for any other type from the supply.",
        color=cards.PowerColor.BROWN,
    )
    gs = eng.state
    gs.current_player = 0
    player = gs.me()
    for food in cards.ALL_FOODS:
        player.food[food] = 0
    player.food[cards.Food.SEED] = 2
    gs.food_supply[cards.Food.FRUIT] = 5

    asked: list[object] = []

    def agent[C: decisions.Choice](
        _engine: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        asked.append(decision)
        if not asked or len(asked) == 1:
            # first prompt: pick which food to discard
            return typing.cast(
                C,
                next(
                    choice
                    for choice in decision.choices
                    if isinstance(choice, decisions.FoodChoice)
                    and choice.food == cards.Food.SEED
                ),
            )
        return typing.cast(
            C,
            next(
                choice
                for choice in decision.choices
                if isinstance(choice, decisions.FoodChoice)
                and choice.food == cards.Food.FRUIT
            ),
        )

    powers.dispatch_power(eng, agent, player, pb, cards.Habitat.WETLAND, "activate")
    assert player.food[cards.Food.SEED] == 1
    assert player.food[cards.Food.FRUIT] == 1


def test_trade_wild_food_skip_does_nothing():
    eng, pb = _make_engine_with_bird(
        "Trade 1 [wild] for any other type from the supply.",
        color=cards.PowerColor.BROWN,
    )
    gs = eng.state
    gs.current_player = 0
    player = gs.me()
    for food in cards.ALL_FOODS:
        player.food[food] = 0
    player.food[cards.Food.SEED] = 1
    food_before = player.food.as_dict()

    def agent[C: decisions.Choice](
        _engine: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        return typing.cast(
            C,
            next(
                choice
                for choice in decision.choices
                if isinstance(choice, decisions.SkipChoice)
            ),
        )

    powers.dispatch_power(eng, agent, player, pb, cards.Habitat.WETLAND, "activate")
    assert player.food.as_dict() == food_before


def test_trade_wild_food_no_food_no_op():
    eng, pb = _make_engine_with_bird(
        "Trade 1 [wild] for any other type from the supply.",
        color=cards.PowerColor.BROWN,
    )
    gs = eng.state
    gs.current_player = 0
    player = gs.me()
    for food in cards.ALL_FOODS:
        player.food[food] = 0

    def agent[C: decisions.Choice](  # pragma: no cover
        _engine: engine.Engine,
        _decision: decisions.Decision[C],
    ) -> C:
        pytest.fail("should not be consulted when player has no food")

    powers.dispatch_power(eng, agent, player, pb, cards.Habitat.WETLAND, "activate")


# ---------------------------------------------------------------------------
# Hermit Thrush — FEWEST_FOREST_GAINS_DIE


def test_fewest_forest_gains_die_only_min_player_gets_food():
    eng, pb = _make_engine_with_bird(
        "Player(s) with the fewest birds in their [forest] gain 1 [die] from birdfeeder.",
        color=cards.PowerColor.BROWN,
    )
    gs = eng.state
    gs.current_player = 0
    # Stash agents so the engine can ask non-active players.
    p0, p1 = gs.players
    # Give P0 zero forest birds, P1 one forest bird (so P0 is "fewest").
    p1.board[cards.Habitat.FOREST].append(state.PlayedBird(bird=pb.bird))
    # Ensure birdfeeder has a known food.
    for food in cards.ALL_FOODS:
        gs.birdfeeder.counts[food] = 0
    gs.birdfeeder.counts[cards.Food.SEED] = 3
    food_before = {
        other_player.id: other_player.food.as_dict() for other_player in gs.players
    }

    def agent[C: decisions.Choice](
        _engine: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        return typing.cast(
            C,
            next(
                choice
                for choice in decision.choices
                if isinstance(choice, decisions.FoodChoice)
                and choice.food == cards.Food.SEED
            ),
        )

    eng.agents = [agent, agent]
    powers.dispatch_power(eng, agent, p0, pb, cards.Habitat.FOREST, "activate")
    assert p0.food[cards.Food.SEED] == food_before[0][cards.Food.SEED] + 1
    assert p1.food[cards.Food.SEED] == food_before[1][cards.Food.SEED]
    assert gs.birdfeeder.counts[cards.Food.SEED] == 2


def test_fewest_forest_gains_die_ties_each_gets_one():
    eng, pb = _make_engine_with_bird(
        "Player(s) with the fewest birds in their [forest] gain 1 [die] from birdfeeder.",
        color=cards.PowerColor.BROWN,
    )
    gs = eng.state
    gs.current_player = 0
    p0, p1 = gs.players
    # Both have zero forest birds -> tie -> both gain a die.
    for food in cards.ALL_FOODS:
        gs.birdfeeder.counts[food] = 0
    gs.birdfeeder.counts[cards.Food.SEED] = 5
    food_before = {
        other_player.id: other_player.food.as_dict() for other_player in gs.players
    }

    def agent[C: decisions.Choice](
        _engine: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        return typing.cast(
            C,
            next(
                choice
                for choice in decision.choices
                if isinstance(choice, decisions.FoodChoice)
                and choice.food == cards.Food.SEED
            ),
        )

    eng.agents = [agent, agent]
    powers.dispatch_power(eng, agent, p0, pb, cards.Habitat.FOREST, "activate")
    assert p0.food[cards.Food.SEED] == food_before[0][cards.Food.SEED] + 1
    assert p1.food[cards.Food.SEED] == food_before[1][cards.Food.SEED] + 1


def test_fewest_forest_gains_die_empty_feeder_no_op():
    eng, pb = _make_engine_with_bird(
        "Player(s) with the fewest birds in their [forest] gain 1 [die] from birdfeeder.",
        color=cards.PowerColor.BROWN,
    )
    gs = eng.state
    gs.current_player = 0
    for food in cards.ALL_FOODS:
        gs.birdfeeder.counts[food] = 0

    def agent[C: decisions.Choice](  # pragma: no cover
        _engine: engine.Engine,
        _decision: decisions.Decision[C],
    ) -> C:
        pytest.fail("should not be consulted when feeder is empty")

    eng.agents = [agent, agent]
    powers.dispatch_power(eng, agent, gs.me(), pb, cards.Habitat.FOREST, "activate")


# ---------------------------------------------------------------------------
# House Wren — PLAY_ADDITIONAL_BIRD_HERE


def test_play_additional_bird_here_grants_extra_play():
    eng, pb = _make_engine_with_bird(
        "Play an additional bird in this bird's habitat. Pay its normal cost.",
    )
    gs = eng.state
    gs.current_player = 0
    player = gs.me()
    before = gs.turn_extra_plays
    powers.dispatch_power(eng, _no_agent, player, pb, cards.Habitat.FOREST, "play")
    assert gs.turn_extra_plays == before + 1


# ---------------------------------------------------------------------------
# American Oystercatcher — DRAW_N_PLUS_ONE_DRAFT


def test_draw_n_plus_one_draft_each_player_picks_one_active_keeps_rest():
    eng, pb = _make_engine_with_bird(
        "Draw [card] equal to the number of players +1. Starting with you and "
        "proceeding clockwise, each player selects 1 of those cards and places "
        "it in their hand. You keep the extra card.",
    )
    gs = eng.state
    gs.current_player = 0
    p0, p1 = gs.players
    p0.hand = []
    p1.hand = []
    deck_before = len(gs.bird_deck)

    def agent[C: decisions.Choice](
        _engine: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        # Always pick the first offered card.
        return decision.choices[0]

    eng.agents = [agent, agent]
    powers.dispatch_power(eng, agent, p0, pb, cards.Habitat.WETLAND, "play")

    # Drew 3 cards, p0 keeps 2 (own pick + leftover), p1 keeps 1.
    assert len(p0.hand) == 2
    assert len(p1.hand) == 1
    assert len(gs.bird_deck) == deck_before - 3


def test_draw_n_plus_one_draft_empty_deck_no_op():
    eng, pb = _make_engine_with_bird(
        "Draw [card] equal to the number of players +1. Starting with you and "
        "proceeding clockwise, each player selects 1 of those cards and places "
        "it in their hand. You keep the extra card.",
    )
    gs = eng.state
    gs.current_player = 0
    p0 = gs.me()
    p0.hand = []
    gs.players[1].hand = []
    gs.bird_deck = []
    gs.bird_discard = []

    def agent[C: decisions.Choice](  # pragma: no cover
        _engine: engine.Engine,
        _decision: decisions.Decision[C],
    ) -> C:
        pytest.fail("should not be consulted when deck is empty")

    eng.agents = [agent, agent]
    powers.dispatch_power(eng, agent, p0, pb, cards.Habitat.WETLAND, "play")
    assert p0.hand == []
    assert gs.players[1].hand == []


# ---------------------------------------------------------------------------
# Specific bird wiring


def test_target_birds_parse_to_expected_kinds():
    from wingspan import cards

    birds, _, _ = cards.load_all()
    by_name = {bird.name: bird for bird in birds}
    expected = {
        "Brant": cards.EffectKind.DRAW_FROM_TRAY_ALL,
        "Green Heron": cards.EffectKind.TRADE_WILD_FOOD,
        "Hermit Thrush": cards.EffectKind.FEWEST_FOREST_GAINS_DIE,
        "House Wren": cards.EffectKind.PLAY_ADDITIONAL_BIRD_HERE,
        "American Oystercatcher": cards.EffectKind.DRAW_N_PLUS_ONE_DRAFT,
    }
    for name, kind in expected.items():
        kinds = [effect.kind for effect in by_name[name].power.effects]
        assert kind in kinds, f"{name}: {kinds}"
