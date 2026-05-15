"""Unit tests for the five misc-unique bird powers.

Each test builds a minimal engine, patches a played bird with the target
power text, and invokes ``Engine._dispatch_power`` directly so we can
assert engine behaviour for the new EffectKinds without driving a full
turn sequence.
"""
from __future__ import annotations

import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan.actions import Choice, Decision, DecisionType
from wingspan.cards import (
    ALL_FOODS, EffectKind, Food, Habitat, PowerColor, parse_power,
)
from wingspan.game import Engine
from wingspan.state import PlayedBird, new_game


def _make_engine_with_bird(
    power_text: str,
    color: PowerColor = PowerColor.WHITE,
    agents=None,
) -> tuple[Engine, PlayedBird]:
    """Build a near-empty engine and stage a played bird carrying ``power_text``."""
    from wingspan import cards
    birds, bonuses, goals = cards.load_all()
    rng = random.Random(0)
    state = new_game(rng, birds, bonuses, goals)
    eng = Engine(state, agents=agents) if agents is not None else Engine(state)
    template = next(b for b in birds if b.color == color)
    bird = type(template)(
        **{
            **{f.name: getattr(template, f.name) for f in template.__dataclass_fields__.values()},
            "color": color,
            "raw_power_text": power_text,
            "power": parse_power(color, power_text),
        }
    )
    pb = PlayedBird(bird=bird)
    return eng, pb


# ---------------------------------------------------------------------------
# Parser tests

def test_parser_draw_from_tray_all():
    p = parse_power(PowerColor.WHITE, "Draw the 3 face-up [card] in the bird tray.")
    assert [e.kind for e in p.effects] == [EffectKind.DRAW_FROM_TRAY_ALL]


def test_parser_trade_wild_food():
    p = parse_power(PowerColor.BROWN, "Trade 1 [wild] for any other type from the supply.")
    assert [e.kind for e in p.effects] == [EffectKind.TRADE_WILD_FOOD]


def test_parser_fewest_forest_gains_die():
    p = parse_power(
        PowerColor.BROWN,
        "Player(s) with the fewest birds in their [forest] gain 1 [die] from birdfeeder.",
    )
    assert [e.kind for e in p.effects] == [EffectKind.FEWEST_FOREST_GAINS_DIE]


def test_parser_play_additional_bird_here():
    p = parse_power(
        PowerColor.WHITE,
        "Play an additional bird in this bird's habitat. Pay its normal cost.",
    )
    assert [e.kind for e in p.effects] == [EffectKind.PLAY_ADDITIONAL_BIRD_HERE]


def test_parser_draw_n_plus_one_draft():
    p = parse_power(
        PowerColor.WHITE,
        "Draw [card] equal to the number of players +1. Starting with you and "
        "proceeding clockwise, each player selects 1 of those cards and places "
        "it in their hand. You keep the extra card.",
    )
    assert [e.kind for e in p.effects] == [EffectKind.DRAW_N_PLUS_ONE_DRAFT]


# ---------------------------------------------------------------------------
# Brant — DRAW_FROM_TRAY_ALL

def test_draw_from_tray_all_takes_all_three_and_refills():
    eng, pb = _make_engine_with_bird("Draw the 3 face-up [card] in the bird tray.")
    state = eng.state
    state.current_player = 0
    p = state.me()
    p.hand = []
    # Force a deterministic tray of 3 known birds.
    original_tray_names = [b.name for b in state.tray]
    deck_before = len(state.bird_deck)
    eng._dispatch_power(lambda _e, _d: None, p, pb, Habitat.WETLAND, "play")
    assert [b.name for b in p.hand] == original_tray_names
    assert len(state.tray) == 3  # refilled
    # 3 cards moved from deck to tray to refill.
    assert len(state.bird_deck) == deck_before - 3


# ---------------------------------------------------------------------------
# Green Heron — TRADE_WILD_FOOD

def test_trade_wild_food_swaps_one_food():
    eng, pb = _make_engine_with_bird(
        "Trade 1 [wild] for any other type from the supply.",
        color=PowerColor.BROWN,
    )
    state = eng.state
    state.current_player = 0
    p = state.me()
    for f in ALL_FOODS:
        p.food[f] = 0
    p.food[Food.SEED] = 2
    state.food_supply[Food.FRUIT] = 5

    asked = []

    def agent(_e: Engine, d: Decision) -> Choice:
        asked.append(d)
        if not asked or len(asked) == 1:
            # first prompt: pick which food to discard
            return next(c for c in d.choices if c.payload == Food.SEED)
        return next(c for c in d.choices if c.payload == Food.FRUIT)

    eng._dispatch_power(agent, p, pb, Habitat.WETLAND, "activate")
    assert p.food[Food.SEED] == 1
    assert p.food[Food.FRUIT] == 1


def test_trade_wild_food_skip_does_nothing():
    eng, pb = _make_engine_with_bird(
        "Trade 1 [wild] for any other type from the supply.",
        color=PowerColor.BROWN,
    )
    state = eng.state
    state.current_player = 0
    p = state.me()
    for f in ALL_FOODS:
        p.food[f] = 0
    p.food[Food.SEED] = 1
    food_before = dict(p.food)

    def agent(_e: Engine, d: Decision) -> Choice:
        return next(c for c in d.choices if c.payload is None)

    eng._dispatch_power(agent, p, pb, Habitat.WETLAND, "activate")
    assert p.food == food_before


def test_trade_wild_food_no_food_no_op():
    eng, pb = _make_engine_with_bird(
        "Trade 1 [wild] for any other type from the supply.",
        color=PowerColor.BROWN,
    )
    state = eng.state
    state.current_player = 0
    p = state.me()
    for f in ALL_FOODS:
        p.food[f] = 0

    def agent(_e: Engine, d: Decision) -> Choice:  # pragma: no cover
        pytest.fail("should not be consulted when player has no food")

    eng._dispatch_power(agent, p, pb, Habitat.WETLAND, "activate")


# ---------------------------------------------------------------------------
# Hermit Thrush — FEWEST_FOREST_GAINS_DIE

def test_fewest_forest_gains_die_only_min_player_gets_food():
    eng, pb = _make_engine_with_bird(
        "Player(s) with the fewest birds in their [forest] gain 1 [die] from birdfeeder.",
        color=PowerColor.BROWN,
    )
    state = eng.state
    state.current_player = 0
    # Stash agents so the engine can ask non-active players.
    p0, p1 = state.players
    # Give P0 zero forest birds, P1 one forest bird (so P0 is "fewest").
    p1.board[Habitat.FOREST].append(PlayedBird(bird=pb.bird))
    # Ensure birdfeeder has a known food.
    for f in ALL_FOODS:
        state.birdfeeder.counts[f] = 0
    state.birdfeeder.counts[Food.SEED] = 3
    food_before = {q.id: dict(q.food) for q in state.players}

    def agent(_e: Engine, d: Decision) -> Choice:
        return next(c for c in d.choices if c.payload == Food.SEED)

    eng.agents = [agent, agent]
    eng._dispatch_power(agent, p0, pb, Habitat.FOREST, "activate")
    assert p0.food[Food.SEED] == food_before[0][Food.SEED] + 1
    assert p1.food[Food.SEED] == food_before[1][Food.SEED]
    assert state.birdfeeder.counts[Food.SEED] == 2


def test_fewest_forest_gains_die_ties_each_gets_one():
    eng, pb = _make_engine_with_bird(
        "Player(s) with the fewest birds in their [forest] gain 1 [die] from birdfeeder.",
        color=PowerColor.BROWN,
    )
    state = eng.state
    state.current_player = 0
    p0, p1 = state.players
    # Both have zero forest birds -> tie -> both gain a die.
    for f in ALL_FOODS:
        state.birdfeeder.counts[f] = 0
    state.birdfeeder.counts[Food.SEED] = 5
    food_before = {q.id: dict(q.food) for q in state.players}

    def agent(_e: Engine, d: Decision) -> Choice:
        return next(c for c in d.choices if c.payload == Food.SEED)

    eng.agents = [agent, agent]
    eng._dispatch_power(agent, p0, pb, Habitat.FOREST, "activate")
    assert p0.food[Food.SEED] == food_before[0][Food.SEED] + 1
    assert p1.food[Food.SEED] == food_before[1][Food.SEED] + 1


def test_fewest_forest_gains_die_empty_feeder_no_op():
    eng, pb = _make_engine_with_bird(
        "Player(s) with the fewest birds in their [forest] gain 1 [die] from birdfeeder.",
        color=PowerColor.BROWN,
    )
    state = eng.state
    state.current_player = 0
    for f in ALL_FOODS:
        state.birdfeeder.counts[f] = 0

    def agent(_e: Engine, d: Decision) -> Choice:  # pragma: no cover
        pytest.fail("should not be consulted when feeder is empty")

    eng.agents = [agent, agent]
    eng._dispatch_power(agent, state.me(), pb, Habitat.FOREST, "activate")


# ---------------------------------------------------------------------------
# House Wren — PLAY_ADDITIONAL_BIRD_HERE

def test_play_additional_bird_here_grants_extra_play():
    eng, pb = _make_engine_with_bird(
        "Play an additional bird in this bird's habitat. Pay its normal cost.",
    )
    state = eng.state
    state.current_player = 0
    p = state.me()
    before = eng.turn_state.extra_plays
    eng._dispatch_power(lambda _e, _d: None, p, pb, Habitat.FOREST, "play")
    assert eng.turn_state.extra_plays == before + 1


# ---------------------------------------------------------------------------
# American Oystercatcher — DRAW_N_PLUS_ONE_DRAFT

def test_draw_n_plus_one_draft_each_player_picks_one_active_keeps_rest():
    eng, pb = _make_engine_with_bird(
        "Draw [card] equal to the number of players +1. Starting with you and "
        "proceeding clockwise, each player selects 1 of those cards and places "
        "it in their hand. You keep the extra card.",
    )
    state = eng.state
    state.current_player = 0
    p0, p1 = state.players
    p0.hand = []
    p1.hand = []
    deck_before = len(state.bird_deck)

    def agent(_e: Engine, d: Decision) -> Choice:
        # Always pick the first offered card.
        return d.choices[0]

    eng.agents = [agent, agent]
    eng._dispatch_power(agent, p0, pb, Habitat.WETLAND, "play")

    # Drew 3 cards, p0 keeps 2 (own pick + leftover), p1 keeps 1.
    assert len(p0.hand) == 2
    assert len(p1.hand) == 1
    assert len(state.bird_deck) == deck_before - 3


def test_draw_n_plus_one_draft_empty_deck_no_op():
    eng, pb = _make_engine_with_bird(
        "Draw [card] equal to the number of players +1. Starting with you and "
        "proceeding clockwise, each player selects 1 of those cards and places "
        "it in their hand. You keep the extra card.",
    )
    state = eng.state
    state.current_player = 0
    p0 = state.me()
    p0.hand = []
    state.players[1].hand = []
    state.bird_deck = []
    state.bird_discard = []

    def agent(_e: Engine, d: Decision) -> Choice:  # pragma: no cover
        pytest.fail("should not be consulted when deck is empty")

    eng.agents = [agent, agent]
    eng._dispatch_power(agent, p0, pb, Habitat.WETLAND, "play")
    assert p0.hand == []
    assert state.players[1].hand == []


# ---------------------------------------------------------------------------
# Specific bird wiring

def test_target_birds_parse_to_expected_kinds():
    from wingspan import cards
    birds, _, _ = cards.load_all()
    by_name = {b.name: b for b in birds}
    expected = {
        "Brant": EffectKind.DRAW_FROM_TRAY_ALL,
        "Green Heron": EffectKind.TRADE_WILD_FOOD,
        "Hermit Thrush": EffectKind.FEWEST_FOREST_GAINS_DIE,
        "House Wren": EffectKind.PLAY_ADDITIONAL_BIRD_HERE,
        "American Oystercatcher": EffectKind.DRAW_N_PLUS_ONE_DRAFT,
    }
    for name, kind in expected.items():
        kinds = [e.kind for e in by_name[name].power.effects]
        assert kind in kinds, f"{name}: {kinds}"
