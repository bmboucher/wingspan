"""Unit tests for individual bird-power effect dispatch.

These build minimal game states and invoke ``Engine._dispatch_power`` directly
so we can assert engine behaviour for a single Effect without driving a full
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
    BonusCard, Effect, EffectKind, Power, PowerColor, parse_power,
)
from wingspan.game import Engine
from wingspan.state import PlayedBird, new_game


def _fake_bonus(i: int) -> BonusCard:
    return BonusCard(id=i, name=f"Bonus{i}", condition="", explanatory="", vp_text="")


def _make_engine_with_bird(power_text: str, bonus_deck: list[BonusCard]) -> tuple[Engine, PlayedBird]:
    """Build a near-empty engine and stage a played bird carrying ``power_text``."""
    from wingspan import cards
    birds, bonuses, goals = cards.load_all()
    rng = random.Random(0)
    state = new_game(rng, birds, bonuses, goals)
    state.bonus_deck = list(bonus_deck)
    state.bonus_discard = []
    eng = Engine(state)
    template = next(b for b in birds if b.color == PowerColor.WHITE)
    bird = type(template)(  # dataclass copy with overridden power
        **{
            **{f.name: getattr(template, f.name) for f in template.__dataclass_fields__.values()},
            "raw_power_text": power_text,
            "power": parse_power(PowerColor.WHITE, power_text),
        }
    )
    pb = PlayedBird(bird=bird)
    return eng, pb


def test_parser_recognises_draw_bonus_keep_one():
    p = parse_power(PowerColor.WHITE, "Draw 2 new bonus cards and keep 1.")
    assert len(p.effects) == 1
    eff = p.effects[0]
    assert eff.kind == EffectKind.DRAW_BONUS_KEEP_ONE
    assert eff.amount == 2
    assert eff.extra == (1,)


def test_draw_bonus_keep_one_keeps_one_discards_rest():
    fakes = [_fake_bonus(i) for i in range(5)]
    eng, pb = _make_engine_with_bird("Draw 2 new bonus cards and keep 1.", fakes)
    player = eng.state.players[0]
    eng.state.current_player = 0
    before_hand = len(player.bonus_cards)

    def agent(_engine: Engine, decision: Decision) -> Choice:
        assert decision.type == DecisionType.BIRD_POWER_PICK_BONUS_TO_KEEP
        return decision.choices[0]

    eng._dispatch_power(agent, player, pb, pb.bird.habitats[0], "play")

    assert len(player.bonus_cards) == before_hand + 1
    assert len(eng.state.bonus_discard) == 1
    # Two cards left the deck (drew 2 of 5).
    assert len(eng.state.bonus_deck) == 3


def test_draw_bonus_keep_one_reshuffles_discard_when_deck_empty():
    eng, pb = _make_engine_with_bird("Draw 2 new bonus cards and keep 1.", [])
    eng.state.bonus_deck = []
    eng.state.bonus_discard = [_fake_bonus(7), _fake_bonus(8)]
    player = eng.state.players[0]
    eng.state.current_player = 0

    def agent(_engine: Engine, decision: Decision) -> Choice:
        return decision.choices[0]

    eng._dispatch_power(agent, player, pb, pb.bird.habitats[0], "play")

    # Drew 2 from reshuffled discard, kept 1, discarded 1.
    assert len(player.bonus_cards) == 1
    assert len(eng.state.bonus_discard) == 1
    assert eng.state.bonus_deck == []


def test_draw_bonus_keep_one_handles_empty_bonus_pool():
    eng, pb = _make_engine_with_bird("Draw 2 new bonus cards and keep 1.", [])
    eng.state.bonus_deck = []
    eng.state.bonus_discard = []
    player = eng.state.players[0]
    eng.state.current_player = 0

    def agent(_engine: Engine, decision: Decision) -> Choice:  # pragma: no cover
        pytest.fail("agent should not be consulted when no bonus cards available")

    eng._dispatch_power(agent, player, pb, pb.bird.habitats[0], "play")

    assert player.bonus_cards == []
    assert eng.state.bonus_deck == []
    assert eng.state.bonus_discard == []


def test_all_fifteen_target_birds_parse_to_draw_bonus_keep_one():
    from wingspan import cards
    targets = {
        "Atlantic Puffin", "Bell's Vireo", "California Condor", "Cassin's Finch",
        "Cerulean Warbler", "Chestnut-Collared Longspur", "Greater Prairie-Chicken",
        "King Rail", "Painted Bunting", "Red-Cockaded Woodpecker", "Roseate Spoonbill",
        "Spotted Owl", "Sprague's Pipit", "Whooping Crane", "Wood Stork",
    }
    birds, _, _ = cards.load_all()
    by_name = {b.name: b for b in birds}
    for name in targets:
        bird = by_name[name]
        kinds = [e.kind for e in bird.power.effects]
        assert EffectKind.DRAW_BONUS_KEEP_ONE in kinds, f"{name}: kinds={kinds}"
