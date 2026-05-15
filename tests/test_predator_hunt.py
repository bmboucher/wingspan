"""Unit tests for PREDATOR_HUNT_BY_WINGSPAN bird power."""
from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import cards
from wingspan.cards import EffectKind, Habitat, PowerColor, parse_power
from wingspan.game import Engine
from wingspan.state import PlayedBird, new_game


PREDATOR_HUNT_BIRDS = {
    "Barred Owl": 75,
    "Cooper's Hawk": 75,
    "Golden Eagle": 100,
    "Great Horned Owl": 100,
    "Greater Roadrunner": 50,
    "Northern Harrier": 75,
    "Peregrine Falcon": 100,
    "Red-Shouldered Hawk": 75,
    "Red-Tailed Hawk": 75,
    "Swainson's Hawk": 75,
}


def _find_bird(birds, name):
    for b in birds:
        if b.name == name:
            return b
    raise AssertionError(f"bird {name!r} not found")


def test_parse_predator_hunt_pattern():
    text = "Look at a [card] from the deck. If less than 75cm, tuck it behind this bird. If not, discard it."
    p = parse_power(PowerColor.BROWN, text)
    assert len(p.effects) == 1
    eff = p.effects[0]
    assert eff.kind == EffectKind.PREDATOR_HUNT
    assert eff.amount == 75


def test_all_ten_predator_hunt_birds_parse():
    birds, _, _ = cards.load_all()
    for name, threshold in PREDATOR_HUNT_BIRDS.items():
        b = _find_bird(birds, name)
        effs = [e for e in b.power.effects if e.kind == EffectKind.PREDATOR_HUNT]
        assert effs, f"{name} missing PREDATOR_HUNT (got {[e.kind for e in b.power.effects]})"
        assert effs[0].amount == threshold


def test_predator_hunt_success_and_failure():
    birds, bonuses, goals = cards.load_all()
    state = new_game(random.Random(42), birds, bonuses, goals)
    eng = Engine(state)

    hawk = _find_bird(birds, "Cooper's Hawk")  # 75cm threshold
    p = state.players[0]
    state.current_player = 0
    pb = PlayedBird(bird=hawk)
    p.board[Habitat.FOREST].append(pb)
    eff = next(e for e in hawk.power.effects if e.kind == EffectKind.PREDATOR_HUNT)

    small = next(b for b in birds if b.wingspan_cm and b.wingspan_cm < 75)
    state.bird_deck.append(small)
    tucked_before = pb.tucked_cards
    deck_before = len(state.bird_deck)
    eng._apply_effect(lambda *_: None, p, pb, Habitat.FOREST, eff, "activate")
    assert pb.tucked_cards == tucked_before + 1
    assert pb.predator_succeeded_this_turn is True
    assert len(state.bird_deck) == deck_before - 1

    pb.predator_succeeded_this_turn = False
    big = next(b for b in birds if b.wingspan_cm and b.wingspan_cm >= 75)
    state.bird_deck.append(big)
    tucked_before = pb.tucked_cards
    discard_before = len(state.bird_discard)
    eng._apply_effect(lambda *_: None, p, pb, Habitat.FOREST, eff, "activate")
    assert pb.tucked_cards == tucked_before
    assert pb.predator_succeeded_this_turn is False
    assert len(state.bird_discard) == discard_before + 1


def test_predator_succeeded_flag_resets_on_turn_start():
    birds, bonuses, goals = cards.load_all()
    state = new_game(random.Random(7), birds, bonuses, goals)
    hawk = _find_bird(birds, "Cooper's Hawk")
    pb0 = PlayedBird(bird=hawk, predator_succeeded_this_turn=True)
    pb1 = PlayedBird(bird=hawk, predator_succeeded_this_turn=True)
    state.players[0].board[Habitat.FOREST].append(pb0)
    state.players[1].board[Habitat.FOREST].append(pb1)
    for q in state.players:
        for row in q.board.values():
            for pb in row:
                pb.predator_succeeded_this_turn = False
    assert pb0.predator_succeeded_this_turn is False
    assert pb1.predator_succeeded_this_turn is False
