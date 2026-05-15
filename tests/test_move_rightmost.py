"""Tests for the MOVE_RIGHTMOST_TO_OTHER_HABITAT bird power."""
from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import cards
from wingspan.actions import Choice, Decision, DecisionType
from wingspan.cards import EffectKind, Habitat, PowerColor
from wingspan.game import Engine
from wingspan.state import PlayedBird, new_game


def _find_bird(birds, name: str):
    for b in birds:
        if b.name == name:
            return b
    raise KeyError(name)


def _first_choice_agent(_eng, d: Decision) -> Choice:
    """Always picks the first choice (so the MOVE decision picks 'move')."""
    return d.choices[0]


def test_power_text_parsed_for_all_eight_birds():
    birds, _, _ = cards.load_all()
    expected = {
        "Bewick's Wren", "Blue Grosbeak", "Chimney Swift", "Common Nighthawk",
        "Lincoln's Sparrow", "Song Sparrow", "White-Crowned Sparrow",
        "Yellow-Breasted Chat",
    }
    matched = set()
    for b in birds:
        if any(e.kind == EffectKind.MOVE_RIGHTMOST_TO_OTHER_HABITAT
               for e in b.power.effects):
            matched.add(b.name)
    assert expected <= matched, f"missing: {expected - matched}"


def test_move_rightmost_moves_bird_to_other_habitat():
    birds, bonuses, goals = cards.load_all()
    rng = random.Random(0)
    state = new_game(rng, birds, bonuses, goals)
    eng = Engine(state)

    # Find a MOVE bird whose habitats include a destination (most do).
    wren = _find_bird(birds, "Bewick's Wren")
    assert Habitat.FOREST in wren.habitats or Habitat.GRASSLAND in wren.habitats

    p = state.players[0]
    # Clear board
    for h in p.board:
        p.board[h].clear()

    # Place the wren as the (only, hence rightmost) bird in its first habitat.
    src_hab = wren.habitats[0]
    pb = PlayedBird(bird=wren)
    p.board[src_hab].append(pb)

    state.current_player = 0
    eng._dispatch_power(_first_choice_agent, p, pb, src_hab, "activate")

    # It should no longer be in source habitat.
    assert pb not in p.board[src_hab]
    # It must be in some other legal habitat for this bird.
    other_habs = [h for h in wren.habitats if h != src_hab]
    located = [h for h in other_habs if pb in p.board[h]]
    assert len(located) == 1, f"bird not relocated, board={p.board}"


def test_move_rightmost_skips_when_not_rightmost():
    birds, bonuses, goals = cards.load_all()
    rng = random.Random(1)
    state = new_game(rng, birds, bonuses, goals)
    eng = Engine(state)

    wren = _find_bird(birds, "Bewick's Wren")
    p = state.players[0]
    for h in p.board:
        p.board[h].clear()

    src_hab = wren.habitats[0]
    pb_wren = PlayedBird(bird=wren)
    # Pick any other core bird as the "rightmost" filler.
    filler = next(b for b in birds if b is not wren and src_hab in b.habitats)
    pb_filler = PlayedBird(bird=filler)
    p.board[src_hab].append(pb_wren)
    p.board[src_hab].append(pb_filler)

    state.current_player = 0
    eng._dispatch_power(_first_choice_agent, p, pb_wren, src_hab, "activate")

    # Should NOT move (not rightmost).
    assert pb_wren in p.board[src_hab]
    assert pb_filler in p.board[src_hab]
