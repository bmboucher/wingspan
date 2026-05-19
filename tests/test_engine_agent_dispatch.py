"""Unit tests for Engine.agents and Engine.agent_for plumbing."""

from __future__ import annotations

import copy
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import cards, engine, state


def _fresh_state():
    birds, bonuses, goals = cards.load_all()
    return state.new_game(random.Random(0), birds, bonuses, goals)


def test_engine_init_without_agents_has_empty_list():
    eng = engine.Engine(_fresh_state())
    assert eng.agents == []


def test_engine_init_with_agents_indexes_by_player_id():
    gs = _fresh_state()
    sentinel_p0 = lambda eng, d: None
    sentinel_p1 = lambda eng, d: None
    eng = engine.Engine(gs, agents=[sentinel_p0, sentinel_p1])
    assert eng.agent_for(gs.players[0]) is sentinel_p0
    assert eng.agent_for(gs.players[1]) is sentinel_p1


def test_engine_agent_for_raises_when_unset():
    gs = _fresh_state()
    eng = engine.Engine(gs)
    with pytest.raises(RuntimeError, match="No agent registered"):
        eng.agent_for(gs.players[0])


def test_engine_init_rejects_length_mismatch():
    gs = _fresh_state()  # 2 players
    with pytest.raises(ValueError, match="does not match players count"):
        engine.Engine(gs, agents=[lambda eng, d: None])  # 1 agent


def test_deepcopy_preserves_agent_routing():
    """Deepcopy of Engine (used by MCTS rollouts) must keep agent routing
    functional — the GameState invariant says state must be deep-copy-cheap,
    but Engine.agents is allowed to be copied/shared."""
    gs = _fresh_state()
    a1, a2 = (lambda eng, d: None), (lambda eng, d: None)
    eng = engine.Engine(gs, agents=[a1, a2])
    eng_copy = copy.deepcopy(eng)
    # agent_for must still resolve on the copy.
    assert eng_copy.agent_for(eng_copy.state.players[0]) is not None
    assert eng_copy.agent_for(eng_copy.state.players[1]) is not None
