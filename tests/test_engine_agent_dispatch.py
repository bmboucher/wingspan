"""Unit tests for Engine.agents and Engine.agent_for plumbing."""
from __future__ import annotations

import copy
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import cards
from wingspan.game import Engine
from wingspan.state import new_game


def _fresh_state():
    birds, bonuses, goals = cards.load_all()
    return new_game(random.Random(0), birds, bonuses, goals)


def test_engine_init_without_agents_has_empty_list():
    eng = Engine(_fresh_state())
    assert eng.agents == []


def test_engine_init_with_agents_indexes_by_player_id():
    state = _fresh_state()
    sentinel_p0 = lambda eng, d: None
    sentinel_p1 = lambda eng, d: None
    eng = Engine(state, agents=[sentinel_p0, sentinel_p1])
    assert eng.agent_for(state.players[0]) is sentinel_p0
    assert eng.agent_for(state.players[1]) is sentinel_p1


def test_engine_agent_for_raises_when_unset():
    state = _fresh_state()
    eng = Engine(state)
    with pytest.raises(RuntimeError, match="No agent registered"):
        eng.agent_for(state.players[0])


def test_engine_init_rejects_length_mismatch():
    state = _fresh_state()  # 2 players
    with pytest.raises(ValueError, match="does not match players count"):
        Engine(state, agents=[lambda eng, d: None])  # 1 agent


def test_play_one_game_reassigns_agents():
    """A reused Engine instance must accept fresh agents on each play_one_game."""
    state = _fresh_state()
    a1, a2 = (lambda eng, d: None), (lambda eng, d: None)
    eng = Engine(state, agents=[a1, a2])
    b1, b2 = (lambda eng, d: None), (lambda eng, d: None)
    # Simulate the unconditional assignment that play_one_game does.
    eng.agents = list((b1, b2))
    assert eng.agent_for(state.players[0]) is b1
    assert eng.agent_for(state.players[1]) is b2


def test_deepcopy_preserves_agent_routing():
    """Deepcopy of Engine (used by MCTS rollouts) must keep agent routing
    functional — the GameState invariant says state must be deep-copy-cheap,
    but Engine.agents is allowed to be copied/shared."""
    state = _fresh_state()
    a1, a2 = (lambda eng, d: None), (lambda eng, d: None)
    eng = Engine(state, agents=[a1, a2])
    eng_copy = copy.deepcopy(eng)
    # agent_for must still resolve on the copy.
    assert eng_copy.agent_for(eng_copy.state.players[0]) is not None
    assert eng_copy.agent_for(eng_copy.state.players[1]) is not None
