"""Unit tests for Engine.agents and Engine.agent_for plumbing."""

from __future__ import annotations

import copy
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import cards, decisions, engine, state


def _fresh_state():
    birds, bonuses, goals = cards.load_all()
    return state.new_game(random.Random(0), birds, bonuses, goals)


def _stub_agent() -> engine.Agent:
    """Return a fresh ``Agent``-typed callable that must never be consulted.
    Each call yields a distinct object so identity-based routing assertions can
    tell two agents apart."""

    def stub[C: decisions.Choice](
        _engine: engine.Engine,
        _decision: decisions.Decision[C],
    ) -> C:
        raise AssertionError("stub agent should not be consulted")

    return stub


def test_engine_init_without_agents_has_empty_list():
    eng = engine.Engine(_fresh_state())
    assert eng.agents == []


def test_engine_init_with_agents_indexes_by_player_id():
    gs = _fresh_state()
    sentinel_p0 = _stub_agent()
    sentinel_p1 = _stub_agent()
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
        engine.Engine(gs, agents=[_stub_agent()])  # 1 agent


def test_deepcopy_preserves_agent_routing():
    """Deepcopy of Engine (used by MCTS rollouts) must keep agent routing
    functional — the GameState invariant says state must be deep-copy-cheap,
    but Engine.agents is allowed to be copied/shared."""
    gs = _fresh_state()
    a1, a2 = _stub_agent(), _stub_agent()
    eng = engine.Engine(gs, agents=[a1, a2])
    eng_copy = copy.deepcopy(eng)
    # agent_for must still resolve on the copy.
    assert eng_copy.agent_for(eng_copy.state.players[0]) is not None
    assert eng_copy.agent_for(eng_copy.state.players[1]) is not None
