"""Shared multiplayer test helpers: N-agent lists and N-player engines.

Centralizes the boilerplate every N-player test needs — a seat count's worth
of agents, and an engine built at that seat count — so individual test files
stay focused on what they are actually asserting. Mirrors the
``stub_agents.py`` convention: a plain module import with no ``sys.path``
handling of its own, relying on pytest's ``pythonpath = ["src"]`` config
(see ``pyproject.toml``).
"""

from __future__ import annotations

import random

from wingspan import agents, engine


def make_agents(n: int, seed: int = 0) -> list[engine.Agent]:
    """``n`` independent random agents, each closed over its own seeded
    ``random.Random`` stream (``seed + i``) so a test can reproduce a
    specific N-player game without the seats sharing RNG state."""
    return [agents.random_agent(random.Random(seed + i)) for i in range(n)]


def make_engine(num_players: int, seed: int = 0) -> engine.Engine:
    """A fresh ``Engine`` over a ``num_players``-seat game, built the same
    way ``Engine.create`` builds today's 2-player default."""
    eng, *_ = engine.Engine.create(seed=seed, num_players=num_players)
    return eng
