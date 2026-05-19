"""Agents that resolve Decisions.

An agent is any callable matching ``wingspan.engine.core.Agent`` — given the
engine and a ``Decision[C]``, it returns the chosen ``C``. The Agent type
itself lives next to the Engine; this package collects implementations:

- ``base`` — the random-policy agent
- ``cli``  — the interactive human (stdin/stdout) agent plus the hotseat
  helper ``mixed_agents``

The public entry points are re-exported so callers can keep writing
``from wingspan.agents import cli_agent``.
"""

from wingspan.agents.base import random_agent
from wingspan.agents.cli import cli_agent, mixed_agents

__all__ = [
    "random_agent",
    "cli_agent",
    "mixed_agents",
]
