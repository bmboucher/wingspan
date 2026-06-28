"""Reusable scripted agents for tests — torch-free.

Exported agents match the :class:`wingspan.engine.core.Agent` protocol and are
safe to use wherever the engine's ``agents=`` list is accepted. Import this
module in tests rather than re-defining local stubs:

    import stub_agents
    eng = engine.Engine(gs, agents=[stub_agents.no_agent, stub_agents.no_agent])

All agents here are typed with the generic ``[C: decisions.Choice]`` call
signature so strict pyright can track the return type through each call site.
"""

from __future__ import annotations

import typing

from wingspan import decisions
from wingspan.engine import core as engine_core


def make_unconsulted_agent() -> engine_core.Agent:
    """Return a new ``Agent``-typed callable that raises if the engine ever
    consults it.

    Returns a fresh object each call so identity-based routing assertions can
    distinguish two seats (``is``-compare them by variable, not by value).
    """

    def _agent[C: decisions.Choice](
        _engine: engine_core.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        raise AssertionError(
            f"agent should not be consulted (got {type(decision).__name__})"
        )

    return typing.cast(engine_core.Agent, _agent)


def no_agent[C: decisions.Choice](
    _engine: engine_core.Engine,
    decision: decisions.Decision[C],
) -> C:
    """Raises on any consultation — use as a module-level agent when identity
    checks are not needed and the agent must never be queried."""
    raise AssertionError(
        f"agent should not be consulted (got {type(decision).__name__})"
    )


def accept_agent[C: decisions.Choice](
    _engine: engine_core.Engine,
    decision: decisions.Decision[C],
) -> C:
    """Picks the first non-skip choice; falls back to the first choice when all
    choices are skips."""
    for choice in decision.choices:
        if not isinstance(choice, decisions.SkipChoice):
            return choice
    return decision.choices[0]


def skip_agent[C: decisions.Choice](
    _engine: engine_core.Engine,
    decision: decisions.Decision[C],
) -> C:
    """Picks the :class:`~decisions.SkipChoice` when one is offered; falls back
    to the first choice when no skip is available."""
    for choice in decision.choices:
        if isinstance(choice, decisions.SkipChoice):
            return choice
    return decision.choices[0]
