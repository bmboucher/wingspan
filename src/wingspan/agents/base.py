"""The random-policy agent.

This is the minimal surface area every agent implementation depends on: a
tiny default agent that just picks a uniformly-random offered choice. Other
modules (e.g. ``cli``) build on this without pulling the interactive bits.

The shared callable type is ``wingspan.engine.core.Agent`` — a Protocol with
a generic ``__call__`` so the returned Choice subtype tracks the Decision's
parameterization automatically.
"""

from __future__ import annotations

import random

from wingspan import decisions
from wingspan.engine import core as engine_core


def random_agent(rng: random.Random | None = None) -> engine_core.Agent:
    """Build an agent that picks uniformly at random from the offered choices.

    A fresh ``random.Random`` is allocated when ``rng`` is omitted so callers
    that want reproducibility can supply a seeded one.
    """
    chooser = rng or random.Random()

    def agent[C: decisions.Choice](
        _engine: engine_core.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        return chooser.choice(decision.choices)

    return agent
