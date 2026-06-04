"""End-to-end smoke test: a full game with random agents completes, produces
final scores, and leaves a substantive game log.

The training-cycle smoke coverage lives in ``test_model_and_self_play.py``,
which runs the production collect → update path end-to-end on CPU.
"""

from __future__ import annotations

import random


def test_random_game_completes():
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from wingspan import agents, engine

    eng, *_ = engine.Engine.create(seed=123)
    rng = random.Random(123)
    engine.Engine.play_one_game(
        eng.state, (agents.random_agent(rng), agents.random_agent(rng))
    )
    assert eng.state.game_over
    for player in eng.state.players:
        assert hasattr(player, "final_score")
        assert isinstance(player.final_score, int)
    assert len(eng.state.log) > 50, "expected a substantive log"
