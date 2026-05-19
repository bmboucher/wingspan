"""End-to-end smoke tests.

These confirm the three completion criteria all run without raising:

1. A full game with random agents completes and produces final scores.
2. The detailed game log is non-empty.
3. One epoch of self-play + a training step completes (CPU-only here).
"""

from __future__ import annotations

import random

import pytest


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
    for p in eng.state.players:
        assert hasattr(p, "final_score")
        assert isinstance(p.final_score, int)
    assert len(eng.state.log) > 50, "expected a substantive log"


def test_train_one_epoch_cpu():
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("torch not installed")
    from wingspan import train

    rc = train.main(
        [
            "--episodes",
            "4",
            "--epochs",
            "1",
            "--device",
            "cpu",
            "--seed",
            "0",
            "--checkpoint",
            "checkpoints/_test.pt",
        ]
    )
    assert rc == 0
