"""Tests for the dense piecewise-linear bonus-card payoff estimate.

``scoring.bonus_linear_value`` interpolates a tiered card's stepped payout
between its ``(count, vp)`` thresholds so the RL encoder gets a gradient that
rewards incremental progress toward the next plateau, instead of the step
function's flat regions. Per-bird cards are already linear in the qualifying
count and so pass straight through.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import cards, state
from wingspan.engine import scoring


def _bonus(bonuses: list[cards.BonusCard], name: str) -> cards.BonusCard:
    return next(bonus for bonus in bonuses if bonus.name == name)


def _player_with_board(
    bonus: cards.BonusCard, board_birds: list[cards.Bird]
) -> state.Player:
    """A player holding ``bonus`` with ``board_birds`` placed in the forest."""
    player = state.Player(id=0, name="P0", bonus_cards=[bonus])
    player.board[cards.Habitat.FOREST] = [
        state.PlayedBird(bird=bird) for bird in board_birds
    ]
    return player


def test_bird_feeder_linear_interpolation():
    """Bird Feeder's anchors are (0,0), (5,3), (8,7): the value ramps linearly
    up to each threshold and holds flat past the last."""
    birds, bonuses, _ = cards.load_all()
    bird_feeder = _bonus(bonuses, "Bird Feeder")
    seed = [bird for bird in birds if "Bird Feeder" in bird.bonus_categories]
    assert len(seed) >= 10

    expected = {0: 0.0, 3: 1.8, 5: 3.0, 6: 3 + 4 / 3, 8: 7.0, 10: 7.0}
    for count, value in expected.items():
        player = _player_with_board(bird_feeder, seed[:count])
        assert math.isclose(scoring.bonus_linear_value(player, bird_feeder), value)


def test_per_bird_card_is_linear():
    """A per-bird card's linear value equals its stepped value: per_bird_vp
    times the qualifying count (Bird Counter: 2 VP each, 4 birds -> 8)."""
    birds, bonuses, _ = cards.load_all()
    bird_counter = _bonus(bonuses, "Bird Counter")
    flocking = [bird for bird in birds if "Bird Counter" in bird.bonus_categories][:4]
    assert len(flocking) == 4
    player = _player_with_board(bird_counter, flocking)
    assert math.isclose(scoring.bonus_linear_value(player, bird_counter), 8.0)
    assert math.isclose(
        scoring.bonus_linear_value(player, bird_counter),
        float(scoring.bonus_score(player, bird_counter)),
    )


def test_linear_value_flat_tail_holds_max():
    """Past the highest threshold the linear value holds at the final VP
    (Bird Feeder: any count >= 8 -> 7.0)."""
    birds, bonuses, _ = cards.load_all()
    bird_feeder = _bonus(bonuses, "Bird Feeder")
    seed = [bird for bird in birds if "Bird Feeder" in bird.bonus_categories][:12]
    assert len(seed) == 12
    player = _player_with_board(bird_feeder, seed)
    assert math.isclose(scoring.bonus_linear_value(player, bird_feeder), 7.0)
