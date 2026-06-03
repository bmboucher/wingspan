"""Tests for the Elo math.

Checks the expected-score symmetry and update direction/magnitude, and that
:func:`elo.replay` is independent of the order results are passed in (it sorts to
a canonical order before applying updates, so the report's Elo is reproducible).
"""

from __future__ import annotations

import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from wingspan.tournament import elo, results, schedule


def test_expected_scores_are_symmetric_and_fair_when_equal() -> None:
    table = elo.EloTable.initial(["a", "b"], init=1500.0, k=24.0)
    assert abs(table.expected("a", "b") + table.expected("b", "a") - 1.0) < 1e-9
    assert abs(table.expected("a", "b") - 0.5) < 1e-9


def test_update_moves_pair_by_equal_and_opposite_amounts() -> None:
    table = elo.EloTable.initial(["a", "b"], init=1500.0, k=24.0)
    table.update("a", "b", score_a=1.0)
    assert table.ratings["a"] > 1500.0 > table.ratings["b"]
    gained = table.ratings["a"] - 1500.0
    lost = 1500.0 - table.ratings["b"]
    assert abs(gained - lost) < 1e-9
    # Equal ratings -> expected 0.5 -> a win moves by k * (1 - 0.5) = 12.
    assert abs(gained - 12.0) < 1e-9


def _game(
    round_index: int,
    pair_index: int,
    orientation: schedule.Orientation,
    player_a: str,
    player_b: str,
    a_score: int,
    b_score: int,
) -> results.GameResult:
    return results.GameResult(
        round_index=round_index,
        pair_index=pair_index,
        orientation=orientation,
        player_a_id=player_a,
        player_b_id=player_b,
        a_score=a_score,
        b_score=b_score,
        a_was_start_player=True,
    )


def test_replay_is_order_independent() -> None:
    games = [
        _game(0, 0, schedule.Orientation.A_SEAT_0, "a", "b", 20, 15),
        _game(0, 0, schedule.Orientation.A_SEAT_1, "a", "b", 12, 18),
        _game(0, 1, schedule.Orientation.A_SEAT_0, "a", "c", 22, 10),
        _game(0, 1, schedule.Orientation.A_SEAT_1, "a", "c", 14, 14),
        _game(0, 2, schedule.Orientation.A_SEAT_0, "b", "c", 19, 21),
        _game(1, 2, schedule.Orientation.A_SEAT_1, "b", "c", 17, 13),
    ]
    ids = ["a", "b", "c"]
    baseline = elo.replay(ids, init=1500.0, k=24.0, game_results=games)
    shuffled = list(games)
    random.Random(7).shuffle(shuffled)
    reshuffled = elo.replay(ids, init=1500.0, k=24.0, game_results=shuffled)
    assert baseline.ratings == reshuffled.ratings
