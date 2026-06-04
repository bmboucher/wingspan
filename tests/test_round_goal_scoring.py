"""Tests for end-of-round goal scoring (``engine.scoring.score_round_goal``).

The 2-player round-goal payout scales by round and follows the printed rule:
the higher category count takes 1st-place VP, the lower takes 2nd, equal counts
share 1st place, and a player whose count is 0 does not place (scores nothing).
These tests pin each of those cases per round so the round-indexed payout table
in ``state.ROUND_GOAL_PAYOUTS_2P`` and the 0-count rule stay correct.
"""

from __future__ import annotations

import random

from wingspan import cards, engine, state
from wingspan.engine import scoring


def _game_with_forest_counts(count_0: int, count_1: int) -> state.GameState:
    """A fresh 2P game whose round goal (every slot) counts forest birds, with
    P0 / P1 holding ``count_0`` / ``count_1`` birds in their forest row and
    nothing elsewhere — so the only thing driving the goal is the given counts."""
    rng = random.Random(0)
    birds, bonuses, goals = cards.load_all()
    game_state = state.new_game(rng, birds, bonuses, goals)
    forest_goal = cards.EndRoundGoal(
        id=0, description="[bird] in [forest]", category="birds_forest", tile_id=0
    )
    game_state.round_goals = [forest_goal] * 4
    any_bird = birds[0]
    counts = (count_0, count_1)
    for i, player in enumerate(game_state.players):
        player.board[cards.Habitat.FOREST] = [
            state.PlayedBird(bird=any_bird) for _ in range(counts[i])
        ]
    return game_state


def _score(count_0: int, count_1: int, round_idx: int) -> tuple[int, int]:
    """Round-goal VP awarded to (P0, P1) for the given forest counts and round."""
    game_state = _game_with_forest_counts(count_0, count_1)
    eng = engine.Engine(game_state)
    scoring.score_round_goal(eng, round_idx)
    return (
        game_state.players[0].round_goal_points,
        game_state.players[1].round_goal_points,
    )


def test_payouts_scale_by_round():
    """A clear 3-vs-1 lead pays the printed 1st/2nd board values per round —
    this is the bug that mattered most: every round used to pay (5, 2)."""
    expected = {0: (4, 1), 1: (5, 2), 2: (6, 3), 3: (7, 4)}
    for round_idx, (win_vp, lose_vp) in expected.items():
        assert _score(3, 1, round_idx) == (win_vp, lose_vp)
        assert _score(1, 3, round_idx) == (lose_vp, win_vp)  # symmetric


def test_zero_count_player_always_scores_zero():
    """A player with a category count of 0 never places, so scores 0 no matter
    the round or the opponent's count. An opponent with >0 takes an uncontested
    1st; when both players are at 0, neither scores."""
    for round_idx in range(4):
        win_vp = state.ROUND_GOAL_PAYOUTS_2P[round_idx][0]
        for opp_count in (0, 1, 5):
            opp_vp = win_vp if opp_count > 0 else 0
            assert _score(0, opp_count, round_idx) == (0, opp_vp)
            assert _score(opp_count, 0, round_idx) == (opp_vp, 0)  # symmetric


def test_tie_splits_first_and_second_rounded_down():
    """Equal positive counts tie for the goal. Per the official rule the tied
    players occupy 1st and 2nd together, so each scores the floor of the
    combined payout — regardless of the (shared) count value."""
    for round_idx in range(4):
        first, second = state.ROUND_GOAL_PAYOUTS_2P[round_idx]
        tie_vp = (first + second) // 2
        for count in (1, 3, 5):
            p0_vp, p1_vp = _score(count, count, round_idx)
            assert p0_vp == p1_vp == tie_vp
    # Concrete per-round tie values: rounds 1..4 -> 2, 3, 4, 5.
    assert [_score(2, 2, round_idx)[0] for round_idx in range(4)] == [2, 3, 4, 5]


def test_round_goal_standing_mirrors_scoring():
    """The live-display standing reports the same VP a player would actually
    score — for a clear lead, the 0-count trailer, and a tie."""
    leading = _game_with_forest_counts(3, 0)
    leading.round_idx = 2  # round 3 pays (6, 3)
    leader = scoring.round_goal_standing(leading, leading.players[0])
    trailer = scoring.round_goal_standing(leading, leading.players[1])
    assert (leader.count, leader.place, leader.vp) == (3, 1, 6)
    assert (trailer.count, trailer.vp) == (0, 0)  # 0 count -> 0 VP

    tied = _game_with_forest_counts(2, 2)
    tied.round_idx = 0  # round 1 tie pays (4 + 1) // 2 = 2 each
    for player in tied.players:
        standing = scoring.round_goal_standing(tied, player)
        assert (standing.count, standing.place, standing.vp) == (2, 1, 2)
