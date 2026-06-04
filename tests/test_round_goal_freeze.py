# pyright: reportPrivateUsage=false
# (these tests read the layout's package-private stripe constants and the
# state encoder's private round-goal stripe builder to isolate the freeze)
"""Tests that a scored round goal is frozen — in the engine and the encoders.

``scoring.score_round_goal`` records each round's outcome on
``GameState.scored_goals`` the moment it pays out; from then on the standings
readers (``round_goal_standing_for_round``, the state encoder's round-goal
stripes) report the frozen at-scoring values no matter how the boards evolve,
and the choice encoder's ``goal_delta`` stripe stays zero for the scored
round (no choice can change a frozen payout).
"""

from __future__ import annotations

import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import cards, decisions, encode, engine, state  # noqa: E402
from wingspan.encode import layout, state_encode  # noqa: E402
from wingspan.engine import scoring  # noqa: E402


def _forest_goal_game(count_0: int, count_1: int) -> state.GameState:
    """A fresh 2P game whose every round goal counts forest birds, with P0 / P1
    holding ``count_0`` / ``count_1`` birds in their forest rows."""
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


def _add_forest_birds(player: state.Player, count: int) -> None:
    bird = player.board[cards.Habitat.FOREST][0].bird
    for _ in range(count):
        player.board[cards.Habitat.FOREST].append(state.PlayedBird(bird=bird))


def _goal_delta_slot(row: np.ndarray, goal_idx: int) -> tuple[float, float]:
    base = layout._OFF_GOAL_DELTA + goal_idx * layout._GOAL_DELTA_SLOT_DIM
    return (
        float(row[base + layout._GOAL_DELTA_COUNT]),
        float(row[base + layout._GOAL_DELTA_VP]),
    )


def test_score_round_goal_appends_frozen_result():
    """Scoring a round records both seats' counts and awarded VP verbatim."""
    game_state = _forest_goal_game(3, 1)
    eng = engine.Engine(game_state)
    assert game_state.scored_goals == []
    scoring.score_round_goal(eng, 0)
    assert len(game_state.scored_goals) == 1
    result = game_state.scored_goals[0]
    assert result.counts == [3, 1]
    assert result.vp_awarded == [4, 1]  # round 1 pays (4, 1)


def test_standing_frozen_against_later_board_changes():
    """After a round is scored, its standing never moves — even when the
    losing seat later overtakes on the live board. Unscored rounds stay live."""
    game_state = _forest_goal_game(3, 1)
    eng = engine.Engine(game_state)
    scoring.score_round_goal(eng, 0)
    _add_forest_birds(game_state.players[1], 4)  # P1: 1 -> 5 forest birds

    frozen = scoring.round_goal_standing_for_round(game_state, game_state.players[0], 0)
    assert (frozen.count, frozen.opp_count, frozen.place, frozen.vp) == (3, 1, 1, 4)
    frozen_opp = scoring.round_goal_standing_for_round(
        game_state, game_state.players[1], 0
    )
    assert (frozen_opp.count, frozen_opp.opp_count, frozen_opp.vp) == (1, 3, 1)

    live = scoring.round_goal_standing_for_round(game_state, game_state.players[0], 1)
    assert (live.count, live.opp_count, live.place) == (3, 5, 2)
    assert live.vp == 2  # round 2 pays (5, 2); P0 now trails


def test_state_encoder_past_round_stripes_freeze():
    """The state vector's round-goal stripes for a scored round stop tracking
    the live board (my count, opp count, and VP all freeze)."""
    game_state = _forest_goal_game(3, 1)
    eng = engine.Engine(game_state)
    scoring.score_round_goal(eng, 0)

    before = state_encode._round_goals_all_rounds(
        game_state, game_state.players[0]
    ).copy()
    _add_forest_birds(game_state.players[1], 4)
    after = state_encode._round_goals_all_rounds(game_state, game_state.players[0])

    slot = layout._ROUND_GOAL_SLOT_DIM
    assert np.array_equal(before[:slot], after[:slot]), "scored round must freeze"
    assert not np.array_equal(
        before[slot : 2 * slot], after[slot : 2 * slot]
    ), "unscored rounds must stay live"


def test_goal_delta_zero_for_scored_rounds_on_bird_rows():
    """A bird candidate's goal_delta prices only unscored rounds: once round 1
    is scored, its slot zeroes out while later rounds keep pricing."""
    game_state = _forest_goal_game(1, 1)
    birds, _, _ = cards.load_all()
    forest_bird = next(bird for bird in birds if cards.Habitat.FOREST in bird.habitats)
    decision = decisions.BirdPowerTuckFromHandDecision(
        player_id=0,
        prompt="t",
        choices=[decisions.BirdChoice(label=forest_bird.name, bird=forest_bird)],
    )

    row = encode.encode_choices(decision, game_state)[0]
    assert _goal_delta_slot(row, 0)[0] > 0.0
    assert _goal_delta_slot(row, 1)[0] > 0.0

    eng = engine.Engine(game_state)
    scoring.score_round_goal(eng, 0)
    row = encode.encode_choices(decision, game_state)[0]
    assert _goal_delta_slot(row, 0) == (0.0, 0.0), "scored round must price zero"
    assert _goal_delta_slot(row, 1)[0] > 0.0, "unscored rounds keep pricing"
