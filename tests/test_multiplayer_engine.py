"""Engine-level N-player tests: game completion at N=3/4, clockwise turn-order
rotation, ``GameState.opponents_clockwise``, and ``state.new_game`` validation.

Per-power N-player behavior (Oystercatcher, Hummingbird) lives alongside its
existing 2-player tests in ``test_powers_misc_unique.py`` /
``test_powers_each_player_die.py``; the N-player scoring kernel
(``placement_payouts`` / ``winners`` / ``determine_winner`` /
``RoundGoalStanding``) lives in ``test_scoring_nplayer.py``. The ~40 existing
2-player engine test files are untouched by Stage 1 — this file is additive.
"""

from __future__ import annotations

import random
import typing

import pytest

import helpers
from wingspan import cards, decisions, engine, state


def test_3p_random_game_completes():
    """A full 3-player random-vs-random game runs to completion: every seat
    ends with a ``final_score`` and no exception is raised."""
    eng = helpers.make_engine(num_players=3, seed=1)
    engine.Engine.play_one_game(eng.state, helpers.make_agents(3, seed=1))
    assert eng.state.game_over
    assert len(eng.state.players) == 3
    for player in eng.state.players:
        assert player.final_score is not None


def test_4p_random_game_completes():
    """A full 4-player random-vs-random game runs to completion: every seat
    ends with a ``final_score`` and no exception is raised."""
    eng = helpers.make_engine(num_players=4, seed=2)
    engine.Engine.play_one_game(eng.state, helpers.make_agents(4, seed=2))
    assert eng.state.game_over
    assert len(eng.state.players) == 4
    for player in eng.state.players:
        assert player.final_score is not None


def test_turn_order_rotates_clockwise_in_round_one_n3():
    """Round 1's ``MainActionDecision``s are asked in strict clockwise order
    off the randomly-chosen first player, wrapping around: ``first,
    (first+1)%3, (first+2)%3, first, ...`` — every seat holds equal cubes in
    round 1, so the round-robin never skips a seat."""
    eng = helpers.make_engine(num_players=3, seed=7)
    base_agents = helpers.make_agents(3, seed=7)
    recorded_ids: list[int] = []

    def _maybe_record(
        live_engine: engine.Engine, decision: decisions.Decision[typing.Any]
    ) -> None:
        # A helper taking the untyped ``Decision[Any]`` shape (rather than
        # narrowing the caller's ``Decision[C]`` in place) so the isinstance
        # check below can't widen ``agent``'s return type into a union at its
        # post-if join point.
        if live_engine.state.round_idx == 0 and isinstance(
            decision, decisions.MainActionDecision
        ):
            recorded_ids.append(decision.player_id)

    def _recorder(base: engine.Agent) -> engine.Agent:
        def agent[C: decisions.Choice](
            live_engine: engine.Engine, decision: decisions.Decision[C]
        ) -> C:
            _maybe_record(live_engine, decision)
            return base(live_engine, decision)

        return agent

    recording_agents = [_recorder(base) for base in base_agents]
    engine.Engine.play_one_game(eng.state, recording_agents)

    first = eng.state.start_player  # round 0's first player is start_player itself
    one_cycle = [(first + offset) % 3 for offset in range(3)]
    assert recorded_ids[:6] == one_cycle + one_cycle


def test_opponents_clockwise_order_n4():
    """``opponents_clockwise`` lists every other seat starting immediately
    after the POV seat, wrapping — at any ``from_id``, and defaulting to
    ``current_player`` when omitted."""
    eng = helpers.make_engine(num_players=4, seed=0)
    gs = eng.state
    assert [player.id for player in gs.opponents_clockwise(from_id=0)] == [1, 2, 3]
    assert [player.id for player in gs.opponents_clockwise(from_id=2)] == [3, 0, 1]
    assert [player.id for player in gs.opponents_clockwise(from_id=3)] == [0, 1, 2]
    gs.current_player = 1
    assert [player.id for player in gs.opponents_clockwise()] == [2, 3, 0]


def test_opponents_clockwise_at_2p_is_the_lone_opponent():
    eng = helpers.make_engine(num_players=2, seed=0)
    gs = eng.state
    assert [player.id for player in gs.opponents_clockwise(from_id=0)] == [1]
    assert [player.id for player in gs.opponents_clockwise(from_id=1)] == [0]


def test_new_game_default_matches_todays_2p_game():
    """Omitting ``player_names``/``num_players`` produces the exact 2-player
    game of today: 2 seats named P0/P1."""
    birds, bonuses, goals = cards.load_all()
    gs = state.new_game(random.Random(0), birds, bonuses, goals)
    assert len(gs.players) == 2
    assert [player.name for player in gs.players] == ["P0", "P1"]


def test_new_game_default_names_scale_with_num_players():
    birds, bonuses, goals = cards.load_all()
    gs = state.new_game(random.Random(0), birds, bonuses, goals, num_players=4)
    assert [player.name for player in gs.players] == ["P0", "P1", "P2", "P3"]


@pytest.mark.parametrize("num_players", [state.MIN_PLAYERS - 1, state.MAX_PLAYERS + 1])
def test_new_game_rejects_out_of_range_num_players(num_players: int):
    birds, bonuses, goals = cards.load_all()
    with pytest.raises(ValueError):
        state.new_game(random.Random(0), birds, bonuses, goals, num_players=num_players)


def test_new_game_rejects_player_names_count_mismatch():
    birds, bonuses, goals = cards.load_all()
    with pytest.raises(ValueError):
        state.new_game(
            random.Random(0),
            birds,
            bonuses,
            goals,
            player_names=("Alice", "Bob"),
            num_players=3,
        )


def test_new_game_accepts_explicit_names_at_every_supported_count():
    birds, bonuses, goals = cards.load_all()
    for num_players in range(state.MIN_PLAYERS, state.MAX_PLAYERS + 1):
        names = tuple(f"Seat{i}" for i in range(num_players))
        gs = state.new_game(
            random.Random(0),
            birds,
            bonuses,
            goals,
            player_names=names,
            num_players=num_players,
        )
        assert [player.name for player in gs.players] == list(names)
