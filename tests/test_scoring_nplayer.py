"""N-player scoring-kernel tests: ``scoring.placement_payouts`` (the
round-goal placement kernel), ``scoring.winners`` / ``determine_winner`` (the
game-winner kernel, including the official food-supply tiebreak), and
``RoundGoalStanding``'s N-player ``other_counts`` / ``place`` fields.

Round-goal *scoring flow* tests (``score_round_goal`` end to end, frozen vs.
live standings) live in ``test_round_goal_scoring.py`` /
``test_round_goal_freeze.py`` and stay 2-player — this file is the kernel
math those flows are built on, exercised directly at 2/3/4 players.
"""

from __future__ import annotations

import random

from wingspan import cards, engine, state
from wingspan.engine import scoring


def _bird(name: str) -> cards.Bird:
    """A minimal, otherwise-inert bird for board-count fixtures."""
    return cards.Bird(
        id=abs(hash(name)) % 100000,
        name=name,
        scientific_name="Testus birdus",
        color=cards.PowerColor.NONE,
        points=1,
        nest=cards.NestType.BOWL,
        egg_limit=5,
        wingspan_cm=50,
        habitats=(cards.Habitat.FOREST,),
        food_cost=cards.BirdCost(),
        flocking=False,
        predator=False,
        raw_power_text="",
        power=cards.Power(color=cards.PowerColor.NONE),
        bonus_categories=(),
    )


def _player_with(player_id: int, score: int, food_total: int) -> state.Player:
    """A minimal ``Player`` with a set ``final_score`` and ``food_total``
    tokens in personal supply (all of one food type — the tiebreak only
    cares about the sum)."""
    player = state.Player(id=player_id, name=f"P{player_id}", final_score=score)
    if food_total:
        player.food[cards.Food.SEED] = food_total
    return player


def _game_with_forest_counts(counts: list[int]) -> state.GameState:
    """A fresh ``len(counts)``-player game whose round goal (every slot)
    counts forest birds, with seat ``i`` holding ``counts[i]`` birds in
    forest and nothing elsewhere."""
    rng = random.Random(0)
    birds, bonuses, goals = cards.load_all()
    gs = state.new_game(rng, birds, bonuses, goals, num_players=len(counts))
    forest_goal = cards.EndRoundGoal(
        id=0, description="[bird] in [forest]", category="birds_forest", tile_id=0
    )
    gs.round_goals = [forest_goal] * 4
    any_bird = birds[0]
    for player, count in zip(gs.players, counts):
        player.board[cards.Habitat.FOREST] = [
            state.PlayedBird(bird=any_bird) for _ in range(count)
        ]
    return gs


#### placement_payouts ####


def _legacy_placement_vp(my_count: int, opp_count: int, first: int, second: int) -> int:
    """Inline copy of the pre-N-player ``scoring._placement_vp`` formula
    (deleted from the module), kept only here so this test pins the
    *documented* legacy 2-player behavior independent of the current
    implementation."""
    if my_count == 0:
        return 0
    if my_count > opp_count:
        return first
    if my_count < opp_count:
        return second
    return (first + second) // 2


def test_placement_payouts_matches_legacy_2p_exhaustively():
    """``placement_payouts`` reduces to the legacy 2-player rule for every
    (my, opp) count pair and every round's payout."""
    for round_idx in range(4):
        first, second = state.ROUND_GOAL_PAYOUTS[round_idx][:2]
        for my_count in range(13):
            for opp_count in range(13):
                expected = (
                    _legacy_placement_vp(my_count, opp_count, first, second),
                    _legacy_placement_vp(opp_count, my_count, first, second),
                )
                actual = scoring.placement_payouts(
                    [my_count, opp_count], (first, second)
                )
                assert tuple(actual) == expected, (
                    f"round {round_idx}, counts=({my_count}, {opp_count}): "
                    f"expected {expected}, got {tuple(actual)}"
                )


def test_placement_payouts_3p_two_way_tie_at_top():
    assert scoring.placement_payouts([5, 5, 3], (5, 2, 1)) == [3, 3, 1]


def test_placement_payouts_3p_two_way_tie_below_top():
    assert scoring.placement_payouts([5, 3, 3], (5, 2, 1)) == [5, 1, 1]


def test_placement_payouts_3p_three_way_tie():
    assert scoring.placement_payouts([5, 5, 5], (5, 2, 1)) == [2, 2, 2]


def test_placement_payouts_zero_seat_never_places_and_others_move_up():
    assert scoring.placement_payouts([5, 0, 3], (5, 2, 1)) == [5, 0, 2]
    assert scoring.placement_payouts([5, 0, 3], (4, 1, 0)) == [4, 0, 1]


def test_placement_payouts_all_zero_scores_all_zero():
    assert scoring.placement_payouts([0, 0, 0], (4, 1, 0)) == [0, 0, 0]


def test_placement_payouts_4way_tie_pads_past_ladder_end_with_zero():
    # The ladder only pays 3 places; a 4-way tie's shared pool is
    # 7 + 4 + 3 + 0 (4th place pads to 0 past the end of the ladder).
    assert scoring.placement_payouts([9, 9, 9, 9], (7, 4, 3)) == [3, 3, 3, 3]


#### winners / determine_winner ####


def test_winners_single_score_leader_wins_outright():
    players = [_player_with(0, 10, 0), _player_with(1, 5, 0)]
    assert scoring.winners(players) == [0]
    assert scoring.determine_winner(players) == 0


def test_winners_score_tie_broken_by_supply_food():
    players = [_player_with(0, 10, 2), _player_with(1, 10, 5)]
    assert scoring.winners(players) == [1]
    assert scoring.determine_winner(players) == 1


def test_winners_full_tie_score_and_food_is_shared_victory():
    players = [_player_with(0, 10, 3), _player_with(1, 10, 3)]
    assert scoring.winners(players) == [0, 1]
    assert scoring.determine_winner(players) == -1


def test_winners_cached_food_on_birds_does_not_count_toward_tiebreak():
    """Only ``Player.food`` (personal supply) counts; a huge cache stacked on
    a played bird must not break the tie."""
    leader = _player_with(0, 10, 1)
    trailer = _player_with(1, 10, 1)
    pb = state.PlayedBird(bird=_bird("Cache Hog"))
    pb.cached_food[cards.Food.SEED] = 99
    trailer.board[cards.Habitat.FOREST].append(pb)
    players = [leader, trailer]
    assert scoring.winners(players) == [0, 1]
    assert scoring.determine_winner(players) == -1


def test_winners_3way_shared_victory_after_food_tiebreak():
    players = [_player_with(seat_id, 10, 4) for seat_id in range(3)]
    assert scoring.winners(players) == [0, 1, 2]
    assert scoring.determine_winner(players) == -1


def test_winners_4p_food_tiebreak_narrows_to_two():
    players = [
        _player_with(0, 20, 1),
        _player_with(1, 20, 5),
        _player_with(2, 20, 5),
        _player_with(3, 15, 9),
    ]
    assert scoring.winners(players) == [1, 2]
    assert scoring.determine_winner(players) == -1


#### RoundGoalStanding at N=3 ####


def test_round_goal_standing_other_counts_clockwise_n3():
    gs = _game_with_forest_counts([3, 1, 5])
    standing = scoring.round_goal_standing_for_round(gs, gs.players[0], 0)
    assert standing.count == 3
    assert standing.other_counts == [1, 5]  # clockwise from seat 0: seat 1, seat 2
    assert standing.opp_count == 5  # max(other_counts)


def test_round_goal_standing_place_and_vp_with_a_tie_n3():
    gs = _game_with_forest_counts([5, 5, 2])
    p0 = scoring.round_goal_standing_for_round(gs, gs.players[0], 0)
    p1 = scoring.round_goal_standing_for_round(gs, gs.players[1], 0)
    p2 = scoring.round_goal_standing_for_round(gs, gs.players[2], 0)
    assert p0.place == 1  # tied for 1st with seat 1
    assert p1.place == 1
    assert p2.place == 3  # both others strictly greater
    # Round 1 payouts (4, 1, 0): the 2-way tie at 1st/2nd shares floor((4+1)/2).
    assert p0.vp == p1.vp == 2
    assert p2.vp == 0


def test_round_goal_standing_scored_round_freezes_other_counts_n3():
    gs = _game_with_forest_counts([3, 1, 5])
    eng = engine.Engine(gs)
    scoring.score_round_goal(eng, 0)
    frozen = scoring.round_goal_standing_for_round(gs, gs.players[0], 0)
    assert frozen.other_counts == [1, 5]
    assert frozen.opp_count == 5
