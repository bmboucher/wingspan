"""Tests for N-player support in the ``gamelog`` event recorder (Stage 4).

Confirms :class:`~wingspan.gamelog.recorder.EventRecorder` records a full
per-seat ``scores`` list (not just two seats) and computes ``margin_before``
as the deciding seat's own margin vs its best other seat at any table size —
and that a 2-seat recording still produces the same tree shape as before the
``score_p0``/``score_p1`` -> ``scores`` schema change (values unchanged).
"""

from __future__ import annotations

import random
import typing

import helpers
from wingspan import agents as agents_module
from wingspan import engine as engine_module
from wingspan.gamelog import models as gamelog_models
from wingspan.gamelog import recorder as gamelog_recorder


def _record_game(num_players: int, seed: int) -> gamelog_models.GameEventTree:
    """Play one random-agent game at ``num_players`` seats through a real
    ``EventRecorder`` and return the resulting tree."""
    probes = tuple(None for _ in range(num_players))
    rec = gamelog_recorder.EventRecorder(probes=probes)
    eng, *_ = engine_module.Engine.create(seed=seed, num_players=num_players)
    agent_list = helpers.make_agents(num_players, seed=seed)
    engine_module.Engine.play_one_game(eng.state, agent_list, event_recorder=rec)
    return rec.root


def _collect_decision_subs(
    tree: gamelog_models.GameEventTree,
) -> list[gamelog_models.DecisionSubEvent]:
    """Every ``DecisionSubEvent`` in the tree, DFS order."""

    def _walk(
        events: typing.Sequence[gamelog_models.GameEvent],
    ) -> list[gamelog_models.SubEvent]:
        result: list[gamelog_models.SubEvent] = []
        for event in events:
            result.extend(event.sub_events)
            result.extend(_walk(event.children))
        return result

    all_subs = _walk([event for phase in tree.phases for event in phase.events])
    return [sub for sub in all_subs if isinstance(sub, gamelog_models.DecisionSubEvent)]


###### 3-player recording ######


def test_3p_decision_scores_length_matches_seat_count():
    """Every recorded decision's ``scores`` list has exactly 3 entries at a
    3-seat table."""
    tree = _record_game(num_players=3, seed=101)
    decision_subs = _collect_decision_subs(tree)
    assert decision_subs, "a full game records at least one genuine decision"
    assert all(len(sub.scores) == 3 for sub in decision_subs)


def test_3p_margin_before_is_own_score_minus_best_other():
    """``margin_before`` is exactly the deciding seat's own score minus the
    *best* other seat's score, for every recorded decision — this is the
    generalized (own-POV) formula that Stage 4 threads through the recorder."""
    tree = _record_game(num_players=3, seed=102)
    decision_subs = _collect_decision_subs(tree)
    assert decision_subs
    for sub in decision_subs:
        # DecisionSubEvent.player_id is always a real seat (never None) — the
        # recorder stamps decision.player_id, and every Decision names a seat.
        assert sub.player_id is not None
        own_score = sub.scores[sub.player_id]
        other_scores = [
            score for idx, score in enumerate(sub.scores) if idx != sub.player_id
        ]
        expected_margin = float(own_score - max(other_scores))
        assert sub.margin_before == expected_margin, (
            f"margin_before mismatch for player {sub.player_id}: "
            f"got {sub.margin_before}, expected {expected_margin} "
            f"(scores={sub.scores})"
        )


def test_3p_final_scoring_event_has_three_breakdowns():
    """The game-end ``FinalScoringEvent`` carries one breakdown per seat."""
    tree = _record_game(num_players=3, seed=103)
    game_end = tree.phases[-1]
    assert game_end.kind == "game_end"
    scoring_events = [
        event
        for event in game_end.events
        if isinstance(event, gamelog_models.FinalScoringEvent)
    ]
    assert len(scoring_events) == 1
    assert len(scoring_events[0].scores) == 3


def test_3p_round_goal_events_have_three_counts_and_vps():
    """Each of the 4 ``RoundGoalEvent``s carries a 3-length ``counts``/``vps``."""
    tree = _record_game(num_players=3, seed=104)
    goal_events = [
        event
        for phase in tree.phases
        for event in phase.events
        if isinstance(event, gamelog_models.RoundGoalEvent)
    ]
    assert len(goal_events) == 4
    for event in goal_events:
        assert len(event.counts) == 3
        assert len(event.vps) == 3


def test_4p_decision_scores_length_matches_seat_count():
    """The same invariant holds at 4 seats (not just 3)."""
    tree = _record_game(num_players=4, seed=105)
    decision_subs = _collect_decision_subs(tree)
    assert decision_subs
    assert all(len(sub.scores) == 4 for sub in decision_subs)


###### 2-player recording: tree shape unchanged ######


def test_2p_recording_tree_shape_matches_pre_nplayer_expectations():
    """A 2-seat recording still has the same tree shape as before the
    ``scores`` schema change: game_start/setup/round/turn/game_end phases,
    4 round-goal events, one 2-seat final scoring event, and every decision's
    ``scores`` list has exactly 2 entries."""
    tree = _record_game(num_players=2, seed=106)
    kinds = [phase.kind for phase in tree.phases]
    assert kinds[0] == "game_start"
    assert kinds[-1] == "game_end"
    assert "round" in kinds
    assert "turn" in kinds

    goal_events = [
        event
        for phase in tree.phases
        for event in phase.events
        if isinstance(event, gamelog_models.RoundGoalEvent)
    ]
    assert len(goal_events) == 4

    game_end = tree.phases[-1]
    scoring_events = [
        event
        for event in game_end.events
        if isinstance(event, gamelog_models.FinalScoringEvent)
    ]
    assert len(scoring_events) == 1
    assert len(scoring_events[0].scores) == 2

    decision_subs = _collect_decision_subs(tree)
    assert decision_subs
    assert all(len(sub.scores) == 2 for sub in decision_subs)


def test_2p_margin_before_reduces_to_legacy_own_minus_opponent():
    """At 2 seats the generalized own-vs-best-other formula reduces exactly to
    the legacy own-minus-opponent value (no behavior change at N=2)."""
    tree = _record_game(num_players=2, seed=107)
    decision_subs = _collect_decision_subs(tree)
    assert decision_subs
    for sub in decision_subs:
        assert sub.player_id is not None
        other = sub.scores[1 - sub.player_id]
        expected_margin = float(sub.scores[sub.player_id] - other)
        assert sub.margin_before == expected_margin


def test_null_recorder_still_noop_regardless_of_seat_count():
    """The null recorder accepts any seat count's calls without raising."""
    rng = random.Random(55)
    eng, *_ = engine_module.Engine.create(seed=55, num_players=3)
    agent_list = [agents_module.random_agent(rng) for _ in range(3)]
    engine_module.Engine.play_one_game(
        eng.state, agent_list, event_recorder=gamelog_recorder.null_recorder()
    )
    assert eng.state.game_over
