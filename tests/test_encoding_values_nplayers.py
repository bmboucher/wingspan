# pyright: reportPrivateUsage=false
# (reads the shared, package-private layout constants — the same intra-package
# coupling convention test_round_goal_freeze.py and state_encode.py itself use)
"""N-player encoding *values* (not just shape): ``player_select``,
``turn_position``, per-opponent round-goal counts, and the card-index block's
seat ordering — all at N=3, where the single-opponent N=2 shape can no longer
hide an off-by-one or a wrong rotation direction.
"""

from __future__ import annotations

import random
import typing

import numpy as np

from wingspan import cards, decisions, encode, engine, state
from wingspan.encode import layout
from wingspan.engine import powers, scoring


def _new_game(num_players: int, seed: int = 0) -> state.GameState:
    rng = random.Random(seed)
    birds, bonuses, goals = cards.load_all()
    return state.new_game(rng, birds, bonuses, goals, num_players=num_players)


# ---------------------------------------------------------------------------
# player_select: (choice.player_id - decider) % N, via a real
# BirdPowerPickGainOrderDecision (mirrors test_powers_each_player_die.py's
# N=3 gate tests, but captures the decision/state pair for direct encoding
# instead of just recording player ids).


def _accept_veto[C: decisions.Choice](decision: decisions.Decision[C]) -> C:
    if isinstance(decision, decisions.AcceptExchangeDecision):
        return typing.cast(
            C,
            next(c for c in decision.choices if isinstance(c, decisions.PayCostChoice)),
        )
    return decision.choices[0]


def test_player_select_one_hot_at_n3_gain_order_decision():
    gs = _new_game(3)
    for food in gs.birdfeeder.counts:
        gs.birdfeeder.counts[food] = 0
    gs.birdfeeder.choice_dice = 0
    gs.birdfeeder.counts[cards.Food.SEED] = 2
    gs.birdfeeder.counts[cards.Food.FRUIT] = 2

    p0, p1, p2 = gs.players
    gs.current_player = p0.id
    carrier = next(
        bird for bird in cards.load_all()[0] if bird.name == "Anna's Hummingbird"
    )
    pb = state.PlayedBird(bird=carrier)
    eff = cards.Effect(
        kind=cards.EffectKind.EACH_PLAYER_GAINS_DIE_CHOOSE_ORDER, amount=1
    )

    captured: dict[str, decisions.Decision[typing.Any]] = {}

    def _capture_order(decision: decisions.Decision[typing.Any]) -> None:
        # Takes the untyped Decision[Any] shape — rather than narrowing the
        # calling agent's own decision: Decision[C] in place — so the
        # isinstance check here can't widen that caller's return type into a
        # union at its post-call join point (mirrors
        # test_powers_each_player_die.py's _record_order_choices).
        if (
            isinstance(decision, decisions.BirdPowerPickGainOrderDecision)
            and "order" not in captured
        ):
            captured["order"] = decision

    def agent_p0[C: decisions.Choice](
        _engine: engine.Engine, decision: decisions.Decision[C]
    ) -> C:
        _capture_order(decision)
        return _accept_veto(decision)

    def agent_other[C: decisions.Choice](
        _engine: engine.Engine, decision: decisions.Decision[C]
    ) -> C:
        return _accept_veto(decision)

    eng = engine.Engine(gs, agents=[agent_p0, agent_other, agent_other])
    powers.apply_effect(eng, agent_p0, p0, pb, carrier.habitats[0], eff, "play")

    order_decision = captured["order"]
    assert order_decision.player_id == p0.id
    assert {c.player_id for c in order_decision.choices} == {p0.id, p1.id, p2.id}

    spec3 = encode.EncodingSpec(num_players=3)
    feats = encode.encode_choices(order_decision, gs, spec3)
    off = layout.off_player_select(spec3)
    assert off is not None
    for row, choice in zip(feats, order_decision.choices):
        expected_offset = (choice.player_id - order_decision.player_id) % 3
        one_hot = row[off : off + 3]
        assert np.count_nonzero(one_hot) == 1
        assert one_hot[expected_offset] == 1.0
        # is_self bit (unrelated stripe) is untouched by player_select.
        is_self_expected = 1.0 if choice.player_id == order_decision.player_id else 0.0
        assert row[layout._OFF_SPECIAL + layout._SPECIAL_IS_SELF] == is_self_expected


def test_player_select_absent_at_n2():
    """The same decision at N=2 has no player_select stripe at all — encoding
    a 2-seat game never trips the (spec.num_players >= 3) branch."""
    gs = _new_game(2)
    p0, p1 = gs.players
    decision = decisions.BirdPowerPickGainOrderDecision(
        player_id=p0.id,
        prompt="order",
        choices=[
            decisions.PlayerIdChoice(label="p0", player_id=p0.id),
            decisions.PlayerIdChoice(label="p1", player_id=p1.id),
        ],
    )
    feats = encode.encode_choices(decision, gs, encode.DEFAULT_SPEC)
    assert feats.shape == (2, encode.choice_feature_dim(encode.DEFAULT_SPEC))
    assert layout.off_player_select(encode.DEFAULT_SPEC) is None


# ---------------------------------------------------------------------------
# turn_position: one-hot of the POV player's clockwise offset from the
# round's first-to-act seat, rotating with round_idx.


def test_turn_position_rotates_with_round_idx():
    gs = _new_game(3)
    spec3 = encode.EncodingSpec(num_players=3)
    cont = layout.state_cont_layout(spec3)
    tp_off = cont.offset_of("turn_position")
    tp_size = cont.size_of("turn_position")

    gs.start_player = 0
    gs.turn_counter = (
        1  # non-setup, so turn_state's one-hot also fires; irrelevant here
    )
    for round_idx in range(4):
        gs.round_idx = round_idx
        for seat in range(3):
            me = gs.players[seat]
            vec = encode.encode_state(gs, decision=None, spec=spec3)
            # decision=None means encode_state falls back to state.current_player
            # for pov; pin it explicitly via current_player instead.
            gs.current_player = seat
            vec = encode.encode_state(gs, decision=None, spec=spec3)
            one_hot = vec[tp_off : tp_off + tp_size]
            expected_offset = (me.id - (gs.start_player + round_idx) % 3) % 3
            assert np.count_nonzero(one_hot) == 1
            assert one_hot[expected_offset] == 1.0


def test_turn_position_matches_engine_round_rotation_order():
    """The seat encode_state marks offset 0 (turn_position[0] == 1) is exactly
    the seat engine.core.Engine._play_round starts the round with."""
    gs = _new_game(3)
    spec3 = encode.EncodingSpec(num_players=3)
    cont = layout.state_cont_layout(spec3)
    tp_off = cont.offset_of("turn_position")

    gs.start_player = 2
    gs.round_idx = 1
    gs.turn_counter = 1
    expected_first = (gs.start_player + gs.round_idx) % 3  # mirrors _play_round
    gs.current_player = expected_first
    vec = encode.encode_state(gs, decision=None, spec=spec3)
    assert vec[tp_off] == 1.0


# ---------------------------------------------------------------------------
# round_goals: other_counts carried clockwise from POV at N=3.


def _forest_goal_game(num_players: int, counts: tuple[int, ...]) -> state.GameState:
    gs = _new_game(num_players)
    forest_goal = cards.EndRoundGoal(
        id=0, description="[bird] in [forest]", category="birds_forest", tile_id=0
    )
    gs.round_goals = [forest_goal] * 4
    any_bird = cards.load_all()[0][0]
    for player, count in zip(gs.players, counts):
        player.board[cards.Habitat.FOREST] = [
            state.PlayedBird(bird=any_bird) for _ in range(count)
        ]
    return gs


def test_round_goal_other_counts_ride_clockwise_at_n3():
    gs = _forest_goal_game(3, (5, 2, 7))
    spec3 = encode.EncodingSpec(num_players=3)
    cont = layout.state_cont_layout(spec3)
    rg_off = cont.offset_of("round_goals")
    slot_dim = layout.round_goal_slot_dim(spec3)

    for pov in range(3):
        gs.current_player = pov
        vec = encode.encode_state(gs, decision=None, spec=spec3)
        base = rg_off  # round 0
        my_count = vec[base + layout._ROUND_GOAL_MY_COUNT] * layout._GOAL_COUNT_SCALE
        assert round(float(my_count)) == (5, 2, 7)[pov]

        others = scoring.round_goal_standing_for_round(
            gs, gs.players[pov], 0
        ).other_counts
        other_base = base + layout._ROUND_GOAL_OPP_COUNT
        for seat_offset, expected in enumerate(others):
            got = vec[other_base + seat_offset] * layout._GOAL_COUNT_SCALE
            assert round(float(got)) == expected

        vp_off = base + layout.round_goal_vp_offset(spec3)
        expected_vp = scoring.round_goal_standing_for_round(gs, gs.players[pov], 0).vp
        assert (
            round(float(vec[vp_off]) * layout._ROUND_GOAL_POINTS_SCALE) == expected_vp
        )
        assert slot_dim == layout.MAX_GOAL_CATEGORIES + 1 + 2 + 1  # my + 2 others + vp


# ---------------------------------------------------------------------------
# Full encode_state at N=3: shape + card-index block seat ordering.


def test_encode_state_n3_shape_and_card_index_ordering():
    gs = _new_game(3, seed=11)
    spec3 = encode.EncodingSpec(num_players=3)
    assert encode.state_size(spec3) == 1299

    bird_a, bird_b, bird_c = cards.load_all()[0][:3]
    gs.players[0].board[cards.Habitat.FOREST] = [state.PlayedBird(bird=bird_a)]
    gs.players[1].board[cards.Habitat.FOREST] = [state.PlayedBird(bird=bird_b)]
    gs.players[2].board[cards.Habitat.FOREST] = [state.PlayedBird(bird=bird_c)]
    gs.current_player = 0

    vec = encode.encode_state(gs, decision=None, spec=spec3)
    assert vec.shape == (1299,)

    off = layout.off_card_index(spec3)
    slots = layout._SLOTS_PER_BOARD
    me_slot0 = vec[off]
    opp1_slot0 = vec[off + slots]
    opp2_slot0 = vec[off + 2 * slots]
    assert int(me_slot0) == cards.bird_index(bird_a) + 1
    assert int(opp1_slot0) == cards.bird_index(bird_b) + 1  # clockwise opp 1 = player 1
    assert int(opp2_slot0) == cards.bird_index(bird_c) + 1  # clockwise opp 2 = player 2

    # Tray slots follow all three boards.
    tray_off = off + 3 * slots
    for slot, bird in enumerate(gs.tray):
        expected = cards.bird_index(bird) + 1 if bird is not None else 0
        assert int(vec[tray_off + slot]) == expected


def test_encode_state_n3_pov_rotation_keeps_clockwise_order():
    """From player 1's POV, opponents clockwise are (2, 0) — not (0, 2)."""
    gs = _new_game(3, seed=12)
    spec3 = encode.EncodingSpec(num_players=3)
    bird_a, bird_b, bird_c = cards.load_all()[0][:3]
    gs.players[0].board[cards.Habitat.FOREST] = [state.PlayedBird(bird=bird_a)]
    gs.players[1].board[cards.Habitat.FOREST] = [state.PlayedBird(bird=bird_b)]
    gs.players[2].board[cards.Habitat.FOREST] = [state.PlayedBird(bird=bird_c)]
    gs.current_player = 1

    vec = encode.encode_state(gs, decision=None, spec=spec3)
    off = layout.off_card_index(spec3)
    slots = layout._SLOTS_PER_BOARD
    assert int(vec[off]) == cards.bird_index(bird_b) + 1  # me = player 1
    assert int(vec[off + slots]) == cards.bird_index(bird_c) + 1  # opp1 clockwise = 2
    assert (
        int(vec[off + 2 * slots]) == cards.bird_index(bird_a) + 1
    )  # opp2 clockwise = 0
