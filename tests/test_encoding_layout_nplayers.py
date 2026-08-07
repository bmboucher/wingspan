# pyright: reportPrivateUsage=false
# (assertions reference package-private layout constants/accessors, e.g.
# _SLOTS_PER_BOARD / _opponent_suffix, the same intra-package coupling
# convention test_encode_spec.py already uses)
"""N-player encoding layout: dims, offsets, and stripe presence.

``EncodingSpec.num_players`` (default 2) is the encoders' second config-driven
axis, alongside ``include_setup``. These tests pin three invariants:

1. **N=2 is the checkpoint-compat anchor** — every stripe offset/size at the
   default spec matches the frozen pre-Stage-2 table, and every new
   ``spec``-aware accessor equals its frozen module-constant counterpart at
   ``DEFAULT_SPEC``.
2. **N=3/N=4 dims follow the documented per-opponent-block arithmetic**: each
   extra opponent adds 343 state dims (food 5 + board 135 + board_summary 6 +
   bonus_count 1 + hand_size 1 + 15 card-index slots + 180 known-hand identity
   multi-hot) plus 4 round-goal opponent-count dims (1 per round); N>=3
   additionally adds a ``turn_position`` one-hot (width N) and a
   ``player_select`` choice stripe (width N).
3. **Stripe adjacency** — each per-opponent replica (``food_opp2``, ...)
   sits immediately after its singular sibling, and ``turn_position`` /
   ``player_select`` exist only at N>=3.
"""

from __future__ import annotations

from wingspan.encode import layout

_SPEC_N2 = layout.EncodingSpec()
_SPEC_N3 = layout.EncodingSpec(num_players=3)
_SPEC_N4 = layout.EncodingSpec(num_players=4)
_SPEC_N5 = layout.EncodingSpec(num_players=5)


# ---------------------------------------------------------------------------
# (a) N=2 frozen table — the checkpoint-compat anchor


def test_n2_state_stripe_table_is_frozen():
    """Exact (offset, size) for the stripes the plan's frozen table names."""
    cont = layout.STATE_CONT_LAYOUT
    expected = {
        "food_opp": (32, 5),
        "board_opp": (182, 135),
        "board_summary_opp": (323, 6),
        "opp_bonus_count": (433, 1),
        "opp_hand_size": (434, 1),
        "round_goals": (444, 92),
        "card_idx_block": (536, 33),
        "hand_multihot": (569, 180),
        "known_hand_opp": (1109, 180),  # v1.8: tail of the multi-hot region
    }
    for name, (offset, size) in expected.items():
        assert cont.offset_of(name) == offset, name
        assert cont.size_of(name) == size, name
    assert layout.OFF_DECISION_TYPE == 1289


def test_n2_totals_match_pre_stage2_dims():
    assert layout.state_feature_dim(_SPEC_N2) == 1309
    setup_spec = layout.EncodingSpec(include_setup=True)
    assert layout.state_feature_dim(setup_spec) == 1310
    assert layout.choice_feature_dim(_SPEC_N2) == 517
    setup_choice_spec = layout.EncodingSpec(include_setup=True, num_players=2)
    assert layout.choice_feature_dim(setup_choice_spec) == 701


def test_n2_has_no_turn_position_or_player_select():
    state_names = {s.name for s in layout.state_cont_layout(_SPEC_N2).stripes}
    choice_names = {s.name for s in layout.choice_base_layout(_SPEC_N2).stripes}
    assert "turn_position" not in state_names
    assert "player_select" not in choice_names
    assert layout.off_player_select(_SPEC_N2) is None


# ---------------------------------------------------------------------------
# (b) accessor(DEFAULT_SPEC) == module constant, for every pair


def test_accessors_equal_default_spec_module_constants():
    assert layout.n_board_index_slots(_SPEC_N2) == layout.N_BOARD_INDEX_SLOTS
    assert layout.n_card_index_slots(_SPEC_N2) == layout.N_CARD_INDEX_SLOTS
    assert layout.off_card_index(_SPEC_N2) == layout.OFF_CARD_INDEX
    assert layout.off_hand_multihot(_SPEC_N2) == layout.OFF_HAND_MULTIHOT
    assert layout.off_decision_type(_SPEC_N2) == layout.OFF_DECISION_TYPE
    assert layout.round_goal_slot_dim(_SPEC_N2) == layout._ROUND_GOAL_SLOT_DIM
    assert layout.round_goal_vp_offset(_SPEC_N2) == layout._ROUND_GOAL_VP
    assert layout.round_goals_stripe_dim(_SPEC_N2) == layout._ROUND_GOALS_STRIPE_DIM
    assert layout.off_board(_SPEC_N2, 0) == layout.OFF_BOARD_ME
    assert layout.off_board(_SPEC_N2, 1) == layout.OFF_BOARD_OPP
    assert layout.state_cont_layout(_SPEC_N2) == layout.STATE_CONT_LAYOUT
    assert layout.choice_base_layout(_SPEC_N2) == layout.CHOICE_BASE_LAYOUT
    full_setup_spec = layout.EncodingSpec(include_setup=True, num_players=2)
    assert layout.choice_full_layout(full_setup_spec) == layout.CHOICE_FULL_LAYOUT


# ---------------------------------------------------------------------------
# (c) N=3 / N=4 totals


def test_n3_totals():
    assert layout.state_feature_dim(_SPEC_N3) == 1659
    assert layout.choice_feature_dim(_SPEC_N3) == 520


def test_n4_totals():
    assert layout.state_feature_dim(_SPEC_N4) == 2007
    assert layout.choice_feature_dim(_SPEC_N4) == 521


def test_n5_totals_follow_the_same_per_opponent_arithmetic():
    # Each additional opponent beyond N=2 adds 163 state dims + 4 round-goal
    # dims + 180 for its known_hand_opp{k} identity multi-hot (v1.8); turn_position
    # grows from non-existent (N=2) to width 3 (N=3, a +3 jump), then widens by 1
    # per additional seat beyond that.
    n2 = layout.state_feature_dim(_SPEC_N2)
    n3 = layout.state_feature_dim(_SPEC_N3)
    n4 = layout.state_feature_dim(_SPEC_N4)
    n5 = layout.state_feature_dim(_SPEC_N5)
    assert n3 - n2 == 163 + 180 + 4 + 3  # turn_position appears at width 3
    assert (
        n4 - n3 == 163 + 180 + 4 + 1
    )  # +1 for turn_position widening by one more seat
    assert n5 - n4 == 163 + 180 + 4 + 1
    assert (
        layout.choice_feature_dim(_SPEC_N5) - layout.choice_feature_dim(_SPEC_N4) == 1
    )


# ---------------------------------------------------------------------------
# (d) per-opponent stripe adjacency


def test_replicated_opponent_stripes_sit_immediately_after_their_singular():
    cont = layout.state_cont_layout(_SPEC_N3)
    for base in (
        "food_opp",
        "board_opp",
        "board_summary_opp",
        "opp_bonus_count",
        "opp_hand_size",
        "known_hand_opp",
    ):
        singular = cont.offset_of(base)
        singular_size = cont.size_of(base)
        replica_name = f"{base}2"
        assert cont.offset_of(replica_name) == singular + singular_size, base
        assert cont.size_of(replica_name) == singular_size, base


def test_n4_has_two_replicas_per_opponent_group_in_clockwise_order():
    cont = layout.state_cont_layout(_SPEC_N4)
    names = [s.name for s in cont.stripes]
    # food_me, food_opp, food_opp2, food_opp3 all appear, in that order.
    food_names = [n for n in names if n.startswith("food_")]
    assert food_names == ["food_me", "food_opp", "food_opp2", "food_opp3"]
    board_names = [n for n in names if n == "board_me" or n.startswith("board_opp")]
    assert board_names == ["board_me", "board_opp", "board_opp2", "board_opp3"]


def test_card_idx_block_width_covers_every_seat_plus_tray():
    from wingspan import state as state_module

    for spec in (_SPEC_N2, _SPEC_N3, _SPEC_N4):
        expected = spec.num_players * layout._SLOTS_PER_BOARD + state_module.TRAY_SIZE
        assert layout.n_card_index_slots(spec) == expected
        assert layout.state_cont_layout(spec).size_of("card_idx_block") == expected


# ---------------------------------------------------------------------------
# (e) turn_position / player_select presence


def test_turn_position_present_only_at_n_ge_3():
    for spec in (_SPEC_N3, _SPEC_N4, _SPEC_N5):
        cont = layout.state_cont_layout(spec)
        assert cont.size_of("turn_position") == spec.num_players
        # Immediately after turn_state.
        turn_state_end = cont.offset_of("turn_state") + cont.size_of("turn_state")
        assert cont.offset_of("turn_position") == turn_state_end

    cont2 = layout.state_cont_layout(_SPEC_N2)
    assert "turn_position" not in {s.name for s in cont2.stripes}


def test_player_select_present_only_at_n_ge_3_and_sized_to_num_players():
    for spec in (_SPEC_N3, _SPEC_N4, _SPEC_N5):
        base_layout = layout.choice_base_layout(spec)
        off = layout.off_player_select(spec)
        assert off is not None
        assert base_layout.size_of("player_select") == spec.num_players
        # Directly after goal_delta_ignoring_eggs (the last base stripe before it).
        tail_off = base_layout.offset_of("goal_delta_ignoring_eggs")
        tail_size = base_layout.size_of("goal_delta_ignoring_eggs")
        assert off == tail_off + tail_size
        # And before any conditional setup stripes.
        setup_spec = layout.EncodingSpec(
            include_setup=True, num_players=spec.num_players
        )
        full_layout = layout.choice_full_layout(setup_spec)
        assert full_layout.offset_of("setup_agg") == off + spec.num_players
