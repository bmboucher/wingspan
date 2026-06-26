# pyright: reportPrivateUsage=false
# (tests read the v0_8 module's package-private frozen offsets and dims to pin
# the shim's block-copy contract; some tests also verify shim output row-by-row
# against game-state-derived values, matching the style of test_compat_shim_v0_7.py)
"""Unit tests for the ``wingspan.compat.v0_8`` combined state-and-choice shim.

Pins the shim's geometry directly across both FRESH changes introduced in v0.9:

**Choice-encoding tests** (board_target 120 → 60, board_idx removed):

* :func:`~wingspan.compat.v0_8.uses_v0_8_choice_encoding` covers exactly v0.1–v0.8
  (not v0.0, not v0.9).
* :func:`~wingspan.compat.v0_8.encode_choices_v08` produces a 395-dim row with
  120-dim ``board_target`` (per-type cached food), a 15-slot ``board_idx`` block,
  and ``bird_id`` zero on board-target rows (occupant encoding not present pre-0.9).
* The live (v0.9) encoder produces 328-dim rows — different width and layout.

**State-encoding tests** (1155 → 1119 compaction):

* :func:`~wingspan.compat.v0_8.uses_pre_v09_state_encoding` covers exactly 0.8 (not
  0.7, not 0.9); versions 0.6 and 0.7 are routed here via their own shims.
* :func:`~wingspan.compat.v0_8.encode_state_v08` produces the 1155-dim pre-0.9
  vector: ``hand_summary`` present (10 dims), 18-dim ``board_summary`` per seat,
  4-dim ``misc_scalars``, all ``round_goals`` slots filled regardless of scoring.
* :func:`~wingspan.compat.v0_8.state_embed_offsets_v08` returns the frozen offsets
  (card_index 562, hand_multihot 595, decision_type 1135, hand_summary 343..353).

**PolicyValueNetV08** covers both axes simultaneously so a single class handles any
pre-0.9 artifact.  Version routing: 0.8 → ``PolicyValueNetV08``,
0.7 → ``PolicyValueNetV07``, 0.9/live → ``PolicyValueNet``.
V06/V07 also produce the 1155-dim state via delegation.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import (
    architecture,
    cards,
    decisions,
    encode,
    engine,
    model,
    state,
    version,
)
from wingspan.compat import v0_6, v0_7, v0_8
from wingspan.encode import layout
from wingspan.training import runmeta

_SMALL = architecture.ModelArchitecture(
    trunk_layers=(8, 8),
    choice_layers=(8, 8),
    head_layers=(),
    value_layers=(),
    card_embed_dim=4,
)

_SPEC = encode.EncodingSpec()
_V08_CHOICE_DIM = v0_8.choice_feature_dim_v08(has_becomes_playable=True)

# Frozen v0.6–v0.8 state vector width under the default spec.
_V08_STATE_DIM = 1155
# Width change from v0.9 compaction.
_COMPACTION_DELTA = 36


# ---------------------------------------------------------------------------
# Version predicates


def test_version_predicate_covers_0_1_through_0_8():
    """``uses_v0_8_choice_encoding`` is True for v0.1 through v0.8."""
    for ver in ("0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8"):
        assert v0_8.uses_v0_8_choice_encoding(ver), f"expected True for {ver}"


def test_version_predicate_excludes_0_0():
    """Pre-0.1 artifacts use the v0.0 shim, not this one."""
    assert not v0_8.uses_v0_8_choice_encoding("0.0")


def test_version_predicate_excludes_live():
    """0.9+ artifacts use the live encoding."""
    assert not v0_8.uses_v0_8_choice_encoding("0.9")
    assert not v0_8.uses_v0_8_choice_encoding("1.0")


def test_version_predicate_covers_0_8():
    """``uses_pre_v09_state_encoding`` is True for exactly 0.8."""
    assert v0_8.uses_pre_v09_state_encoding("0.8")


def test_version_predicate_excludes_0_7():
    """0.7 artifacts are handled by the v0_7 shim (which delegates state here)."""
    assert not v0_8.uses_pre_v09_state_encoding("0.7")


def test_version_predicate_excludes_0_9_and_beyond():
    """0.9+ artifacts use the live compacted encoding."""
    assert not v0_8.uses_pre_v09_state_encoding("0.9")
    assert not v0_8.uses_pre_v09_state_encoding("1.0")


def test_version_predicate_excludes_pre_0_8():
    """Pre-0.8 artifacts fall under earlier shims."""
    assert not v0_8.uses_pre_v09_state_encoding("0.6")
    assert not v0_8.uses_pre_v09_state_encoding("0.5")
    assert not v0_8.uses_pre_v09_state_encoding("0.0")


# ---------------------------------------------------------------------------
# Frozen choice-dim formula


def test_state_feature_dim_v08_is_1155():
    """The frozen pre-0.9 state width is exactly 1155 (live 1119 + 36)."""
    assert v0_8.state_feature_dim_v08() == _V08_STATE_DIM
    assert v0_8.state_feature_dim_v08() == encode.state_size() + _COMPACTION_DELTA


# ---------------------------------------------------------------------------
# Row geometry (encode_choices_v08)


def test_encode_choices_v08_row_width():
    """``encode_choices_v08`` rows are 395 dims wide (not the 328-dim live rows)."""
    eng, birds, *_ = engine.Engine.create(seed=3)
    bird = next(b for b in birds if b.habitats)
    decision = decisions.PlayBirdDecision(
        player_id=0,
        prompt="play",
        choices=[
            decisions.PlayBirdChoice(label="x", bird=bird, habitat=bird.habitats[0])
        ],
    )
    rows = v0_8.encode_choices_v08(decision, eng.state, _SPEC)
    assert rows.shape == (1, _V08_CHOICE_DIM)
    live_rows = encode.encode_choices(decision, eng.state)
    assert live_rows.shape[1] == encode.choice_feature_dim()
    assert live_rows.shape[1] != _V08_CHOICE_DIM


def test_encode_choices_v08_board_target_is_120_dims():
    """The ``board_target`` block in a v0.8 row is 120 dims (15 slots × 8 scalars),
    not the 60-dim live block."""
    eng, *_ = engine.Engine.create(seed=5)
    eng.state.players[0].board[cards.Habitat.GRASSLAND] = [
        state.PlayedBird(bird=engine.Engine.create(seed=0)[1][0])
    ]
    target = decisions.BoardTargetChoice(
        label="x", habitat=cards.Habitat.GRASSLAND, slot=0
    )
    decision = decisions.LayEggDecision(player_id=0, prompt="lay", choices=[target])
    row = v0_8.encode_choices_v08(decision, eng.state, _SPEC)[0]

    # board_target sits at offset 18 in both v0.8 and live; width differs.
    board_block = row[v0_8._OFF_BOARD_V08 : v0_8._OFF_MAIN_ACTION_V08]
    assert board_block.shape == (v0_8._BOARD_TARGET_DIM_V08,)


def test_encode_choices_v08_board_idx_is_15_slots():
    """A board-target v0.8 row fills the 15-slot ``board_idx`` block with the
    occupant's bird index, and ``bird_id`` at offset 172 is zero (pre-0.9)."""
    eng, birds, *_ = engine.Engine.create(seed=7)
    eng.state.players[0].board[cards.Habitat.FOREST] = [state.PlayedBird(bird=birds[0])]
    target = decisions.BoardTargetChoice(
        label="x", habitat=cards.Habitat.FOREST, slot=0
    )
    decision = decisions.LayEggDecision(player_id=0, prompt="lay", choices=[target])
    row = v0_8.encode_choices_v08(decision, eng.state, _SPEC)[0]

    forest_slot = list(cards.ALL_HABITATS).index(cards.Habitat.FOREST) * state.ROW_SLOTS
    board_idx_block = row[v0_8._OFF_BOARD_IDX_V08 : v0_8._OFF_BIRD_ID_V08]
    assert board_idx_block[forest_slot] == cards.bird_index(birds[0]) + 1
    # bird_id is zero on board-target rows in v0.8 (occupant was in board_idx).
    assert row[v0_8._OFF_BIRD_ID_V08] == 0.0


def test_encode_choices_v08_bird_id_carries_candidate_on_placement_rows():
    """On play-bird rows, ``bird_id`` at the v0.8 offset carries the candidate's
    index (as in both v0.8 and live — only the offset differs)."""
    eng, birds, *_ = engine.Engine.create(seed=3)
    bird = next(b for b in birds if b.habitats)
    decision = decisions.PlayBirdDecision(
        player_id=0,
        prompt="play",
        choices=[
            decisions.PlayBirdChoice(label="x", bird=bird, habitat=bird.habitats[0])
        ],
    )
    row = v0_8.encode_choices_v08(decision, eng.state, _SPEC)[0]
    assert row[v0_8._OFF_BIRD_ID_V08] == cards.bird_index(bird) + 1


# ---------------------------------------------------------------------------
# encode_state_v08: shape, dtype, and structural differences from live


def test_encode_state_v08_output_is_1155_dims():
    """``encode_state_v08`` produces a 1155-dim float32 array."""
    eng, *_ = engine.Engine.create(seed=42)
    vec = v0_8.encode_state_v08(eng.state)
    assert vec.shape == (_V08_STATE_DIM,)
    assert vec.dtype == np.float32


def test_encode_state_v08_is_36_dims_wider_than_live():
    """The v0.8 frozen vector is exactly 36 dims wider than the live v0.9 one."""
    eng, *_ = engine.Engine.create(seed=7)
    v08_vec = v0_8.encode_state_v08(eng.state)
    live_vec = encode.encode_state(eng.state)
    assert len(v08_vec) - len(live_vec) == _COMPACTION_DELTA


def test_encode_state_v08_hand_summary_present():
    """The v0.8 frozen vector is 1155-dim; the live v0.9 vector is 1119-dim.

    The 36-dim gap includes the removed 10-dim hand_summary stripe, confirming
    the stripe is present in the frozen vector and absent from the live one."""
    eng, birds, *_ = engine.Engine.create(seed=3)
    eng.state.players[0].hand = birds[:3]

    v08_vec = v0_8.encode_state_v08(eng.state)
    live_vec = encode.encode_state(eng.state)

    assert len(v08_vec) == _V08_STATE_DIM
    assert len(live_vec) == encode.state_size()
    assert len(v08_vec) - len(live_vec) == _COMPACTION_DELTA


def test_encode_state_v08_board_summary_is_18_dims_per_seat():
    """The v0.8 frozen vector has 18-dim board_summary per seat (full_stats=True).

    The live v0.9 encoder keeps only 6 dims (row_length + total_eggs per habitat).
    We compare total vector lengths and check that the delta is exactly 36 (2 seats
    × 12 dropped dims + 10 hand_summary + 2 misc)."""
    eng, *_ = engine.Engine.create(seed=42)
    v08_vec = v0_8.encode_state_v08(eng.state)
    live_vec = encode.encode_state(eng.state)
    assert (
        len(v08_vec) - len(live_vec) == _COMPACTION_DELTA
    )  # 24 board + 10 hand + 2 misc


def test_encode_state_v08_scored_round_slot_nonzero():
    """In the v0.8 frozen vector a scored round's goal slot is filled (non-zero).

    The live v0.9 encoder zeros the scored-round slot. We manufacture a scored
    round directly by appending to ``scored_goals`` then verify the round_goals
    stripe differs between the two encoders."""
    from wingspan import state as state_module

    eng, *_ = engine.Engine.create(seed=5)

    # Force round 0 to appear "scored" without playing — append a dummy result.
    dummy_result = state_module.RoundGoalResult(counts=[3, 2], vp_awarded=[5, 1])
    eng.state.scored_goals.append(dummy_result)

    v08_vec = v0_8.encode_state_v08(eng.state)
    live_vec = encode.encode_state(eng.state)

    round_goals_off = layout.STATE_CONT_LAYOUT.offset_of("round_goals")
    round_goals_dim = layout.STATE_CONT_LAYOUT.size_of("round_goals")

    # The full round_goals region should differ: v0.8 fills scored rounds, live zeros them.
    assert not np.array_equal(
        v08_vec[round_goals_off : round_goals_off + round_goals_dim],
        live_vec[round_goals_off : round_goals_off + round_goals_dim],
    ), "round_goals region should differ between v0.8 (fill-all) and live (zero-scored)"


# ---------------------------------------------------------------------------
# state_embed_offsets_v08


def test_state_embed_offsets_v08_card_index():
    """The frozen card_index offset is 562 (36 more than the live v0.9 offset)."""
    offsets = v0_8.state_embed_offsets_v08()
    assert offsets.card_index == 562
    assert offsets.card_index == encode.OFF_CARD_INDEX + _COMPACTION_DELTA


def test_state_embed_offsets_v08_hand_multihot():
    """The frozen hand_multihot offset is 595 (36 more than the live v0.9 offset)."""
    offsets = v0_8.state_embed_offsets_v08()
    assert offsets.hand_multihot == 595
    assert offsets.hand_multihot == encode.OFF_HAND_MULTIHOT + _COMPACTION_DELTA


def test_state_embed_offsets_v08_decision_type():
    """The frozen decision_type offset is 1135 (36 more than the live v0.9 offset)."""
    offsets = v0_8.state_embed_offsets_v08()
    assert offsets.decision_type == 1135
    assert offsets.decision_type == encode.OFF_DECISION_TYPE + _COMPACTION_DELTA


def test_state_embed_offsets_v08_hand_summary_span():
    """The frozen hand_summary offset is 343 and hand_summary_end is 353."""
    offsets = v0_8.state_embed_offsets_v08()
    assert offsets.hand_summary == 343
    assert offsets.hand_summary_end == 353
    assert offsets.hand_summary_end - offsets.hand_summary == layout.HAND_SUMMARY_DIM


def test_state_embed_offsets_live_has_no_hand_summary():
    """The live v0.9 net returns hand_summary == hand_summary_end == 0 (stripe absent)."""
    net = model.PolicyValueNet(
        arch=_SMALL,
        state_dim=encode.state_size(),
        choice_dim=encode.choice_feature_dim(),
    )
    offsets = net._state_embed_offsets()
    assert offsets.hand_summary == 0
    assert offsets.hand_summary_end == 0


# ---------------------------------------------------------------------------
# PolicyValueNetV08


def test_policy_value_net_v08_default_choice_dim():
    """``PolicyValueNetV08`` defaults to the v0.8 frozen dims (state 1155, choice 395)."""
    net = v0_8.PolicyValueNetV08(arch=_SMALL)
    assert net.choice_dim == _V08_CHOICE_DIM
    assert net.choice_dim != encode.choice_feature_dim()
    assert net.state_dim == _V08_STATE_DIM
    assert net.state_dim != encode.state_size()


def test_policy_value_net_v08_encode_state_uses_frozen_encoder():
    """``PolicyValueNetV08.encode_state`` produces the 1155-dim frozen vector."""
    import wingspan.decisions as decisions_module

    eng, *_ = engine.Engine.create(seed=11)
    net = v0_8.PolicyValueNetV08(arch=_SMALL)
    decision: decisions_module.Decision[decisions_module.Choice] | None = None
    enc = net.encode_state(eng.state, decision)  # type: ignore[arg-type]
    assert enc.shape == (_V08_STATE_DIM,)
    assert np.array_equal(enc, v0_8.encode_state_v08(eng.state, decision))


def test_policy_value_net_v08_forward_pass_finite():
    """A batch of frozen-dim inputs (state 1155, choice 395) produces finite logits and value."""
    net = v0_8.PolicyValueNetV08(arch=_SMALL)
    net.eval()
    batch_size, n_choices = 2, 4
    state_vec = torch.zeros(batch_size, net.state_dim)
    choice_vec = torch.randn(batch_size, n_choices, net.choice_dim)
    mask = torch.ones(batch_size, n_choices)
    family = torch.zeros(batch_size, dtype=torch.long)
    with torch.no_grad():
        logits, value = net(state_vec, choice_vec, mask, family)
    assert logits.shape == (batch_size, n_choices)
    assert value.shape == (batch_size,)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()


def test_policy_value_net_v08_encode_choices_matches_shim():
    """``PolicyValueNetV08.encode_choices`` produces the same rows as
    ``encode_choices_v08``."""
    eng, birds, *_ = engine.Engine.create(seed=3)
    bird = next(b for b in birds if b.habitats)
    decision = decisions.PlayBirdDecision(
        player_id=0,
        prompt="play",
        choices=[
            decisions.PlayBirdChoice(label="x", bird=bird, habitat=bird.habitats[0])
        ],
    )
    net = v0_8.PolicyValueNetV08(arch=_SMALL)
    net_rows = net.encode_choices(decision, eng.state)  # type: ignore[arg-type]
    shim_rows = v0_8.encode_choices_v08(decision, eng.state, _SPEC)
    assert np.array_equal(net_rows, shim_rows)


# ---------------------------------------------------------------------------
# V06/V07 also produce 1155-dim state via delegation


def test_v06_encode_state_is_1155_dim():
    """``PolicyValueNetV06.encode_state`` delegates to v0_8 and returns 1155 dims."""
    import wingspan.decisions as decisions_module

    eng, *_ = engine.Engine.create(seed=99)
    net = v0_6.PolicyValueNetV06(arch=_SMALL)
    decision: decisions_module.Decision[decisions_module.Choice] | None = None
    enc = net.encode_state(eng.state, decision)  # type: ignore[arg-type]
    assert enc.shape == (_V08_STATE_DIM,)


def test_v07_encode_state_is_1155_dim():
    """``PolicyValueNetV07.encode_state`` delegates to v0_8 and returns 1155 dims."""
    import wingspan.decisions as decisions_module

    eng, *_ = engine.Engine.create(seed=99)
    net = v0_7.PolicyValueNetV07(arch=_SMALL)
    decision: decisions_module.Decision[decisions_module.Choice] | None = None
    enc = net.encode_state(eng.state, decision)  # type: ignore[arg-type]
    assert enc.shape == (_V08_STATE_DIM,)


# ---------------------------------------------------------------------------
# Version routing


def test_from_model_config_routes_0_8_to_v08():
    """A v0.8 descriptor reconstructs as ``PolicyValueNetV08`` with frozen dims."""
    v08_config = runmeta.ModelConfig(
        run_name="routing-v08",
        state_dim=v0_8.state_feature_dim_v08(),
        choice_dim=_V08_CHOICE_DIM,
        family_order=("main_action",),
        architecture=_SMALL,
        include_setup=False,
        version="0.8",
    )
    net = model.PolicyValueNet.from_model_config(v08_config)
    assert isinstance(net, v0_8.PolicyValueNetV08)
    assert net.state_dim == _V08_STATE_DIM
    assert net.choice_dim == _V08_CHOICE_DIM


def test_from_model_config_routes_0_7_to_v07():
    """A v0.7 descriptor reconstructs as ``PolicyValueNetV07`` (not V08)."""
    v07_config = runmeta.ModelConfig(
        run_name="routing-v07",
        state_dim=v0_8.state_feature_dim_v08(),
        choice_dim=_V08_CHOICE_DIM,
        family_order=("main_action",),
        architecture=_SMALL,
        include_setup=False,
        version="0.7",
    )
    net = model.PolicyValueNet.from_model_config(v07_config)
    assert isinstance(net, v0_7.PolicyValueNetV07)
    assert not isinstance(net, v0_8.PolicyValueNetV08)


def test_from_model_config_routes_live_to_base_net():
    """A current-version (0.9+) descriptor reconstructs as the live ``PolicyValueNet``."""
    live_config = runmeta.ModelConfig(
        run_name="routing-live",
        state_dim=encode.state_size(),
        choice_dim=encode.choice_feature_dim(),
        family_order=("main_action",),
        architecture=_SMALL,
        include_setup=False,
        version=version.MODEL_VERSION,
    )
    net = model.PolicyValueNet.from_model_config(live_config)
    assert type(net) is model.PolicyValueNet
    assert not isinstance(net, v0_8.PolicyValueNetV08)
    assert not isinstance(net, v0_7.PolicyValueNetV07)
    assert not isinstance(net, v0_6.PolicyValueNetV06)
