# pyright: reportPrivateUsage=false
# (tests call the shim's package-private geometry constants and check live layout
# constants — a deliberate compat coupling matching test_compat_shim_v0_3.py)
"""Unit tests for the ``wingspan.compat.v0_4`` state+choice encoding shim.

The end-to-end proof that a real pre-0.6 checkpoint loads and plays lives in
``test_compat_v0_4.py`` (the pinned fixture — to be captured after the next
short 0.6 training run, deferred per VERSIONING.md). These tests pin the shim's
geometry directly:

* The frozen state dim is 795 (live 1155 minus the 2 × 180 playability stripes).
* The frozen choice row is 180 narrower than live (no ``becomes_playable`` stripe).
* ``encode_state_v04`` omits the two playability multi-hots.
* ``encode_choices_v04`` omits the ``becomes_playable`` stripe from each row.
* ``PolicyValueNetV04`` builds at the frozen dims and slices at the frozen offsets.
* The version predicate covers 0.4 and 0.5 but not 0.6.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import architecture, encode, engine, model, version
from wingspan.compat import v0_4
from wingspan.encode import layout
from wingspan.training import runmeta

_SMALL = architecture.ModelArchitecture(
    trunk_layers=(8, 8),
    choice_layers=(8, 8),
    head_layers=(),
    value_layers=(),
    card_embed_dim=4,
)

# The v0.4/v0.5 fixture's expected state geometry.
_V04_STATE_DIM = 795
# Playability stripes added in v0.6: 2 × 180 dims.
_PLAYABLE_STRIPE_DIM = 360
# Pre-0.6 choice row is 180 dims narrower (no becomes_playable stripe).
_BECOMES_PLAYABLE_DIM = 180


# ---------------------------------------------------------------------------
# Version predicate


def test_version_predicate_covers_0_4_and_0_5():
    """``uses_v0_4_encoding`` is True for 0.4 and 0.5 (encoding-identical) and
    False for anything outside that range."""
    assert v0_4.uses_v0_4_encoding("0.4")
    assert v0_4.uses_v0_4_encoding("0.5")


def test_version_predicate_excludes_0_6_and_beyond():
    """0.6+ artifacts use the live encoding (no shim needed)."""
    assert not v0_4.uses_v0_4_encoding("0.6")
    assert not v0_4.uses_v0_4_encoding("0.7")
    assert not v0_4.uses_v0_4_encoding("1.0")


def test_version_predicate_excludes_pre_0_4():
    """Pre-0.4 artifacts fall under earlier shims, not this one."""
    assert not v0_4.uses_v0_4_encoding("0.0")
    assert not v0_4.uses_v0_4_encoding("0.3")


# ---------------------------------------------------------------------------
# Frozen state dims


def test_state_feature_dim_v04_is_795():
    """The frozen v0.4/v0.5 state width is exactly 795 (live 1155 minus 360).

    The 360-dim gap is exactly N_HAND_PLAYABLE_MULTIHOTS * HAND_MULTIHOT_DIM:
    the two new playability multi-hot stripes added in 0.6.
    """
    assert v0_4.state_feature_dim_v04() == _V04_STATE_DIM
    assert encode.state_size() - v0_4.state_feature_dim_v04() == _PLAYABLE_STRIPE_DIM


def test_state_feature_dim_v04_gap_is_playability_stripes():
    """The live → frozen state dim gap equals N_HAND_PLAYABLE_MULTIHOTS * HAND_MULTIHOT_DIM."""
    expected_delta = layout.N_HAND_PLAYABLE_MULTIHOTS * layout.HAND_MULTIHOT_DIM
    assert encode.state_size() - v0_4.state_feature_dim_v04() == expected_delta


# ---------------------------------------------------------------------------
# Frozen choice dims


def test_choice_feature_dim_v04_is_180_narrower():
    """The frozen pre-0.6 choice row is exactly 180 dims narrower than live."""
    assert (
        encode.choice_feature_dim() - v0_4.choice_feature_dim_v04()
        == _BECOMES_PLAYABLE_DIM
    )
    assert (
        encode.choice_feature_dim() - v0_4.choice_feature_dim_v04()
        == layout.CHOICE_BECOMES_PLAYABLE_DIM
    )


# ---------------------------------------------------------------------------
# Embed offsets


def test_state_embed_offsets_v04_hand_multihot_unchanged():
    """The ``hand_multihot`` offset is identical between the v0.4 and live vectors.

    The playability stripes sit AFTER hand_multihot (they were inserted between
    hand_multihot and the decision-type tail), so hand_multihot's position is
    the same in both era vectors."""
    offsets = v0_4.state_embed_offsets_v04()
    assert offsets.hand_multihot == encode.OFF_HAND_MULTIHOT


def test_state_embed_offsets_v04_decision_type_shifted_360():
    """The ``decision_type`` offset is 360 less in the v0.4 frozen vector.

    In the live vector two 180-dim playability stripes sit between hand_multihot
    and the decision-type one-hot; in the v0.4 vector they are absent, so the
    decision-type one-hot starts 360 columns earlier."""
    offsets = v0_4.state_embed_offsets_v04()
    assert offsets.decision_type == encode.OFF_DECISION_TYPE - _PLAYABLE_STRIPE_DIM


def test_state_embed_offsets_v04_card_index_and_hand_summary_unchanged():
    """card_index and hand_summary are before the insertion point — unchanged."""
    offsets = v0_4.state_embed_offsets_v04()
    assert offsets.card_index == encode.OFF_CARD_INDEX
    assert offsets.hand_summary == encode.HAND_SUMMARY_OFFSET


def test_state_embed_offsets_v04_hand_multihot_and_decision_type_contiguous():
    """In the v0.4 frozen vector hand_multihot and decision_type are directly
    adjacent — no intervening playability stripes."""
    offsets = v0_4.state_embed_offsets_v04()
    assert offsets.hand_multihot + layout.HAND_MULTIHOT_DIM == offsets.decision_type


# ---------------------------------------------------------------------------
# encode_state_v04 output shape and structure


def test_encode_state_v04_output_is_795_dims():
    """``encode_state_v04`` produces a 795-dim float32 array."""
    eng, *_ = engine.Engine.create(seed=42)
    vec = v0_4.encode_state_v04(eng.state)
    assert vec.shape == (_V04_STATE_DIM,)
    assert vec.dtype == np.float32


def test_encode_state_v04_is_360_dims_narrower_than_live():
    """v0.4 state vector is exactly 360 dims narrower than the live one."""
    eng, *_ = engine.Engine.create(seed=7)
    v04_vec = v0_4.encode_state_v04(eng.state)
    live_vec = encode.encode_state(eng.state)
    assert len(live_vec) - len(v04_vec) == _PLAYABLE_STRIPE_DIM


def test_encode_state_v04_hand_multihot_matches_live():
    """The hand multi-hot bytes at the frozen offset equal the live encoder's
    hand multi-hot — the playability stripes are AFTER hand_multihot, so
    everything up through hand_multihot is byte-identical between the two."""
    eng, birds, *_ = engine.Engine.create(seed=3)
    # Give the first player a real hand so the multi-hot has 1-bits.
    eng.state.players[0].hand = birds[:3]

    v04_vec = v0_4.encode_state_v04(eng.state)
    live_vec = encode.encode_state(eng.state)

    off = encode.OFF_HAND_MULTIHOT
    dim = layout.HAND_MULTIHOT_DIM
    assert np.array_equal(v04_vec[off : off + dim], live_vec[off : off + dim])


# ---------------------------------------------------------------------------
# encode_choices_v04 output shape


def test_encode_choices_v04_is_180_dims_narrower_per_row():
    """Each choice row from ``encode_choices_v04`` is 180 dims narrower than live."""
    import wingspan.decisions as decisions_module

    eng, *_ = engine.Engine.create(seed=9)
    decision = decisions_module.MainActionDecision(
        player_id=0,
        prompt="x",
        choices=[
            decisions_module.MainActionChoice(
                label="a", action=decisions_module.MainAction.GAIN_FOOD
            )
        ],
    )
    live_rows = encode.encode_choices(decision, eng.state)
    v04_rows = v0_4.encode_choices_v04(decision, eng.state)
    assert live_rows.shape[0] == v04_rows.shape[0]  # same number of choices
    assert live_rows.shape[1] - v04_rows.shape[1] == _BECOMES_PLAYABLE_DIM


# ---------------------------------------------------------------------------
# PolicyValueNetV04


def test_policy_value_net_v04_state_dim():
    """``PolicyValueNetV04`` built with the frozen 795 state_dim carries that dim."""
    net = v0_4.PolicyValueNetV04(arch=_SMALL, state_dim=v0_4.state_feature_dim_v04())
    assert net.state_dim == _V04_STATE_DIM
    assert net.state_dim != encode.state_size()


def test_policy_value_net_v04_choice_dim():
    """``PolicyValueNetV04`` built with the frozen choice_dim carries that dim."""
    net = v0_4.PolicyValueNetV04(
        arch=_SMALL,
        state_dim=v0_4.state_feature_dim_v04(),
        choice_dim=v0_4.choice_feature_dim_v04(),
    )
    assert net.choice_dim == v0_4.choice_feature_dim_v04()
    assert net.choice_dim != encode.choice_feature_dim()


def test_policy_value_net_v04_state_embed_offsets():
    """The v0.4 net returns the frozen state embed offsets (decision_type shifted -360)."""
    net = v0_4.PolicyValueNetV04(arch=_SMALL, state_dim=v0_4.state_feature_dim_v04())
    frozen = net._state_embed_offsets()
    assert frozen == v0_4.state_embed_offsets_v04()
    assert frozen.decision_type == encode.OFF_DECISION_TYPE - _PLAYABLE_STRIPE_DIM
    assert frozen.hand_multihot == encode.OFF_HAND_MULTIHOT  # unchanged


def test_policy_value_net_v04_choice_embed_offsets_becomes_playable_none():
    """The v0.4 net returns ``becomes_playable=None`` from ``_choice_embed_offsets``.

    Pre-0.6 choice vectors have no ``becomes_playable`` stripe; returning None here
    causes ``_embed_choices`` to skip the stripe and avoid reading garbage data."""
    net = v0_4.PolicyValueNetV04(arch=_SMALL, state_dim=v0_4.state_feature_dim_v04())
    cho = net._choice_embed_offsets()
    assert cho.becomes_playable is None


def test_policy_value_net_v04_encode_state_uses_frozen_encoder():
    """``PolicyValueNetV04.encode_state`` delegates to ``encode_state_v04``."""
    import wingspan.decisions as decisions_module

    eng, *_ = engine.Engine.create(seed=11)
    net = v0_4.PolicyValueNetV04(arch=_SMALL, state_dim=v0_4.state_feature_dim_v04())
    decision: decisions_module.Decision[decisions_module.Choice] | None = None
    enc = net.encode_state(eng.state, decision)  # type: ignore[arg-type]
    assert enc.shape == (_V04_STATE_DIM,)
    assert np.array_equal(enc, v0_4.encode_state_v04(eng.state, decision))


def test_policy_value_net_v04_forward_pass_finite():
    """A batch of synthetic 795-dim state inputs produces finite logits and value."""
    net = v0_4.PolicyValueNetV04(
        arch=_SMALL,
        state_dim=v0_4.state_feature_dim_v04(),
        choice_dim=v0_4.choice_feature_dim_v04(),
    )
    net.eval()
    batch_size, n_choices = 2, 4
    state_vec = torch.zeros(batch_size, net.state_dim)
    choices = torch.randn(batch_size, n_choices, net.choice_dim)
    mask = torch.ones(batch_size, n_choices)
    family = torch.zeros(batch_size, dtype=torch.long)
    with torch.no_grad():
        logits, value = net(state_vec, choices, mask, family)
    assert logits.shape == (batch_size, n_choices)
    assert value.shape == (batch_size,)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()


# ---------------------------------------------------------------------------
# Version routing


def test_from_model_config_routes_v04_to_shim():
    """A v0.4 descriptor reconstructs as ``PolicyValueNetV04``."""
    v04_config = runmeta.ModelConfig(
        run_name="routing-v04",
        state_dim=v0_4.state_feature_dim_v04(),
        choice_dim=v0_4.choice_feature_dim_v04(),
        family_order=("main_action",),
        architecture=_SMALL,
        include_setup=False,
        version="0.4",
    )
    v04_net = model.PolicyValueNet.from_model_config(v04_config)
    assert isinstance(v04_net, v0_4.PolicyValueNetV04)
    assert v04_net.state_dim == _V04_STATE_DIM


def test_from_model_config_routes_v05_to_shim():
    """A v0.5 descriptor also reconstructs as ``PolicyValueNetV04`` (same encoding)."""
    v05_config = runmeta.ModelConfig(
        run_name="routing-v05",
        state_dim=v0_4.state_feature_dim_v04(),
        choice_dim=v0_4.choice_feature_dim_v04(),
        family_order=("main_action",),
        architecture=_SMALL,
        include_setup=False,
        version="0.5",
    )
    v05_net = model.PolicyValueNet.from_model_config(v05_config)
    assert isinstance(v05_net, v0_4.PolicyValueNetV04)


def test_from_model_config_routes_v06_to_live():
    """A 0.6 descriptor reconstructs as the live ``PolicyValueNet`` (not the shim)."""
    live_config = runmeta.ModelConfig(
        run_name="routing-live",
        state_dim=encode.state_size(),
        choice_dim=encode.choice_feature_dim(),
        family_order=("main_action",),
        architecture=_SMALL,
        include_setup=False,
        version=version.MODEL_VERSION,
    )
    live_net = model.PolicyValueNet.from_model_config(live_config)
    assert not isinstance(live_net, v0_4.PolicyValueNetV04)
