# pyright: reportPrivateUsage=false
# (tests call the shim's package-private _TOTAL_DIM_DELTA and check the live
# layout's stripe constants — a deliberate compat coupling, matching the
# convention in test_compat_shim_v0_0.py and compat/v0_3.py)
"""Unit tests for the ``wingspan.compat.v0_3`` state-encoding shim.

The end-to-end proof that a real pre-0.4 checkpoint loads and plays lives in
``test_compat_v0_3.py`` (the pinned fixture — to be captured after v0.4 merges);
these tests pin the shim's geometry directly: the frozen dims match the v0.3
era, the live → v0.3 state transform omits the turn_state stripe and splices the
frozen 26-dim misc stripe, and ``PolicyValueNetV03`` builds and encodes in the
frozen geometry.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import architecture, encode, engine, model, version
from wingspan.compat import v0_3
from wingspan.encode import layout
from wingspan.training import runmeta

_SMALL = architecture.ModelArchitecture(
    trunk_layers=(8, 8),
    choice_layers=(8, 8),
    head_layers=(),
    value_layers=(),
    card_embed_dim=4,
)

# The v0.3 fixture's expected state geometry.
_V03_STATE_DIM = 790
_V03_DIM_DELTA = -5  # live 795 - frozen 790


# ---------------------------------------------------------------------------
# Frozen dims and the version predicate


def test_version_predicate_selects_only_0_3_range():
    """``uses_v0_3_state_encoding`` is True iff the artifact version is 0.3
    and False for 0.2, 0.4 (the change point), or any 1.x."""
    assert v0_3.uses_v0_3_state_encoding("0.3")
    assert not v0_3.uses_v0_3_state_encoding("0.2")
    assert not v0_3.uses_v0_3_state_encoding(v0_3.MISC_SCALARS_CHANGED_IN)
    assert not v0_3.uses_v0_3_state_encoding("0.4")
    assert not v0_3.uses_v0_3_state_encoding("1.0")


def test_state_feature_dim_v03_is_790():
    """The frozen v0.3 state width is exactly 790 (live 795 minus 5)."""
    assert v0_3.state_feature_dim_v03() == _V03_STATE_DIM
    assert v0_3.state_feature_dim_v03() == encode.state_size() + _V03_DIM_DELTA
    assert v0_3._TOTAL_DIM_DELTA == _V03_DIM_DELTA


# ---------------------------------------------------------------------------
# Embed offsets


def test_state_embed_offsets_v03_shift_by_minus_5():
    """The v0.3 embed offsets sit 5 columns left of the live v0.4 offsets.

    v0.4 added a 27-dim turn_state stripe and shrank misc from 26 → 4 dims,
    net +5 before the card-index block; the v0.3 vector is 5 dims narrower.
    The widths of the sliced regions coincide, so slicing live would corrupt
    the trunk input silently rather than crash."""
    off_card, off_hand, off_decision = v0_3.state_embed_offsets_v03()
    assert off_card == encode.OFF_CARD_INDEX + _V03_DIM_DELTA
    assert off_hand == encode.OFF_HAND_MULTIHOT + _V03_DIM_DELTA
    assert off_decision == encode.OFF_DECISION_TYPE + _V03_DIM_DELTA
    # The three regions must remain contiguous (widths unchanged between eras).
    assert off_card + encode.N_CARD_INDEX_SLOTS == off_hand
    assert off_hand + encode.HAND_MULTIHOT_DIM == off_decision


# ---------------------------------------------------------------------------
# encode_state_v03 output shape and structure


def test_encode_state_v03_output_is_790_dims():
    """``encode_state_v03`` concatenates the frozen 790-dim state vector.

    This is the v0.3 format: no leading turn_state stripe, the misc_scalars
    stripe is the frozen 26-dim one-hot version rather than the live 4-dim
    one, everything else is identical to the live encoder."""
    eng, *_ = engine.Engine.create(seed=42)
    vec = v0_3.encode_state_v03(eng.state)
    assert vec.shape == (
        _V03_STATE_DIM,
    ), f"Expected 790-dim v0.3 state vector, got {vec.shape}"
    assert vec.dtype == np.float32


def test_encode_state_v03_is_5_dims_narrower_than_live():
    """v0.3 state vector must be exactly 5 dims narrower than the live one."""
    eng, *_ = engine.Engine.create(seed=7)
    v03_vec = v0_3.encode_state_v03(eng.state)
    live_vec = encode.encode_state(eng.state)
    assert len(live_vec) - len(v03_vec) == 5


# ---------------------------------------------------------------------------
# turn_state stripe behaviour (new in v0.4, absent from v0.3)


def test_turn_state_stripe_all_zeros_during_setup():
    """During game setup (``turn_counter == 0``) the leading 26-dim turn
    one-hot in the live state vector must be all zeros (setup position)."""
    from wingspan.encode import state_encode

    eng, *_ = engine.Engine.create(seed=99)
    # A freshly created engine is in setup (turn_counter == 0).
    assert eng.state.turn_counter == 0
    me = eng.state.players[0]
    turn_vec = state_encode._summary_turn_state(eng.state, me)
    assert turn_vec.shape == (layout.N_PLAYER_TURNS + 1,)
    # The 26-dim one-hot prefix must be all zeros during setup.
    assert np.all(
        turn_vec[: layout.N_PLAYER_TURNS] == 0.0
    ), f"Expected all-zeros turn one-hot during setup; got {turn_vec}"


# ---------------------------------------------------------------------------
# state_stripe_layout_v03 structure


def test_state_stripe_layout_v03_omits_turn_state():
    """The v0.3 stripe layout has no ``turn_state`` stripe."""
    frozen_layout = v0_3.state_stripe_layout_v03()
    stripe_names = [stripe.name for stripe in frozen_layout.stripes]
    assert (
        "turn_state" not in stripe_names
    ), "v0.3 stripe layout must not contain the v0.4 turn_state stripe"


def test_state_stripe_layout_v03_has_26_dim_misc():
    """The v0.3 stripe layout's ``misc_scalars`` is 26 dims (frozen one-hot
    version), not the live 4-dim scalar-only version."""
    frozen_layout = v0_3.state_stripe_layout_v03()
    misc = next(
        (stripe for stripe in frozen_layout.stripes if stripe.name == "misc_scalars"),
        None,
    )
    assert misc is not None, "Expected misc_scalars stripe in v0.3 layout"
    assert (
        misc.size == 26
    ), f"Expected 26-dim frozen misc_scalars in v0.3 layout, got {misc.size}"
    # Sub-fields should include the one-hot descriptors (round, cubes).
    sub_names = [sub.name for sub in misc.sub_fields]
    assert "round_index" in sub_names
    assert "my_action_cubes" in sub_names
    assert "opp_action_cubes" in sub_names


def test_state_stripe_layout_v03_total_is_5_less_than_live():
    """The v0.3 layout's total_size is 5 less than the live layout's.

    ``VectorLayout.total_size`` includes the post-embedding representation
    (card-index stripes store embed-dim columns, not raw indices), so the
    absolute value is not 790 — but the 5-dim gap between v0.3 and v0.4 must
    be preserved so the trunk input remains consistently shaped."""
    from wingspan.encode.stripes import state as live_state_stripes

    frozen_layout = v0_3.state_stripe_layout_v03()
    live_layout = live_state_stripes.state_stripe_layout()
    assert frozen_layout.total_size == live_layout.total_size + _V03_DIM_DELTA, (
        f"Expected v0.3 total_size to be 5 less than live "
        f"({live_layout.total_size}), got {frozen_layout.total_size}"
    )


# ---------------------------------------------------------------------------
# PolicyValueNetV03 — frozen-era net


def test_policy_value_net_v03_state_dim():
    """``PolicyValueNetV03`` built with the frozen state_dim carries 790 dims.

    The constructor's ``state_dim`` parameter defaults to the live encoder's
    width — pass the frozen dim explicitly (as ``from_model_config`` does when
    loading a v0.3 artifact) to get the 790-dim trunk."""
    net = v0_3.PolicyValueNetV03(arch=_SMALL, state_dim=v0_3.state_feature_dim_v03())
    assert net.state_dim == _V03_STATE_DIM
    assert net.state_dim != encode.state_size()


def test_policy_value_net_v03_embed_offsets():
    """The v0.3 net overrides ``_state_embed_offsets`` with the frozen -5
    shifted offsets — not the live 795-dim ones."""
    net = v0_3.PolicyValueNetV03(arch=_SMALL, state_dim=v0_3.state_feature_dim_v03())
    assert net._state_embed_offsets() == v0_3.state_embed_offsets_v03()
    assert net._state_embed_offsets() != (
        encode.OFF_CARD_INDEX,
        encode.OFF_HAND_MULTIHOT,
        encode.OFF_DECISION_TYPE,
    )


def test_policy_value_net_v03_encode_state_uses_frozen_encoder():
    """``PolicyValueNetV03.encode_state`` produces a 790-dim vector (calls
    ``encode_state_v03``), not the live 795-dim encoder."""
    eng, *_ = engine.Engine.create(seed=11)
    net = v0_3.PolicyValueNetV03(arch=_SMALL, state_dim=v0_3.state_feature_dim_v03())
    import wingspan.decisions as decisions_module

    decision: decisions_module.Decision[decisions_module.Choice] | None = None
    # encode_state on the net must return a 790-dim array.
    enc = net.encode_state(eng.state, decision)  # type: ignore[arg-type]
    assert enc.shape == (_V03_STATE_DIM,)
    assert np.array_equal(enc, v0_3.encode_state_v03(eng.state, decision))


def test_policy_value_net_v03_forward_pass():
    """A batch of synthetic 790-dim inputs flows through the frozen net to
    finite logits and value — the frozen state trunk accepts its own geometry."""
    net = v0_3.PolicyValueNetV03(arch=_SMALL, state_dim=v0_3.state_feature_dim_v03())
    net.eval()
    batch_size, n_choices = 2, 4
    state_vec = torch.zeros(batch_size, net.state_dim)  # 790
    choices = torch.randn(batch_size, n_choices, net.choice_dim)
    mask = torch.ones(batch_size, n_choices)
    family = torch.zeros(batch_size, dtype=torch.long)
    with torch.no_grad():
        logits, value = net(state_vec, choices, mask, family)
    assert logits.shape == (batch_size, n_choices)
    assert value.shape == (batch_size,)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()


def test_from_model_config_routes_v0_3_to_shim():
    """A v0.3 descriptor reconstructs as ``PolicyValueNetV03`` (frozen 790-dim
    state); a current-version (0.4) descriptor reconstructs as the live net."""
    v03_config = runmeta.ModelConfig(
        run_name="routing-v03",
        state_dim=v0_3.state_feature_dim_v03(),
        choice_dim=encode.choice_feature_dim(),
        family_order=("main_action",),
        architecture=_SMALL,
        include_setup=False,
        version="0.3",
    )
    v03_net = model.PolicyValueNet.from_model_config(v03_config)
    assert isinstance(v03_net, v0_3.PolicyValueNetV03)
    assert v03_net.state_dim == _V03_STATE_DIM

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
    assert not isinstance(live_net, v0_3.PolicyValueNetV03)
