# pyright: reportPrivateUsage=false
# (tests access _state_embed_offsets / _choice_embed_offsets to pin the seams)
"""Tests for the pre-1.4 -> v1.4 compat shim.

v1.4 folded two independent encoding changes into one era, and
``compat.v1_3.PolicyValueNetV1_3`` reverses both for a pre-1.4 checkpoint:

* the two food-unlock **state** stripes (``hand_food_unlock_me`` /
  ``tray_food_unlock_me``) appended to the continuous state prefix, and
* the 1-dim ``resets_feeder`` **choice** stripe appended after
  ``becomes_unplayable``.

The shim strips each from the live-encoded vectors and freezes the pre-1.4 state-
and choice-embed offsets so a pre-1.4 checkpoint computes identically to what it
did at training time.

Unlike v1.0 (which had no real artifacts and used a freshly-built tensor), a real
v1.3 checkpoint's geometry is exercised end-to-end by
``test_v1_3_stamped_checkpoint_round_trips`` — it builds an era net, saves it with a
v1.3 stamp, reloads through the production ``load_policy_net`` path (which hands the
constructor the era's already-narrow dims), and forward-passes. That is the test the
synthetic v1.0 tensor test cannot be: it would fail on any double-subtraction of a
stripe width.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wingspan import (
    architecture,
    compat,
    decisions,
    encode,
    engine,
    model,
    state,
    version,
)
from wingspan.compat import v1_0 as compat_v1_0
from wingspan.compat import v1_3 as compat_v1_3
from wingspan.model import core
from wingspan.players import loaders
from wingspan.training import config

_STATE_STRIPE_WIDTH = 2 * encode.STATE_FOOD_UNLOCK_DIM  # both 5-wide state stripes


def _small_arch() -> architecture.ModelArchitecture:
    return architecture.ModelArchitecture(
        trunk_layers=(8, 8),
        choice_layers=(8, 8),
        head_layers=(),
        value_layers=(),
        card_embed_dim=4,
    )


def _era_shim(
    era: str = "1.3",
    arch: architecture.ModelArchitecture | None = None,
    spec: encode.EncodingSpec = encode.DEFAULT_SPEC,
) -> compat_v1_3.PolicyValueNetV1_3:
    """A v1_3 shim built at ``era``'s (narrow) dims — exactly how the load path
    (``encoding_dims_for_era`` -> constructor) builds it."""
    arch = arch or _small_arch()
    state_dim, choice_dim = compat.encoding_dims_for_era(era, spec)
    return compat_v1_3.PolicyValueNetV1_3(
        state_dim=state_dim, choice_dim=choice_dim, arch=arch, spec=spec
    )


def _decision() -> decisions.MainActionDecision:
    return decisions.MainActionDecision(
        player_id=0,
        prompt="action",
        choices=[
            decisions.MainActionChoice(
                label="food", action=decisions.MainAction.GAIN_FOOD
            ),
            decisions.MainActionChoice(
                label="eggs", action=decisions.MainAction.LAY_EGGS
            ),
        ],
    )


def _forward(
    net: core.PolicyValueNet,
    decision: decisions.MainActionDecision,
    game_state: state.GameState,
) -> None:
    state_vec = net.encode_state(game_state, decision)
    choice_feats = net.encode_choices(decision, game_state)
    family_idx = decisions.family_index_for(type(decision))
    logits, value = net(
        torch.from_numpy(state_vec).unsqueeze(0),
        torch.from_numpy(choice_feats).unsqueeze(0),
        torch.ones(1, choice_feats.shape[0]),
        torch.tensor([family_idx], dtype=torch.long),
    )
    assert logits.shape == (1, choice_feats.shape[0])
    assert value.shape == (1,)


# ---------------------------------------------------------------------------
# class_for_version routing


class TestClassForVersionRouting:
    def test_v1_1_through_v1_3_route_to_shim(self) -> None:
        for era in ("1.1", "1.2", "1.3"):
            assert (
                core.PolicyValueNet.class_for_version(era)
                is compat_v1_3.PolicyValueNetV1_3
            )

    def test_v1_0_routes_to_v1_0_shim(self) -> None:
        assert (
            core.PolicyValueNet.class_for_version("1.0")
            is compat_v1_0.PolicyValueNetV1_0
        )

    def test_current_version_returns_live_class(self) -> None:
        assert (
            core.PolicyValueNet.class_for_version(version.MODEL_VERSION)
            is core.PolicyValueNet
        )


# ---------------------------------------------------------------------------
# encoding_dims_for_era: both the state and choice narrowings


class TestEncodingDimsForEra:
    def test_state_dim_narrower_by_ten_for_pre_1_4(self) -> None:
        spec = encode.DEFAULT_SPEC
        live_state = encode.state_size(spec)
        for era in ("1.0", "1.1", "1.2", "1.3"):
            state_dim, _ = compat.encoding_dims_for_era(era, spec)
            assert live_state - state_dim == _STATE_STRIPE_WIDTH

    def test_choice_dim_narrower_by_resets_feeder_for_pre_1_4(self) -> None:
        spec = encode.DEFAULT_SPEC
        live_choice = encode.choice_feature_dim(spec)
        for era in ("1.1", "1.2", "1.3"):
            _, choice_dim = compat.encoding_dims_for_era(era, spec)
            assert live_choice - choice_dim == encode.CHOICE_RESETS_FEEDER_DIM

    def test_dims_are_live_for_current_era(self) -> None:
        spec = encode.DEFAULT_SPEC
        state_dim, choice_dim = compat.encoding_dims_for_era(
            version.MODEL_VERSION, spec
        )
        assert state_dim == encode.state_size(spec)
        assert choice_dim == encode.choice_feature_dim(spec)

    def test_v1_0_choice_dim_drops_both_choice_stripes(self) -> None:
        """v1.0 predates both the v1.1 becomes_unplayable stripe and the v1.4
        resets_feeder stripe, so its choice_dim drops both."""
        spec = encode.DEFAULT_SPEC
        _, choice_dim = compat.encoding_dims_for_era("1.0", spec)
        assert encode.choice_feature_dim(spec) - choice_dim == (
            encode.CHOICE_BECOMES_UNPLAYABLE_DIM + encode.CHOICE_RESETS_FEEDER_DIM
        )


# ---------------------------------------------------------------------------
# The v1_3 shim strips the two state stripes


class TestV1_3StateStripeStripping:
    def test_encode_state_narrower_than_live_by_stripe_width(self) -> None:
        eng, *_ = engine.Engine.create(seed=100)
        shim = _era_shim()
        decision = _decision()
        live_len = encode.encode_state(eng.state, decision).shape[0]
        shim_len = shim.encode_state(eng.state, decision).shape[0]
        assert live_len - shim_len == _STATE_STRIPE_WIDTH

    def test_encode_state_matches_live_without_stripes(self) -> None:
        eng, *_ = engine.Engine.create(seed=100)
        shim = _era_shim()
        decision = _decision()
        live = encode.encode_state(eng.state, decision)
        start = encode.STATE_HAND_FOOD_UNLOCK_OFFSET
        live_stripped = np.delete(
            live, slice(start, start + _STATE_STRIPE_WIDTH), axis=0
        )
        shim_out = shim.encode_state(eng.state, decision)
        assert shim_out.shape == live_stripped.shape
        assert np.array_equal(shim_out, live_stripped)

    def test_state_embed_offsets_shifted_left(self) -> None:
        arch = _small_arch()
        live_off = core.PolicyValueNet(arch=arch)._state_embed_offsets()
        shim_off = _era_shim(arch=arch)._state_embed_offsets()
        assert live_off.card_index - shim_off.card_index == _STATE_STRIPE_WIDTH
        assert live_off.hand_multihot - shim_off.hand_multihot == _STATE_STRIPE_WIDTH
        assert live_off.decision_type - shim_off.decision_type == _STATE_STRIPE_WIDTH


# ---------------------------------------------------------------------------
# The v1_3 shim strips the resets_feeder choice stripe


class TestV1_3ChoiceStripeStripping:
    def test_encode_choices_narrower_than_live_by_resets_feeder(self) -> None:
        eng, *_ = engine.Engine.create(seed=100)
        shim = _era_shim()
        decision = _decision()
        live_cols = encode.encode_choices(decision, eng.state).shape[1]
        shim_cols = shim.encode_choices(decision, eng.state).shape[1]
        assert live_cols - shim_cols == encode.CHOICE_RESETS_FEEDER_DIM

    def test_encode_choices_matches_live_without_resets_feeder(self) -> None:
        eng, *_ = engine.Engine.create(seed=100)
        shim = _era_shim()
        decision = _decision()
        live_full = encode.encode_choices(decision, eng.state)
        start = encode.CHOICE_RESETS_FEEDER_OFFSET
        end = start + encode.CHOICE_RESETS_FEEDER_DIM
        live_stripped = np.delete(live_full, slice(start, end), axis=1)
        shim_out = shim.encode_choices(decision, eng.state)
        assert shim_out.shape == live_stripped.shape
        assert np.array_equal(shim_out, live_stripped)

    def test_becomes_unplayable_kept_and_unshifted(self) -> None:
        """The v1.3 shim keeps becomes_unplayable at the live offset (it precedes
        the new stripe); only kept_multihot shifts."""
        arch = _small_arch()
        live = core.PolicyValueNet(arch=arch)._choice_embed_offsets()
        shim = _era_shim(arch=arch)._choice_embed_offsets()
        assert shim.becomes_unplayable is not None
        assert shim.becomes_unplayable == live.becomes_unplayable
        assert shim.becomes_playable == live.becomes_playable
        assert shim.bird_id == live.bird_id

    def test_kept_multihot_offset_shifted_left(self) -> None:
        """With include_setup, kept_multihot shifts left by exactly the stripe width."""
        arch = _small_arch()
        spec = encode.EncodingSpec(include_setup=True)
        live = core.PolicyValueNet(spec=spec, arch=arch)._choice_embed_offsets()
        shim = _era_shim(arch=arch, spec=spec)._choice_embed_offsets()
        assert live.kept_multihot is not None
        assert shim.kept_multihot is not None
        assert (
            live.kept_multihot - shim.kept_multihot == encode.CHOICE_RESETS_FEEDER_DIM
        )


# ---------------------------------------------------------------------------
# End-to-end forward passes at era dims (the load-path geometry)


class TestForwardAtEraDims:
    def test_v1_3_forward_pass_runs_at_era_dims(self) -> None:
        eng, *_ = engine.Engine.create(seed=101)
        _forward(_era_shim(), _decision(), eng.state)

    def test_v1_3_forward_pass_runs_at_live_default_dims(self) -> None:
        """Constructing with default (live) dims still works — the shim derives its
        true encoder widths from ``self.spec``, so the live-dim test style remains
        valid alongside the era-dim load path."""
        eng, *_ = engine.Engine.create(seed=101)
        _forward(
            compat_v1_3.PolicyValueNetV1_3(arch=_small_arch()), _decision(), eng.state
        )


# ---------------------------------------------------------------------------
# v1.0 inherits both the state-stripe removal and the resets_feeder removal


class TestV1_0InheritsPre1_4Strips:
    def test_v1_0_subclasses_v1_3(self) -> None:
        assert issubclass(
            compat_v1_0.PolicyValueNetV1_0, compat_v1_3.PolicyValueNetV1_3
        )

    def _v1_0_era_net(
        self, spec: encode.EncodingSpec = encode.DEFAULT_SPEC
    ) -> compat_v1_0.PolicyValueNetV1_0:
        state_dim, choice_dim = compat.encoding_dims_for_era("1.0", spec)
        return compat_v1_0.PolicyValueNetV1_0(
            state_dim=state_dim, choice_dim=choice_dim, arch=_small_arch(), spec=spec
        )

    def test_v1_0_encode_state_strips_the_stripes(self) -> None:
        eng, *_ = engine.Engine.create(seed=102)
        net = self._v1_0_era_net()
        decision = _decision()
        live = encode.encode_state(eng.state, decision)
        start = encode.STATE_HAND_FOOD_UNLOCK_OFFSET
        live_stripped = np.delete(
            live, slice(start, start + _STATE_STRIPE_WIDTH), axis=0
        )
        assert np.array_equal(net.encode_state(eng.state, decision), live_stripped)

    def test_v1_0_forward_pass_at_era_dims(self) -> None:
        eng, *_ = engine.Engine.create(seed=103)
        _forward(self._v1_0_era_net(), _decision(), eng.state)


# ---------------------------------------------------------------------------
# Real load-path round-trip (fixture-equivalent)


def test_v1_3_stamped_checkpoint_round_trips(tmp_path: pathlib.Path) -> None:
    """A v1.3-stamped checkpoint loads under v1.4 via ``load_policy_net`` and
    forward-passes. The loader hands the shim the era's already-narrow dims (from
    ``encoding_dims_for_era``); ``load_state_dict`` succeeds and inference runs,
    proving the shim does not double-subtract either stripe width."""
    base = config.RunConfig(
        misc=config.MiscConfig(device="cpu"),
        run=config.RunSettings(
            run_name="v13-roundtrip",
            checkpoint_dir=str(tmp_path),
            games_per_iter=2,
            eval_games=2,
        ),
        architecture=config.ArchitectureConfig(
            main=config.MainNetArchitecture(
                trunk_layers=(8, 8),
                choice_layers=(8, 8),
                head_layers=(),
                value_layers=(),
                card_embed_dim=4,
                card_encoder_layers=(),
                hand_encoder_layers=(8,),
            ),
        ),
    )
    cfg = config.with_encoding_version(base, "1.3")
    assert cfg.encoding_version == "1.3"
    assert cfg.state_dim == encode.state_size(cfg.encoding_spec) - _STATE_STRIPE_WIDTH
    assert (
        cfg.choice_dim
        == encode.choice_feature_dim(cfg.encoding_spec)
        - encode.CHOICE_RESETS_FEEDER_DIM
    )

    net_cls = model.PolicyValueNet.class_for_version(cfg.encoding_version)
    assert net_cls is compat_v1_3.PolicyValueNetV1_3
    net = net_cls(
        state_dim=cfg.state_dim,
        choice_dim=cfg.choice_dim,
        num_families=len(cfg.family_order),
        arch=cfg.arch,
        spec=cfg.encoding_spec,
    )

    ckpt = tmp_path / "v13.pt"
    torch.save(
        {"config": cfg.model_dump(), "model": net.state_dict(), "version": "1.3"},
        ckpt,
    )

    loaded, saved_cfg = loaders.load_policy_net(ckpt, torch.device("cpu"))
    assert isinstance(loaded, compat_v1_3.PolicyValueNetV1_3)
    assert saved_cfg.encoding_version == "1.3"

    eng, *_ = engine.Engine.create(seed=200)
    _forward(loaded, _decision(), eng.state)
