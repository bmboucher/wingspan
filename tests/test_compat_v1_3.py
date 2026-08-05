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
from wingspan.encode import stripes
from wingspan.encode.stripes import descriptors
from wingspan.model import core
from wingspan.players import loaders
from wingspan.reporting import encode_viewer
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
        """Pre-1.4 eras predate both resets_feeder (v1.4) and
        goal_delta_ignoring_eggs (v1.6), so both narrowings compose."""
        spec = encode.DEFAULT_SPEC
        live_choice = encode.choice_feature_dim(spec)
        for era in ("1.1", "1.2", "1.3"):
            _, choice_dim = compat.encoding_dims_for_era(era, spec)
            assert live_choice - choice_dim == (
                encode.CHOICE_RESETS_FEEDER_DIM
                + encode.CHOICE_GOAL_DELTA_IGNORING_EGGS_DIM
            )

    def test_dims_are_live_for_current_era(self) -> None:
        spec = encode.DEFAULT_SPEC
        state_dim, choice_dim = compat.encoding_dims_for_era(
            version.MODEL_VERSION, spec
        )
        assert state_dim == encode.state_size(spec)
        assert choice_dim == encode.choice_feature_dim(spec)

    def test_v1_0_choice_dim_drops_both_choice_stripes(self) -> None:
        """v1.0 predates the v1.1 becomes_unplayable stripe, the v1.4
        resets_feeder stripe, and the v1.6 goal_delta_ignoring_eggs stripe, so
        its choice_dim drops all three."""
        spec = encode.DEFAULT_SPEC
        _, choice_dim = compat.encoding_dims_for_era("1.0", spec)
        assert encode.choice_feature_dim(spec) - choice_dim == (
            encode.CHOICE_BECOMES_UNPLAYABLE_DIM
            + encode.CHOICE_RESETS_FEEDER_DIM
            + encode.CHOICE_GOAL_DELTA_IGNORING_EGGS_DIM
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
        """v1_3 strips both its own resets_feeder column and the
        goal_delta_ignoring_eggs tail it inherits from the v1_5 parent."""
        eng, *_ = engine.Engine.create(seed=100)
        shim = _era_shim()
        decision = _decision()
        live_cols = encode.encode_choices(decision, eng.state).shape[1]
        shim_cols = shim.encode_choices(decision, eng.state).shape[1]
        assert live_cols - shim_cols == (
            encode.CHOICE_RESETS_FEEDER_DIM + encode.CHOICE_GOAL_DELTA_IGNORING_EGGS_DIM
        )

    def test_encode_choices_matches_live_without_resets_feeder(self) -> None:
        """Strip both the resets_feeder column and the goal_delta_ignoring_eggs
        tail from the live rows before comparing — the inherited v1_5 strip
        runs before v1_3's own resets_feeder strip in the super() chain."""
        eng, *_ = engine.Engine.create(seed=100)
        shim = _era_shim()
        decision = _decision()
        live_full = encode.encode_choices(decision, eng.state)
        tail_start = encode.CHOICE_GOAL_DELTA_IGNORING_EGGS_OFFSET
        tail_end = tail_start + encode.CHOICE_GOAL_DELTA_IGNORING_EGGS_DIM
        live_stripped = np.delete(live_full, slice(tail_start, tail_end), axis=1)
        start = encode.CHOICE_RESETS_FEEDER_OFFSET
        end = start + encode.CHOICE_RESETS_FEEDER_DIM
        live_stripped = np.delete(live_stripped, slice(start, end), axis=1)
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
        """With include_setup, kept_multihot shifts left by resets_feeder's width
        plus the inherited goal_delta_ignoring_eggs width."""
        arch = _small_arch()
        spec = encode.EncodingSpec(include_setup=True)
        live = core.PolicyValueNet(spec=spec, arch=arch)._choice_embed_offsets()
        shim = _era_shim(arch=arch, spec=spec)._choice_embed_offsets()
        assert live.kept_multihot is not None
        assert shim.kept_multihot is not None
        assert live.kept_multihot - shim.kept_multihot == (
            encode.CHOICE_RESETS_FEEDER_DIM + encode.CHOICE_GOAL_DELTA_IGNORING_EGGS_DIM
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
# Era-owned raw stripe layouts (the game-log encoding viewer's decode seam)


class TestEraStripeLayouts:
    """Each net's ``raw_*_stripe_layout()`` must describe *its own* encoder's
    output. Decoding a compat-era vector with the live layout silently shifts
    every stripe past the removed columns (phantom hand birds, tray cards in
    board slots — the game.html bug); these tests pin the era-owned layouts and
    the viewer's use of them."""

    def _v1_0_net(self) -> compat_v1_0.PolicyValueNetV1_0:
        state_dim, choice_dim = compat.encoding_dims_for_era("1.0", encode.DEFAULT_SPEC)
        return compat_v1_0.PolicyValueNetV1_0(
            state_dim=state_dim, choice_dim=choice_dim, arch=_small_arch()
        )

    def test_without_stripes_shifts_offsets_and_total(self) -> None:
        vl = descriptors.VectorLayout.from_stripe_specs(
            [
                descriptors.StripeSpec(name=name, size=size)
                for name, size in (("a", 3), ("b", 5), ("c", 2), ("d", 4))
            ]
        )
        out = vl.without_stripes(("b", "d"))
        assert out.total_size == 5
        assert [(s.name, s.offset, s.size) for s in out.stripes] == [
            ("a", 0, 3),
            ("c", 3, 2),
        ]

    def test_without_stripes_unknown_name_raises(self) -> None:
        vl = descriptors.VectorLayout.from_stripe_specs(
            [descriptors.StripeSpec(name="a", size=3)]
        )
        with pytest.raises(KeyError):
            vl.without_stripes(("nope",))

    def test_live_layouts_match_live_dims(self) -> None:
        net = core.PolicyValueNet(arch=_small_arch())
        assert net.raw_state_stripe_layout().total_size == encode.state_size(net.spec)
        assert net.raw_choice_stripe_layout().total_size == encode.choice_feature_dim(
            net.spec
        )

    def test_era_layout_dims_match_encoding_dims_for_era(self) -> None:
        for era, net in (("1.3", _era_shim()), ("1.0", self._v1_0_net())):
            state_dim, choice_dim = compat.encoding_dims_for_era(era, net.spec)
            assert net.raw_state_stripe_layout().total_size == state_dim, era
            assert net.raw_choice_stripe_layout().total_size == choice_dim, era

    def test_era_layouts_match_encoder_output_widths(self) -> None:
        eng, *_ = engine.Engine.create(seed=104)
        decision = _decision()
        for net in (_era_shim(), self._v1_0_net()):
            state_vec = net.encode_state(eng.state, decision)
            choice_rows = net.encode_choices(decision, eng.state)
            assert net.raw_state_stripe_layout().total_size == state_vec.shape[0]
            assert net.raw_choice_stripe_layout().total_size == choice_rows.shape[1]

    def test_v1_3_layouts_drop_the_v1_4_and_inherited_v1_6_stripes(self) -> None:
        shim = _era_shim()
        state_names = {s.name for s in shim.raw_state_stripe_layout().stripes}
        assert "hand_food_unlock_me" not in state_names
        assert "tray_food_unlock_me" not in state_names
        choice_names = {s.name for s in shim.raw_choice_stripe_layout().stripes}
        assert "resets_feeder" not in choice_names
        assert "goal_delta_ignoring_eggs" not in choice_names  # inherited via v1_5
        assert "becomes_unplayable" in choice_names  # v1.1 stripe still present

    def test_v1_0_choice_layout_also_drops_becomes_unplayable(self) -> None:
        names = {s.name for s in self._v1_0_net().raw_choice_stripe_layout().stripes}
        assert "becomes_unplayable" not in names
        assert "resets_feeder" not in names
        assert "goal_delta_ignoring_eggs" not in names
        assert "becomes_playable" in names

    def test_era_state_offsets_shift_left_past_removed_stripes(self) -> None:
        shim = _era_shim()
        live = stripes.raw_state_stripe_layout(shim.spec)
        era = shim.raw_state_stripe_layout()
        assert era.offset_of("food_me") == live.offset_of("food_me")
        assert (
            live.offset_of("hand_multihot") - era.offset_of("hand_multihot")
            == _STATE_STRIPE_WIDTH
        )
        assert (
            live.offset_of("decision_type") - era.offset_of("decision_type")
            == _STATE_STRIPE_WIDTH
        )

    def test_era_vector_decodes_hand_via_own_layout(self) -> None:
        """THE regression: a v1.3 net's state vector decoded through the net's
        own layout names the actual hand; decoded through the live layout it
        used to show phantom birds — now the width mismatch raises instead."""
        eng, birds, *_ = engine.Engine.create(seed=105)
        hand = [birds[8], birds[101], birds[152]]
        eng.state.players[0].hand = list(hand)
        shim = _era_shim()
        vec = shim.encode_state(eng.state, _decision()).tolist()

        result = encode_viewer.extract_state_stripes(
            vec, include_setup=False, vector_layout=shim.raw_state_stripe_layout()
        )
        hand_stripes = [s for s in result if s.name == "hand_multihot"]
        assert hand_stripes, "hand_multihot stripe missing"
        label = hand_stripes[0].sub_fields[0].decoded_label
        assert label is not None
        for bird in hand:
            assert bird.name in label, f"'{bird.name}' missing from: {label}"

        with pytest.raises(ValueError, match="different encoding era"):
            encode_viewer.extract_state_stripes(vec, include_setup=False)


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
    assert cfg.choice_dim == (
        encode.choice_feature_dim(cfg.encoding_spec)
        - encode.CHOICE_RESETS_FEEDER_DIM
        - encode.CHOICE_GOAL_DELTA_IGNORING_EGGS_DIM
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
