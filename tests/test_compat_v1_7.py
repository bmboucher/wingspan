# pyright: reportPrivateUsage=false
# (tests access _state_embed_offsets to pin the StateEmbedOffsets seam,
# matching the test_compat_v1_3.py convention)
"""Tests for the pre-1.8 -> v1.8 compat shim.

v1.8 appends a per-opponent ``known_hand_opp`` 180-wide identity multi-hot
to the state vector's multi-hot region (immediately after
``hand_playable_eggs_me``, before the trailing ``decision_type`` one-hot).
``compat.v1_7.PolicyValueNetV1_7`` strips it from pre-1.8 state vectors and
freezes the pre-1.8 ``StateEmbedOffsets`` — only ``decision_type`` shifts
left; ``card_index`` / ``hand_multihot`` precede the new stripe and are
UNCHANGED, the opposite shift shape from ``compat.v1_3``'s food-unlock strip
(which precedes ``card_index`` and shifts all three offsets).

v1.8 touches state encoding only: no choice-side change, and no
``SetupNetV1_7`` — v1.8 leaves the choice vector and the setup model
entirely untouched, so this module has no setup-net section (only a small
pin confirming ``SetupNet.class_for_version("1.7")`` did not change).

As for every prior era, a committed LFS checkpoint fixture is deferred:
``test_v1_7_stamped_checkpoint_round_trips`` builds a v1.7-era net, saves it
with a v1.7 stamp, and round-trip-loads it through the production
``players.loaders.load_policy_net`` path — exercising the narrow-dims load
path (``encoding_dims_for_era`` -> constructor -> ``load_state_dict`` ->
forward) so any double-subtraction of the stripe width would fail the test.
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
from wingspan.compat import v1_4 as compat_v1_4
from wingspan.compat import v1_5 as compat_v1_5
from wingspan.compat import v1_6 as compat_v1_6
from wingspan.compat import v1_7 as compat_v1_7
from wingspan.model import core
from wingspan.players import loaders
from wingspan.training import config
from wingspan.training import setup_net as setup_net_module


def _small_arch() -> architecture.ModelArchitecture:
    return architecture.ModelArchitecture(
        trunk_layers=(8, 8),
        choice_layers=(8, 8),
        head_layers=(),
        value_layers=(),
        card_embed_dim=4,
    )


def _era_shim(
    era: str = "1.7",
    arch: architecture.ModelArchitecture | None = None,
    spec: encode.EncodingSpec = encode.DEFAULT_SPEC,
) -> compat_v1_7.PolicyValueNetV1_7:
    """A v1_7 shim built at ``era``'s (narrow) dims — exactly how the load path
    (``encoding_dims_for_era`` -> constructor) builds it."""
    arch = arch or _small_arch()
    state_dim, choice_dim = compat.encoding_dims_for_era(era, spec)
    return compat_v1_7.PolicyValueNetV1_7(
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
# (1) class_for_version routing + the full inheritance chain


class TestClassForVersionRouting:
    def test_v1_7_routes_to_shim(self) -> None:
        assert (
            core.PolicyValueNet.class_for_version("1.7")
            is compat_v1_7.PolicyValueNetV1_7
        )

    def test_earlier_eras_keep_their_own_shims(self) -> None:
        assert (
            core.PolicyValueNet.class_for_version("1.6")
            is compat_v1_6.PolicyValueNetV1_6
        )
        assert (
            core.PolicyValueNet.class_for_version("1.5")
            is compat_v1_5.PolicyValueNetV1_5
        )
        assert (
            core.PolicyValueNet.class_for_version("1.4")
            is compat_v1_4.PolicyValueNetV1_4
        )
        for era in ("1.1", "1.2", "1.3"):
            assert (
                core.PolicyValueNet.class_for_version(era)
                is compat_v1_3.PolicyValueNetV1_3
            )
        assert (
            core.PolicyValueNet.class_for_version("1.0")
            is compat_v1_0.PolicyValueNetV1_0
        )

    def test_current_version_returns_live_class(self) -> None:
        assert (
            core.PolicyValueNet.class_for_version(version.MODEL_VERSION)
            is core.PolicyValueNet
        )

    def test_inheritance_chain_v1_0_through_v1_7_to_core(self) -> None:
        """v1_0 subclass v1_3 subclass v1_4 subclass v1_5 subclass v1_6
        subclass v1_7 subclass core.PolicyValueNet — the full re-chain this
        bump extends by inserting v1_7 between v1_6 and core."""
        assert issubclass(
            compat_v1_0.PolicyValueNetV1_0, compat_v1_3.PolicyValueNetV1_3
        )
        assert issubclass(
            compat_v1_3.PolicyValueNetV1_3, compat_v1_4.PolicyValueNetV1_4
        )
        assert issubclass(
            compat_v1_4.PolicyValueNetV1_4, compat_v1_5.PolicyValueNetV1_5
        )
        assert issubclass(
            compat_v1_5.PolicyValueNetV1_5, compat_v1_6.PolicyValueNetV1_6
        )
        assert issubclass(
            compat_v1_6.PolicyValueNetV1_6, compat_v1_7.PolicyValueNetV1_7
        )
        assert issubclass(compat_v1_7.PolicyValueNetV1_7, core.PolicyValueNet)


# ---------------------------------------------------------------------------
# (2) encoding_dims_for_era: the composed (state_dim, choice_dim) per era


class TestEncodingDimsForEra:
    def test_composed_dims_across_every_era(self) -> None:
        """Every era's dims, composed bottom-up from the known stripe widths
        (each individually documented on ``compat.encoding_dims_for_era``)
        rather than hardcoded, so a future stripe addition cannot silently
        desync this test from the real branches."""
        spec = encode.DEFAULT_SPEC
        live_state = encode.state_size(spec)
        live_choice = encode.choice_feature_dim(spec)
        assert (live_state, live_choice) == (1309, 517)

        known_hand_opp = encode.STATE_KNOWN_HAND_OPP_DIM
        goal_delta_tail = encode.CHOICE_GOAL_DELTA_IGNORING_EGGS_DIM
        food_unlock = 2 * encode.STATE_FOOD_UNLOCK_DIM
        resets_feeder = encode.CHOICE_RESETS_FEEDER_DIM
        becomes_unplayable = encode.CHOICE_BECOMES_UNPLAYABLE_DIM

        expected: dict[str, tuple[int, int]] = {
            version.MODEL_VERSION: (live_state, live_choice),
            "1.7": (live_state - known_hand_opp, live_choice),
            "1.6": (live_state - known_hand_opp, live_choice),
            "1.5": (live_state - known_hand_opp, live_choice - goal_delta_tail),
            "1.4": (live_state - known_hand_opp, live_choice - goal_delta_tail),
            "1.3": (
                live_state - known_hand_opp - food_unlock,
                live_choice - goal_delta_tail - resets_feeder,
            ),
            "1.0": (
                live_state - known_hand_opp - food_unlock,
                live_choice - goal_delta_tail - resets_feeder - becomes_unplayable,
            ),
        }
        for era, dims in expected.items():
            assert compat.encoding_dims_for_era(era, spec) == dims, era


# ---------------------------------------------------------------------------
# (3) The v1_7 shim strips the known_hand_opp state stripe


class TestKnownHandOppStripStripping:
    def test_encode_state_matches_live_with_stripe_deleted(self) -> None:
        eng, *_ = engine.Engine.create(seed=100)
        shim = _era_shim()
        decision = _decision()
        live = encode.encode_state(eng.state, decision)
        start = encode.STATE_KNOWN_HAND_OPP_OFFSET
        end = start + encode.STATE_KNOWN_HAND_OPP_DIM
        live_stripped = np.delete(live, slice(start, end), axis=0)
        shim_out = shim.encode_state(eng.state, decision)
        assert shim_out.shape == live_stripped.shape
        assert np.array_equal(shim_out, live_stripped)

    def test_width_is_live_minus_known_hand_opp_dim(self) -> None:
        eng, *_ = engine.Engine.create(seed=100)
        shim = _era_shim()
        decision = _decision()
        live_len = encode.encode_state(eng.state, decision).shape[0]
        shim_len = shim.encode_state(eng.state, decision).shape[0]
        assert live_len - shim_len == encode.STATE_KNOWN_HAND_OPP_DIM


# ---------------------------------------------------------------------------
# (4) StateEmbedOffsets: only decision_type shifts; v1_3 composes its own
# food-unlock shift on top of v1_7's inherited decision_type shift


class TestStateEmbedOffsets:
    def test_card_index_and_hand_multihot_unchanged(self) -> None:
        """The new stripe is appended after BOTH playability multi-hots, so
        card_index / hand_multihot — which precede hand_playable_me — never
        move (the opposite shape from v1_3's food-unlock strip, which
        precedes card_index and shifts it)."""
        arch = _small_arch()
        live_off = core.PolicyValueNet(arch=arch)._state_embed_offsets()
        shim_off = _era_shim(arch=arch)._state_embed_offsets()
        assert shim_off.card_index == live_off.card_index
        assert shim_off.hand_multihot == live_off.hand_multihot
        assert shim_off.hand_summary == live_off.hand_summary
        assert shim_off.hand_summary_end == live_off.hand_summary_end

    def test_decision_type_shifted_left_by_known_hand_opp_dim(self) -> None:
        arch = _small_arch()
        live_off = core.PolicyValueNet(arch=arch)._state_embed_offsets()
        shim_off = _era_shim(arch=arch)._state_embed_offsets()
        assert (
            live_off.decision_type - shim_off.decision_type
            == encode.STATE_KNOWN_HAND_OPP_DIM
        )

    def test_v1_3_composes_its_own_shift_on_top_of_v1_7s(self) -> None:
        """v1_3's card_index / hand_multihot shift by its own food-unlock
        width only (known_hand_opp never touches them); decision_type
        carries BOTH shifts — v1_3's own food-unlock width plus the
        inherited v1_7 known_hand_opp width."""
        arch = _small_arch()
        live_off = core.PolicyValueNet(arch=arch)._state_embed_offsets()
        v1_3_off = compat_v1_3.PolicyValueNetV1_3(arch=arch)._state_embed_offsets()
        food_unlock = 2 * encode.STATE_FOOD_UNLOCK_DIM
        assert live_off.card_index - v1_3_off.card_index == food_unlock
        assert live_off.hand_multihot - v1_3_off.hand_multihot == food_unlock
        assert live_off.decision_type - v1_3_off.decision_type == (
            food_unlock + encode.STATE_KNOWN_HAND_OPP_DIM
        )


# ---------------------------------------------------------------------------
# (5) Forward pass at era dims (the load-path geometry) and at live default
# dims (the _true_state_dim-derived construction still works either way)


class TestForwardAtEraDims:
    def test_forward_pass_runs_at_era_dims(self) -> None:
        eng, *_ = engine.Engine.create(seed=101)
        _forward(_era_shim(), _decision(), eng.state)

    def test_forward_pass_runs_at_live_default_dims(self) -> None:
        """Constructing with default (live) dims still works — the shim
        derives its true encoder width from ``self.spec``, so the live-dim
        test style remains valid alongside the era-dim load path."""
        eng, *_ = engine.Engine.create(seed=101)
        _forward(
            compat_v1_7.PolicyValueNetV1_7(arch=_small_arch()), _decision(), eng.state
        )


# ---------------------------------------------------------------------------
# (6) Era-owned raw state stripe layout


class TestRawStateStripeLayout:
    def test_v1_7_layout_lacks_known_hand_opp(self) -> None:
        names = {s.name for s in _era_shim().raw_state_stripe_layout().stripes}
        assert "known_hand_opp" not in names

    def test_total_size_matches_era_state_dim(self) -> None:
        state_dim, _ = compat.encoding_dims_for_era("1.7", encode.DEFAULT_SPEC)
        assert _era_shim().raw_state_stripe_layout().total_size == state_dim

    def test_every_older_shim_layout_also_lacks_known_hand_opp(self) -> None:
        for net in (
            compat_v1_6.PolicyValueNetV1_6(arch=_small_arch()),
            compat_v1_3.PolicyValueNetV1_3(arch=_small_arch()),
            compat_v1_0.PolicyValueNetV1_0(arch=_small_arch()),
        ):
            names = {s.name for s in net.raw_state_stripe_layout().stripes}
            assert "known_hand_opp" not in names, type(net).__name__

    def test_without_stripes_on_unknown_name_raises(self) -> None:
        with pytest.raises(KeyError):
            _era_shim().raw_state_stripe_layout().without_stripes(("nope",))


# ---------------------------------------------------------------------------
# (7) Real load-path round-trip (fixture-equivalent)


def test_v1_7_stamped_checkpoint_round_trips(tmp_path: pathlib.Path) -> None:
    """A v1.7-stamped checkpoint loads under v1.8 via ``load_policy_net`` and
    forward-passes. The loader hands the shim the era's already-narrow state
    dim (from ``encoding_dims_for_era``); ``load_state_dict`` succeeds and
    inference runs, proving the shim does not double-subtract the stripe
    width."""
    base = config.RunConfig(
        misc=config.MiscConfig(device="cpu"),
        run=config.RunSettings(
            run_name="v17-roundtrip",
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
    cfg = config.with_encoding_version(base, "1.7")
    assert cfg.encoding_version == "1.7"
    assert cfg.state_dim == (
        encode.state_size(cfg.encoding_spec) - encode.STATE_KNOWN_HAND_OPP_DIM
    )
    assert cfg.choice_dim == encode.choice_feature_dim(cfg.encoding_spec)

    net_cls = model.PolicyValueNet.class_for_version(cfg.encoding_version)
    assert net_cls is compat_v1_7.PolicyValueNetV1_7
    net = net_cls(
        state_dim=cfg.state_dim,
        choice_dim=cfg.choice_dim,
        num_families=len(cfg.family_order),
        arch=cfg.arch,
        spec=cfg.encoding_spec,
    )

    ckpt = tmp_path / "v17.pt"
    torch.save(
        {"config": cfg.model_dump(), "model": net.state_dict(), "version": "1.7"},
        ckpt,
    )

    loaded, saved_cfg = loaders.load_policy_net(ckpt, torch.device("cpu"))
    assert isinstance(loaded, compat_v1_7.PolicyValueNetV1_7)
    assert not isinstance(loaded, compat_v1_6.PolicyValueNetV1_6)
    assert saved_cfg.encoding_version == "1.7"

    eng, *_ = engine.Engine.create(seed=200)
    _forward(loaded, _decision(), eng.state)


# ---------------------------------------------------------------------------
# (8) No SetupNet mirror: v1.8 leaves setup encoding untouched


def test_setup_net_class_for_version_routing_unchanged_at_era_1_7() -> None:
    """v1.8 does not touch setup encoding, so there is no ``SetupNetV1_7``
    and ``SetupNet.class_for_version``'s routing for era 1.7 does not change:
    its cascade already topped out at ``minor <= 6 -> SetupNetV1_6`` before
    this bump (era 1.7 fell through to the live ``SetupNet``), and it still
    does — this pins that the routing did not silently change."""
    assert (
        setup_net_module.SetupNet.class_for_version("1.7") is setup_net_module.SetupNet
    )
