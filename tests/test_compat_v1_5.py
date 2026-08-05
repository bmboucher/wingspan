# pyright: reportPrivateUsage=false
# (tests access _choice_embed_offsets to pin the ChoiceEmbedOffsets seam,
# matching the test_compat_v1_3.py / test_compat_v1_4.py convention)
"""Tests for the pre-1.6 -> v1.6 compat shim: main-net choice geometry, and the
setup net's ``goal_affinity`` value freeze.

v1.6 appends an 8-dim ``goal_delta_ignoring_eggs`` choice stripe as the new
last base stripe (immediately after ``resets_feeder``): per round goal, a
``(count_delta, vp_delta)`` pair pricing the hypothesis that this row's bird
is eventually played (a slot must be open) and egg-populated to whatever
level best advances the goal. ``compat.v1_5.PolicyValueNetV1_5`` strips it
after live encoding and shifts only the trailing ``kept_multihot`` region —
the geometry-narrowing analogue of the ``v1_3`` / ``v1_0`` column strips,
applied to one stripe. ``compat.v1_4.PolicyValueNetV1_4`` now subclasses this
net directly (was the live net), so every pre-1.6 era inherits the tail-strip
too, on top of its own stripes and value refills.

As for v1.0 / v1.3 / v1.4, a committed LFS checkpoint fixture is deferred:
``test_v1_5_stamped_checkpoint_round_trips`` builds a v1.5-era net, saves it
with a v1.5 stamp, and reloads it through the production ``load_policy_net``
path, then forward-passes.

**Setup half.** v1.6 also changed the separate setup model's
``goal_affinity`` stripe (:mod:`wingspan.setup_model.encode`, step 7) from
:func:`wingspan.engine.scoring.goal_count_delta_for_bird` (zero for every
egg-driven category) to :func:`wingspan.engine.scoring.goal_affinity_for_kept`
(the played-and-optimally-egg-populated hand-level bound). No dims change, so
this is a value-only freeze exactly like v1.5's main-net ``goal_delta`` fix —
``compat.v1_5.SetupNetV1_5`` is the first :class:`wingspan.training.setup_net.SetupNet`
compat shim, overriding only ``encode_candidate``.
``test_v1_5_stamped_setup_checkpoint_round_trips`` mirrors the main-net
round-trip test for the setup side.
"""

from __future__ import annotations

import pathlib
import typing

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wingspan import (
    architecture,
    cards,
    compat,
    decisions,
    encode,
    engine,
    model,
    setup_model,
    state,
    version,
)
from wingspan.compat import v1_3 as compat_v1_3
from wingspan.compat import v1_4 as compat_v1_4
from wingspan.compat import v1_5 as compat_v1_5
from wingspan.model import core
from wingspan.players import loaders
from wingspan.training import artifacts, config, loop_checkpoint, runmeta
from wingspan.training import setup_net as setup_net_module
from wingspan.training import setup_runmeta

_TAIL_SLICE = slice(
    encode.CHOICE_GOAL_DELTA_IGNORING_EGGS_OFFSET,
    encode.CHOICE_GOAL_DELTA_IGNORING_EGGS_OFFSET
    + encode.CHOICE_GOAL_DELTA_IGNORING_EGGS_DIM,
)


def _small_arch() -> architecture.ModelArchitecture:
    return architecture.ModelArchitecture(
        trunk_layers=(8, 8),
        choice_layers=(8, 8),
        head_layers=(),
        value_layers=(),
        card_embed_dim=4,
    )


def _era_shim(
    arch: architecture.ModelArchitecture | None = None,
    spec: encode.EncodingSpec = encode.DEFAULT_SPEC,
) -> compat_v1_5.PolicyValueNetV1_5:
    """A v1_5 shim built at era 1.5's (narrow) dims — exactly how the load path
    (``encoding_dims_for_era`` -> constructor) builds it."""
    arch = arch or _small_arch()
    state_dim, choice_dim = compat.encoding_dims_for_era("1.5", spec)
    return compat_v1_5.PolicyValueNetV1_5(
        state_dim=state_dim, choice_dim=choice_dim, arch=arch, spec=spec
    )


def _v1_4_era_net(
    spec: encode.EncodingSpec = encode.DEFAULT_SPEC,
) -> compat_v1_4.PolicyValueNetV1_4:
    """A v1_4 shim built at era 1.4's (v1.6-narrowed) dims."""
    state_dim, choice_dim = compat.encoding_dims_for_era("1.4", spec)
    return compat_v1_4.PolicyValueNetV1_4(
        state_dim=state_dim, choice_dim=choice_dim, arch=_small_arch(), spec=spec
    )


def _v1_3_era_net(
    spec: encode.EncodingSpec = encode.DEFAULT_SPEC,
) -> compat_v1_3.PolicyValueNetV1_3:
    """A v1_3 shim built at era 1.3's dims (both v1.4 stripes plus the
    inherited v1.6 tail stripped)."""
    state_dim, choice_dim = compat.encoding_dims_for_era("1.3", spec)
    return compat_v1_3.PolicyValueNetV1_3(
        state_dim=state_dim, choice_dim=choice_dim, arch=_small_arch(), spec=spec
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
    decision: decisions.Decision[typing.Any],
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
# (a) class_for_version routing


class TestClassForVersionRouting:
    def test_v1_5_routes_to_shim(self) -> None:
        assert (
            core.PolicyValueNet.class_for_version("1.5")
            is compat_v1_5.PolicyValueNetV1_5
        )

    def test_v1_4_still_routes_to_its_own_shim_which_subclasses_v1_5(self) -> None:
        """Era 1.4's routing target is unchanged (its own shim class), but that
        class now subclasses v1_5 (was the live net directly)."""
        assert (
            core.PolicyValueNet.class_for_version("1.4")
            is compat_v1_4.PolicyValueNetV1_4
        )
        assert issubclass(
            compat_v1_4.PolicyValueNetV1_4, compat_v1_5.PolicyValueNetV1_5
        )

    def test_current_version_returns_live_class(self) -> None:
        assert (
            core.PolicyValueNet.class_for_version(version.MODEL_VERSION)
            is core.PolicyValueNet
        )


# ---------------------------------------------------------------------------
# (b) encoding_dims_for_era: the new choice narrowing, and composition with
# the older resets_feeder narrowing


class TestEncodingDimsForEra:
    def test_choice_dim_narrower_by_goal_delta_ignoring_eggs_for_pre_1_6(
        self,
    ) -> None:
        spec = encode.DEFAULT_SPEC
        live_choice = encode.choice_feature_dim(spec)
        for era in ("1.4", "1.5"):
            state_dim, choice_dim = compat.encoding_dims_for_era(era, spec)
            assert state_dim == encode.state_size(spec)
            assert (
                live_choice - choice_dim == encode.CHOICE_GOAL_DELTA_IGNORING_EGGS_DIM
            )

    def test_era_1_3_composes_the_v1_6_and_v1_4_narrowings(self) -> None:
        """Era 1.3 predates both the v1.6 tail stripe and the v1.4
        resets_feeder stripe, so the two narrowings sum."""
        spec = encode.DEFAULT_SPEC
        live_choice = encode.choice_feature_dim(spec)
        _, choice_dim = compat.encoding_dims_for_era("1.3", spec)
        assert live_choice - choice_dim == (
            encode.CHOICE_GOAL_DELTA_IGNORING_EGGS_DIM + encode.CHOICE_RESETS_FEEDER_DIM
        )

    def test_dims_are_live_for_current_era(self) -> None:
        spec = encode.DEFAULT_SPEC
        state_dim, choice_dim = compat.encoding_dims_for_era(
            version.MODEL_VERSION, spec
        )
        assert state_dim == encode.state_size(spec)
        assert choice_dim == encode.choice_feature_dim(spec)


# ---------------------------------------------------------------------------
# (c) behavior freeze: shim output byte-equals live output with the tail
# stripped


class TestChoiceGeometryFreeze:
    def test_encode_choices_matches_live_with_tail_stripped(self) -> None:
        eng, *_ = engine.Engine.create(seed=100)
        shim = _era_shim()
        decision = _decision()
        live_full = encode.encode_choices(decision, eng.state)
        live_stripped = np.delete(live_full, _TAIL_SLICE, axis=1)
        shim_out = shim.encode_choices(decision, eng.state)
        assert shim_out.shape == live_stripped.shape
        assert np.array_equal(shim_out, live_stripped)

    def test_encode_choices_narrower_than_live_by_tail_dim(self) -> None:
        eng, *_ = engine.Engine.create(seed=100)
        shim = _era_shim()
        decision = _decision()
        live_cols = encode.encode_choices(decision, eng.state).shape[1]
        shim_cols = shim.encode_choices(decision, eng.state).shape[1]
        assert live_cols - shim_cols == encode.CHOICE_GOAL_DELTA_IGNORING_EGGS_DIM

    def test_bird_id_and_becomes_offsets_unshifted(self) -> None:
        """bird_id / becomes_playable / becomes_unplayable precede the tail
        stripe, so their offsets are identical between live and the shim."""
        arch = _small_arch()
        live = core.PolicyValueNet(arch=arch)._choice_embed_offsets()
        shim = _era_shim(arch=arch)._choice_embed_offsets()
        assert shim.bird_id == live.bird_id
        assert shim.becomes_playable == live.becomes_playable
        assert shim.becomes_unplayable == live.becomes_unplayable

    def test_kept_multihot_shifted_left_with_include_setup(self) -> None:
        """With include_setup, kept_multihot shifts left by exactly the tail
        stripe's width."""
        arch = _small_arch()
        spec = encode.EncodingSpec(include_setup=True)
        live = core.PolicyValueNet(spec=spec, arch=arch)._choice_embed_offsets()
        shim = _era_shim(arch=arch, spec=spec)._choice_embed_offsets()
        assert live.kept_multihot is not None
        assert shim.kept_multihot is not None
        assert (
            live.kept_multihot - shim.kept_multihot
            == encode.CHOICE_GOAL_DELTA_IGNORING_EGGS_DIM
        )


# ---------------------------------------------------------------------------
# Forward pass at era dims (the load-path geometry) and at live default dims


class TestForwardAtEraDims:
    def test_v1_5_forward_pass_runs_at_era_dims(self) -> None:
        eng, *_ = engine.Engine.create(seed=101)
        _forward(_era_shim(), _decision(), eng.state)

    def test_v1_5_forward_pass_runs_at_live_default_dims(self) -> None:
        """Constructing with default (live) dims still works — the shim derives
        its true encoder width from ``self.spec``, so the live-dim test style
        remains valid alongside the era-dim load path."""
        eng, *_ = engine.Engine.create(seed=101)
        _forward(
            compat_v1_5.PolicyValueNetV1_5(arch=_small_arch()), _decision(), eng.state
        )


# ---------------------------------------------------------------------------
# (e) raw_choice_stripe_layout().total_size per era


class TestRawChoiceStripeLayoutPerEra:
    def test_total_size_matches_encoding_dims_for_era(self) -> None:
        cases: list[tuple[str, core.PolicyValueNet]] = [
            ("1.5", _era_shim()),
            ("1.4", _v1_4_era_net()),
            ("1.3", _v1_3_era_net()),
        ]
        for era, net in cases:
            _, choice_dim = compat.encoding_dims_for_era(era, encode.DEFAULT_SPEC)
            assert net.raw_choice_stripe_layout().total_size == choice_dim, era

    def test_goal_delta_ignoring_eggs_absent_from_every_pre_1_6_layout(self) -> None:
        for net in (_era_shim(), _v1_4_era_net(), _v1_3_era_net()):
            names = {s.name for s in net.raw_choice_stripe_layout().stripes}
            assert "goal_delta_ignoring_eggs" not in names


# ---------------------------------------------------------------------------
# (d) Real load-path round-trip (fixture-equivalent)


def test_v1_5_stamped_checkpoint_round_trips(tmp_path: pathlib.Path) -> None:
    """A v1.5-stamped checkpoint loads under live code via ``load_policy_net``
    as the shim class (at era 1.5's dims — live minus the
    ``goal_delta_ignoring_eggs`` tail) and forward-passes."""
    base = config.RunConfig(
        misc=config.MiscConfig(device="cpu"),
        run=config.RunSettings(
            run_name="v15-roundtrip",
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
    cfg = config.with_encoding_version(base, "1.5")
    assert cfg.encoding_version == "1.5"
    assert cfg.state_dim == encode.state_size(cfg.encoding_spec)
    assert cfg.choice_dim == (
        encode.choice_feature_dim(cfg.encoding_spec)
        - encode.CHOICE_GOAL_DELTA_IGNORING_EGGS_DIM
    )

    net_cls = model.PolicyValueNet.class_for_version(cfg.encoding_version)
    assert net_cls is compat_v1_5.PolicyValueNetV1_5
    net = net_cls(
        state_dim=cfg.state_dim,
        choice_dim=cfg.choice_dim,
        num_families=len(cfg.family_order),
        arch=cfg.arch,
        spec=cfg.encoding_spec,
    )

    ckpt = tmp_path / "v15.pt"
    torch.save(
        {"config": cfg.model_dump(), "model": net.state_dict(), "version": "1.5"},
        ckpt,
    )

    loaded, saved_cfg = loaders.load_policy_net(ckpt, torch.device("cpu"))
    assert isinstance(loaded, compat_v1_5.PolicyValueNetV1_5)
    assert saved_cfg.encoding_version == "1.5"

    eng, *_ = engine.Engine.create(seed=200)
    _forward(loaded, _decision(), eng.state)


# ---------------------------------------------------------------------------
# Setup-net half: SetupNetV1_5's goal_affinity value freeze
# ---------------------------------------------------------------------------

_SETUP_BIRDS, *_ = cards.load_all()


def _small_main_arch_for_setup() -> architecture.ModelArchitecture:
    """A tiny main architecture — shapes the setup net's frozen embedder
    copies. Mirrors the ``MainNetArchitecture`` used by
    ``test_v1_5_stamped_checkpoint_round_trips``."""
    return architecture.ModelArchitecture(
        trunk_layers=(8, 8),
        choice_layers=(8, 8),
        head_layers=(),
        value_layers=(),
        card_embed_dim=4,
        card_encoder_layers=(),
        hand_encoder_layers=(8,),
    )


def _small_setup_arch() -> setup_model.SetupArchitecture:
    return setup_model.SetupArchitecture(
        trunk_layers=(8,), choice_layers=(8,), head_layers=(8,), value_layers=()
    )


def _setup_context(goal_categories: tuple[str, ...]) -> setup_model.SetupContext:
    return setup_model.SetupContext(
        tray_birds=(None, None, None),
        birdfeeder_counts=(0, 0, 0, 0, 0, 0),
        round_goal_categories=goal_categories,
    )


def _star_nest_candidate() -> (
    tuple[setup_model.SetupCandidate, setup_model.SetupContext]
):
    """A single-bird keep whose bird is star-nested (wild) against a
    bowl-with-eggs (nest) goal: nonzero under the live played-and-
    egg-populated pricing, zero under the pre-1.6 play-instant pricing (a
    freshly played bird has no eggs)."""
    star_bird = _SETUP_BIRDS[0].model_copy(
        update={"nest": cards.NestType.STAR, "egg_limit": 4}
    )
    candidate = setup_model.SetupCandidate(
        kept_cards=(star_bird,), kept_foods=(), bonus_card=None
    )
    context = _setup_context(("bowl_birds_with_eggs",) + ("birds_forest",) * 3)
    return candidate, context


# ---------------------------------------------------------------------------
# (a) SetupNet.class_for_version routing


class TestSetupClassForVersionRouting:
    def test_v1_5_routes_to_shim(self) -> None:
        assert (
            setup_net_module.SetupNet.class_for_version("1.5")
            is compat_v1_5.SetupNetV1_5
        )

    def test_v1_4_also_routes_to_shim(self) -> None:
        """Era 1.4 predates the setup net's two-tower restructure (a v1.3
        shape change) but routing it through the v1.5 shim first is harmless:
        a real 1.4 ``setup.pt`` would fail at ``load_state_dict`` regardless
        of which class builds it, and ``load_setup_net`` already turns that
        into a clear "retrain the setup model" error."""
        assert (
            setup_net_module.SetupNet.class_for_version("1.4")
            is compat_v1_5.SetupNetV1_5
        )

    def test_current_version_returns_live_class(self) -> None:
        assert (
            setup_net_module.SetupNet.class_for_version(version.MODEL_VERSION)
            is setup_net_module.SetupNet
        )


# ---------------------------------------------------------------------------
# (b) behavior freeze: the shim's goal_affinity equals the OLD (egg-blind)
# pricing while the live net's is nonzero; every other position matches.
# (c) geometry: shim output width equals the live encoding's total_dim.


class TestSetupGoalAffinityFreeze:
    def test_shim_freezes_egg_blind_pricing_live_net_does_not(self) -> None:
        candidate, context = _star_nest_candidate()
        encoding = setup_model.SetupEncoding()
        main_arch = _small_main_arch_for_setup()
        setup_arch = _small_setup_arch()

        live_net = setup_net_module.SetupNet(
            encoding=encoding, arch=setup_arch, main_arch=main_arch
        )
        shim_net = compat_v1_5.SetupNetV1_5(
            encoding=encoding, arch=setup_arch, main_arch=main_arch
        )

        live_vec = live_net.encode_candidate(candidate, context)
        shim_vec = shim_net.encode_candidate(candidate, context)

        base = encoding.off_goal_affinity
        assert live_vec[base] != 0.0  # played-and-egg-populated: star wild
        assert shim_vec[base] == 0.0  # frozen pre-1.6 play-instant pricing

        # Every other position — including the other 3 goal_affinity scalars,
        # which are all-zero either way for the padding goal categories — is
        # byte-identical between the shim and the live net.
        stripe_width = encoding.off_turn1_playable - base
        outside_stripe = np.ones(live_vec.shape[0], dtype=bool)
        outside_stripe[base : base + stripe_width] = False
        assert np.array_equal(live_vec[outside_stripe], shim_vec[outside_stripe])
        assert np.array_equal(
            live_vec[base + 1 : base + stripe_width],
            shim_vec[base + 1 : base + stripe_width],
        )

    def test_shim_output_width_matches_live_encoding(self) -> None:
        candidate, context = _star_nest_candidate()
        encoding = setup_model.SetupEncoding()
        shim_net = compat_v1_5.SetupNetV1_5(
            encoding=encoding,
            arch=_small_setup_arch(),
            main_arch=_small_main_arch_for_setup(),
        )
        vec = shim_net.encode_candidate(candidate, context)
        assert vec.shape == (encoding.total_dim,)


# ---------------------------------------------------------------------------
# (d) Real load-path round-trip (fixture-equivalent), setup side


def test_v1_5_stamped_setup_checkpoint_round_trips(tmp_path: pathlib.Path) -> None:
    """A v1.5-stamped setup checkpoint loads under live code via
    ``load_setup_net`` as ``SetupNetV1_5`` and freezes the pre-1.6 egg-blind
    ``goal_affinity`` pricing — the setup-side twin of
    ``test_v1_5_stamped_checkpoint_round_trips``."""
    base_cfg = config.RunConfig(
        misc=config.MiscConfig(device="cpu"),
        run=config.RunSettings(
            run_name="v15-setup-roundtrip", checkpoint_dir=str(tmp_path)
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
            setup=config.SetupNetArchitecture(head_layers=(8,)),
        ),
    )
    cfg = config.with_encoding_version(base_cfg, "1.5")
    assert cfg.encoding_version == "1.5"

    runmeta.write_run_config(
        str(tmp_path),
        cfg,
        stamp="t0",
        started_at="t0",
        git_sha=None,
        resumed_from_iteration=0,
    )
    descriptor = setup_runmeta.read_setup_config(str(tmp_path))
    assert descriptor.version == "1.5"

    net_cls = setup_net_module.SetupNet.class_for_version(cfg.encoding_version)
    assert net_cls is compat_v1_5.SetupNetV1_5
    net = net_cls.from_setup_config(descriptor)

    setup_payload: dict[str, object] = {
        "setup_model": net.state_dict(),
        "version": "1.5",
    }
    loop_checkpoint.atomic_save(setup_payload, tmp_path / artifacts.SETUP_CKPT)

    loaded = loaders.load_setup_net(tmp_path, torch.device("cpu"))
    assert isinstance(loaded, compat_v1_5.SetupNetV1_5)

    candidate, context = _star_nest_candidate()
    vec = loaded.encode_candidate(candidate, context)
    assert vec[loaded.encoding.off_goal_affinity] == 0.0


# ---------------------------------------------------------------------------
# (e) setup_architecture_key: leads with the era, so a same-shape setup net
# at a different era reads as incompatible.


class TestSetupArchitectureKeyEra:
    def test_leading_element_is_the_era_and_differs_across_eras(self) -> None:
        base_cfg = config.RunConfig()
        era_cfg = config.with_encoding_version(base_cfg, "1.5")
        live_cfg = config.with_encoding_version(base_cfg, version.MODEL_VERSION)

        assert era_cfg.setup_architecture_key[0] == "1.5"
        assert live_cfg.setup_architecture_key[0] == version.MODEL_VERSION
        assert era_cfg.setup_architecture_key != live_cfg.setup_architecture_key
