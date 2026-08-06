# pyright: reportPrivateUsage=false
# (tests read the package-private bonus_value layout offsets to pin the frozen
# scalars, matching the test_encode.py / test_compat_v1_5.py convention)
"""Tests for the pre-1.7 -> v1.7 compat shim: the static (egg-blind) bonus
potential value freeze, on both nets.

v1.7 made the bonus *potential* counters optimistic about the egg-counting
dynamic cards (``scoring.bonus_potential_count``): a not-yet-played bird whose
``egg_limit`` reaches the card's threshold (Breeding Manager at 4, Oologist at
1) now counts. Pre-1.7 encoders counted only the static ``bonus_categories``
tag, which no dynamic card carries — so those cards' potentials read 0. No
dims change on either net: ``compat.v1_6.PolicyValueNetV1_6`` overrides only
``encode_choices`` (regenerating the static ``hand_potential`` /
``tray_potential`` on bonus-carrying rows) and ``compat.v1_6.SetupNetV1_6``
overrides only ``encode_candidate`` (regenerating the static
``bonus_card_affinity`` pair / ``kept_bonus_value`` 4-vector).
``compat.v1_5`` re-chains to subclass both, so every earlier era freezes the
static potentials too.

As for every prior era, a committed LFS checkpoint fixture is deferred: the
round-trip tests build v1.6-stamped nets and reload them through the
production ``load_policy_net`` / ``load_setup_net`` paths.
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
from wingspan.compat import v1_5 as compat_v1_5
from wingspan.compat import v1_6 as compat_v1_6
from wingspan.encode import layout
from wingspan.engine import scoring
from wingspan.model import core
from wingspan.players import loaders
from wingspan.training import artifacts, config, loop_checkpoint, runmeta
from wingspan.training import setup_net as setup_net_module
from wingspan.training import setup_runmeta

_BIRDS, _BONUSES, _GOALS = cards.load_all()
_BONUS_BY_NAME = {bonus_card.name: bonus_card for bonus_card in _BONUSES}
_BIG_NEST = [bird for bird in _BIRDS if bird.egg_limit >= 4]

_HAND_IDX = layout._OFF_BONUS_VALUE + layout._BONUS_VALUE_HAND
_TRAY_IDX = layout._OFF_BONUS_VALUE + layout._BONUS_VALUE_TRAY


def _small_arch() -> architecture.ModelArchitecture:
    return architecture.ModelArchitecture(
        trunk_layers=(8, 8),
        choice_layers=(8, 8),
        head_layers=(),
        value_layers=(),
        card_embed_dim=4,
    )


def _era_shim(
    spec: encode.EncodingSpec = encode.DEFAULT_SPEC,
) -> compat_v1_6.PolicyValueNetV1_6:
    """A v1_6 shim built at era 1.6's dims — which equal live (behavior-only
    era), exactly how the load path builds it."""
    state_dim, choice_dim = compat.encoding_dims_for_era("1.6", spec)
    return compat_v1_6.PolicyValueNetV1_6(
        state_dim=state_dim, choice_dim=choice_dim, arch=_small_arch(), spec=spec
    )


def _pick_bonus_decision(
    bonus_card: cards.BonusCard,
) -> decisions.BirdPowerPickBonusCardDecision:
    return decisions.BirdPowerPickBonusCardDecision(
        player_id=0,
        prompt="x",
        choices=[
            decisions.BonusCardChoice(label=bonus_card.name, bonus_card=bonus_card)
        ],
    )


def _eggy_hand_state() -> state.GameState:
    """A game state whose deciding player holds three 4+-egg-capacity birds and
    whose tray shows one more — nonzero Breeding Manager potentials under the
    live v1.7 pricing, zero under the static pre-1.7 pricing."""
    eng, *_ = engine.Engine.create(seed=110)
    eng.state.players[0].hand = list(_BIG_NEST[:3])
    eng.state.tray = [_BIG_NEST[3], None, None]
    return eng.state


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
# (a) class_for_version routing, both nets, and the v1_5 re-chain


class TestClassForVersionRouting:
    def test_v1_6_routes_to_shim(self) -> None:
        assert (
            core.PolicyValueNet.class_for_version("1.6")
            is compat_v1_6.PolicyValueNetV1_6
        )
        assert (
            setup_net_module.SetupNet.class_for_version("1.6")
            is compat_v1_6.SetupNetV1_6
        )

    def test_v1_5_still_routes_to_its_own_shim_which_subclasses_v1_6(self) -> None:
        """Era 1.5's routing targets are unchanged (their own shim classes),
        but those classes now subclass the v1_6 shims (were the live nets)."""
        assert (
            core.PolicyValueNet.class_for_version("1.5")
            is compat_v1_5.PolicyValueNetV1_5
        )
        assert issubclass(
            compat_v1_5.PolicyValueNetV1_5, compat_v1_6.PolicyValueNetV1_6
        )
        assert (
            setup_net_module.SetupNet.class_for_version("1.5")
            is compat_v1_5.SetupNetV1_5
        )
        assert issubclass(compat_v1_5.SetupNetV1_5, compat_v1_6.SetupNetV1_6)

    def test_current_version_returns_live_classes(self) -> None:
        assert (
            core.PolicyValueNet.class_for_version(version.MODEL_VERSION)
            is core.PolicyValueNet
        )
        assert (
            setup_net_module.SetupNet.class_for_version(version.MODEL_VERSION)
            is setup_net_module.SetupNet
        )


# ---------------------------------------------------------------------------
# (b) behavior-only era: dims are live, no dims-router branch


class TestEncodingDimsForEra:
    def test_era_1_6_dims_equal_live(self) -> None:
        spec = encode.DEFAULT_SPEC
        state_dim, choice_dim = compat.encoding_dims_for_era("1.6", spec)
        assert state_dim == encode.state_size(spec)
        assert choice_dim == encode.choice_feature_dim(spec)


# ---------------------------------------------------------------------------
# (c) main-net value freeze: the shim's bonus potentials equal the OLD
# (static) pricing while the live encoder's are nonzero; every other column
# is byte-identical.


class TestMainNetBonusPotentialFreeze:
    def test_shim_freezes_static_potentials_live_encoder_does_not(self) -> None:
        game_state = _eggy_hand_state()
        decision = _pick_bonus_decision(_BONUS_BY_NAME["Breeding Manager"])
        live_rows = encode.encode_choices(decision, game_state)
        shim_rows = _era_shim().encode_choices(decision, game_state)
        assert live_rows.shape == shim_rows.shape  # behavior-only: no narrowing

        assert np.isclose(live_rows[0][_HAND_IDX], 3.0 / layout._BONUS_COUNT_SCALE)
        assert np.isclose(live_rows[0][_TRAY_IDX], 1.0 / layout._BONUS_COUNT_SCALE)
        assert shim_rows[0][_HAND_IDX] == 0.0
        assert shim_rows[0][_TRAY_IDX] == 0.0

        outside = np.ones(live_rows.shape[1], dtype=bool)
        outside[[_HAND_IDX, _TRAY_IDX]] = False
        assert np.array_equal(live_rows[0][outside], shim_rows[0][outside])

    def test_shim_preserves_hand_counting_card_full_source(self) -> None:
        """The refill re-runs the generic static predicate, not a zeroing:
        Visionary Leader's full-hand ``hand_potential`` — unchanged across
        eras — survives the shim byte-identically."""
        game_state = _eggy_hand_state()
        decision = _pick_bonus_decision(_BONUS_BY_NAME["Visionary Leader"])
        live_rows = encode.encode_choices(decision, game_state)
        shim_rows = _era_shim().encode_choices(decision, game_state)
        assert np.isclose(live_rows[0][_HAND_IDX], 3.0 / layout._BONUS_COUNT_SCALE)
        assert np.array_equal(live_rows, shim_rows)

    def test_shim_matches_live_on_static_card_rows(self) -> None:
        """A static type-counting card's potentials never changed, so the shim
        output equals the live encoding byte-for-byte."""
        game_state = _eggy_hand_state()
        decision = _pick_bonus_decision(_BONUS_BY_NAME["Bird Feeder"])
        live_rows = encode.encode_choices(decision, game_state)
        shim_rows = _era_shim().encode_choices(decision, game_state)
        assert np.array_equal(live_rows, shim_rows)

    def test_v1_5_era_net_strips_tail_and_freezes_potentials(self) -> None:
        """A v1.5-era net composes both shims: rows are 8 narrower (its own
        tail-strip) AND the bonus potentials are static (inherited from
        v1_6) — the refill offsets precede the stripped tail."""
        game_state = _eggy_hand_state()
        decision = _pick_bonus_decision(_BONUS_BY_NAME["Breeding Manager"])
        spec = encode.DEFAULT_SPEC
        state_dim, choice_dim = compat.encoding_dims_for_era("1.5", spec)
        v1_5_net = compat_v1_5.PolicyValueNetV1_5(
            state_dim=state_dim, choice_dim=choice_dim, arch=_small_arch(), spec=spec
        )
        rows = v1_5_net.encode_choices(decision, game_state)
        assert rows.shape[1] == (
            encode.choice_feature_dim(spec) - encode.CHOICE_GOAL_DELTA_IGNORING_EGGS_DIM
        )
        assert rows[0][_HAND_IDX] == 0.0
        assert rows[0][_TRAY_IDX] == 0.0

    def test_forward_pass_runs_at_era_dims(self) -> None:
        game_state = _eggy_hand_state()
        _forward(
            _era_shim(),
            _pick_bonus_decision(_BONUS_BY_NAME["Breeding Manager"]),
            game_state,
        )


# ---------------------------------------------------------------------------
# (d) setup-net value freeze, folded and split modes


def _small_main_arch_for_setup() -> architecture.ModelArchitecture:
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


def _setup_nets(
    encoding: setup_model.SetupEncoding,
) -> tuple[setup_net_module.SetupNet, compat_v1_6.SetupNetV1_6]:
    live_net = setup_net_module.SetupNet(
        encoding=encoding,
        arch=_small_setup_arch(),
        main_arch=_small_main_arch_for_setup(),
    )
    shim_net = compat_v1_6.SetupNetV1_6(
        encoding=encoding,
        arch=_small_setup_arch(),
        main_arch=_small_main_arch_for_setup(),
    )
    return live_net, shim_net


class TestSetupBonusPricingFreeze:
    def test_folded_kept_bonus_value_freezes_all_four_scalars(self) -> None:
        """Breeding Manager kept with two 4+-egg-capacity birds: the live
        encoder prices qual/stepped/linear at the optimistic count and the
        tray potential at the tray's egg-capable bird; the shim regenerates
        the static zeros. Every other position is byte-identical."""
        breeding_manager = _BONUS_BY_NAME["Breeding Manager"]
        candidate = setup_model.SetupCandidate(
            kept_cards=(_BIG_NEST[0], _BIG_NEST[1]),
            kept_foods=(cards.Food.SEED, cards.Food.FISH, cards.Food.FRUIT),
            bonus_card=breeding_manager,
        )
        context = setup_model.SetupContext(
            tray_birds=(_BIG_NEST[2], None, None),
            birdfeeder_counts=(0, 0, 0, 0, 0, 0),
            round_goal_categories=("birds_forest",) * 4,
        )
        encoding = setup_model.SetupEncoding()
        live_net, shim_net = _setup_nets(encoding)
        live_vec = live_net.encode_candidate(candidate, context)
        shim_vec = shim_net.encode_candidate(candidate, context)

        base = encoding.off_bonus_value
        assert np.isclose(live_vec[base], 2.0 / layout._BONUS_COUNT_SCALE)
        assert np.isclose(
            live_vec[base + 1],
            scoring.bonus_score_for_count(breeding_manager, 2)
            / layout._BONUS_VALUE_SCALE,
        )
        assert live_vec[base + 2] > 0.0
        assert np.isclose(live_vec[base + 3], 1.0 / layout._BONUS_COUNT_SCALE)
        assert np.all(shim_vec[base : base + 4] == 0.0)

        outside = np.ones(live_vec.shape[0], dtype=bool)
        outside[base : base + 4] = False
        assert np.array_equal(live_vec[outside], shim_vec[outside])

    def test_split_bonus_card_affinity_freezes_min_max(self) -> None:
        """Split mode: with Breeding Manager and a static tagged card dealt,
        the live affinity pair prices the egg card by egg capacity while the
        shim regenerates the static counts."""
        from wingspan.setup_model import architecture as arch_module

        breeding_manager = _BONUS_BY_NAME["Breeding Manager"]
        bird_feeder = _BONUS_BY_NAME["Bird Feeder"]
        tagged = next(
            bird for bird in _BIRDS if bird_feeder.name in bird.bonus_categories
        )
        high_a = _BIRDS[0].model_copy(update={"egg_limit": 4, "bonus_categories": ()})
        high_b = _BIRDS[1].model_copy(update={"egg_limit": 5, "bonus_categories": ()})
        tagged_low = tagged.model_copy(update={"egg_limit": 2})
        candidate = setup_model.SetupCandidate(
            kept_cards=(high_a, high_b, tagged_low),
            kept_foods=(cards.Food.SEED, cards.Food.FISH),
            bonus_card=None,
        )
        context = setup_model.SetupContext(
            tray_birds=(None, None, None),
            birdfeeder_counts=(0, 0, 0, 0, 0, 0),
            round_goal_categories=("birds_forest",) * 4,
            dealt_bonus_cards=(breeding_manager, bird_feeder),
        )
        encoding = setup_model.SetupEncoding(split_bonus=True)
        live_net, shim_net = _setup_nets(encoding)
        live_vec = live_net.encode_candidate(candidate, context)
        shim_vec = shim_net.encode_candidate(candidate, context)

        base = encoding.off_bonus_block + arch_module._BONUS_DIM
        # Live: Bird Feeder counts its tagged keep (1), Breeding Manager both
        # 4+-egg keeps (2). Shim: the egg card falls back to its (empty) tag.
        assert np.isclose(live_vec[base + 0], 1.0 / layout._BONUS_COUNT_SCALE)
        assert np.isclose(live_vec[base + 1], 2.0 / layout._BONUS_COUNT_SCALE)
        assert shim_vec[base + 0] == 0.0
        assert np.isclose(shim_vec[base + 1], 1.0 / layout._BONUS_COUNT_SCALE)

        outside = np.ones(live_vec.shape[0], dtype=bool)
        outside[base : base + 2] = False
        assert np.array_equal(live_vec[outside], shim_vec[outside])


# ---------------------------------------------------------------------------
# (e) Real load-path round-trips (fixture-equivalent), both nets


def test_v1_6_stamped_checkpoint_round_trips(tmp_path: pathlib.Path) -> None:
    """A v1.6-stamped checkpoint loads under live code via ``load_policy_net``
    as the shim class (at live dims — behavior-only era) and forward-passes."""
    base = config.RunConfig(
        misc=config.MiscConfig(device="cpu"),
        run=config.RunSettings(
            run_name="v16-roundtrip",
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
    cfg = config.with_encoding_version(base, "1.6")
    assert cfg.encoding_version == "1.6"
    assert cfg.state_dim == encode.state_size(cfg.encoding_spec)
    assert cfg.choice_dim == encode.choice_feature_dim(cfg.encoding_spec)

    net_cls = model.PolicyValueNet.class_for_version(cfg.encoding_version)
    assert net_cls is compat_v1_6.PolicyValueNetV1_6
    net = net_cls(
        state_dim=cfg.state_dim,
        choice_dim=cfg.choice_dim,
        num_families=len(cfg.family_order),
        arch=cfg.arch,
        spec=cfg.encoding_spec,
    )

    ckpt = tmp_path / "v16.pt"
    torch.save(
        {"config": cfg.model_dump(), "model": net.state_dict(), "version": "1.6"},
        ckpt,
    )

    loaded, saved_cfg = loaders.load_policy_net(ckpt, torch.device("cpu"))
    assert isinstance(loaded, compat_v1_6.PolicyValueNetV1_6)
    assert not isinstance(loaded, compat_v1_5.PolicyValueNetV1_5)
    assert saved_cfg.encoding_version == "1.6"

    game_state = _eggy_hand_state()
    decision = _pick_bonus_decision(_BONUS_BY_NAME["Breeding Manager"])
    assert loaded.encode_choices(decision, game_state)[0][_HAND_IDX] == 0.0
    _forward(loaded, decision, game_state)


def test_v1_6_stamped_setup_checkpoint_round_trips(tmp_path: pathlib.Path) -> None:
    """A v1.6-stamped setup checkpoint loads under live code via
    ``load_setup_net`` as ``SetupNetV1_6`` and freezes the static bonus
    pricing — the setup-side twin of the main-net round-trip above."""
    base_cfg = config.RunConfig(
        misc=config.MiscConfig(device="cpu"),
        run=config.RunSettings(
            run_name="v16-setup-roundtrip", checkpoint_dir=str(tmp_path)
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
    cfg = config.with_encoding_version(base_cfg, "1.6")
    assert cfg.encoding_version == "1.6"

    runmeta.write_run_config(
        str(tmp_path),
        cfg,
        stamp="t0",
        started_at="t0",
        git_sha=None,
        resumed_from_iteration=0,
    )
    descriptor = setup_runmeta.read_setup_config(str(tmp_path))
    assert descriptor.version == "1.6"

    net_cls = setup_net_module.SetupNet.class_for_version(cfg.encoding_version)
    assert net_cls is compat_v1_6.SetupNetV1_6
    net = net_cls.from_setup_config(descriptor)

    setup_payload: dict[str, object] = {
        "setup_model": net.state_dict(),
        "version": "1.6",
    }
    loop_checkpoint.atomic_save(setup_payload, tmp_path / artifacts.SETUP_CKPT)

    loaded = loaders.load_setup_net(tmp_path, torch.device("cpu"))
    assert isinstance(loaded, compat_v1_6.SetupNetV1_6)
    assert not isinstance(loaded, compat_v1_5.SetupNetV1_5)

    # Freeze assert against whichever bonus-block shape the production config
    # carries (split_bonus defers the kept bonus, so the candidate then keeps
    # none and the dealt Breeding Manager is priced by the affinity pair).
    from wingspan.setup_model import architecture as setup_arch_module

    breeding_manager = _BONUS_BY_NAME["Breeding Manager"]
    candidate = setup_model.SetupCandidate(
        kept_cards=(_BIG_NEST[0], _BIG_NEST[1]),
        kept_foods=(cards.Food.SEED, cards.Food.FISH, cards.Food.FRUIT),
        bonus_card=None if loaded.encoding.split_bonus else breeding_manager,
    )
    context = setup_model.SetupContext(
        tray_birds=(None, None, None),
        birdfeeder_counts=(0, 0, 0, 0, 0, 0),
        round_goal_categories=("birds_forest",) * 4,
        dealt_bonus_cards=(breeding_manager,),
    )
    vec = loaded.encode_candidate(candidate, context)
    if loaded.encoding.split_bonus:
        base = loaded.encoding.off_bonus_block + setup_arch_module._BONUS_DIM
        assert vec[base] == 0.0  # live pricing would read 2/5 (both keeps 4+ eggs)
        assert vec[base + 1] == 0.0
    else:
        assert vec[loaded.encoding.off_bonus_value] == 0.0


# ---------------------------------------------------------------------------
# (f) architecture keys: lead with the era, so a same-shape net at era 1.6
# reads as incompatible with a live (1.7) run.


class TestArchitectureKeyEra:
    def test_keys_differ_between_1_6_and_live(self) -> None:
        base_cfg = config.RunConfig()
        era_cfg = config.with_encoding_version(base_cfg, "1.6")
        live_cfg = config.with_encoding_version(base_cfg, version.MODEL_VERSION)

        assert era_cfg.setup_architecture_key[0] == "1.6"
        assert live_cfg.setup_architecture_key[0] == version.MODEL_VERSION
        assert era_cfg.setup_architecture_key != live_cfg.setup_architecture_key
        assert era_cfg.architecture_key != live_cfg.architecture_key
