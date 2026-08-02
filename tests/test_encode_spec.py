# pyright: reportPrivateUsage=false
# (a few assertions reference package-private layout constants, e.g. _SETUP_DIM)
"""The config-driven encoding shape (``EncodingSpec.include_setup``).

The main model's state/choice vectors — and the model's head count — depend on
one config axis: whether the opening (``SetupDecision``) is scored by the main
net or delegated to the separate setup model. The setup-only pieces are kept
*last* in every order, so excluding them is a clean truncation; these tests pin
that invariant and the round-trip through ``model_config.json``.
"""

from __future__ import annotations

import pydantic
import pytest

from wingspan import (  # noqa: E402
    architecture,
    compat,
    decisions,
    encode,
    model,
    version,
)
from wingspan.encode import layout, stripes  # noqa: E402
from wingspan.training import config, runmeta  # noqa: E402

_INCLUDE = encode.EncodingSpec(include_setup=True)
_EXCLUDE = encode.EncodingSpec(include_setup=False)
_SMALL = architecture.ModelArchitecture(
    trunk_layers=(8, 8),
    choice_layers=(8, 8),
    head_layers=(),
    value_layers=(),
    card_embed_dim=4,
)
_SMALL_ATTN_POSITIONS = architecture.ModelArchitecture(
    trunk_layers=(8, 8),
    choice_layers=(8, 8),
    head_layers=(),
    value_layers=(),
    card_embed_dim=4,
    use_board_attention=True,
    board_attention_positions=True,
)


def test_default_spec_excludes_setup():
    # The bare default matches a default run (use_setup_model=True ⇒ setup off the
    # main net), so bare nets / encoders agree with a default configured run.
    assert encode.DEFAULT_SPEC.include_setup is False


def test_state_size_differs_by_one_decision_type_column():
    # Only the trailing decision-type one-hot widens — by the SetupDecision column.
    assert encode.state_size(_INCLUDE) == encode.state_size(_EXCLUDE) + 1


def test_choice_dim_differs_by_the_trailing_setup_stripes():
    assert (
        encode.choice_feature_dim(_INCLUDE)
        == encode.choice_feature_dim(_EXCLUDE)
        + layout._SETUP_DIM
        + layout._KEPT_MULTIHOT_DIM
    )


def test_card_region_offsets_are_spec_invariant():
    # board_hab + board_col sit immediately before bird_id, and both precede
    # the trailing conditional setup stripes, so the model's slice offsets
    # never move with the spec (the kept_multihot region is by construction
    # the include_setup row's final columns).
    assert (
        encode.CHOICE_BOARD_COL_OFFSET + encode.CHOICE_BOARD_COL_DIM
        == encode.CHOICE_BIRD_ID_OFFSET
    )
    assert (
        encode.CHOICE_BOARD_HAB_OFFSET + encode.CHOICE_BOARD_HAB_DIM
        == encode.CHOICE_BOARD_COL_OFFSET
    )
    assert (
        encode.CHOICE_KEPT_MULTIHOT_OFFSET + encode.CHOICE_KEPT_MULTIHOT_DIM
        == encode.choice_feature_dim(_INCLUDE)
    )


def test_active_decision_classes_drop_only_setup_and_keep_it_last():
    full = decisions.active_decision_classes(True)
    reduced = decisions.active_decision_classes(False)
    assert set(full) - set(reduced) == {decisions.SetupDecision}
    assert full[-1] is decisions.SetupDecision


def test_active_families_drop_only_setup_and_keep_it_last():
    full = decisions.active_decision_families(True)
    reduced = decisions.active_decision_families(False)
    assert set(full) - set(reduced) == {decisions.DecisionFamily.SETUP}
    assert full[-1] is decisions.DecisionFamily.SETUP


def test_family_index_is_stable_across_the_setup_axis():
    # SETUP is last, so every non-setup decision keeps a head index in range for
    # the setup-excluded model — family_index_for needs no spec.
    for decision_class in decisions.active_decision_classes(False):
        assert decisions.family_index_for(decision_class) < encode.num_families(
            _EXCLUDE
        )


def test_stripe_layouts_sum_and_show_setup_only_when_included():
    card_embed_dim = 8
    for spec in (_INCLUDE, _EXCLUDE):
        # Pre-0.6 compat path: n_playable=0, has_becomes_playable=False
        state_layout = stripes.state_stripe_layout(spec, card_embed_dim)
        choice_layout = stripes.choice_stripe_layout(
            spec, card_embed_dim, has_becomes_playable=False
        )
        # The report shows the post-embedding network input, so the stripe sizes
        # sum to the trunk / choice-encoder first-Linear width, not the raw encoder
        # output width.
        assert (
            sum(s.size for s in state_layout.stripes)
            == state_layout.total_size
            == encode.trunk_input_dim(encode.state_size(spec), card_embed_dim)
        )
        # Pre-0.6 raw dim omits both playable stripes, so the formula is called
        # with choice_feature_dim - CHOICE_BECOMES_PLAYABLE_DIM - CHOICE_BECOMES_UNPLAYABLE_DIM.
        raw_pre06_dim = (
            encode.choice_feature_dim(spec)
            - layout.CHOICE_BECOMES_PLAYABLE_DIM
            - layout.CHOICE_BECOMES_UNPLAYABLE_DIM
        )
        assert (
            sum(s.size for s in choice_layout.stripes)
            == choice_layout.total_size
            == encode.choice_input_dim(
                raw_pre06_dim,
                card_embed_dim,
                include_setup=spec.include_setup,
                has_becomes_playable=False,
                has_becomes_unplayable=False,
            )
        )
        choice_names = {s.name for s in choice_layout.stripes}
        assert ("setup_agg" in choice_names) == spec.include_setup
        assert ("kept_multihot" in choice_names) == spec.include_setup
        assert "becomes_playable" not in choice_names
        assert "becomes_unplayable" not in choice_names

        # Live v0.6+ path: n_playable=N, has_becomes_playable=True
        n_playable = encode.N_HAND_PLAYABLE_MULTIHOTS
        state_layout_live = stripes.state_stripe_layout(
            spec, card_embed_dim, n_playable_multihots=n_playable
        )
        choice_layout_live = stripes.choice_stripe_layout(spec, card_embed_dim)
        assert (
            sum(s.size for s in state_layout_live.stripes)
            == state_layout_live.total_size
            == encode.trunk_input_dim(
                encode.state_size(spec), card_embed_dim, n_playable_multihots=n_playable
            )
        )
        # Playable multi-hots must appear embedded at card_embed_dim, not 180
        live_stripe_by_name = {s.name: s for s in state_layout_live.stripes}
        assert live_stripe_by_name["hand_playable_me"].size == card_embed_dim
        assert live_stripe_by_name["hand_playable_eggs_me"].size == card_embed_dim
        assert (
            sum(s.size for s in choice_layout_live.stripes)
            == choice_layout_live.total_size
            == encode.choice_input_dim(
                encode.choice_feature_dim(spec),
                card_embed_dim,
                include_setup=spec.include_setup,
            )
        )
        choice_live_names = {s.name for s in choice_layout_live.stripes}
        assert ("setup_agg" in choice_live_names) == spec.include_setup
        assert ("kept_multihot" in choice_live_names) == spec.include_setup
        assert "becomes_playable" in choice_live_names
        assert "becomes_unplayable" in choice_live_names


def test_model_builds_for_both_specs():
    inc = model.PolicyValueNet(spec=_INCLUDE, arch=_SMALL)
    exc = model.PolicyValueNet(spec=_EXCLUDE, arch=_SMALL)
    assert inc.include_setup is True and exc.include_setup is False
    assert len(inc.scorers) == len(decisions.active_decision_families(True))
    assert len(exc.scorers) == len(decisions.active_decision_families(False))
    assert inc.choice_dim == encode.choice_feature_dim(_INCLUDE)
    assert exc.choice_dim == encode.choice_feature_dim(_EXCLUDE)
    assert inc.state_dim == encode.state_size(_INCLUDE)
    assert exc.state_dim == encode.state_size(_EXCLUDE)


def test_from_model_config_round_trips_include_setup():
    for include in (True, False):
        spec = encode.EncodingSpec(include_setup=include)
        descriptor = runmeta.ModelConfig(
            run_name="t",
            state_dim=encode.state_size(spec),
            choice_dim=encode.choice_feature_dim(spec),
            family_order=tuple(
                f.value for f in decisions.active_decision_families(include)
            ),
            architecture=_SMALL,
            include_setup=include,
        )
        net = model.PolicyValueNet.from_model_config(descriptor)
        assert net.include_setup is include
        assert net.choice_dim == encode.choice_feature_dim(spec)
        assert len(net.scorers) == len(decisions.active_decision_families(include))


def _make_current_era_descriptor(*, include_setup: bool = False) -> runmeta.ModelConfig:
    """Build a minimal ModelConfig at the current MODEL_VERSION for era-routing tests."""
    spec = encode.EncodingSpec(include_setup=include_setup)
    return runmeta.ModelConfig(
        run_name="t",
        state_dim=encode.state_size(spec),
        choice_dim=encode.choice_feature_dim(spec),
        family_order=tuple(
            f.value for f in decisions.active_decision_families(include_setup)
        ),
        architecture=_SMALL,
        include_setup=include_setup,
        version=version.MODEL_VERSION,
    )


def test_state_layout_for_matches_param_report_trunk_in():
    """state_layout_for().total_size equals trunk_input_dim() for current-era descriptors."""
    for include_setup in (False, True):
        descriptor = _make_current_era_descriptor(include_setup=include_setup)
        state_layout = runmeta.state_layout_for(descriptor)
        arch = descriptor.architecture

        # Independent trunk_in mirroring param_report_for, passing all arch
        # params so pooled_hand_width and other knobs match. With no pre-1.0
        # shims every descriptor is live, so both playability multi-hots are present.
        n_playable = encode.N_HAND_PLAYABLE_MULTIHOTS
        expected_trunk_in = encode.trunk_input_dim(
            descriptor.state_dim,
            arch.card_embed_dim,
            use_distinct_hand_model=arch.use_distinct_hand_model,
            hand_embed_dim=arch.hand_embed_dim,
            pooled_hand_width=arch.pooled_hand_width,
            tray_set_embedding=arch.tray_set_embedding,
            n_playable_multihots=n_playable,
        )
        assert state_layout.total_size == expected_trunk_in
        assert sum(s.size for s in state_layout.stripes) == state_layout.total_size


def test_state_layout_for_matches_param_report_trunk_in_with_board_attention_positions():
    """state_layout_for().total_size equals trunk_input_dim(..., board_position_dim=...)
    for a descriptor whose architecture has board_attention_positions on, and
    param_report_for's board-attention block accounts for the wider token."""
    spec = encode.EncodingSpec(include_setup=False)
    descriptor = runmeta.ModelConfig(
        run_name="t",
        state_dim=encode.state_size(spec),
        choice_dim=encode.choice_feature_dim(spec),
        family_order=tuple(f.value for f in decisions.active_decision_families(False)),
        architecture=_SMALL_ATTN_POSITIONS,
        include_setup=False,
        version=version.MODEL_VERSION,
    )
    state_layout = runmeta.state_layout_for(descriptor)
    arch = descriptor.architecture
    n_playable = encode.N_HAND_PLAYABLE_MULTIHOTS
    expected_trunk_in = encode.trunk_input_dim(
        descriptor.state_dim,
        arch.card_embed_dim,
        use_distinct_hand_model=arch.use_distinct_hand_model,
        hand_embed_dim=arch.hand_embed_dim,
        pooled_hand_width=arch.pooled_hand_width,
        tray_set_embedding=arch.tray_set_embedding,
        n_playable_multihots=n_playable,
        board_position_dim=encode.BOARD_POSITION_DIM,
    )
    assert state_layout.total_size == expected_trunk_in
    assert sum(s.size for s in state_layout.stripes) == state_layout.total_size

    param_report = runmeta.param_report_for(descriptor)
    assert param_report.board_attention is not None
    # W = 4 (card_embed_dim) + 9 (slot_scalar_dim) + 8 (board_position_dim) = 21.
    expected_attn_params = 2 * (4 * 21 * 21 + 4 * 21)
    assert param_report.board_attention.total == expected_attn_params


def test_choice_layout_for_matches_param_report_choice_in():
    """choice_layout_for().total_size equals choice_input_dim_for() for current-era descriptors."""
    for include_setup in (False, True):
        descriptor = _make_current_era_descriptor(include_setup=include_setup)
        choice_layout = runmeta.choice_layout_for(descriptor)
        expected_choice_in = runmeta.choice_input_dim_for(descriptor)
        assert choice_layout.total_size == expected_choice_in
        assert sum(s.size for s in choice_layout.stripes) == choice_layout.total_size


def test_config_syncs_dims_to_use_setup_model_and_is_fresh_on_toggle():
    on = config.RunConfig(
        misc=config.MiscConfig(device="cpu"),
        architecture=config.ArchitectureConfig(use_setup_model=True),
    )
    off = config.RunConfig(
        misc=config.MiscConfig(device="cpu"),
        architecture=config.ArchitectureConfig(use_setup_model=False),
    )
    assert on.state_dim == encode.state_size(_EXCLUDE)
    assert off.state_dim == encode.state_size(_INCLUDE)
    assert on.choice_dim == encode.choice_feature_dim(_EXCLUDE)
    assert off.choice_dim == encode.choice_feature_dim(_INCLUDE)
    # Toggling use_setup_model now reshapes the main net, so it is a FRESH change.
    assert on.architecture_key != off.architecture_key


#### num_players ####


def test_config_syncs_dims_to_num_players():
    n3 = config.RunConfig(
        misc=config.MiscConfig(device="cpu"),
        architecture=config.ArchitectureConfig(num_players=3),
    )
    assert n3.state_dim == 1299
    assert n3.choice_dim == 512
    assert n3.num_players == 3
    assert n3.arch.num_players == 3
    assert n3.encoding_spec.num_players == 3


def test_num_players_defaults_to_2():
    default = config.RunConfig(misc=config.MiscConfig(device="cpu"))
    assert default.architecture.num_players == 2
    assert default.num_players == 2
    assert default.state_dim == encode.state_size(_EXCLUDE)
    assert default.choice_dim == encode.choice_feature_dim(_EXCLUDE)


def test_num_players_is_fresh_and_joins_shape_key():
    n2 = config.RunConfig(
        misc=config.MiscConfig(device="cpu"),
        architecture=config.ArchitectureConfig(num_players=2),
    )
    n3 = config.RunConfig(
        misc=config.MiscConfig(device="cpu"),
        architecture=config.ArchitectureConfig(num_players=3),
    )
    assert n2.architecture_key != n3.architecture_key
    assert n2.arch.shape_key.num_players == 2
    assert n3.arch.shape_key.num_players == 3
    assert n2.arch.shape_key != n3.arch.shape_key


def test_num_players_out_of_range_rejected():
    for bad in (1, 6):
        with pytest.raises(pydantic.ValidationError):
            config.ArchitectureConfig(num_players=bad)


def test_rehydration_defaults_num_players_absent_parses_to_2():
    """A config dict that predates ``num_players`` (the keys deleted from a
    default dump, simulating a pre-Stage-2 run_config_<stamp>.json) parses
    back to num_players=2 and its architecture_key equals a fresh default
    config's — the resume-protection proof (alongside the untouched
    test_era_pinned_resume.py). Both sides recompute their architecture_key
    under *current* code, so a pre-Stage-2 run and a fresh default run still
    agree — even though the key's own repr now includes num_players and so
    differs from what it would have printed before this change."""
    default = config.RunConfig(misc=config.MiscConfig(device="cpu"))
    data = default.model_dump()
    del data["architecture"]["num_players"]
    assert "num_players" not in data["architecture"]

    reconstituted = config.RunConfig.model_validate(data)
    assert reconstituted.architecture.num_players == 2
    assert reconstituted.architecture_key == default.architecture_key


def test_compat_era_refuses_num_players_3():
    """A superseded (non-live) encoding era can never be N-player: every
    compat-shimmed era predates num_players by construction."""
    with pytest.raises(version.IncompatibleArtifactError):
        compat.encoding_dims_for_era("1.3", encode.EncodingSpec(num_players=3))
    # num_players=2 at the same era is unaffected (existing 2P resume path).
    dims = compat.encoding_dims_for_era("1.3", encode.EncodingSpec(num_players=2))
    assert dims == (
        encode.state_size(_EXCLUDE) - 10,
        encode.choice_feature_dim(_EXCLUDE) - 1,
    )
