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

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import architecture, decisions, encode, model  # noqa: E402
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
    # The board-index block sits immediately before the candidate bird-index
    # column, and both precede the trailing conditional setup stripes, so the
    # model's slice offsets never move with the spec (the kept_multihot region
    # is by construction the include_setup row's final columns).
    assert (
        encode.CHOICE_BOARD_IDX_OFFSET + encode.CHOICE_BOARD_IDX_SLOTS
        == encode.CHOICE_BIRD_ID_OFFSET
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
        state_layout = stripes.state_stripe_layout(spec, card_embed_dim)
        choice_layout = stripes.choice_stripe_layout(spec, card_embed_dim)
        # The report shows the post-embedding network input, so the stripe sizes
        # sum to the trunk / choice-encoder first-Linear width, not the raw encoder
        # output width.
        assert (
            sum(s.size for s in state_layout.stripes)
            == state_layout.total_size
            == encode.trunk_input_dim(encode.state_size(spec), card_embed_dim)
        )
        assert (
            sum(s.size for s in choice_layout.stripes)
            == choice_layout.total_size
            == encode.choice_input_dim(
                encode.choice_feature_dim(spec),
                card_embed_dim,
                include_setup=spec.include_setup,
            )
        )
        choice_names = {s.name for s in choice_layout.stripes}
        assert ("setup_agg" in choice_names) == spec.include_setup
        assert ("kept_multihot" in choice_names) == spec.include_setup


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


def test_config_syncs_dims_to_use_setup_model_and_is_fresh_on_toggle():
    on = config.TrainConfig(device="cpu", use_setup_model=True)
    off = config.TrainConfig(device="cpu", use_setup_model=False)
    assert on.state_dim == encode.state_size(_EXCLUDE)
    assert off.state_dim == encode.state_size(_INCLUDE)
    assert on.choice_dim == encode.choice_feature_dim(_EXCLUDE)
    assert off.choice_dim == encode.choice_feature_dim(_INCLUDE)
    # Toggling use_setup_model now reshapes the main net, so it is a FRESH change.
    assert on.architecture_key != off.architecture_key
