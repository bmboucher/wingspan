"""Tests for the setup model's stripe registry and its model-summary sections.

Cover that the stripe layout stays in sync with the encoder (total size,
contiguous offsets, sub-fields within their stripe span) and that the HTML
report documents the setup model both when it is active and when it is off.
"""

from __future__ import annotations

from wingspan import architecture, decisions, encode, setup_model, version  # noqa: E402
from wingspan.encode import stripes as encode_stripes  # noqa: E402
from wingspan.reporting import html as report  # noqa: E402
from wingspan.setup_model import architecture as arch_module  # noqa: E402


def test_layout_total_matches_feature_dim():
    # Default encoding includes include_playable_kept_cards=True (488 dims).
    # SETUP_FEATURE_DIM (308) is the no-flags base; use encoding.total_dim here.
    encoding = arch_module.SetupEncoding()
    layout = setup_model.setup_stripe_layout(encoding)
    assert layout.total_size == encoding.total_dim
    assert sum(stripe.size for stripe in layout.stripes) == layout.total_size


def test_stripe_offsets_are_contiguous():
    layout = setup_model.setup_stripe_layout()
    expected_offset = 0
    for stripe in layout.stripes:
        assert stripe.offset == expected_offset
        expected_offset += stripe.size
    assert expected_offset == layout.total_size


def test_playable_kept_cards_stripe_layout_with_flag_on():
    """With include_playable_kept_cards=True the layout stays contiguous and sums correctly."""
    encoding = arch_module.SetupEncoding(include_playable_kept_cards=True)
    layout = setup_model.setup_stripe_layout(encoding)
    assert layout.total_size == encoding.total_dim
    assert sum(stripe.size for stripe in layout.stripes) == layout.total_size
    # The playable_kept_cards stripe must be present.
    stripe_names = [stripe.name for stripe in layout.stripes]
    assert "playable_kept_cards" in stripe_names


def test_state_and_choice_stripe_layouts_sum_correctly():
    """setup_state_stripe_layout / setup_choice_stripe_layout each sum to their dim
    function, and together to the fused readout width (the partition identity)."""
    encoding = arch_module.SetupEncoding(include_playable_kept_cards=True)
    from wingspan.setup_model import stripes as setup_stripes

    main_arch = architecture.ModelArchitecture()
    state_layout = setup_stripes.setup_state_stripe_layout(encoding, main_arch)
    choice_layout = setup_stripes.setup_choice_stripe_layout(encoding, main_arch)
    state_in = arch_module.setup_state_input_dim(encoding, main_arch)
    choice_in = arch_module.setup_choice_input_dim(encoding, main_arch)
    assert state_layout.total_size == state_in
    assert sum(stripe.size for stripe in state_layout.stripes) == state_in
    assert choice_layout.total_size == choice_in
    assert sum(stripe.size for stripe in choice_layout.stripes) == choice_in


def test_split_food_plus_playable_kept_cards_layout():
    """Combine split_food=True + include_playable_kept_cards=True — layout stays consistent."""
    encoding = arch_module.SetupEncoding(
        split_food=True, include_playable_kept_cards=True
    )
    layout = setup_model.setup_stripe_layout(encoding)
    assert layout.total_size == encoding.total_dim
    assert sum(stripe.size for stripe in layout.stripes) == layout.total_size


def test_sub_fields_stay_within_their_stripe():
    layout = setup_model.setup_stripe_layout()
    for stripe in layout.stripes:
        for sub_field in stripe.sub_fields:
            assert sub_field.relative_offset >= 0
            assert sub_field.relative_offset + sub_field.size <= stripe.size


def test_html_report_documents_active_setup_model():
    html = _report_html(use_setup_model=True)
    assert "id='setup_state'" in html
    assert "id='setup_choice'" in html
    assert "Setup State Vector" in html
    assert "Setup Choice Vector" in html
    assert "STATE TRUNK" in html
    assert "CHOICE TRUNK" in html
    assert "SETUP VALUE" in html
    assert "SETUP POLICY" in html
    assert "(separate)" in html
    assert "not active this run" not in html


def test_html_report_documents_inactive_setup_model():
    html = _report_html(use_setup_model=False)
    assert "id='setup_state'" in html
    assert "id='setup_choice'" in html
    assert "Setup State Vector" in html
    assert "Setup Choice Vector" in html
    assert "STATE TRUNK" in html
    assert "CHOICE TRUNK" in html
    assert "SETUP VALUE" in html
    assert "not active this run" in html


def test_html_report_arch_svg_content():
    html = _report_html(use_setup_model=True)
    arch = architecture.ModelArchitecture()
    # All seven blocks are drawn; default config uses pooling so hand is MULTI-CARD POOLING.
    assert "SINGLE-CARD ENCODER" in html
    assert "MULTI-CARD POOLING" in html
    assert "STATE ENCODER" in html
    assert "POLICY HEAD" in html
    # Encoder fan-out copy labels (card -> state board path, pooled hand -> setup).
    assert f"×{encode.N_CARD_INDEX_SLOTS}" in html
    assert "kept + playable" in html
    # "trained in-game only" note only appears when use_distinct_hand_model=True;
    # the default is now pooled (False), so this note is absent for a bare config.
    # Parameter counts are exact bare integers — no "123k", no Σ, no commas.
    first_width = arch.card_encoder_layers[0]
    first_linear = encode.CARD_FEATURE_DIM * first_width + first_width
    assert str(first_linear) in html
    assert f"{first_linear:,}" not in html
    assert "Σ" not in html
    # The trunk's M feeds both the value head and the decision head.
    assert html.count(f"M={arch.trunk_embed_width}") >= 2


def test_html_report_arch_svg_setup_off():
    html = _report_html(use_setup_model=False)
    assert "MULTI-CARD POOLING" in html
    assert "STATE TRUNK" in html
    assert "CHOICE TRUNK" in html


###### PRIVATE #######


def _report_html(use_setup_model: bool) -> str:
    """Generate the model-summary HTML for a baseline net with setup on/off."""
    spec = encode.spec_for(use_setup_model)
    state_dim = encode.state_size(spec)
    choice_dim = encode.choice_feature_dim(spec)
    arch = architecture.ModelArchitecture()
    family_order = tuple(
        family.value
        for family in decisions.active_decision_families(spec.include_setup)
    )
    param_report = architecture.count_parameters(
        arch,
        card_feat_in=encode.CARD_FEATURE_DIM,
        trunk_in=encode.trunk_input_dim(
            state_dim,
            arch.card_embed_dim,
            use_distinct_hand_model=arch.use_distinct_hand_model,
            hand_embed_dim=arch.hand_embed_dim,
            tray_set_embedding=arch.tray_set_embedding,
        ),
        choice_in=encode.choice_input_dim(
            choice_dim, arch.card_embed_dim, include_setup=spec.include_setup
        ),
        num_families=len(family_order),
        hand_feat_in=encode.HAND_ENCODER_INPUT_DIM,
    )
    return report.generate_html_report(
        encode_stripes.state_stripe_layout(spec, arch.card_embed_dim),
        encode_stripes.choice_stripe_layout(spec, arch.card_embed_dim),
        param_report,
        arch,
        setup_encoding=setup_model.SetupEncoding(),
        setup_arch=setup_model.SetupArchitecture(),
        use_setup_model=use_setup_model,
        state_dim=state_dim,
        choice_dim=choice_dim,
        family_order=family_order,
        run_name="test-setup-stripes",
        model_version=version.MODEL_VERSION,
    )
