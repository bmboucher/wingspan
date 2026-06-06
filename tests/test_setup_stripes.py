"""Tests for the setup model's stripe registry and its model-summary sections.

Cover that the stripe layout stays in sync with the encoder (total size,
contiguous offsets, sub-fields within their stripe span) and that the HTML
report documents the setup model both when it is active and when it is off.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import architecture, decisions, encode, setup_model, version  # noqa: E402
from wingspan.encode import stripes as encode_stripes  # noqa: E402
from wingspan.reporting import html as report  # noqa: E402


def test_layout_total_matches_feature_dim():
    layout = setup_model.setup_stripe_layout()
    assert layout.total_size == setup_model.SETUP_FEATURE_DIM
    assert sum(stripe.size for stripe in layout.stripes) == layout.total_size


def test_stripe_offsets_are_contiguous():
    layout = setup_model.setup_stripe_layout()
    expected_offset = 0
    for stripe in layout.stripes:
        assert stripe.offset == expected_offset
        expected_offset += stripe.size
    assert expected_offset == layout.total_size


def test_sub_fields_stay_within_their_stripe():
    layout = setup_model.setup_stripe_layout()
    for stripe in layout.stripes:
        for sub_field in stripe.sub_fields:
            assert sub_field.relative_offset >= 0
            assert sub_field.relative_offset + sub_field.size <= stripe.size


def test_html_report_documents_active_setup_model():
    html = _report_html(use_setup_model=True)
    assert "id='setup'" in html
    assert "Setup Vector" in html
    assert "SETUP MODEL" in html
    assert str(setup_model.SETUP_FEATURE_DIM) in html
    assert "(separate)" in html
    assert "not active this run" not in html
    assert "off this run" not in html


def test_html_report_documents_inactive_setup_model():
    html = _report_html(use_setup_model=False)
    assert "id='setup'" in html
    assert "Setup Vector" in html
    assert "SETUP MODEL" in html
    assert "not active this run" in html
    assert "off this run" in html


def test_html_report_arch_svg_content():
    html = _report_html(use_setup_model=True)
    arch = architecture.ModelArchitecture()
    # All seven blocks are drawn, including the multi-card encoder.
    assert "SINGLE-CARD ENCODER" in html
    assert "MULTI-CARD ENCODER" in html
    assert "STATE ENCODER" in html
    assert "DECISION HEAD" in html
    # Encoder fan-out copy labels (card -> trunk, hand -> setup).
    assert f"×{encode.N_CARD_INDEX_SLOTS}" in html
    assert "kept + tray set" in html
    # The top-row encoders carry the freezing side note (default arch has the
    # distinct hand model on, so the blanket note is accurate).
    assert "trained in-game only" in html
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
    assert "MULTI-CARD ENCODER" in html
    assert "SETUP MODEL" in html


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
        choice_in=encode.choice_input_dim(choice_dim, arch.card_embed_dim),
        num_families=len(family_order),
        hand_feat_in=encode.HAND_ENCODER_INPUT_DIM,
    )
    return report.generate_html_report(
        encode_stripes.state_stripe_layout(spec, arch.card_embed_dim),
        encode_stripes.choice_stripe_layout(spec, arch.card_embed_dim),
        param_report,
        arch,
        setup_layout=setup_model.setup_stripe_layout(),
        setup_arch=setup_model.SetupArchitecture(),
        use_setup_model=use_setup_model,
        state_dim=state_dim,
        choice_dim=choice_dim,
        family_order=family_order,
        run_name="test-setup-stripes",
        model_version=version.MODEL_VERSION,
    )
