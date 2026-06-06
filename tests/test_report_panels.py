"""Tests for the interactive model-summary report and its new vector layouts.

Cover the two raw encoder-input stripe layouts (single-card features and the
multi-card set input) staying in sync with the encode constants, and the
diagram-as-menu HTML structure: the architecture section leading the page, the
detail sections rendered as hidden panels, the SVG click hooks (``data-panel``
on the five input boxes, ``data-params-block`` on the parameter counts), the
parameter table's per-block jump anchors, and the nav losing its section tabs.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import architecture, decisions, encode, setup_model, version  # noqa: E402
from wingspan.encode import stripes as encode_stripes  # noqa: E402
from wingspan.reporting import html as report  # noqa: E402

# The five clickable input boxes' panel ids, in diagram order, and the
# parameter-table block keys the default architecture renders anchors for.
_PANEL_IDS = ("card", "hand", "state", "choice", "setup")
_PARAMS_BLOCK_KEYS = ("embed", "hand", "trunk", "choice", "scorer", "value", "total")


def test_card_feature_layout_matches_encoder_constant():
    layout = encode_stripes.card_feature_stripe_layout()
    assert layout.total_size == encode.CARD_FEATURE_DIM
    assert sum(stripe.size for stripe in layout.stripes) == layout.total_size


def test_card_feature_layout_offsets_are_contiguous():
    layout = encode_stripes.card_feature_stripe_layout()
    expected_offset = 0
    for stripe in layout.stripes:
        assert stripe.offset == expected_offset
        expected_offset += stripe.size
    assert expected_offset == layout.total_size


def test_card_attr_sub_fields_tile_the_stripe():
    layout = encode_stripes.card_feature_stripe_layout()
    attrs = layout.stripes[0]
    assert attrs.name == "bird_attrs"
    assert sum(sub_field.size for sub_field in attrs.sub_fields) == attrs.size
    for sub_field in attrs.sub_fields:
        assert sub_field.relative_offset >= 0
        assert sub_field.relative_offset + sub_field.size <= attrs.size


def test_hand_encoder_layout_matches_encoder_constant():
    layout = encode_stripes.hand_encoder_input_stripe_layout()
    assert layout.total_size == encode.HAND_ENCODER_INPUT_DIM
    assert sum(stripe.size for stripe in layout.stripes) == layout.total_size
    expected_offset = 0
    for stripe in layout.stripes:
        assert stripe.offset == expected_offset
        expected_offset += stripe.size


def test_hand_encoder_layout_reuses_hand_summary_drilldown():
    layout = encode_stripes.hand_encoder_input_stripe_layout()
    summary = layout.stripes[-1]
    assert summary.name == "hand_summary"
    assert sum(sub_field.size for sub_field in summary.sub_fields) == summary.size


def test_html_arch_section_leads_and_panels_start_hidden():
    html = _report_html(use_setup_model=True)
    arch_div = "<div class='section' id='arch'>"
    assert arch_div in html
    for panel_id in (*_PANEL_IDS, "params"):
        panel_div = f"<div class='section panel' id='{panel_id}' hidden>"
        assert panel_div in html
        # The diagram leads: every panel comes after the arch section.
        assert html.index(arch_div) < html.index(panel_div)
    assert "Card Feature Vector" in html
    assert "Card Set Vector" in html


def test_html_svg_click_hooks():
    html = _report_html(use_setup_model=True)
    # Exactly the five input boxes are clickable — output boxes and the heads'
    # intermediate-embedding inputs are not.
    assert html.count("data-panel=") == len(_PANEL_IDS)
    for panel_id in _PANEL_IDS:
        assert f'data-panel="{panel_id}"' in html
    assert "arch-click" in html
    assert "arch-paramclick" in html
    for block_key in ("embed", "trunk", "total"):
        assert f'data-params-block="{block_key}"' in html


def test_html_params_table_jump_anchors():
    html = _report_html(use_setup_model=True)
    for block_key in _PARAMS_BLOCK_KEYS:
        assert f"id='params-block-{block_key}'" in html


def test_html_nav_has_no_section_tabs_and_script_is_inline():
    html = _report_html(use_setup_model=True)
    assert "nav-brand" in html
    assert "href='#state'" not in html
    assert "href='#arch'" not in html
    assert "<script>" in html
    assert "data-params-block" in html


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
        setup_layout=setup_model.setup_stripe_layout(),
        setup_arch=setup_model.SetupArchitecture(),
        use_setup_model=use_setup_model,
        state_dim=state_dim,
        choice_dim=choice_dim,
        family_order=family_order,
        run_name="test-report-panels",
        model_version=version.MODEL_VERSION,
    )
