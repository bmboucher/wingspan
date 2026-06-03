"""Standalone HTML model-summary report generator.

Produces a self-contained ``.html`` file (no external assets) documenting
the full structure of a training run's network: the element-by-element
breakdown of the state and choice vectors, the network architecture, and the
per-layer parameter accounting.  The file is meant to be opened in a browser
and supports drill-down into complex stripes (board slots, round-goal rounds)
via HTML5 ``<details>``/``<summary>`` elements — no JavaScript required.

The public entry point is :func:`generate_html_report`.  The training loop
writes the result alongside the other JSON sidecars at startup; the
``wingspan-inspect --html`` flag exposes it from the CLI.
"""

from __future__ import annotations

import collections
import html as html_lib

from wingspan import architecture
from wingspan.encode import stripes as encode_stripes
from wingspan.training.charts import text_helpers

# ---------------------------------------------------------------------------
# Accent colors — one per report section.

_ACCENT_STATE = "#3b82f6"
_ACCENT_CHOICE = "#22c55e"
_ACCENT_ARCH = "#a855f7"
_ACCENT_PARAMS = "#f97316"

# Minimum sub-field count that triggers grouped (nested) display rather than a
# flat table.  Board stripes have 135 sub-fields across 15 slots, so they are
# always grouped; small stripes (≤ threshold) display flat.
_GROUP_THRESHOLD = 24

# ---------------------------------------------------------------------------
# SVG architecture diagram — color palette.

_SVG_BG = "#f1f5f9"
_SVG_BLOCK_FILL = "#ffffff"
_SVG_BLOCK_STROKE = "#e2e8f0"
_SVG_ARROW = "#94a3b8"
_SVG_TEXT_TITLE = "#1e293b"
_SVG_TEXT_DIM = "#64748b"
_SVG_TEXT_MUTED = "#94a3b8"
_SVG_PILL_BG = "#1e293b"
_SVG_PILL_FG = "#e2e8f0"
_SVG_LINEAR_COLOR = "#3b82f6"
_SVG_ACT_COLOR = "#22c55e"

_ACCENT_CARD = "#a855f7"
_ACCENT_TRUNK_SVG = "#3b82f6"
_ACCENT_CHOICE_SVG = "#0ea5e9"
_ACCENT_VALUE = "#10b981"
_ACCENT_DECISION = "#f97316"
_ACCENT_DECISION_BADGE_BG = "#fde8d8"

# SVG geometry constants.

_SVG_W = 860
_SVG_GUTTER = 20
_SVG_COL_W = 390
_SVG_COL_R = 450  # x of right column = GUTTER + COL_W + 40px gap
_SVG_ACCENT_W = 4  # left-border accent bar width
_SVG_RX_BLK = 8  # block corner radius
_SVG_RX_ROW = 4  # mini-row corner radius
_SVG_ROW_H = 28  # mini-row height
_SVG_ROW_STRIDE = 33  # mini-row height + 5px gap

# Block internal layout (all relative to block top-left):
#   PAD_T | title (16px) + gap (6px) + subtitle (16px) = HDR_H | HDR_GAP
#   | rows (ROW_STRIDE each) | FTR_GAP | footer text | PAD_B
_SVG_BLK_PAD_T = 14
_SVG_BLK_HDR_H = 38
_SVG_BLK_HDR_GAP = 6
_SVG_BLK_FTR_GAP = 6
_SVG_BLK_FTR_H = 28
_SVG_BLK_PAD_B = 14

# ---------------------------------------------------------------------------
# Inline CSS — fully self-contained; no external resources.

_CSS = """\
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: #f8fafc; color: #1e293b; font-size: 14px; line-height: 1.5;
}
nav {
  position: sticky; top: 0; background: #1e293b; color: #e2e8f0;
  padding: 10px 24px; display: flex; gap: 16px; align-items: center;
  z-index: 100; box-shadow: 0 2px 8px rgba(0,0,0,.15); flex-wrap: wrap;
}
.nav-brand { font-weight: 700; font-size: 15px; color: #f8fafc; margin-right: 4px; }
.nav-run   { font-size: 11px; color: #94a3b8; font-family: monospace; }
nav a {
  color: #cbd5e1; font-size: 13px; padding: 4px 10px; border-radius: 4px;
  text-decoration: none; transition: background .15s;
}
nav a:hover { background: #334155; color: #f8fafc; }
.container { max-width: 1400px; margin: 0 auto; padding: 28px 24px 60px; }
.section { margin-bottom: 44px; }
.section-header {
  font-size: 18px; font-weight: 700; margin-bottom: 6px;
  padding-left: 12px; border-left: 4px solid currentColor;
}
.section-sub { font-size: 12px; color: #64748b; margin-bottom: 14px; margin-left: 16px; }
.tbl-wrap { overflow-x: auto; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,.06); }
table { width: 100%; border-collapse: collapse; background: #fff; font-size: 13px; }
th {
  background: #f1f5f9; font-weight: 600; text-align: left;
  padding: 8px 12px; border-bottom: 2px solid #e2e8f0; white-space: nowrap;
}
td { padding: 7px 12px; border-bottom: 1px solid #f1f5f9; vertical-align: top; }
tr:last-child > td { border-bottom: none; }
tr:hover > td { background: #f8fafc; }
.mono  { font-family: 'Courier New', Courier, monospace; font-size: 12px; }
.dim   { color: #64748b; font-size: 12px; }
.right { text-align: right; }
.badge {
  display: inline-block; padding: 1px 7px; border-radius: 9999px;
  font-size: 11px; font-weight: 600; background: #e2e8f0; color: #334155;
  white-space: nowrap;
}
.notes-text { color: #64748b; font-size: 11px; margin-top: 3px; }
details { cursor: default; }
details > summary {
  list-style: none; cursor: pointer; display: inline-flex;
  align-items: center; gap: 5px; user-select: none;
}
details > summary::-webkit-details-marker { display: none; }
details > summary::before {
  content: '▶'; font-size: 9px; transition: transform .15s; color: #94a3b8;
  display: inline-block;
}
details[open] > summary::before { transform: rotate(90deg); }
.sub-wrap {
  margin-top: 8px; padding: 0 0 4px 10px;
  border-left: 2px solid #e2e8f0; overflow-x: auto;
}
.sub-tbl { width: 100%; border-collapse: collapse; font-size: 12px; }
.sub-tbl th {
  background: #f8fafc; font-weight: 600; text-align: left;
  padding: 5px 8px; border-bottom: 1px solid #e2e8f0; white-space: nowrap;
}
.sub-tbl td { padding: 4px 8px; border-bottom: 1px solid #f8fafc; vertical-align: top; }
.sub-tbl tr:last-child > td { border-bottom: none; }
.sub-tbl tr:hover > td { background: #f0f9ff; }
.grp-details { margin-bottom: 6px; }
.grp-summary {
  color: #475569; font-size: 12px; font-weight: 600; padding: 3px 2px;
}
.grp-tbl-wrap { margin-top: 4px; padding-left: 8px; }
.subtotal-row > td {
  background: #f1f5f9; font-weight: 700; border-top: 1px solid #e2e8f0;
}
.total-row > td { background: #1e293b; color: #f8fafc; font-weight: 700; }
.arch-svg-wrap { background: #f1f5f9; border-radius: 8px; padding: 24px 20px; overflow-x: auto; }
"""


# ---------------------------------------------------------------------------
# Public API


def generate_html_report(
    state_layout: encode_stripes.VectorLayout,
    choice_layout: encode_stripes.VectorLayout,
    param_report: architecture.ParamReport,
    arch: architecture.ModelArchitecture,
    *,
    state_dim: int,
    choice_dim: int,
    family_order: tuple[str, ...],
    run_name: str,
) -> str:
    """Return a self-contained HTML document string for the model summary.

    All CSS is embedded inline; no external resources are referenced, so the
    file opens correctly when copied anywhere or viewed offline.
    """
    body = "\n".join(
        [
            _vector_section(state_layout, "state", "State Vector", _ACCENT_STATE),
            _vector_section(choice_layout, "choice", "Choice Vector", _ACCENT_CHOICE),
            _arch_section(arch, param_report, state_dim, choice_dim, family_order),
            _params_section(param_report),
        ]
    )
    return _wrap(
        title=f"Model Summary — {html_lib.escape(run_name)}",
        run_name=run_name,
        body=body,
    )


###### PRIVATE #######

#### HTML shell ####


def _wrap(*, title: str, run_name: str, body: str) -> str:
    nav = _nav(run_name)
    return (
        f"<!DOCTYPE html>\n<html lang='en'>\n<head>\n"
        f"<meta charset='utf-8'>\n"
        f"<meta name='viewport' content='width=device-width, initial-scale=1'>\n"
        f"<title>{title}</title>\n"
        f"<style>\n{_CSS}\n</style>\n"
        f"</head>\n<body>\n"
        f"{nav}\n"
        f"<div class='container'>\n{body}\n</div>\n"
        f"</body>\n</html>\n"
    )


def _nav(run_name: str) -> str:
    links = [
        ("#state", "State Vector"),
        ("#choice", "Choice Vector"),
        ("#arch", "Architecture"),
        ("#params", "Parameters"),
    ]
    link_html = " ".join(
        f"<a href='{href}'>{html_lib.escape(label)}</a>" for href, label in links
    )
    return (
        f"<nav>"
        f"<span class='nav-brand'>Wingspan</span>"
        f"<span class='nav-run'>{html_lib.escape(run_name)}</span>"
        f"&nbsp;&nbsp;{link_html}"
        f"</nav>"
    )


#### Vector sections ####


def _vector_section(
    layout: encode_stripes.VectorLayout,
    section_id: str,
    title: str,
    accent: str,
) -> str:
    """Build a complete section for a state or choice vector layout."""
    expanded_count = sum(1 for s in layout.stripes if s.sub_fields)
    sub = (
        f"{layout.total_size} elements (post-embedding network input) · "
        f"{len(layout.stripes)} stripes · {expanded_count} with drill-down"
    )
    return (
        f"<div class='section' id='{section_id}'>"
        f"<div class='section-header' style='color:{accent}'>{html_lib.escape(title)}</div>"
        f"<div class='section-sub'>{sub}</div>"
        f"<div class='tbl-wrap'>"
        f"<table>"
        f"<thead><tr>"
        f"<th>Name</th>"
        f"<th class='right'>Offset</th>"
        f"<th class='right'>Size</th>"
        f"<th>Encoding</th>"
        f"<th>Range</th>"
        f"<th>Description</th>"
        f"</tr></thead>"
        f"<tbody>"
        + "".join(_stripe_row(stripe, accent) for stripe in layout.stripes)
        + "</tbody></table></div></div>"
    )


def _stripe_row(stripe: encode_stripes.StripeDescriptor, accent: str) -> str:
    """One <tr> for a single stripe, with an optional inline sub-fields block."""
    # Name cell: plain or wrapped in <details> if sub-fields exist.
    if stripe.sub_fields:
        sub_html = _sub_fields_block(stripe, accent)
        name_cell = (
            f"<details>"
            f"<summary class='mono' style='font-weight:600'>"
            f"{html_lib.escape(stripe.name)}"
            f"</summary>"
            f"{sub_html}"
            f"</details>"
        )
    else:
        name_cell = f"<span class='mono' style='font-weight:600'>{html_lib.escape(stripe.name)}</span>"

    desc_html = html_lib.escape(stripe.description)
    if stripe.notes:
        desc_html += f"<div class='notes-text'>{html_lib.escape(stripe.notes)}</div>"

    return (
        f"<tr>"
        f"<td>{name_cell}</td>"
        f"<td class='right dim mono'>{stripe.offset}</td>"
        f"<td class='right mono'>{stripe.size}</td>"
        f"<td><span class='badge'>{html_lib.escape(stripe.encoding)}</span></td>"
        f"<td class='dim mono'>{html_lib.escape(stripe.value_range)}</td>"
        f"<td>{desc_html}</td>"
        f"</tr>"
    )


def _sub_fields_block(stripe: encode_stripes.StripeDescriptor, accent: str) -> str:
    """The expanded sub-field content shown when a stripe's <details> is open.

    Uses nested <details> per group when the sub-field count exceeds
    ``_GROUP_THRESHOLD`` (flat table otherwise).
    """
    sub_fields = stripe.sub_fields
    use_groups = len(sub_fields) > _GROUP_THRESHOLD and any(
        sf.group for sf in sub_fields
    )

    if use_groups:
        inner = _grouped_sub_fields(sub_fields, stripe.offset)
    else:
        inner = _flat_sub_table(sub_fields, stripe.offset)

    return f"<div class='sub-wrap'>{inner}</div>"


def _flat_sub_table(
    sub_fields: tuple[encode_stripes.SubFieldDescriptor, ...],
    stripe_offset: int,
) -> str:
    """A plain table listing all sub-fields with no grouping."""
    header = (
        "<table class='sub-tbl'>"
        "<thead><tr>"
        "<th>Name</th>"
        "<th class='right'>Abs&nbsp;Offset</th>"
        "<th class='right'>Size</th>"
        "<th>Encoding</th>"
        "<th>Range</th>"
        "<th>Description</th>"
        "</tr></thead><tbody>"
    )
    rows = "".join(_sub_field_row(sf, stripe_offset) for sf in sub_fields)
    return f"{header}{rows}</tbody></table>"


def _grouped_sub_fields(
    sub_fields: tuple[encode_stripes.SubFieldDescriptor, ...],
    stripe_offset: int,
) -> str:
    """Render sub-fields using nested <details> per group.

    When every group has an identical shape (same per-member layout, differing only
    in group label and absolute offset — e.g. the 15 identical board slots), the
    shape is shown once as a representative "per-slot layout" with a ``×N``
    annotation instead of being repeated N times.
    """
    # Collect sub-fields into ordered groups.
    groups: dict[str, list[encode_stripes.SubFieldDescriptor]] = (
        collections.OrderedDict()
    )
    ungrouped: list[encode_stripes.SubFieldDescriptor] = []
    for sf in sub_fields:
        if sf.group:
            groups.setdefault(sf.group, []).append(sf)
        else:
            ungrouped.append(sf)

    group_items = list(groups.items())
    parts: list[str] = []

    # Collapse N structurally-identical groups to one representative; otherwise
    # render each group as its own collapsible <details>.
    if len(group_items) > 1 and _groups_identical(group_items):
        parts.append(_collapsed_repeated_group(group_items, stripe_offset))
    else:
        for group_name, members in group_items:
            parts.append(_render_one_group(group_name, members, stripe_offset))

    # Any ungrouped sub-fields go at the end as a flat table.
    if ungrouped:
        parts.append(_flat_sub_table(tuple(ungrouped), stripe_offset))

    return "".join(parts)


def _render_one_group(
    group_name: str,
    members: list[encode_stripes.SubFieldDescriptor],
    stripe_offset: int,
) -> str:
    """One named group as a collapsible <details>, with absolute offsets."""
    count = sum(sf.size for sf in members)
    first_abs = stripe_offset + members[0].relative_offset
    summary = (
        f"<summary class='grp-summary'>"
        f"{html_lib.escape(group_name)}"
        f"&nbsp;<span class='dim'>({count} element{'s' if count != 1 else ''},"
        f" offset {first_abs})</span>"
        f"</summary>"
    )
    table_rows = "".join(_sub_field_row(sf, stripe_offset) for sf in members)
    table_html = (
        f"<div class='grp-tbl-wrap'>"
        f"<table class='sub-tbl'><thead><tr>"
        f"<th>Name</th><th class='right'>Abs&nbsp;Offset</th>"
        f"<th class='right'>Size</th><th>Encoding</th>"
        f"<th>Range</th><th>Description</th>"
        f"</tr></thead><tbody>{table_rows}</tbody></table>"
        f"</div>"
    )
    return f"<details class='grp-details'>{summary}{table_html}</details>"


def _group_signature(
    members: list[encode_stripes.SubFieldDescriptor],
) -> tuple[tuple[str, int, str, str, str, str | None, int], ...]:
    """A group's shape, ignoring its label and absolute position: per member, the
    name suffix (after the group prefix), size, encoding, range, description, notes,
    and offset relative to the group's start."""
    base = members[0].relative_offset
    return tuple(
        (
            sf.name.split(".", 1)[-1],
            sf.size,
            sf.encoding,
            sf.value_range,
            sf.description,
            sf.notes,
            sf.relative_offset - base,
        )
        for sf in members
    )


def _groups_identical(
    group_items: list[tuple[str, list[encode_stripes.SubFieldDescriptor]]],
) -> bool:
    """Whether every group has the same shape (so one representative suffices)."""
    signatures = [_group_signature(members) for _, members in group_items]
    return all(sig == signatures[0] for sig in signatures[1:])


def _collapsed_repeated_group(
    group_items: list[tuple[str, list[encode_stripes.SubFieldDescriptor]]],
    stripe_offset: int,
) -> str:
    """Render N identical groups once: the shared per-member layout (with
    within-group relative offsets) plus a ``×N`` annotation naming the repeats."""
    group_names = [name for name, _ in group_items]
    representative = group_items[0][1]
    base = representative[0].relative_offset
    stride = sum(sf.size for sf in representative)
    first_abs = stripe_offset + base
    span = f"{html_lib.escape(group_names[0])} … {html_lib.escape(group_names[-1])}"
    summary = (
        f"<summary class='grp-summary'>per-slot layout "
        f"<span class='dim'>(×{len(group_items)}: {span}; {stride} element"
        f"{'s' if stride != 1 else ''} each, stride {stride}, first at "
        f"offset {first_abs})</span></summary>"
    )
    table_rows = "".join(_repeated_member_row(sf, base) for sf in representative)
    table_html = (
        f"<div class='grp-tbl-wrap'>"
        f"<table class='sub-tbl'><thead><tr>"
        f"<th>Name</th><th class='right'>Slot&nbsp;Offset</th>"
        f"<th class='right'>Size</th><th>Encoding</th>"
        f"<th>Range</th><th>Description</th>"
        f"</tr></thead><tbody>{table_rows}</tbody></table>"
        f"</div>"
    )
    return f"<details class='grp-details' open>{summary}{table_html}</details>"


def _repeated_member_row(sf: encode_stripes.SubFieldDescriptor, group_base: int) -> str:
    """One <tr> for a representative group's member, showing its within-group
    (relative) offset rather than an absolute one."""
    rel = sf.relative_offset - group_base
    name = sf.name.split(".", 1)[-1]
    desc_html = html_lib.escape(sf.description)
    if sf.notes:
        desc_html += f"<div class='notes-text'>{html_lib.escape(sf.notes)}</div>"
    return (
        f"<tr>"
        f"<td class='mono'>{html_lib.escape(name)}</td>"
        f"<td class='right dim mono'>+{rel}</td>"
        f"<td class='right dim mono'>{sf.size}</td>"
        f"<td class='dim'>{html_lib.escape(sf.encoding)}</td>"
        f"<td class='dim mono'>{html_lib.escape(sf.value_range)}</td>"
        f"<td>{desc_html}</td>"
        f"</tr>"
    )


def _sub_field_row(sf: encode_stripes.SubFieldDescriptor, stripe_offset: int) -> str:
    """One <tr> for a single sub-field."""
    abs_off = stripe_offset + sf.relative_offset
    desc_html = html_lib.escape(sf.description)
    if sf.notes:
        desc_html += f"<div class='notes-text'>{html_lib.escape(sf.notes)}</div>"
    return (
        f"<tr>"
        f"<td class='mono'>{html_lib.escape(sf.name)}</td>"
        f"<td class='right dim mono'>{abs_off}</td>"
        f"<td class='right dim mono'>{sf.size}</td>"
        f"<td class='dim'>{html_lib.escape(sf.encoding)}</td>"
        f"<td class='dim mono'>{html_lib.escape(sf.value_range)}</td>"
        f"<td>{desc_html}</td>"
        f"</tr>"
    )


#### Architecture section ####


def _arch_section(
    arch: architecture.ModelArchitecture,
    param_report: architecture.ParamReport,
    state_dim: int,
    choice_dim: int,
    family_order: tuple[str, ...],
) -> str:
    svg = _build_arch_svg(arch, param_report, state_dim, choice_dim, family_order)
    return (
        f"<div class='section' id='arch'>"
        f"<div class='section-header' style='color:{_ACCENT_ARCH}'>Architecture</div>"
        f"<div class='section-sub'>"
        f"Network flow — embed → trunk → choice encoder → scorer heads → value head"
        f"</div>"
        f"<div class='arch-svg-wrap'>{svg}</div>"
        f"</div>"
    )


#### Parameters section ####


def _params_section(report: architecture.ParamReport) -> str:
    total = report.total
    sub = (
        f"{text_helpers.human_count(total)} total trainable parameters &nbsp;·&nbsp; "
        f"{len(report.blocks)} blocks"
    )
    return (
        f"<div class='section' id='params'>"
        f"<div class='section-header' style='color:{_ACCENT_PARAMS}'>Parameters</div>"
        f"<div class='section-sub'>{sub}</div>"
        f"<div class='tbl-wrap'>"
        f"<table>"
        f"<thead><tr>"
        f"<th>Block</th>"
        f"<th>Layer</th>"
        f"<th class='right'>Params</th>"
        f"<th class='right'>%&nbsp;Total</th>"
        f"<th class='right'>Cumulative</th>"
        f"</tr></thead>"
        f"<tbody>" + _params_rows(report) + "</tbody></table></div></div>"
    )


def _params_rows(report: architecture.ParamReport) -> str:
    total = report.total
    running = 0
    rows: list[str] = []

    for block in report.blocks:
        block_label = block.label
        if block.multiplier > 1:
            block_label = f"{block.label} ×{block.multiplier}"

        # Per-layer rows for this block.
        for idx, layer in enumerate(block.layers):
            layer_label = f"Linear  {layer.in_features} → {layer.out_features}"
            layer_params = layer.linear * block.multiplier
            running += layer_params
            pct = f"{100.0 * layer_params / max(total, 1):.1f}%"
            rows.append(
                f"<tr>"
                f"<td class='mono'>{html_lib.escape(block_label) if idx == 0 else ''}</td>"
                f"<td class='mono dim'>{html_lib.escape(layer_label)}</td>"
                f"<td class='right mono'>{text_helpers.human_count(layer_params)}</td>"
                f"<td class='right dim'>{pct}</td>"
                f"<td class='right dim mono'>{text_helpers.human_count(running)}</td>"
                f"</tr>"
            )
            if layer.norm > 0:
                norm_params = layer.norm * block.multiplier
                running += norm_params
                norm_pct = f"{100.0 * norm_params / max(total, 1):.1f}%"
                rows.append(
                    f"<tr>"
                    f"<td></td>"
                    f"<td class='mono dim'>LayerNorm  {layer.out_features}</td>"
                    f"<td class='right mono'>{text_helpers.human_count(norm_params)}</td>"
                    f"<td class='right dim'>{norm_pct}</td>"
                    f"<td class='right dim mono'>{text_helpers.human_count(running)}</td>"
                    f"</tr>"
                )

        # Block subtotal.
        block_pct = f"{100.0 * block.total / max(total, 1):.1f}%"
        rows.append(
            f"<tr class='subtotal-row'>"
            f"<td></td>"
            f"<td>Subtotal {html_lib.escape(block_label)}</td>"
            f"<td class='right'>{text_helpers.human_count(block.total)}</td>"
            f"<td class='right'>{block_pct}</td>"
            f"<td></td>"
            f"</tr>"
        )

    # Grand total.
    rows.append(
        f"<tr class='total-row'>"
        f"<td>TOTAL</td>"
        f"<td></td>"
        f"<td class='right'>{text_helpers.human_count(total)}</td>"
        f"<td class='right'>100%</td>"
        f"<td></td>"
        f"</tr>"
    )
    return "".join(rows)


#### SVG architecture diagram ####


def _blk_h(num_rows: int) -> int:
    """Total pixel height of a block containing ``num_rows`` mini-rows."""
    return (
        _SVG_BLK_PAD_T
        + _SVG_BLK_HDR_H
        + _SVG_BLK_HDR_GAP
        + num_rows * _SVG_ROW_STRIDE
        + _SVG_BLK_FTR_GAP
        + _SVG_BLK_FTR_H
        + _SVG_BLK_PAD_B
    )


def _has_act_after(is_trunk: bool, is_final: bool) -> bool:
    """Mirror of arch_diagram._has_activation: trunk gets activation after every
    layer; all other blocks only on non-final layers."""
    if is_trunk:
        return True
    return not is_final


def _layer_rows_svg(
    layers: tuple[architecture.LayerParam, ...],
    activation: str,
    *,
    is_trunk: bool,
) -> list[tuple[str, str, str]]:
    """Return (style, label, badge) tuples for the mini-box rows in a block.

    ``style`` is ``"linear"`` or ``"act"``.  ``badge`` is a human-readable
    param count for linear rows, empty string for activation rows.
    """
    rows: list[tuple[str, str, str]] = []
    num_layers = len(layers)
    for idx, layer in enumerate(layers):
        badge = text_helpers.human_count(layer.linear)
        rows.append(("linear", f"Linear →{layer.out_features}", badge))
        if _has_act_after(is_trunk, is_final=(idx == num_layers - 1)):
            rows.append(("act", activation, ""))
    return rows


def _svg_block(
    x: int,
    y: int,
    width: int,
    height: int,
    accent: str,
    title: str,
    subtitle: str,
    rows: list[tuple[str, str, str]],
    sigma: int,
    out_label: str,
    out_color: str,
    tooltip: str,
    *,
    dashed: bool = False,
    stack: int = 0,
) -> str:
    """Render one network block as an SVG ``<g>`` element.

    ``stack`` > 1 adds shadow rects behind the block (the ×N stacked-card
    effect for the decision-head template).  ``dashed`` applies a dashed
    stroke to the main block border.
    """
    parts: list[str] = []

    # Shadow rects for stacked-card effect (drawn behind, largest first).
    if stack > 1:
        for offset in (6, 3):
            parts.append(
                f'<rect x="{x + offset}" y="{y + offset}" '
                f'width="{width}" height="{height}" rx="{_SVG_RX_BLK}" '
                f'fill="{_SVG_BLOCK_FILL}" stroke="{_SVG_BLOCK_STROKE}" stroke-width="1"/>'
            )

    dash_attr = ' stroke-dasharray="5,3"' if dashed else ""
    clip_id = f"c{x}-{y}"

    # Clip path so the left accent bar respects the block's rounded corners.
    parts.append(
        f'<defs><clipPath id="{clip_id}">'
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="{_SVG_RX_BLK}"/>'
        f"</clipPath></defs>"
    )

    # Main block group with accessible tooltip.
    parts.append(f"<g>")
    parts.append(f"<title>{html_lib.escape(tooltip)}</title>")

    # Main block rect.
    parts.append(
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="{_SVG_RX_BLK}" '
        f'fill="{_SVG_BLOCK_FILL}" stroke="{_SVG_BLOCK_STROKE}" stroke-width="1"{dash_attr}/>'
    )

    # Left accent bar (clipped to block shape).
    parts.append(
        f'<rect x="{x}" y="{y}" width="{_SVG_ACCENT_W}" height="{height}" '
        f'fill="{accent}" clip-path="url(#{clip_id})"/>'
    )

    # Title and subtitle text.
    tx = x + _SVG_ACCENT_W + 10
    ty_title = y + _SVG_BLK_PAD_T + 14
    ty_sub = ty_title + 18
    mono = "'Courier New',monospace"
    sans = "-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif"

    parts.append(
        f'<text x="{tx}" y="{ty_title}" font-family="{sans}" '
        f'font-size="13" font-weight="700" fill="{_SVG_TEXT_TITLE}">'
        f"{html_lib.escape(title)}</text>"
    )

    # ×N badge for repeated blocks (decision head).
    if stack > 1:
        badge_label = f"×{stack}"
        badge_w = len(badge_label) * 7 + 12
        badge_rx = x + width - badge_w - 10
        badge_ry = y + _SVG_BLK_PAD_T + 2
        parts.append(
            f'<rect x="{badge_rx}" y="{badge_ry}" width="{badge_w}" height="18" '
            f'rx="9" fill="{_ACCENT_DECISION_BADGE_BG}"/>'
            f'<text x="{badge_rx + badge_w // 2}" y="{badge_ry + 13}" '
            f'font-family="{sans}" font-size="10" font-weight="700" '
            f'fill="{_ACCENT_DECISION}" text-anchor="middle">'
            f"{html_lib.escape(badge_label)}</text>"
        )

    parts.append(
        f'<text x="{tx}" y="{ty_sub}" font-family="{mono}" '
        f'font-size="11" fill="{_SVG_TEXT_DIM}">{html_lib.escape(subtitle)}</text>'
    )

    # Mini-rows for each layer operation.
    row_x = x + _SVG_ACCENT_W + 8
    row_w = width - _SVG_ACCENT_W - 16
    row_y0 = y + _SVG_BLK_PAD_T + _SVG_BLK_HDR_H + _SVG_BLK_HDR_GAP

    for row_idx, (row_type, label, badge) in enumerate(rows):
        ry = row_y0 + row_idx * _SVG_ROW_STRIDE
        color = _SVG_LINEAR_COLOR if row_type == "linear" else _SVG_ACT_COLOR

        parts.append(
            f'<rect x="{row_x}" y="{ry}" width="{row_w}" height="{_SVG_ROW_H}" '
            f'rx="{_SVG_RX_ROW}" fill="{color}" fill-opacity="0.06" '
            f'stroke="{color}" stroke-width="1.5"/>'
        )
        parts.append(
            f'<text x="{row_x + 10}" y="{ry + 19}" font-family="{mono}" '
            f'font-size="11" font-weight="600" fill="{color}">'
            f"{html_lib.escape(label)}</text>"
        )

        # Param pill for linear layers.
        if badge:
            pill_w = len(badge) * 7 + 12
            pill_x = row_x + row_w - pill_w - 4
            pill_y = ry + 5
            parts.append(
                f'<rect x="{pill_x}" y="{pill_y}" width="{pill_w}" height="18" '
                f'rx="9" fill="{_SVG_PILL_BG}"/>'
                f'<text x="{pill_x + pill_w // 2}" y="{pill_y + 13}" '
                f'font-family="{mono}" font-size="10" fill="{_SVG_PILL_FG}" '
                f'text-anchor="middle">{html_lib.escape(badge)}</text>'
            )

    # Footer: Σ (left) and output annotation (right).
    footer_y = y + height - _SVG_BLK_PAD_B - 3
    parts.append(
        f'<text x="{tx}" y="{footer_y}" font-family="{mono}" '
        f'font-size="11" fill="{_SVG_TEXT_DIM}">'
        f"Σ {text_helpers.human_count(sigma)}</text>"
    )
    parts.append(
        f'<text x="{x + width - 10}" y="{footer_y}" font-family="{mono}" '
        f'font-size="11" font-weight="700" fill="{out_color}" text-anchor="end">'
        f"{html_lib.escape(out_label)}</text>"
    )

    parts.append("</g>")
    return "\n".join(parts)


def _svg_arrows(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    label: str = "",
) -> str:
    """A vertical arrow from ``(x1, y1)`` to ``(x2, y2)`` with an optional
    dimension label placed to the right of the midpoint."""
    mid_y = (y1 + y2) // 2
    arrow = (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{_SVG_ARROW}" stroke-width="1.5" marker-end="url(#arr)"/>'
    )
    if not label:
        return arrow
    label_part = (
        f'<text x="{x1 + 4}" y="{mid_y - 3}" '
        f'font-family="\'Courier New\',monospace" font-size="10" '
        f'fill="{_SVG_TEXT_MUTED}">{html_lib.escape(label)}</text>'
    )
    return f"{arrow}\n{label_part}"


def _build_arch_svg(
    arch: architecture.ModelArchitecture,
    param_report: architecture.ParamReport,
    state_dim: int,
    choice_dim: int,
    family_order: tuple[str, ...],
) -> str:
    """Return a self-contained ``<svg>`` string for the architecture diagram."""
    activation = arch.activation.value
    num_families = len(family_order)

    # Build layer-row lists for each block.  The trunk uses BODY_TRUNK rules
    # (activation after every layer); all other blocks use BODY_CHOICE / READOUT
    # rules (no activation after the final layer).
    card_rows = _layer_rows_svg(param_report.embed.layers, activation, is_trunk=False)
    trunk_rows = _layer_rows_svg(param_report.trunk.layers, activation, is_trunk=True)
    choice_rows = _layer_rows_svg(
        param_report.choice.layers, activation, is_trunk=False
    )
    value_rows = _layer_rows_svg(param_report.value.layers, activation, is_trunk=False)
    # Decision head: param_report.scorer.layers contains per-head layers (multiplier
    # accounts for the family count separately).
    decision_rows = _layer_rows_svg(
        param_report.scorer.layers, activation, is_trunk=False
    )

    # Block pixel heights.
    card_h = _blk_h(len(card_rows))
    trunk_h = _blk_h(len(trunk_rows))
    choice_h = _blk_h(len(choice_rows))
    value_h = _blk_h(len(value_rows))
    decision_h = _blk_h(len(decision_rows))
    parallel_h = max(trunk_h, choice_h)
    output_h = max(value_h, decision_h)

    # Vertical layout: hint → card → fan-out → parallel zone → merge → output → total.
    y_card = 22
    y_conn1 = y_card + card_h
    y_parallel = y_conn1 + 46
    y_conn2 = y_parallel + parallel_h
    y_output = y_conn2 + 46
    y_total = y_output + output_h + 22
    svg_h = y_total + 22

    # Horizontal centers for connector routing.
    left_cx = _SVG_GUTTER + _SVG_COL_W // 2  # center of left column
    right_cx = _SVG_COL_R + _SVG_COL_W // 2  # center of right column
    center_x = _SVG_W // 2  # horizontal center of canvas

    parts: list[str] = []

    # SVG root element.
    trunk_m = arch.trunk_embed_width
    choice_n = arch.choice_embed_width
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_SVG_W} {svg_h}" '
        f'width="100%" style="display:block;max-width:{_SVG_W}px;" '
        f'role="img" aria-label="PolicyValueNet architecture: shared Card Encoder '
        f"feeding State Trunk (M={trunk_m}) and Choice Encoder (N={choice_n}), "
        f"merging into Value Head and {num_families} Decision Heads, "
        f'{text_helpers.human_count(param_report.total)} params total">'
    )
    parts.append(
        f"<title>PolicyValueNet · "
        f"{text_helpers.human_count(param_report.total)} params total · "
        f"M={trunk_m} N={choice_n} ×{num_families} heads</title>"
    )

    # Canvas background.
    parts.append(f'<rect width="{_SVG_W}" height="{svg_h}" fill="{_SVG_BG}"/>')

    # Arrowhead marker.
    parts.append(
        "<defs>"
        '<marker id="arr" viewBox="0 0 8 8" refX="7" refY="4" '
        'markerWidth="5" markerHeight="5" orient="auto">'
        '<path d="M0,0.5 L7,4 L0,7.5 Z" fill="#94a3b8"/>'
        "</marker>"
        "</defs>"
    )

    # Setup hint line.
    mono = "'Courier New',monospace"
    parts.append(
        f'<text x="{_SVG_GUTTER}" y="14" font-family="{mono}" '
        f'font-size="11" fill="{_SVG_TEXT_MUTED}">'
        f'setup model · <tspan font-style="italic">'
        f"off (handled by the in-game policy)</tspan></text>"
    )

    # Card Encoder block (full canvas width).
    card_in = (
        param_report.embed.layers[0].in_features if param_report.embed.layers else 0
    )
    card_out = arch.card_embed_dim
    parts.append(
        _svg_block(
            x=_SVG_GUTTER,
            y=y_card,
            width=_SVG_W - 2 * _SVG_GUTTER,
            height=card_h,
            accent=_ACCENT_CARD,
            title="CARD ENCODER · per-card MLP",
            subtitle=f"in {card_in} (attrs ⊕ id)",
            rows=card_rows,
            sigma=param_report.embed.total,
            out_label=f"→ {card_out}",
            out_color=_ACCENT_CARD,
            tooltip=(
                f"Card Encoder · {text_helpers.human_count(param_report.embed.total)} params "
                f"· {card_in} → {card_out}"
            ),
        )
    )

    # Fan-out connectors: card encoder → trunk and choice encoder.
    fan_bar_y = y_conn1 + 24
    parts.append(
        f'<g aria-hidden="true">'
        f'<line x1="{center_x}" y1="{y_conn1}" x2="{center_x}" y2="{fan_bar_y}" '
        f'stroke="{_SVG_ARROW}" stroke-width="1.5"/>'
        f'<line x1="{left_cx}" y1="{fan_bar_y}" x2="{right_cx}" y2="{fan_bar_y}" '
        f'stroke="{_SVG_ARROW}" stroke-width="1.5"/>'
        + _svg_arrows(left_cx, fan_bar_y, left_cx, y_parallel - 1)
        + "\n"
        + _svg_arrows(right_cx, fan_bar_y, right_cx, y_parallel - 1)
        + "\n</g>"
    )

    # State Trunk block (left column).
    trunk_in = (
        param_report.trunk.layers[0].in_features
        if param_report.trunk.layers
        else state_dim
    )
    parts.append(
        _svg_block(
            x=_SVG_GUTTER,
            y=y_parallel,
            width=_SVG_COL_W,
            height=parallel_h,
            accent=_ACCENT_TRUNK_SVG,
            title="STATE TRUNK",
            subtitle=f"in {trunk_in}",
            rows=trunk_rows,
            sigma=param_report.trunk.total,
            out_label=f"M = {trunk_m}",
            out_color=_ACCENT_TRUNK_SVG,
            tooltip=(
                f"State Trunk · {text_helpers.human_count(param_report.trunk.total)} params "
                f"· {trunk_in} → {trunk_m}"
            ),
        )
    )

    # Choice Encoder block (right column).
    choice_in = (
        param_report.choice.layers[0].in_features
        if param_report.choice.layers
        else choice_dim
    )
    parts.append(
        _svg_block(
            x=_SVG_COL_R,
            y=y_parallel,
            width=_SVG_COL_W,
            height=parallel_h,
            accent=_ACCENT_CHOICE_SVG,
            title="CHOICE ENC",
            subtitle=f"in {choice_in}",
            rows=choice_rows,
            sigma=param_report.choice.total,
            out_label=f"N = {choice_n}",
            out_color=_ACCENT_CHOICE_SVG,
            tooltip=(
                f"Choice Encoder · {text_helpers.human_count(param_report.choice.total)} params "
                f"· {choice_in} → {choice_n}"
            ),
        )
    )

    # Merge connectors: trunk and choice encoder → output heads.
    merge_bar_y = y_conn2 + 24
    mn = trunk_m + choice_n
    parts.append(
        f'<g aria-hidden="true">'
        f'<line x1="{left_cx}" y1="{y_conn2}" x2="{left_cx}" y2="{merge_bar_y}" '
        f'stroke="{_SVG_ARROW}" stroke-width="1.5"/>'
        f'<line x1="{right_cx}" y1="{y_conn2}" x2="{right_cx}" y2="{merge_bar_y}" '
        f'stroke="{_SVG_ARROW}" stroke-width="1.5"/>'
        + _svg_arrows(left_cx, merge_bar_y, left_cx, y_output - 1, label=f"M={trunk_m}")
        + "\n"
        + _svg_arrows(right_cx, merge_bar_y, right_cx, y_output - 1, label=f"M+N={mn}")
        + "\n</g>"
    )

    # Value Head block (left column).
    value_in = trunk_m
    parts.append(
        _svg_block(
            x=_SVG_GUTTER,
            y=y_output,
            width=_SVG_COL_W,
            height=output_h,
            accent=_ACCENT_VALUE,
            title="VALUE HEAD",
            subtitle=f"in M = {value_in}",
            rows=value_rows,
            sigma=param_report.value.total,
            out_label="→ scalar",
            out_color=_ACCENT_VALUE,
            tooltip=(
                f"Value Head · {text_helpers.human_count(param_report.value.total)} params "
                f"· {value_in} → 1"
            ),
        )
    )

    # Decision Head block (right column, with stacked-card effect and ×N badge).
    scorer = param_report.scorer
    per_head_total = scorer.total // scorer.multiplier
    parts.append(
        _svg_block(
            x=_SVG_COL_R,
            y=y_output,
            width=_SVG_COL_W,
            height=output_h,
            accent=_ACCENT_DECISION,
            title="DECISION HEAD",
            subtitle=f"in M+N = {mn}",
            rows=decision_rows,
            sigma=per_head_total,
            out_label=f"→ {text_helpers.human_count(scorer.total)} total",
            out_color=_ACCENT_DECISION,
            tooltip=(
                f"Decision Head ×{num_families} · "
                f"{text_helpers.human_count(per_head_total)} params each · "
                f"{text_helpers.human_count(scorer.total)} total · {mn} → 1"
            ),
            dashed=True,
            stack=num_families,
        )
    )

    # Grand total line.
    sans = "-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif"
    total_str = text_helpers.human_count(param_report.total)
    parts.append(
        f'<text x="{center_x}" y="{y_total}" font-family="{sans}" '
        f'font-size="13" font-weight="700" fill="{_ACCENT_ARCH}" text-anchor="middle">'
        f"TOTAL ≈ {total_str} params</text>"
    )

    parts.append("</svg>")
    return "\n".join(parts)
