"""Standalone HTML model-summary report generator.

Produces a self-contained ``.html`` file (no external assets) documenting
the full structure of a training run's network: the element-by-element
breakdown of the state, choice, and setup vectors, the network architecture
(including the separately-trained setup model — the SVG diagram itself is
built by :mod:`wingspan.report_svg`), and the per-layer parameter
accounting.  The file is meant to be opened in a browser and supports
drill-down into complex stripes (board slots, round-goal rounds) via HTML5
``<details>``/``<summary>`` elements — no JavaScript required.

The public entry point is :func:`generate_html_report`.  The training loop
writes the result alongside the other JSON sidecars at startup; the
``wingspan-inspect --html`` flag exposes it from the CLI.
"""

from __future__ import annotations

import collections
import html as html_lib

from wingspan import architecture, report_svg, setup_model
from wingspan.encode import stripes as encode_stripes
from wingspan.training.charts import text_helpers

# ---------------------------------------------------------------------------
# Accent colors — one per report section.

_ACCENT_STATE = "#3b82f6"
_ACCENT_CHOICE = "#22c55e"
_ACCENT_SETUP = "#14b8a6"
_ACCENT_ARCH = "#a855f7"
_ACCENT_PARAMS = "#f97316"

# Minimum sub-field count that triggers grouped (nested) display rather than a
# flat table.  Board stripes have 135 sub-fields across 15 slots, so they are
# always grouped; small stripes (≤ threshold) display flat.
_GROUP_THRESHOLD = 24

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
    setup_layout: encode_stripes.VectorLayout,
    setup_arch: setup_model.SetupArchitecture,
    use_setup_model: bool,
    state_dim: int,
    choice_dim: int,
    family_order: tuple[str, ...],
    run_name: str,
) -> str:
    """Return a self-contained HTML document string for the model summary.

    All CSS is embedded inline; no external resources are referenced, so the
    file opens correctly when copied anywhere or viewed offline.

    The separately-trained setup model is always documented — its input-vector
    breakdown (``setup_layout``) and its block in the architecture diagram
    (``setup_arch``) — even when ``use_setup_model`` is False (the section and
    block are then annotated as inactive rather than omitted).
    """
    setup_param = setup_model.count_setup_parameters(
        setup_arch, feature_dim=setup_layout.total_size, main_arch=arch
    )
    setup_annotation = (
        ""
        if use_setup_model
        else "not active this run — opening keep scored by the in-game policy"
    )
    body = "\n".join(
        [
            _vector_section(state_layout, "state", "State Vector", _ACCENT_STATE),
            _vector_section(choice_layout, "choice", "Choice Vector", _ACCENT_CHOICE),
            _vector_section(
                setup_layout,
                "setup",
                "Setup Vector",
                _ACCENT_SETUP,
                input_note=(
                    "raw network input — multi-hot / one-hot / count blocks, "
                    "no card embedding"
                ),
                annotation=setup_annotation,
            ),
            _arch_section(
                arch,
                param_report,
                family_order,
                setup_param=setup_param,
                setup_arch=setup_arch,
                use_setup_model=use_setup_model,
            ),
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
        ("#setup", "Setup Vector"),
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
    *,
    input_note: str = "post-embedding network input",
    annotation: str = "",
) -> str:
    """Build a complete section for a state, choice, or setup vector layout.

    ``input_note`` qualifies the element count in the subtitle (the state /
    choice vectors are shown post-embedding; the setup vector is raw).  A
    non-empty ``annotation`` is appended to the subtitle — used to flag the
    setup section when the setup model is not active this run.
    """
    expanded_count = sum(1 for stripe in layout.stripes if stripe.sub_fields)
    sub = (
        f"{layout.total_size} elements ({input_note}) · "
        f"{len(layout.stripes)} stripes · {expanded_count} with drill-down"
    )
    if annotation:
        sub += f" · {html_lib.escape(annotation)}"
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
    family_order: tuple[str, ...],
    *,
    setup_param: architecture.BlockParam,
    setup_arch: setup_model.SetupArchitecture,
    use_setup_model: bool,
) -> str:
    svg = report_svg.build_arch_svg(
        arch,
        param_report,
        family_order,
        setup_param=setup_param,
        setup_arch=setup_arch,
        use_setup_model=use_setup_model,
    )
    return (
        f"<div class='section' id='arch'>"
        f"<div class='section-header' style='color:{_ACCENT_ARCH}'>Architecture</div>"
        f"<div class='section-sub'>"
        f"Network flow — single-card + multi-card encoders → state encoder"
        f" ⊕ choice encoder → decision heads + value head · connected setup net"
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
