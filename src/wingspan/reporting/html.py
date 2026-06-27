"""Standalone HTML model-summary report generator.

Produces a self-contained ``.html`` file (no external assets) documenting
the full structure of a training run's network.  The architecture diagram
(built by :mod:`wingspan.reporting.svg`) is the page's main menu: it is the only
thing visible on load, and clicking one of its input boxes rolls out the
matching vector's element-by-element breakdown below it (card features, card
set, state, choice, and setup vectors — one panel at a time; clicking the
active box again collapses it), while clicking any parameter count rolls out
the per-layer parameter table jumped to that block's rows.  A small inline
script drives the panel switching; complex stripes (board slots, round-goal
rounds) still drill down via HTML5 ``<details>``/``<summary>`` elements.

The public entry point is :func:`generate_html_report`.  The training loop
writes the result alongside the other JSON sidecars at startup; the
``wingspan-inspect --html`` flag exposes it from the CLI.
"""

from __future__ import annotations

import collections
import html as html_lib

from wingspan import architecture, cards, setup_model
from wingspan.encode import stripes as encode_stripes
from wingspan.reporting import card_view, encode_viewer, game_log_html
from wingspan.reporting import svg as report_svg
from wingspan.training.charts import text_helpers

# ---------------------------------------------------------------------------
# Accent colors — one per report section (the card / hand panels reuse their
# diagram blocks' accents so the color coding carries through).

_ACCENT_CARD = "#a855f7"
_ACCENT_HAND = "#d946ef"
_ACCENT_STATE = "#3b82f6"
_ACCENT_CHOICE = "#22c55e"
_ACCENT_SETUP = "#14b8a6"
_ACCENT_ARCH = "#a855f7"
_ACCENT_PARAMS = "#f97316"

# Prefix of the parameter table's per-block row anchors: a diagram parameter
# count with ``data-params-block="trunk"`` jumps to ``id="params-block-trunk"``.
# The inline ``_SCRIPT`` repeats this prefix — keep the two in sync.
_PARAMS_ANCHOR_PREFIX = "params-block-"

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
.click-hint { font-size: 12px; color: #64748b; margin: 10px 4px 0; }
[hidden] { display: none; }
.panel[hidden] { display: none; }
.arch-click, .arch-paramclick { cursor: pointer; }
.arch-click:hover rect { stroke: #818cf8; stroke-width: 2; }
.arch-click.selected rect { stroke: #4338ca; stroke-width: 2.5; }
.arch-paramclick:hover text { fill: #1e293b; }
.arch-paramclick.selected text { fill: #1e293b; text-decoration: underline; }
@keyframes rowflash { from { background: #fde68a; } to { background: transparent; } }
tr.flash > td { animation: rowflash 1.4s ease-out; }
.nav-tabs { display: flex; gap: 0; margin-left: 12px; }
.nav-tab {
  background: none; border: none; color: #94a3b8; font-size: 13px;
  padding: 4px 14px; cursor: pointer; border-radius: 4px; font-weight: 600;
}
.nav-tab.active { background: #334155; color: #f8fafc; }
.nav-tab:hover:not(.active) { color: #e2e8f0; }
.birds-grid { display: flex; flex-wrap: wrap; gap: 10px; padding: 20px 0; }
"""

# ---------------------------------------------------------------------------
# Inline JS — the diagram-as-menu behaviour. One delegated listener on the SVG
# wrap routes clicks: ``data-panel`` toggles the matching detail panel (one at
# a time, re-click collapses); ``data-params-block`` opens the Parameters panel
# and flashes that block's anchor row (id prefix = ``_PARAMS_ANCHOR_PREFIX``).

_SCRIPT = """\
(function () {
  'use strict';
  var svgWrap = document.querySelector('.arch-svg-wrap');
  if (!svgWrap) return;
  var panels = document.querySelectorAll('.panel');
  var openId = null;

  function closeAll() {
    panels.forEach(function (panel) { panel.hidden = true; });
    document.querySelectorAll('.arch-click.selected, .arch-paramclick.selected')
      .forEach(function (el) { el.classList.remove('selected'); });
    openId = null;
  }

  function openPanel(id, clicked) {
    closeAll();
    var panel = document.getElementById(id);
    if (!panel) return null;
    panel.hidden = false;
    openId = id;
    clicked.classList.add('selected');
    return panel;
  }

  function flashParamsRow(key) {
    var row = document.getElementById('params-block-' + key);
    if (!row) return;
    row.classList.remove('flash');
    void row.offsetWidth; /* restart the animation */
    row.classList.add('flash');
    row.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }

  svgWrap.addEventListener('click', function (event) {
    var target = event.target.closest('.arch-click, .arch-paramclick');
    if (!target) return;
    var panelId = target.getAttribute('data-panel');
    var paramsKey = target.getAttribute('data-params-block');
    if (panelId) {
      if (openId === panelId) { closeAll(); return; }
      var panel = openPanel(panelId, target);
      if (panel) panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } else if (paramsKey) {
      openPanel('params', target);
      flashParamsRow(paramsKey);
    }
  });
})();
"""

# Encoding-viewer modal for the Birds tab (single "Card Attributes" section).
_ENC_MODAL_HTML = (
    "<div id='enc-modal' role='dialog' aria-modal='true' aria-label='Card encoding'>"
    "<div id='enc-dialog'>"
    "<div id='enc-header'>"
    "<span id='enc-opt-label'></span>"
    "<button id='enc-close' title='Close (Esc)'>&#x2715;</button>"
    "</div>"
    "<div id='enc-body'>"
    "<h3 class='enc-section-title'>Card Attributes</h3>"
    "<div id='enc-card'></div>"
    "</div>"
    "</div>"
    "</div>"
)

# Tab switching + birds grid + encoding modal bootstrap.
_BIRDS_JS = """\
(function () {
  'use strict';
  var _modelSection = document.getElementById('tab-model');
  var _birdsSection = document.getElementById('tab-birds');

  // Tab switching.
  document.querySelectorAll('.nav-tab').forEach(function (btn) {
    btn.addEventListener('click', function () {
      document.querySelectorAll('.nav-tab').forEach(function (b) { b.classList.remove('active'); });
      btn.classList.add('active');
      var which = btn.getAttribute('data-tab');
      _modelSection.hidden = (which !== 'model');
      _birdsSection.hidden = (which !== 'birds');
    });
  });

  // Build the birds grid from embedded JSON.
  var _catalog = JSON.parse(document.getElementById('birds-data').textContent);
  var _grid = document.getElementById('birds-grid');
  if (_grid && _catalog.birds) {
    _catalog.birds.forEach(function (entry) {
      var wrap = document.createElement('div');
      wrap.innerHTML = cardCellHtml(entry.card);
      var cell = wrap.firstChild;
      cell.style.cursor = 'pointer';
      cell.addEventListener('click', function () { _openBirdModal(entry); });
      _grid.appendChild(cell);
    });
  }

  // Encoding modal for bird cards.
  var _encModal = document.getElementById('enc-modal');
  var _encCard = document.getElementById('enc-card');
  var _encLabel = document.getElementById('enc-opt-label');
  var _encClose = document.getElementById('enc-close');

  function _openBirdModal(entry) {
    _encLabel.textContent = entry.card.name;
    _encCard.innerHTML = (entry.stripes && entry.stripes.length)
      ? renderStripes(entry.stripes)
      : '<p class="enc-none">No non-identity attributes encoded.</p>';
    _encModal.classList.add('open');
  }

  if (_encClose) _encClose.addEventListener('click', function () { _encModal.classList.remove('open'); });
  if (_encModal) _encModal.addEventListener('click', function (e) {
    if (e.target === _encModal) _encModal.classList.remove('open');
  });
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && _encModal && _encModal.classList.contains('open'))
      _encModal.classList.remove('open');
  });
})();
"""


# ---------------------------------------------------------------------------
# Public API


def generate_html_report(
    state_layout: encode_stripes.VectorLayout,
    choice_layout: encode_stripes.VectorLayout,
    param_report: architecture.ParamReport,
    arch: architecture.ModelArchitecture,
    *,
    setup_encoding: setup_model.SetupEncoding,
    setup_arch: setup_model.SetupArchitecture,
    use_setup_model: bool,
    state_dim: int,
    choice_dim: int,
    family_order: tuple[str, ...],
    run_name: str,
    model_version: str,
) -> str:
    """Return a self-contained HTML document string for the model summary.

    All CSS and JS are embedded inline; no external resources are referenced,
    so the file opens correctly when copied anywhere or viewed offline.  The
    architecture diagram leads the page and serves as its menu; every vector
    breakdown and the parameter table start as hidden panels it reveals.

    The separately-trained setup model is always documented — its input-vector
    breakdown and its block in the architecture diagram (``setup_arch``) — even
    when ``use_setup_model`` is False (the section and block are then annotated
    as inactive rather than omitted). ``setup_encoding`` supplies both the raw
    feature-vector dimension for parameter accounting and the post-embedding
    layout for the vector breakdown table.
    """
    # The post-embedding layout is what the readout MLP actually receives (and
    # what the breakdown table displays). The raw total_dim is what
    # count_setup_parameters expects as feature_dim — passing the post-embedding
    # size instead would double-apply the embedding transform, inflating the
    # reported input count.
    setup_layout = setup_model.setup_readout_stripe_layout(setup_encoding, arch)
    setup_param = setup_model.count_setup_parameters(
        setup_arch,
        feature_dim=setup_encoding.total_dim,
        main_arch=arch,
        encoding=setup_encoding,
    )
    setup_annotation = (
        ""
        if use_setup_model
        else "not active this run — opening keep scored by the in-game policy"
    )
    body = "\n".join(
        [
            _arch_section(
                arch,
                param_report,
                family_order,
                setup_param=setup_param,
                setup_arch=setup_arch,
                use_setup_model=use_setup_model,
            ),
            _vector_section(
                encode_stripes.card_feature_stripe_layout(),
                report_svg.PANEL_CARD,
                "Card Feature Vector",
                _ACCENT_CARD,
                input_note=(
                    "raw single-card encoder input — static attributes + "
                    "identity one-hot, one row of the shared card table"
                ),
            ),
            _vector_section(
                encode_stripes.hand_encoder_input_stripe_layout(),
                report_svg.PANEL_HAND,
                "Card Set Vector",
                _ACCENT_HAND,
                input_note=(
                    "raw multi-card encoder input — set multi-hot + summary "
                    "stats (own hand / setup keep / tray set)"
                ),
            ),
            _vector_section(
                state_layout, report_svg.PANEL_STATE, "State Vector", _ACCENT_STATE
            ),
            _vector_section(
                choice_layout, report_svg.PANEL_CHOICE, "Choice Vector", _ACCENT_CHOICE
            ),
            _vector_section(
                setup_layout,
                report_svg.PANEL_SETUP,
                "Setup Vector",
                _ACCENT_SETUP,
                input_note=(
                    "post-embedding network input — raw candidate features after "
                    "the kept-card multi-hot and tray indices are replaced by "
                    "frozen encoder embeddings (matches the arch diagram's "
                    "‘in N’)"
                ),
                annotation=setup_annotation,
            ),
            _params_section(param_report),
        ]
    )
    birds_catalog = _birds_payload()
    return _wrap(
        title=f"Model Summary — {html_lib.escape(run_name)}",
        run_name=run_name,
        model_version=model_version,
        body=body,
        birds_catalog=birds_catalog,
    )


###### PRIVATE #######

#### HTML shell ####


def _wrap(
    *,
    title: str,
    run_name: str,
    model_version: str,
    body: str,
    birds_catalog: game_log_html.BirdCatalog,
) -> str:
    """Assemble the full HTML document: nav + tab sections + modal + scripts."""
    nav = _nav(run_name, model_version)
    birds_json = birds_catalog.model_dump_json().replace("<", "\\u003c")
    all_css = _CSS + "\n" + card_view.CARD_CSS + "\n" + card_view.STRIPE_VIEWER_CSS
    all_script = (
        card_view.CARD_JS
        + "\n"
        + card_view.STRIPE_VIEWER_JS
        + "\n"
        + _SCRIPT
        + "\n"
        + _BIRDS_JS
    )
    return (
        f"<!DOCTYPE html>\n<html lang='en'>\n<head>\n"
        f"<meta charset='utf-8'>\n"
        f"<meta name='viewport' content='width=device-width, initial-scale=1'>\n"
        f"<title>{title}</title>\n"
        f"<style>\n{all_css}\n</style>\n"
        f"</head>\n<body>\n"
        f"{nav}\n"
        f"<div id='tab-model'><div class='container'>\n{body}\n</div></div>\n"
        f"<div id='tab-birds' hidden><div class='container'>"
        f"<div class='birds-grid' id='birds-grid'></div>"
        f"</div></div>\n"
        f"{_ENC_MODAL_HTML}\n"
        f"<script type='application/json' id='birds-data'>{birds_json}</script>\n"
        f"<script>\n{all_script}\n</script>\n"
        f"</body>\n</html>\n"
    )


def _nav(run_name: str, model_version: str) -> str:
    """The slim header bar: brand + run name + artifact version + Model/Birds tab buttons."""
    return (
        f"<nav>"
        f"<span class='nav-brand'>Wingspan</span>"
        f"<span class='nav-run'>{html_lib.escape(run_name)}</span>"
        f"<span class='nav-run'>v{html_lib.escape(model_version)}</span>"
        f"<div class='nav-tabs'>"
        f"<button class='nav-tab active' data-tab='model'>Model</button>"
        f"<button class='nav-tab' data-tab='birds'>Birds</button>"
        f"</div>"
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
    """Build a complete vector-layout section as a hidden panel.

    The panel starts hidden and is revealed by clicking the diagram input box
    whose ``data-panel`` equals ``section_id`` (the ``report_svg.PANEL_*``
    contract).  ``input_note`` qualifies the element count in the subtitle (the
    state / choice vectors are shown post-embedding; the card, card-set, and
    setup vectors are raw).  A non-empty ``annotation`` is appended to the
    subtitle — used to flag the setup section when the setup model is not
    active this run.
    """
    expanded_count = sum(1 for stripe in layout.stripes if stripe.sub_fields)
    sub = (
        f"{layout.total_size} elements ({input_note}) · "
        f"{len(layout.stripes)} stripes · {expanded_count} with drill-down"
    )
    if annotation:
        sub += f" · {html_lib.escape(annotation)}"
    return (
        f"<div class='section panel' id='{section_id}' hidden>"
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
    setup_param: setup_model.SetupParamReport,
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
        f"<div class='click-hint'>Click an input box to inspect that vector's "
        f"element-by-element breakdown, or any parameter count to open the "
        f"parameter table at that block.</div>"
        f"</div>"
    )


#### Parameters section ####


def _params_section(report: architecture.ParamReport) -> str:
    """The per-layer parameter table as a hidden panel; each block's first row
    carries the anchor a diagram parameter count jumps to."""
    total = report.total
    sub = (
        f"{text_helpers.human_count(total)} total trainable parameters &nbsp;·&nbsp; "
        f"{len(report.blocks)} blocks"
    )
    return (
        f"<div class='section panel' id='{report_svg.PANEL_PARAMS}' hidden>"
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

        # The block's jump anchor goes on its first rendered row (the subtotal
        # row when a block has no per-layer rows, e.g. the per-family scorer).
        anchor_attr = f" id='{_PARAMS_ANCHOR_PREFIX}{_params_block_key(block.label)}'"

        # Per-layer rows for this block.
        for idx, layer in enumerate(block.layers):
            layer_label = f"Linear  {layer.in_features} → {layer.out_features}"
            layer_params = layer.linear * block.multiplier
            running += layer_params
            pct = f"{100.0 * layer_params / max(total, 1):.1f}%"
            row_anchor, anchor_attr = anchor_attr, ""
            rows.append(
                f"<tr{row_anchor}>"
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
            f"<tr class='subtotal-row'{anchor_attr}>"
            f"<td></td>"
            f"<td>Subtotal {html_lib.escape(block_label)}</td>"
            f"<td class='right'>{text_helpers.human_count(block.total)}</td>"
            f"<td class='right'>{block_pct}</td>"
            f"<td></td>"
            f"</tr>"
        )

    # Grand total.
    rows.append(
        f"<tr class='total-row' "
        f"id='{_PARAMS_ANCHOR_PREFIX}{report_svg.PARAMS_BLOCK_TOTAL}'>"
        f"<td>TOTAL</td>"
        f"<td></td>"
        f"<td class='right'>{text_helpers.human_count(total)}</td>"
        f"<td class='right'>100%</td>"
        f"<td></td>"
        f"</tr>"
    )
    return "".join(rows)


def _params_block_key(label: str) -> str:
    """The anchor / ``data-params-block`` key for a block label — the lowercased
    ``BlockParam.label``, matching what ``report_svg`` attaches to the diagram's
    parameter counts."""
    return label.lower()


#### Birds tab ####


def _birds_payload() -> game_log_html.BirdCatalog:
    """Build the BirdCatalog payload for the Birds tab grid.

    Loops over all 180 core birds in canonical order, pairing each bird's static
    card display info with its non-identity attribute stripes.  Called once at
    report generation time; the result is serialized to embedded JSON.  Does not
    touch the engine or PyTorch — purely card metadata + numpy."""
    return game_log_html.BirdCatalog(
        birds=[
            game_log_html.BirdCatalogEntry(
                card=card_view.bird_cell_info(bird),
                stripes=encode_viewer.extract_card_attr_stripes(bird),
            )
            for bird in cards.birds_ordered()
        ]
    )
