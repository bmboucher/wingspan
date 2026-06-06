"""The live ARCHITECTURE diagram for the FLIGHT PLAN configurator.

:class:`ArchitectureDiagram` is a width/height-aware ``rich`` renderable that
draws the working network as a **wiring diagram** rather than a flat list. It
reads :class:`state.ConfiguratorState` fresh each frame, so it reacts live as
fields are edited. The wiring, top to bottom:

* the separate :class:`setup_net.SetupNet` as an **unconnected** box at the top
  (only when ``use_setup_model`` is on — it is trained on its own and is not part
  of the in-game policy graph);
* the shared **card encoder** (the per-card MLP that maps each card's static
  attributes ⊕ identity to its embedding) as a full-width box, with arrows fanning
  out to both trunks and a small box above each trunk carrying the count of
  *additional* (non-card) inputs it also consumes;
* the **state trunk** and the per-choice **encoder** drawn **side by side**
  (they are parallel bodies that both consume the card encoder), each layer a
  color-coded mini-box with its parameter count inlaid in the bottom border;
* the two trunks merge — the trunk's ``M`` feeds the **value** head, the ``M+N``
  concat feeds the **decision** (scorer) head, drawn once as a *dashed* template
  tagged ``×N`` because one identical head is instantiated per judgment family.

Layer mini-boxes are color-coded by operation kind (Linear / LayerNorm /
activation / Dropout) and transcribe ``model._build_body`` / ``_build_readout``:
body blocks (trunk / choice) apply LayerNorm — when enabled — after every Linear;
the trunk always keeps a final activation; the card / choice / hand encoders keep
one too when ``arch.encoder_final_activation`` is True (new runs), or omit it
when False (old checkpoints); readout heads (scorer / value, and the setup net)
never LayerNorm and end in a bare ``Linear →1``. Parameter counts come from
:func:`architecture.count_parameters` (the main net) and
:func:`setup_model.count_setup_parameters` (the setup net), each pinned to
``sum(p.numel())`` of the real module by a test. Below a two-column width floor
the diagram degrades to a compact single-column text list.
"""

from __future__ import annotations

import dataclasses
import enum
import typing

import pydantic
import rich.console as rich_console
from rich import text

from wingspan import architecture, encode, setup_model
from wingspan.training import theme
from wingspan.training.charts import text_helpers

if typing.TYPE_CHECKING:
    from wingspan.training import config
    from wingspan.training.configure import state

# Below this inner width the two-column box diagram cannot fit, so the renderable
# degrades to the compact single-column text list.
_MIN_TWO_COL_WIDTH = 34
# Blank columns between the two side-by-side columns.
_COL_GAP = 2
# A run of this many consecutive identical layers collapses to one ``×N`` group.
_COLLAPSE_RUN = 2
# Mini-boxes never grow past this so a full-width block (the setup net) keeps tidy
# left-aligned cards rather than one box stretched across the panel.
_OP_CARD_MAX_W = 22
# Rows the small "additional inputs" box adds above each trunk (title, bottom,
# arrow). The trunk-focus anchor skips them so the viewport centers on the trunk
# box itself, not its input wiring.
_EXTRA_HEADER_ROWS = 3

# Solid box-drawing glyphs (thin rules).
_TL, _TR, _BL, _BR = "┌", "┐", "└", "┘"
_H, _V = "─", "│"
_TAP_DOWN = "┬"  # a box bottom border tapping down into a connector
_JUNCT_UP = "┴"  # a connector joining up into a box bottom
_TEE_R, _TEE_L = "├", "┤"
_ARROW = "▼"

# Dashed glyphs for the duplicated decision-head template.
_DH, _DV = "┄", "┊"

# Clipped-viewport indicators (end="" so they replace a row without adding one).
_SCROLL_MORE_UP = "  ▲ more"
_SCROLL_MORE_DOWN = "  ▼ more"

# Which selected field lights up a whole BOX (gold border + title).
_BOX_FOCUS_ATTRS: dict[str, set[str]] = {
    "setup": {"setup_hidden_layers", "use_setup_model"},
    "embed": {"card_embed_dim", "card_encoder_layers"},
    "trunk": {"trunk_layers"},
    "choice": {"choice_layers"},
    "scorer": {"head_layers"},
    "value": {"value_layers"},
}
# The shared op handles brighten their matching mini-boxes instead. The main net
# and the setup net each have their own activation / dropout handles.
_MAIN_OP_FIELDS: dict[str, str] = {
    "activation": "activation",
    "dropout": "dropout",
    "layernorm": "layernorm",
}
_SETUP_OP_FIELDS: dict[str, str] = {
    "activation": "setup_activation",
    "dropout": "setup_dropout",
}
_MAIN_OP_ATTRS = set(_MAIN_OP_FIELDS.values())
_SETUP_OP_ATTRS = set(_SETUP_OP_FIELDS.values())

# Compact activation labels for the narrow op-cards (the rest are already short).
_SHORT_ACTIVATION: dict[str, str] = {"leaky_relu": "lrelu"}

# Per-structural-box accent border (gold overrides it when the box is focused).
_SETUP_BORDER = theme.SETUP_MARK
_CARD_BORDER = theme.GAUGE_MEM_PROC  # the shared embedding — violet
_DECISION_BORDER = theme.BORDER_EVAL  # the duplicated head template — sky blue
_BODY_BORDER = theme.BORDER_DEFAULT
_EXTRA_BORDER = theme.TEXT_MUTED  # the small "additional inputs" box above a trunk


class _OpKind(enum.StrEnum):
    """The four per-layer operations the diagram draws as color-coded mini-boxes."""

    LINEAR = "linear"
    LAYERNORM = "layernorm"
    ACTIVATION = "activation"
    DROPOUT = "dropout"


class _BlockKind(enum.StrEnum):
    """Which model rule-set a block's layers follow (``model._build_body`` vs
    ``_build_readout``). Drives activation / LayerNorm applicability per layer.
    The setup net follows the readout rules with its own activation / dropout."""

    BODY_TRUNK = "body_trunk"
    BODY_CHOICE = "body_choice"
    READOUT = "readout"


class _OpEntry(pydantic.BaseModel):
    """One drawable layer mini-box: its operation ``kind``, the full + ``short``
    display labels (the short one is used when the box is too narrow), the inlaid
    parameter count (``None`` for the zero-parameter activation / dropout ops), the
    collapsed-run length (``run`` >= 2 draws a ``×N`` tag), and whether the field
    behind this op kind is focused (so the box brightens)."""

    kind: _OpKind
    label: str
    short: str
    param: int | None = None
    run: int = 1
    focused: bool = False


# Per-op mini-box border + text color (the "color coding" that tells the four
# layer types apart at a glance).
_OP_COLOR: dict[_OpKind, str] = {
    _OpKind.LINEAR: theme.GAUGE_MEM,  # wetland blue — the weight layers
    _OpKind.LAYERNORM: theme.GAUGE_UTIL,  # teal
    _OpKind.ACTIVATION: theme.GOOD,  # green
    _OpKind.DROPOUT: theme.CAUTION,  # amber
}


def viewport(
    rows: list[text.Text], selected_row: int, height: int
) -> tuple[list[text.Text], bool, bool]:
    """The slice of ``rows`` to show, keeping ``selected_row`` visible, plus
    whether rows are clipped above / below. Shared by the form list and this
    diagram."""
    if height <= 0 or len(rows) <= height:
        return list(rows), False, False
    first = min(max(selected_row - height // 2, 0), len(rows) - height)
    window = list(rows[first : first + height])
    return window, first > 0, first + height < len(rows)


class ArchitectureDiagram:
    """The ARCHITECTURE panel body: the working network as a wiring diagram,
    scrolled to keep the focused block visible and degraded to a compact text list
    when the panel is too narrow for two columns."""

    def __init__(self, view: state.ConfiguratorState):
        self.view = view

    def __rich_console__(
        self, console: rich_console.Console, options: rich_console.ConsoleOptions
    ) -> rich_console.RenderResult:
        content_w = options.max_width
        if content_w < _MIN_TWO_COL_WIDTH:
            rows, selected_row = _compact_rows(self.view)
        else:
            rows, selected_row = _diagram_rows(self.view, content_w)
        height = options.height or options.max_height or len(rows)
        window, scroll_up, scroll_down = viewport(rows, selected_row, height)
        if scroll_up:
            window[0] = text.Text(_SCROLL_MORE_UP, style=theme.TEXT_MUTED, end="")
        if scroll_down:
            window[-1] = text.Text(_SCROLL_MORE_DOWN, style=theme.TEXT_MUTED, end="")
        for index, line in enumerate(window):
            if index:
                yield text.Text("\n", end="")
            yield line


###### PRIVATE #######

#### Row assembly ####


def _diagram_rows(
    view: state.ConfiguratorState, content_w: int
) -> tuple[list[text.Text], int]:
    """The full ordered row list — the setup box, the card encoder, the fan-out,
    the side-by-side trunks, the merge, the value / decision heads, and the
    total — plus the row to scroll toward for focus."""
    cfg = view.working
    _, _, left_center, right_center = _columns(content_w)
    rows: list[text.Text] = []
    block_start: dict[str, int] = {}

    def add(section: str, block_rows: list[text.Text]) -> None:
        block_start[section] = len(rows)
        rows.extend(block_rows)

    if cfg.use_setup_model:
        add("setup", _setup_box(view, content_w))
    else:
        add("setup", [_setup_off_line()])
    rows.append(_blank())

    add("embed", _card_encoder_box(view, content_w))
    rows.extend(_fanout_rows(content_w, left_center, right_center))
    add("trunk", _trunk_region(view, content_w))
    # Center on the trunk box, not its additional-inputs header above it.
    block_start["trunk"] += _EXTRA_HEADER_ROWS
    block_start["choice"] = block_start["trunk"]
    rows.extend(_merge_rows(content_w, left_center, right_center))
    add("scorer", _heads_region(view, content_w))
    block_start["value"] = block_start["scorer"]

    rows.append(_blank())
    rows.append(_total_row(view))
    return rows, _anchor(view, block_start)


def _anchor(view: state.ConfiguratorState, block_start: dict[str, int]) -> int:
    """The row index the viewport centers on: the focused block's title, or the
    trunk / setup block (the first block with op rows) when a shared op handle is
    focused."""
    attr = view.selected_attr
    for section, attrs in _BOX_FOCUS_ATTRS.items():
        if attr in attrs:
            return block_start.get(section, 0)
    if attr in _MAIN_OP_ATTRS:
        return block_start.get("trunk", 0)
    if attr in _SETUP_OP_ATTRS:
        return block_start.get("setup", 0)
    return 0


def _columns(content_w: int) -> tuple[int, int, int, int]:
    """Split ``content_w`` into the two column widths (left / right) plus the two
    column-center x positions the connectors route to."""
    col_w = (content_w - _COL_GAP) // 2
    right_w = content_w - _COL_GAP - col_w
    left_center = col_w // 2
    right_center = col_w + _COL_GAP + right_w // 2
    return col_w, right_w, left_center, right_center


#### Compact fallback ####


def _compact_rows(view: state.ConfiguratorState) -> tuple[list[text.Text], int]:
    """Sub-``_MIN_TWO_COL_WIDTH`` fallback: one labeled width-chain line per block
    (with ``+LN`` / ``+drop`` tags) plus the total — still live, just chromeless."""
    cfg = view.working
    extras: list[str] = []
    if cfg.layernorm:
        extras.append("+LN")
    if cfg.dropout > 0.0:
        extras.append(f"+d{_fmt_dropout(cfg.dropout)}")
    tags = ("  " + " ".join(extras)) if extras else ""
    trunk_in = _trunk_in(cfg)
    choice_in = _choice_in(cfg)
    concat = cfg.arch.trunk_embed_width + cfg.arch.choice_embed_width
    setup_chain = (
        _chain(_setup_readout_in(cfg), (*cfg.setup_hidden_layers, 1))
        if cfg.use_setup_model
        else "off"
    )
    rows = [
        _compact_line("SETUP", setup_chain),
        _compact_line(
            "CARD",
            _chain(
                encode.CARD_FEATURE_DIM, cfg.card_encoder_layers + (cfg.card_embed_dim,)
            ),
        ),
        _compact_line("TRUNK", _chain(trunk_in, cfg.trunk_layers), tags),
        _compact_line("CHOICE", _chain(choice_in, cfg.choice_layers), tags),
        _compact_line(
            "VALUE", _chain(cfg.arch.trunk_embed_width, (*cfg.value_layers, 1))
        ),
        _compact_line(
            "DECIDE",
            f"{_chain(concat, (*cfg.head_layers, 1))} ×{len(cfg.family_order)}",
        ),
        _total_row(view),
    ]
    return rows, 0


def _compact_line(label: str, chain: str, tags: str = "") -> text.Text:
    """One ``LABEL  chain  +tags`` row for the narrow fallback."""
    line = text.Text(no_wrap=True, end="")
    line.append(f"{label:<7}", style=f"bold {theme.TEXT_MUTED}")
    line.append(chain, style=theme.TEXT_PRIMARY)
    if tags:
        line.append(tags, style=theme.CAUTION)
    return line


def _chain(in_dim: int, widths: tuple[int, ...]) -> str:
    """A ``2381→128→128`` input-to-output width string."""
    return "→".join(str(value) for value in (in_dim, *widths))


#### Full-width boxes (setup, card encoder) ####


def _setup_box(view: state.ConfiguratorState, content_w: int) -> list[text.Text]:
    """The separate setup net — an unconnected readout MLP value-regressor over the
    setup-candidate features (``SETUP_FEATURE_DIM → setup_hidden_layers → 1``)."""
    cfg = view.working
    block = _setup_block(view)
    entries = _block_op_entries(
        view,
        (*cfg.setup_hidden_layers, 1),
        _BlockKind.READOUT,
        block.layers,
        _SETUP_OP_FIELDS,
        activation=str(cfg.setup_activation),
        dropout=cfg.setup_dropout,
        layernorm=False,
    )
    caption = [
        (
            f"in {_setup_readout_in(cfg)} "
            f"(embedded {setup_model.SETUP_FEATURE_DIM}-dim candidate)",
            theme.TEXT_DIM2,
        )
    ]
    return _model_block(
        view,
        section="setup",
        title="SETUP MODEL · keep",
        in_caption=caption,
        entries=entries,
        sigma_total=block.total,
        out_caption=None,
        width=content_w,
        tap=False,
        dashed=False,
    )


def _setup_off_line() -> text.Text:
    """The one-line placeholder shown when ``use_setup_model`` is off."""
    line = text.Text(no_wrap=True, end="")
    line.append("setup model · ", style=theme.TEXT_MUTED)
    line.append("off", style=theme.TEXT_DIM2)
    line.append("  (keep handled by the in-game policy)", style=theme.TEXT_MUTED)
    return line


def _card_encoder_box(view: state.ConfiguratorState, content_w: int) -> list[text.Text]:
    """The shared per-card encoder MLP — maps each card's ``[static attributes ⊕
    identity one-hot]`` (``CARD_FEATURE_DIM``) to its ``card_embed_dim`` vector,
    yielding the per-card table every board / tray / hand / choice slot looks up.
    A full-width body block whose bottom taps down into the fan-out to both trunks.
    Applies a final activation when ``encoder_final_activation`` is True."""
    cfg = view.working
    report = _param_report(view)
    widths = cfg.card_encoder_layers + (cfg.card_embed_dim,)
    entries = _block_op_entries(
        view,
        widths,
        _BlockKind.BODY_CHOICE,
        report.embed.layers,
        _MAIN_OP_FIELDS,
        activation=str(cfg.activation),
        dropout=cfg.dropout,
        layernorm=cfg.layernorm,
        final_activation=cfg.encoder_final_activation,
    )
    return _model_block(
        view,
        section="embed",
        title="CARD ENCODER · per-card MLP",
        in_caption=[(f"in {encode.CARD_FEATURE_DIM} (attrs ⊕ id)", theme.TEXT_DIM2)],
        entries=entries,
        sigma_total=report.embed.total,
        out_caption=("→ ", str(cfg.card_embed_dim)),
        width=content_w,
        tap=True,
        dashed=False,
    )


#### Connectors (fan-out, merge) ####


def _fanout_rows(
    content_w: int, left_center: int, right_center: int
) -> list[text.Text]:
    """Two rows wiring the card encoder down to both trunks: a branch from the
    center to the two column centers, then the down-arrows landing on each trunk's
    additional-inputs box."""
    center = content_w // 2
    chars = [" "] * content_w
    for index in range(left_center, right_center + 1):
        chars[index] = _H
    chars[left_center] = _TL
    chars[right_center] = _TR
    if left_center < center < right_center:
        chars[center] = _JUNCT_UP
    branch = text.Text("".join(chars), style=theme.TEXT_MUTED, no_wrap=True, end="")
    arrows = _marks_row(
        content_w, {left_center: _ARROW, right_center: _ARROW}, theme.TEXT_MUTED
    )
    return [branch, arrows]


def _merge_rows(content_w: int, left_center: int, right_center: int) -> list[text.Text]:
    """Three rows merging the two trunks into the heads: stems down from each, a
    cross carrying the trunk's ``M`` rightward to join the choice ``N``, then the
    two head arrows labeled ``M`` (value) and ``M+N`` (decision)."""
    stems = _marks_row(content_w, {left_center: _V, right_center: _V}, theme.TEXT_MUTED)
    chars = [" "] * content_w
    for index in range(left_center, right_center + 1):
        chars[index] = _H
    chars[left_center] = _TEE_R
    chars[right_center] = _TEE_L
    cross = text.Text("".join(chars), style=theme.TEXT_MUTED, no_wrap=True, end="")
    arrows = _label_arrow_row(content_w, left_center, right_center, " M", " M+N")
    return [stems, cross, arrows]


#### Two-column regions (trunks, heads) ####


def _trunk_region(view: state.ConfiguratorState, content_w: int) -> list[text.Text]:
    """The state trunk and the per-choice encoder, drawn side by side with their
    bottom borders aligned so the merge connector lands cleanly."""
    col_w, right_w, _, _ = _columns(content_w)
    report = _param_report(view)
    left = _state_trunk_column(view, col_w, report)
    right = _choice_encoder_column(view, right_w, report)
    height = max(len(left), len(right))
    left, right = _align_bottoms(left, height), _align_bottoms(right, height)
    return text_helpers.join_columns([(left, col_w), (right, right_w)], _COL_GAP)


def _state_trunk_column(
    view: state.ConfiguratorState, width: int, report: architecture.ParamReport
) -> list[text.Text]:
    """The state trunk box (a body block keeping an activation on its final layer),
    fed by the card encoder plus its additional non-card features."""
    cfg = view.working
    trunk_in = _trunk_in(cfg)
    extra = cfg.state_dim - encode.N_CARD_INDEX_SLOTS - encode.HAND_MULTIHOT_DIM
    entries = _block_op_entries(
        view,
        cfg.trunk_layers,
        _BlockKind.BODY_TRUNK,
        report.trunk.layers,
        _MAIN_OP_FIELDS,
        activation=str(cfg.activation),
        dropout=cfg.dropout,
        layernorm=cfg.layernorm,
    )
    block = _model_block(
        view,
        section="trunk",
        title="STATE TRUNK",
        in_caption=[(f"in {trunk_in}", theme.TEXT_DIM2)],
        entries=entries,
        sigma_total=report.trunk.total,
        out_caption=("M = ", str(cfg.trunk_layers[-1])),
        width=width,
        tap=True,
        dashed=False,
    )
    return _extra_input_header(width, extra) + block


def _choice_encoder_column(
    view: state.ConfiguratorState, width: int, report: architecture.ParamReport
) -> list[text.Text]:
    """The per-choice encoder box, fed by the card encoder plus its additional
    features. Applies a final activation when ``encoder_final_activation`` is
    True, matching the trunk; otherwise ends in a bare Linear."""
    cfg = view.working
    choice_in = _choice_in(cfg)
    extra = _choice_extra(cfg)
    entries = _block_op_entries(
        view,
        cfg.choice_layers,
        _BlockKind.BODY_CHOICE,
        report.choice.layers,
        _MAIN_OP_FIELDS,
        activation=str(cfg.activation),
        dropout=cfg.dropout,
        layernorm=cfg.layernorm,
        final_activation=cfg.encoder_final_activation,
    )
    block = _model_block(
        view,
        section="choice",
        title="CHOICE ENC",
        in_caption=[(f"in {choice_in}", theme.TEXT_DIM2)],
        entries=entries,
        sigma_total=report.choice.total,
        out_caption=("N = ", str(cfg.choice_layers[-1])),
        width=width,
        tap=True,
        dashed=False,
    )
    return _extra_input_header(width, extra) + block


def _extra_input_header(width: int, extra: int) -> list[text.Text]:
    """The small "additional inputs" box that sits above a trunk: the count of
    features the card encoder does *not* supply, with an arrow down into the
    trunk (where it joins the fanned-out card embeddings)."""
    return [
        _box_title(f"+{extra} feats", width, _EXTRA_BORDER, dashed=False),
        _box_bottom(width, _EXTRA_BORDER, tap=True, dashed=False),
        _marks_row(width, {width // 2: _ARROW}, theme.TEXT_MUTED),
    ]


def _heads_region(view: state.ConfiguratorState, content_w: int) -> list[text.Text]:
    """The value head and the duplicated decision-head template, side by side."""
    col_w, right_w, _, _ = _columns(content_w)
    report = _param_report(view)
    left = _value_column(view, col_w, report)
    right = _decision_column(view, right_w, report)
    height = max(len(left), len(right))
    left, right = _align_bottoms(left, height), _align_bottoms(right, height)
    return text_helpers.join_columns([(left, col_w), (right, right_w)], _COL_GAP)


def _value_column(
    view: state.ConfiguratorState, width: int, report: architecture.ParamReport
) -> list[text.Text]:
    """The value head — a readout MLP over the trunk ``M`` alone."""
    cfg = view.working
    trunk_m = cfg.arch.trunk_embed_width
    entries = _block_op_entries(
        view,
        (*cfg.value_layers, 1),
        _BlockKind.READOUT,
        report.value.layers,
        _MAIN_OP_FIELDS,
        activation=str(cfg.activation),
        dropout=cfg.dropout,
        layernorm=False,
    )
    return _model_block(
        view,
        section="value",
        title="VALUE",
        in_caption=[(f"in M={trunk_m}", theme.TEXT_DIM2)],
        entries=entries,
        sigma_total=report.value.total,
        out_caption=None,
        width=width,
        tap=False,
        dashed=False,
    )


def _decision_column(
    view: state.ConfiguratorState, width: int, report: architecture.ParamReport
) -> list[text.Text]:
    """The per-family decision (scorer) head — a readout MLP over the ``M+N``
    concat, drawn once as a dashed template tagged with its family multiplier."""
    cfg = view.working
    concat = cfg.arch.trunk_embed_width + cfg.arch.choice_embed_width
    families = len(cfg.family_order)
    per_head = sum(layer.params for layer in report.scorer.layers)
    entries = _block_op_entries(
        view,
        (*cfg.head_layers, 1),
        _BlockKind.READOUT,
        report.scorer.layers,
        _MAIN_OP_FIELDS,
        activation=str(cfg.activation),
        dropout=cfg.dropout,
        layernorm=False,
    )
    caption = [(f"in {concat} ", theme.TEXT_DIM2), (f"×{families}", _DECISION_BORDER)]
    return _model_block(
        view,
        section="scorer",
        title="DECISION",
        in_caption=caption,
        entries=entries,
        sigma_total=per_head,
        out_caption=("→ ", text_helpers.human_count(report.scorer.total)),
        width=width,
        tap=False,
        dashed=True,
    )


#### Generic model block (title -> caption -> op cards -> Σ -> bottom) ####


def _model_block(
    view: state.ConfiguratorState,
    *,
    section: str,
    title: str,
    in_caption: list[tuple[str, str]],
    entries: list[_OpEntry],
    sigma_total: int,
    out_caption: tuple[str, str] | None,
    width: int,
    tap: bool,
    dashed: bool,
) -> list[text.Text]:
    """One model / submodel box: its titled border, the input caption, a mini-box
    per layer operation, the parameter subtotal (plus an optional output-width
    caption), and the bottom border."""
    focused = _box_focused(view, section)
    border = _border_style(focused, _section_accent(section, dashed))
    mb_w = min(width - 4, _OP_CARD_MAX_W)
    rows = [_box_title(title, width, border, dashed)]
    rows.append(_content_row(width, border, in_caption, None, dashed))
    for entry in entries:
        for mini in _op_card_rows(entry, mb_w):
            rows.append(_wrap_in_block(width, border, mini, dashed))
    rows.append(_sigma_row(width, border, sigma_total, out_caption, dashed))
    rows.append(_box_bottom(width, border, tap=tap, dashed=dashed))
    return rows


#### Layer engine (transcribes model._build_body / _build_readout) ####


def _block_op_entries(
    view: state.ConfiguratorState,
    widths: tuple[int, ...],
    kind: _BlockKind,
    layer_params: tuple[architecture.LayerParam, ...],
    op_fields: dict[str, str],
    *,
    activation: str,
    dropout: float,
    layernorm: bool,
    final_activation: bool = False,
) -> list[_OpEntry]:
    """The flat, run-collapsed list of mini-box entries for a block: each layer
    expands to its ordered ops (Linear, optional LayerNorm, optional activation +
    Dropout), runs of identical layers fold to one ``×N`` group, and each op
    carries its parameter count and focus state. ``final_activation`` mirrors
    ``arch.encoder_final_activation`` for ``BODY_CHOICE`` blocks."""
    per_layer = [
        _layer_ops(
            width,
            layer_params[index],
            index == len(widths) - 1,
            kind,
            op_fields,
            view.selected_attr,
            layernorm=layernorm,
            dropout=dropout,
            activation=activation,
            final_activation=final_activation,
        )
        for index, width in enumerate(widths)
    ]
    return _collapse(per_layer, widths)


def _layer_ops(
    width: int,
    params: architecture.LayerParam,
    is_final: bool,
    kind: _BlockKind,
    op_fields: dict[str, str],
    selected_attr: str,
    *,
    layernorm: bool,
    dropout: float,
    activation: str,
    final_activation: bool = False,
) -> list[_OpEntry]:
    """The ordered op entries one layer expands to, following the model's per-block
    rules (LayerNorm after every body Linear; activation on every trunk layer and
    on every encoder layer when ``final_activation`` is True, otherwise only on
    non-final layers; readouts never on the final layer; dropout only where
    an activation is present)."""
    ops = [
        _OpEntry(
            kind=_OpKind.LINEAR,
            label=f"Linear →{width}",
            short=f"→{width}",
            param=params.linear,
        )
    ]
    if layernorm and kind in (_BlockKind.BODY_TRUNK, _BlockKind.BODY_CHOICE):
        ops.append(
            _OpEntry(
                kind=_OpKind.LAYERNORM, label="LayerNorm", short="LN", param=params.norm
            )
        )
    if _has_activation(kind, is_final, final_activation=final_activation):
        ops.append(
            _OpEntry(
                kind=_OpKind.ACTIVATION,
                label=activation,
                short=_SHORT_ACTIVATION.get(activation, activation),
            )
        )
        if dropout > 0.0:
            ops.append(
                _OpEntry(
                    kind=_OpKind.DROPOUT,
                    label=f"Dropout {_fmt_dropout(dropout)}",
                    short=f"d{_fmt_dropout(dropout)}",
                )
            )
    for entry in ops:
        entry.focused = selected_attr == op_fields.get(entry.kind.value)
    return ops


def _has_activation(
    kind: _BlockKind, is_final: bool, *, final_activation: bool = False
) -> bool:
    """Whether a layer carries an activation: the trunk keeps one on every layer;
    BODY_CHOICE blocks keep one on every layer when ``final_activation`` is True
    (``arch.encoder_final_activation``), otherwise only on non-final layers;
    readout heads always drop it on their final layer."""
    if kind is _BlockKind.BODY_TRUNK:
        return True
    if kind is _BlockKind.BODY_CHOICE and final_activation:
        return True
    return not is_final


def _collapse(
    per_layer: list[list[_OpEntry]], widths: tuple[int, ...]
) -> list[_OpEntry]:
    """Fold consecutive layers with identical width + op-label sequence into one
    group, stamping the run length onto each of that group's op entries."""
    out: list[_OpEntry] = []
    index = 0
    while index < len(per_layer):
        key = (widths[index], tuple(op.label for op in per_layer[index]))
        run_end = index + 1
        while run_end < len(per_layer):
            other = (widths[run_end], tuple(op.label for op in per_layer[run_end]))
            if other != key:
                break
            run_end += 1
        run = run_end - index
        for entry in per_layer[index]:
            out.append(entry.model_copy(update={"run": run}))
        index = run_end
    return out


#### Mini-box drawing (one bordered box per layer operation) ####


def _op_card_rows(entry: _OpEntry, width: int) -> list[text.Text]:
    """A layer operation as a 3-line bordered mini-box: a top border, the op label
    (full or short to fit, with a ``×N`` tag when collapsed), and a bottom border
    that inlays the parameter count for the parametric ops. Colored by op kind,
    brightened when the op's handle is focused."""
    color = theme.TEXT_BRIGHT if entry.focused else _OP_COLOR[entry.kind]
    style = f"bold {color}" if entry.focused else color
    inner = width - 4
    tag = f"×{entry.run}" if entry.run >= _COLLAPSE_RUN else ""
    available = max(0, inner - (len(tag) + 1 if tag else 0))
    label = entry.label if len(entry.label) <= available else entry.short
    label = label[:available]

    top = text.Text(_TL + _H * (width - 2) + _TR, style=style, no_wrap=True, end="")
    mid = text.Text(no_wrap=True, end="")
    mid.append(_V + " ", style=style)
    mid.append(label, style=style)
    mid.append(" " * max(0, inner - len(label) - len(tag)), style=style)
    if tag:
        mid.append(tag, style=theme.TEXT_MUTED)
    mid.append(" " + _V, style=style)
    return [top, mid, _op_card_bottom(width, style, entry.param)]


def _op_card_bottom(width: int, style: str, param: int | None) -> text.Text:
    """A mini-box bottom border, inlaying ``[count]`` flush-right for the
    parametric ops and a plain rule for the zero-parameter ops."""
    if param is None:
        return text.Text(
            _BL + _H * (width - 2) + _BR, style=style, no_wrap=True, end=""
        )
    tag = f"[{text_helpers.human_count(param)}]"
    fill = (width - 2) - len(tag) - 1
    line = text.Text(no_wrap=True, end="")
    line.append(_BL + _H * max(0, fill), style=style)
    line.append(tag, style=theme.TEXT_DIM2)
    line.append(_H + _BR, style=style)
    return line


#### Low-level box / row drawing ####


def _box_title(label: str, width: int, border: str, dashed: bool) -> text.Text:
    """A box's top border carrying its ``label`` (``┌─ STATE TRUNK ──┐``)."""
    horizontal = _DH if dashed else _H
    label = label[: max(0, width - 5)]
    title = text.Text(no_wrap=True, end="")
    title.append(_TL + horizontal + " ", style=border)
    title.append(f"{label} ", style=f"bold {border}")
    dashes = width - title.cell_len - 1
    title.append(horizontal * max(0, dashes) + _TR, style=border)
    return title


def _box_bottom(width: int, border: str, *, tap: bool, dashed: bool) -> text.Text:
    """A box's bottom border, with an optional centered ``┬`` tap into a
    connector."""
    horizontal = _DH if dashed else _H
    chars = [horizontal] * (width - 2)
    if tap and chars:
        chars[(width - 2) // 2] = _TAP_DOWN
    return text.Text(_BL + "".join(chars) + _BR, style=border, no_wrap=True, end="")


def _content_row(
    width: int,
    border: str,
    left: list[tuple[str, str]],
    right: list[tuple[str, str]] | None = None,
    dashed: bool = False,
) -> text.Text:
    """A bordered content row: ``│ <left> … <right> │``. The left is clipped to the
    inner width (so a long caption can never overflow a narrow column) and the
    right segments are right-aligned, dropped when they would not fit."""
    wall = _DV if dashed else _V
    inner = width - 4
    left, left_len = _clip_segments(left, inner)
    right_len = sum(len(segment) for segment, _ in (right or []))
    line = text.Text(no_wrap=True, end="")
    line.append(f"{wall} ", style=border)
    for segment, style in left:
        line.append(segment, style=style)
    if right and left_len + 1 + right_len <= inner:
        line.append(" " * (inner - left_len - right_len), style=border)
        for segment, style in right:
            line.append(segment, style=style)
    else:
        line.append(" " * max(0, inner - left_len), style=border)
    line.append(f" {wall}", style=border)
    return line


def _clip_segments(
    segments: list[tuple[str, str]], max_len: int
) -> tuple[list[tuple[str, str]], int]:
    """Truncate a left-segment list to ``max_len`` cells, splitting the segment
    that straddles the boundary."""
    out: list[tuple[str, str]] = []
    total = 0
    for segment, style in segments:
        if total >= max_len:
            break
        piece = segment[: max_len - total]
        out.append((piece, style))
        total += len(piece)
    return out, total


def _wrap_in_block(
    width: int, border: str, inner_text: text.Text, dashed: bool
) -> text.Text:
    """Embed a pre-built mini-box row as the content of a block row, padding it to
    the block's inner width: ``│ <mini-box> │``."""
    wall = _DV if dashed else _V
    line = text.Text(no_wrap=True, end="")
    line.append(f"{wall} ", style=border)
    line.append_text(inner_text)
    pad = (width - 4) - inner_text.cell_len
    if pad > 0:
        line.append(" " * pad)
    line.append(f" {wall}", style=border)
    return line


def _sigma_row(
    width: int,
    border: str,
    total: int,
    out_caption: tuple[str, str] | None,
    dashed: bool,
) -> text.Text:
    """A block's parameter-subtotal row (``Σ 573k``) with an optional right-aligned
    output-width caption (``M = 128``)."""
    left = [
        ("Σ ", theme.TEXT_MUTED),
        (text_helpers.human_count(total), theme.TEXT_PRIMARY),
    ]
    right = None
    if out_caption is not None:
        label, value = out_caption
        right = [(label, theme.TEXT_MUTED), (value, theme.CAUTION)]
    return _content_row(width, border, left, right, dashed)


#### Connector-row primitives ####


def _marks_row(width: int, marks: dict[int, str], style: str) -> text.Text:
    """A single-style row of spaces with ``marks`` (position -> glyph) placed."""
    chars = [" "] * width
    for index, glyph in marks.items():
        if 0 <= index < width:
            chars[index] = glyph
    return text.Text("".join(chars), style=style, no_wrap=True, end="")


def _label_arrow_row(
    width: int, left_center: int, right_center: int, left_label: str, right_label: str
) -> text.Text:
    """An arrows row: ``▼`` at each column center followed by a short label."""
    chars = [" "] * width
    _put(chars, left_center, _ARROW)
    _put_text(chars, left_center + 1, left_label)
    _put(chars, right_center, _ARROW)
    _put_text(chars, right_center + 1, right_label)
    return text.Text("".join(chars), style=theme.TEXT_MUTED, no_wrap=True, end="")


def _put(chars: list[str], index: int, glyph: str) -> None:
    if 0 <= index < len(chars):
        chars[index] = glyph


def _put_text(chars: list[str], start: int, value: str) -> None:
    for offset, char in enumerate(value):
        _put(chars, start + offset, char)


def _align_bottoms(rows: list[text.Text], height: int) -> list[text.Text]:
    """Pad a column to ``height`` by inserting blank rows just above its bottom
    border, so two unequal-height columns keep their bottoms on the same row."""
    missing = height - len(rows)
    if missing <= 0:
        return rows
    return rows[:-1] + [_blank() for _ in range(missing)] + rows[-1:]


def _blank() -> text.Text:
    return text.Text("", no_wrap=True, end="")


def _total_row(view: state.ConfiguratorState) -> text.Text:
    """The bottom total-parameter line (``TOTAL ≈ 618k params · setup 40k``)."""
    report = _param_report(view)
    line = text.Text(no_wrap=True, end="")
    line.append("TOTAL ", style=f"bold {theme.BORDER_HEADLINE}")
    line.append(
        f"≈ {text_helpers.human_count(report.total)}", style=f"bold {theme.TEXT_BRIGHT}"
    )
    line.append(" params", style=theme.TEXT_MUTED)
    if view.working.use_setup_model:
        setup = _setup_block(view)
        line.append(
            f"  · setup {text_helpers.human_count(setup.total)} (separate)",
            style=theme.TEXT_MUTED,
        )
    return line


#### Parameters + focus + formatting ####


def _param_report(view: state.ConfiguratorState) -> architecture.ParamReport:
    """The working config's main-net parameter accounting (per layer / block /
    total)."""
    cfg = view.working
    return architecture.count_parameters(
        cfg.arch,
        card_feat_in=encode.CARD_FEATURE_DIM,
        trunk_in=_trunk_in(cfg),
        choice_in=_choice_in(cfg),
        num_families=len(cfg.family_order),
        hand_feat_in=encode.HAND_ENCODER_INPUT_DIM,
    )


def _trunk_in(cfg: config.TrainConfig) -> int:
    """The working config's post-embedding trunk input width, every embedding
    knob threaded."""
    return encode.trunk_input_dim(
        cfg.state_dim,
        cfg.card_embed_dim,
        use_distinct_hand_model=cfg.use_distinct_hand_model,
        hand_embed_dim=cfg.hand_embed_dim,
        tray_set_embedding=cfg.tray_set_embedding,
    )


def _choice_in(cfg: config.TrainConfig | _StaticConfig) -> int:
    """The choice encoder's first-``Linear`` input width. The static adapter
    carries a precomputed value (era-routed by the caller through the
    descriptor seam — ``runmeta.choice_input_dim_for``); a live
    ``TrainConfig`` (the interactive configurator) is always the current
    era."""
    if isinstance(cfg, _StaticConfig):
        return cfg.choice_in
    return encode.choice_input_dim(
        cfg.choice_dim,
        cfg.card_embed_dim,
        include_setup=cfg.encoding_spec.include_setup,
    )


def _choice_extra(cfg: config.TrainConfig | _StaticConfig) -> int:
    """The choice encoder's passthrough "additional inputs" count — every card
    region (the candidate identity, the board-index block, and — when setup is
    in the main net — the kept-set multi-hot) is embedded, so the extra count
    excludes them all. Era-precomputed on the static adapter
    (``runmeta.choice_extra_for``)."""
    if isinstance(cfg, _StaticConfig):
        return cfg.choice_extra
    return encode.choice_passthrough_dim(
        cfg.choice_dim, include_setup=cfg.encoding_spec.include_setup
    )


def _setup_readout_in(cfg: config.TrainConfig) -> int:
    """The setup net's readout-MLP input width under the working main
    architecture (whose embedder copies size the embedded candidate)."""
    return setup_model.setup_readout_input_dim(setup_model.SETUP_FEATURE_DIM, cfg.arch)


def _setup_block(view: state.ConfiguratorState) -> architecture.BlockParam:
    """The separate setup net's parameter accounting (the frozen embedder copies
    are shaped by the working main architecture)."""
    return setup_model.count_setup_parameters(
        view.working.setup_arch,
        feature_dim=setup_model.SETUP_FEATURE_DIM,
        main_arch=view.working.arch,
    )


def _box_focused(view: state.ConfiguratorState, section: str) -> bool:
    """Whether the focused field is the one that owns ``section``'s box."""
    return view.selected_attr in _BOX_FOCUS_ATTRS.get(section, set())


def _section_accent(section: str, dashed: bool) -> str:
    """The unfocused border color for a structural box."""
    if section == "setup":
        return _SETUP_BORDER
    if section == "embed":
        return _CARD_BORDER
    if dashed:
        return _DECISION_BORDER
    return _BODY_BORDER


def _border_style(focused: bool, accent: str) -> str:
    """A box's border color: gold when focused, its section accent otherwise."""
    return theme.BORDER_HEADLINE if focused else accent


def _fmt_dropout(dropout: float) -> str:
    """Compact dropout probability (``0.15`` → ``.15``)."""
    return f"{dropout:g}".lstrip("0")


#### Static rendering (for wingspan-inspect, no focus state) ####


@dataclasses.dataclass
class _StaticConfig:
    """Minimal working-config adapter for focus-free diagram rendering.

    Exposes the attributes the diagram draw functions read from a real
    ``TrainConfig``, so :func:`render_static` can drive them without
    constructing a full training config. The choice-encoder widths are
    *precomputed* by the caller (era-routed through the descriptor seam)
    rather than derived from the live encoder — see :func:`_choice_in` /
    :func:`_choice_extra` — so an old run's diagram shows its own geometry."""

    state_dim: int
    choice_dim: int
    choice_in: int
    choice_extra: int
    card_embed_dim: int
    use_distinct_hand_model: bool
    hand_embed_dim: int | None
    tray_set_embedding: bool
    trunk_layers: architecture.Widths
    choice_layers: architecture.Widths
    head_layers: architecture.Widths
    value_layers: architecture.Widths
    card_encoder_layers: architecture.Widths
    activation: architecture.ActivationName
    layernorm: bool
    dropout: float
    encoder_final_activation: bool
    arch: architecture.ModelArchitecture
    family_order: tuple[str, ...]
    setup_arch: setup_model.SetupArchitecture
    setup_hidden_layers: architecture.Widths
    setup_activation: architecture.ActivationName
    setup_dropout: float
    use_setup_model: bool = False


@dataclasses.dataclass
class _StaticView:
    """Minimal view adapter for focus-free diagram rendering."""

    working: _StaticConfig
    selected_attr: str = ""


def render_static(
    arch: architecture.ModelArchitecture,
    state_dim: int,
    choice_dim: int,
    family_order: tuple[str, ...],
    *,
    choice_in: int,
    choice_extra: int,
    use_setup_model: bool = False,
    setup_arch: setup_model.SetupArchitecture | None = None,
    width: int = 48,
) -> list[text.Text]:
    """Render the architecture block diagram without interactive focus state.

    Returns the same box-and-arrow rows the FLIGHT PLAN ARCHITECTURE panel draws
    — ``EMBED → TRUNK → CHOICE → CONCAT → SCORER`` (with value-head tap) and the
    total-param line — as a plain list of Rich ``Text`` rows ready to print.
    ``choice_in`` / ``choice_extra`` are the choice encoder's input and
    passthrough widths, supplied by the caller era-routed through the
    descriptor seam (``runmeta.choice_input_dim_for`` /
    ``runmeta.choice_extra_for``) so an old run's diagram never assumes the
    live encoding. When ``use_setup_model`` the separate setup net is drawn
    from ``setup_arch`` (the default topology when ``None``). ``width`` is the
    box interior column budget (default 48).
    """
    resolved_setup_arch = (
        setup_arch if setup_arch is not None else setup_model.SetupArchitecture()
    )
    cfg = _StaticConfig(
        state_dim=state_dim,
        choice_dim=choice_dim,
        choice_in=choice_in,
        choice_extra=choice_extra,
        card_embed_dim=arch.card_embed_dim,
        use_distinct_hand_model=arch.use_distinct_hand_model,
        hand_embed_dim=arch.hand_embed_dim,
        tray_set_embedding=arch.tray_set_embedding,
        trunk_layers=arch.trunk_layers,
        choice_layers=arch.choice_layers,
        head_layers=arch.head_layers,
        value_layers=arch.value_layers,
        card_encoder_layers=arch.card_encoder_layers,
        activation=arch.activation,
        layernorm=arch.layernorm,
        dropout=arch.dropout,
        encoder_final_activation=arch.encoder_final_activation,
        arch=arch,
        family_order=family_order,
        setup_arch=resolved_setup_arch,
        setup_hidden_layers=resolved_setup_arch.hidden_layers,
        setup_activation=resolved_setup_arch.activation,
        setup_dropout=resolved_setup_arch.dropout,
        use_setup_model=use_setup_model,
    )
    view = _StaticView(working=cfg)
    rows, _ = _diagram_rows(typing.cast("state.ConfiguratorState", view), width)
    return rows
