"""The live ARCHITECTURE diagram for the FLIGHT PLAN configurator.

:class:`ArchitectureDiagram` is a width/height-aware ``rich`` renderable that
draws the working :class:`model.PolicyValueNet` topology as a single-column
box-and-arrow flow — ``EMBED → TRUNK → CHOICE → CONCAT → SCORER`` with the value
head tapped off the trunk's ``M`` output. The trunk ends at width ``M`` and the
choice encoder at ``N`` (independent); their outputs concatenate to ``M+N`` for
the scorer heads. It reads :class:`state.ConfiguratorState`
fresh each frame, so it reacts live as fields are edited: turning on dropout makes
a ``Dropout`` row appear in every affected block, toggling LayerNorm shows/hides
``LayerNorm`` rows, changing the activation relabels its rows, adding/removing a
layer adds/removes a box row, and the per-layer / per-block / total parameter
counts update with every change. The focused field highlights its block (gold
border) or, for the shared op handles, brightens the matching op rows everywhere.

The layer rows are transcribed from ``model._build_body`` / ``_build_readout``:
body blocks (trunk / choice) apply LayerNorm — when enabled — after every Linear;
the trunk keeps an activation on its final layer while the choice encoder does
not; readout heads (scorer / value) never LayerNorm and end in a bare
``Linear →1``. Parameter counts come from :func:`architecture.count_parameters`,
which is pinned to ``sum(p.numel())`` of the real net by a test.
"""

from __future__ import annotations

import enum
import typing

import pydantic
import rich.console as rich_console
from rich import text

from wingspan import architecture, encode
from wingspan.training import theme
from wingspan.training.charts import text_helpers

if typing.TYPE_CHECKING:
    from wingspan.training.configure import state

# Below this inner width the box chrome is dropped for a compact text fallback.
_MIN_BOX_WIDTH = 18
# A run of this many consecutive identical layers collapses to one ``×N`` group.
_COLLAPSE_RUN = 2

# Box-drawing + flow glyphs (thin rules; a single centered arrow per connector).
_TITLE_OPEN = "┌─ "
_TITLE_CLOSE = "┐"
_BOTTOM_OPEN = "└"
_BOTTOM_CLOSE = "┘"
_H_RULE = "─"
_WALL = "│"
_RAIL_MID = "├ "
_RAIL_END = "└ "
_ARROW = "▼"

# Clipped-viewport indicators (end="" so they replace a row without adding one).
_SCROLL_MORE_UP = "  ▲ more"
_SCROLL_MORE_DOWN = "  ▼ more"

# Which selected field lights up a whole BOX (gold border + title).
_BOX_FOCUS_ATTRS: dict[str, set[str]] = {
    "embed": {"card_embed_dim"},
    "trunk": {"trunk_layers"},
    "choice": {"choice_layers"},
    "scorer": {"head_layers"},
    "value": {"value_layers"},
}
# The shared op handles brighten their matching rows across every block instead.
_OP_FOCUS_ATTRS = {"activation", "dropout", "layernorm"}


class _OpKind(enum.StrEnum):
    """The four per-layer operations the diagram can draw as rail rows."""

    LINEAR = "linear"
    LAYERNORM = "layernorm"
    ACTIVATION = "activation"
    DROPOUT = "dropout"


class _BlockKind(enum.StrEnum):
    """Which model rule-set a block's layers follow (``model._build_body`` vs
    ``_build_readout``). Drives activation / LayerNorm applicability per layer."""

    BODY_TRUNK = "body_trunk"
    BODY_CHOICE = "body_choice"
    READOUT = "readout"


class _Op(pydantic.BaseModel):
    """One drawable layer operation: its display ``label`` and its ``kind`` (which
    selects the row color and the focus field, rather than sniffing the label)."""

    kind: _OpKind
    label: str


_OP_COLOR: dict[_OpKind, str] = {
    _OpKind.LINEAR: theme.TEXT_PRIMARY,
    _OpKind.LAYERNORM: theme.GAUGE_UTIL,
    _OpKind.ACTIVATION: theme.TEXT_DIM2,
    _OpKind.DROPOUT: theme.CAUTION,
}
_OP_FIELD: dict[_OpKind, str] = {
    _OpKind.ACTIVATION: "activation",
    _OpKind.DROPOUT: "dropout",
    _OpKind.LAYERNORM: "layernorm",
}


def viewport(
    rows: list[text.Text], selected_row: int, height: int
) -> tuple[list[text.Text], bool, bool]:
    """The slice of ``rows`` to show, keeping ``selected_row`` visible, plus
    whether rows are clipped above / below. (Relocated from ``screen._viewport``;
    shared by the form list and this diagram.)"""
    if height <= 0 or len(rows) <= height:
        return list(rows), False, False
    first = min(max(selected_row - height // 2, 0), len(rows) - height)
    window = list(rows[first : first + height])
    return window, first > 0, first + height < len(rows)


class ArchitectureDiagram:
    """The ARCHITECTURE panel body: the working network as a box-and-arrow flow,
    scrolled to keep the focused block visible and degraded to a compact text
    list when the panel is too narrow for boxes."""

    def __init__(self, view: state.ConfiguratorState):
        self.view = view

    def __rich_console__(
        self, console: rich_console.Console, options: rich_console.ConsoleOptions
    ) -> rich_console.RenderResult:
        content_w = options.max_width
        if content_w < _MIN_BOX_WIDTH:
            rows, selected_row = _compact_rows(self.view, content_w)
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
    """The full ordered row list — every block, the connectors between them, the
    value-head tap, and the total — plus the row to scroll toward for focus."""
    rows: list[text.Text] = []
    block_start: dict[str, int] = {}

    def add_block(section: str, block_rows: list[text.Text]) -> None:
        block_start[section] = len(rows)
        rows.extend(block_rows)

    add_block("embed", _embed_box(view, content_w))
    rows.extend(_connector(content_w))
    add_block("trunk", _trunk_box(view, content_w))
    rows.extend(_connector(content_w))
    add_block("choice", _choice_box(view, content_w))
    rows.extend(_connector(content_w))
    add_block("concat", _concat_box(view, content_w))
    rows.extend(_connector(content_w))
    add_block("scorer", _scorer_box(view, content_w))
    rows.append(_value_divider(content_w))
    add_block("value", _value_box(view, content_w))
    rows.append(text.Text("", end=""))
    rows.append(_total_row(view, content_w))
    return rows, _anchor(view, block_start)


def _anchor(view: state.ConfiguratorState, block_start: dict[str, int]) -> int:
    """The row index the viewport centers on: the focused block's title, or the
    trunk (the first block with op rows) when a shared op handle is focused."""
    attr = view.selected_attr
    for section, attrs in _BOX_FOCUS_ATTRS.items():
        if attr in attrs:
            return block_start.get(section, 0)
    if attr in _OP_FOCUS_ATTRS:
        return block_start.get("trunk", 0)
    return 0


def _compact_rows(
    view: state.ConfiguratorState, width: int
) -> tuple[list[text.Text], int]:
    """Sub-``_MIN_BOX_WIDTH`` fallback: one labeled width-chain line per block
    (with ``+LN`` / ``+drop`` tags) plus the total — still live, just chromeless."""
    cfg = view.working
    extras: list[str] = []
    if cfg.layernorm:
        extras.append("+LN")
    if cfg.dropout > 0.0:
        extras.append(f"+d{_fmt_dropout(cfg.dropout)}")
    tags = ("  " + " ".join(extras)) if extras else ""
    trunk_in = encode.trunk_input_dim(cfg.state_dim, cfg.card_embed_dim)
    choice_in = encode.choice_input_dim(cfg.choice_dim, cfg.card_embed_dim)
    rows = [
        _compact_line("EMBED", f"{encode.HAND_MULTIHOT_DIM + 1}×{cfg.card_embed_dim}"),
        _compact_line("TRUNK", _chain(trunk_in, cfg.trunk_layers), tags),
        _compact_line("CHOICE", _chain(choice_in, cfg.choice_layers), tags),
        _compact_line("SCORER", f"M+N→1 ×{len(cfg.family_order)}"),
        _compact_line("VALUE", "M→1"),
        _total_row(view, width),
    ]
    return rows, 0


def _compact_line(label: str, chain: str, tags: str = "") -> text.Text:
    """One ``LABEL  chain  +tags`` row for the narrow fallback (the ``+LN`` /
    ``+drop`` tags appear / disappear with the config, just like the boxes)."""
    line = text.Text(no_wrap=True, end="")
    line.append(f"{label:<7}", style=f"bold {theme.TEXT_MUTED}")
    line.append(chain, style=theme.TEXT_PRIMARY)
    if tags:
        line.append(tags, style=theme.CAUTION)
    return line


def _chain(in_dim: int, widths: tuple[int, ...]) -> str:
    """A ``2381→128→128`` input-to-output width string."""
    return "→".join(str(value) for value in (in_dim, *widths))


#### Block builders ####


def _embed_box(view: state.ConfiguratorState, content_w: int) -> list[text.Text]:
    """The shared card-embedding table: ``N birds × D`` and its parameter count."""
    report = _param_report(view)
    focused = _box_focused(view, "embed")
    border = _border_style(focused)
    n_birds = encode.HAND_MULTIHOT_DIM + 1
    left = [
        (f"{n_birds} birds", theme.TEXT_PRIMARY),
        (f" ×{view.working.card_embed_dim}", theme.TEXT_DIM2),
    ]
    right = [(text_helpers.human_count(report.embed.total), theme.TEXT_DIM2)]
    return [
        _box_title("EMBED", content_w, focused),
        _content_row(content_w, border, left, right),
        _box_bottom(content_w, focused),
    ]


def _trunk_box(view: state.ConfiguratorState, content_w: int) -> list[text.Text]:
    """The state trunk. Its ``in`` is the post-embedding width (the card-index
    block swapped for ``card_embed_dim`` vectors), so it matches the first Linear's
    parameter count and grows with ``card_embed_dim``."""
    cfg = view.working
    trunk_in = encode.trunk_input_dim(cfg.state_dim, cfg.card_embed_dim)
    return _body_box(
        view,
        "trunk",
        "TRUNK · state",
        trunk_in,
        cfg.trunk_layers,
        _BlockKind.BODY_TRUNK,
        _param_report(view).trunk,
        "M",
        content_w,
    )


def _choice_box(view: state.ConfiguratorState, content_w: int) -> list[text.Text]:
    """The per-choice encoder (final layer is a bare ``Linear``, no activation).
    Its ``in`` is the post-embedding width — the bird one-hot swapped for one
    ``card_embed_dim`` vector — to stay consistent with the first Linear's count."""
    cfg = view.working
    choice_in = encode.choice_input_dim(cfg.choice_dim, cfg.card_embed_dim)
    return _body_box(
        view,
        "choice",
        "CHOICE · cand",
        choice_in,
        cfg.choice_layers,
        _BlockKind.BODY_CHOICE,
        _param_report(view).choice,
        "N",
        content_w,
    )


def _concat_box(view: state.ConfiguratorState, content_w: int) -> list[text.Text]:
    """The trunk/choice merge into the ``M+N`` vector each scorer reads."""
    border = _border_style(False)
    arch = view.working.arch
    concat = arch.trunk_embed_width + arch.choice_embed_width
    return [
        _box_title("CONCAT", content_w, False),
        _content_row(content_w, border, [("trunk M + choice N", theme.TEXT_MUTED)]),
        _content_row(
            content_w,
            border,
            [("M+N = ", theme.TEXT_MUTED), (str(concat), theme.CAUTION)],
        ),
        _box_bottom(content_w, False),
    ]


def _scorer_box(view: state.ConfiguratorState, content_w: int) -> list[text.Text]:
    """The per-family scorer bank: ``M+N → head_layers → 1``, one head per family."""
    cfg = view.working
    concat = cfg.arch.trunk_embed_width + cfg.arch.choice_embed_width
    return _readout_box(
        view,
        "scorer",
        f"SCORER ×{len(cfg.family_order)}",
        f"in M+N = {concat}",
        cfg.head_layers,
        _param_report(view).scorer,
        content_w,
    )


def _value_box(view: state.ConfiguratorState, content_w: int) -> list[text.Text]:
    """The value head: ``M → value_layers → 1`` (reads the trunk ``M`` alone, not
    the ``M+N`` concat)."""
    cfg = view.working
    trunk_m = cfg.arch.trunk_embed_width
    return _readout_box(
        view,
        "value",
        "VALUE",
        f"in M = {trunk_m}",
        cfg.value_layers,
        _param_report(view).value,
        content_w,
    )


def _body_box(
    view: state.ConfiguratorState,
    section: str,
    label: str,
    in_dim: int,
    widths: tuple[int, ...],
    kind: _BlockKind,
    block: architecture.BlockParam,
    out_label: str,
    content_w: int,
) -> list[text.Text]:
    """A body block (trunk / choice): title, ``in`` caption, per-layer rows, the
    block parameter subtotal, the output-width line (``M`` for the trunk, ``N`` for
    the choice encoder), and the bottom border."""
    focused = _box_focused(view, section)
    border = _border_style(focused)
    rows = [_box_title(label, content_w, focused)]
    rows.append(_content_row(content_w, border, [(f"in {in_dim}", theme.TEXT_DIM2)]))
    rows.extend(_layer_rows(view, widths, kind, block.layers, content_w, border))
    rows.append(_sigma_row(block.total, content_w, border))
    rows.append(
        _content_row(
            content_w, border, [(f"{out_label} = {widths[-1]}", theme.CAUTION)]
        )
    )
    rows.append(_box_bottom(content_w, focused))
    return rows


def _readout_box(
    view: state.ConfiguratorState,
    section: str,
    label: str,
    in_caption: str,
    hidden: tuple[int, ...],
    block: architecture.BlockParam,
    content_w: int,
) -> list[text.Text]:
    """A readout block (scorer / value head): the hidden widths plus a final
    ``Linear →1`` row, with the block parameter subtotal."""
    focused = _box_focused(view, section)
    border = _border_style(focused)
    render_widths = (*hidden, 1)
    rows = [_box_title(label, content_w, focused)]
    rows.append(_content_row(content_w, border, [(in_caption, theme.TEXT_DIM2)]))
    rows.extend(
        _layer_rows(
            view, render_widths, _BlockKind.READOUT, block.layers, content_w, border
        )
    )
    rows.append(_sigma_row(block.total, content_w, border))
    rows.append(_box_bottom(content_w, focused))
    return rows


#### Layer engine (transcribes model._build_body / _build_readout) ####


def _layer_rows(
    view: state.ConfiguratorState,
    widths: tuple[int, ...],
    kind: _BlockKind,
    layer_params: tuple[architecture.LayerParam, ...],
    content_w: int,
    border: str,
) -> list[text.Text]:
    """Tree-railed op rows for a block, collapsing runs of identical layers to a
    single ``×N`` group. Each ``Linear`` row carries its parameter count; the very
    last op row of the block uses the ``└`` rail."""
    cfg = view.working
    activation = str(cfg.activation)
    per_layer = [
        (
            _ops_for_layer(
                width,
                index == len(widths) - 1,
                kind,
                layernorm=cfg.layernorm,
                dropout=cfg.dropout,
                activation=activation,
            ),
            layer_params[index].linear,
        )
        for index, width in enumerate(widths)
    ]
    entries = _flatten_groups(_collapse_layers(per_layer, widths))
    rows: list[text.Text] = []
    for index, (op, tag, param) in enumerate(entries):
        rows.append(
            _rail_row(
                view, op, index == len(entries) - 1, tag, param, content_w, border
            )
        )
    return rows


def _ops_for_layer(
    width: int,
    is_final: bool,
    kind: _BlockKind,
    *,
    layernorm: bool,
    dropout: float,
    activation: str,
) -> list[_Op]:
    """The ordered ops one layer expands to, following the model's per-block rules:
    LayerNorm after every body Linear (never in readouts); activation on every
    trunk layer but only the non-final layer of the choice encoder / readouts;
    dropout only where an activation is present."""
    ops = [_Op(kind=_OpKind.LINEAR, label=f"Linear →{width}")]
    if layernorm and kind in (_BlockKind.BODY_TRUNK, _BlockKind.BODY_CHOICE):
        ops.append(_Op(kind=_OpKind.LAYERNORM, label="LayerNorm"))
    if _has_activation(kind, is_final):
        ops.append(_Op(kind=_OpKind.ACTIVATION, label=activation))
        if dropout > 0.0:
            ops.append(
                _Op(kind=_OpKind.DROPOUT, label=f"Dropout {_fmt_dropout(dropout)}")
            )
    return ops


def _has_activation(kind: _BlockKind, is_final: bool) -> bool:
    """Whether a layer carries an activation: the trunk keeps one on every layer;
    the choice encoder and readout heads drop it on their final layer."""
    if kind is _BlockKind.BODY_TRUNK:
        return True
    return not is_final


def _collapse_layers(
    per_layer: list[tuple[list[_Op], int]], widths: tuple[int, ...]
) -> list[tuple[list[_Op], int, int]]:
    """Group consecutive layers with identical width and op sequence, returning
    ``(ops, linear_params, run_length)`` per group."""
    groups: list[tuple[list[_Op], int, int]] = []
    index = 0
    while index < len(per_layer):
        ops, linear = per_layer[index]
        key = (widths[index], tuple(op.label for op in ops))
        run_end = index + 1
        while run_end < len(per_layer):
            next_ops, _ = per_layer[run_end]
            if (widths[run_end], tuple(op.label for op in next_ops)) != key:
                break
            run_end += 1
        groups.append((ops, linear, run_end - index))
        index = run_end
    return groups


def _flatten_groups(
    groups: list[tuple[list[_Op], int, int]],
) -> list[tuple[_Op, str, str]]:
    """Flatten collapsed groups into ``(op, ×N tag, param string)`` entries — the
    ``×N`` tag rides the group's last op row, the count rides each ``Linear``."""
    entries: list[tuple[_Op, str, str]] = []
    for ops, linear, run in groups:
        for op_index, op in enumerate(ops):
            tag = f"×{run}" if run >= _COLLAPSE_RUN and op_index == len(ops) - 1 else ""
            param = (
                text_helpers.human_count(linear) if op.kind is _OpKind.LINEAR else ""
            )
            entries.append((op, tag, param))
    return entries


#### Low-level drawing ####


def _rail_row(
    view: state.ConfiguratorState,
    op: _Op,
    is_last: bool,
    tag: str,
    param: str,
    content_w: int,
    border: str,
) -> text.Text:
    """One tree-rail op row inside a box: ``├ Linear →128   557k`` (``└`` if it is
    the block's last op). The op label brightens when its field is focused."""
    rail = _RAIL_END if is_last else _RAIL_MID
    left = [
        (rail, border),
        (op.label, _op_style(op.kind, _op_is_focused(view, op.kind))),
    ]
    right: list[tuple[str, str]] = []
    if param:
        right.append((param, theme.TEXT_DIM2))
    if tag:
        right.append((f" {tag}" if right else tag, theme.TEXT_MUTED))
    return _content_row(content_w, border, left, right)


def _sigma_row(total: int, content_w: int, border: str) -> text.Text:
    """A block's parameter-subtotal row (``Σ 573k``)."""
    left = [
        ("Σ ", theme.TEXT_MUTED),
        (text_helpers.human_count(total), theme.TEXT_PRIMARY),
    ]
    return _content_row(content_w, border, left)


def _content_row(
    content_w: int,
    border: str,
    left: list[tuple[str, str]],
    right: list[tuple[str, str]] | None = None,
) -> text.Text:
    """A bordered content row: ``│ <left> … <right> │``. The right segments are
    right-aligned and silently dropped when they would not fit the inner width."""
    inner = content_w - 4
    left_len = sum(len(segment) for segment, _ in left)
    right_len = sum(len(segment) for segment, _ in (right or []))
    line = text.Text(no_wrap=True, end="")
    line.append(f"{_WALL} ", style=border)
    for segment, style in left:
        line.append(segment, style=style)
    if right and left_len + 1 + right_len <= inner:
        line.append(" " * (inner - left_len - right_len), style=border)
        for segment, style in right:
            line.append(segment, style=style)
    else:
        line.append(" " * max(0, inner - left_len), style=border)
    line.append(f" {_WALL}", style=border)
    return line


def _box_title(label: str, content_w: int, focused: bool) -> text.Text:
    """A box's top border carrying its ``label`` (``┌─ TRUNK · state ──┐``)."""
    border = _border_style(focused)
    label = label[: max(0, content_w - 5)]
    title = text.Text(no_wrap=True, end="")
    title.append(_TITLE_OPEN, style=border)
    title.append(f"{label} ", style=f"bold {border}")
    dashes = content_w - title.cell_len - 1
    title.append(_H_RULE * max(0, dashes) + _TITLE_CLOSE, style=border)
    return title


def _box_bottom(content_w: int, focused: bool) -> text.Text:
    """A box's bottom border."""
    border = _border_style(focused)
    return text.Text(
        _BOTTOM_OPEN + _H_RULE * max(0, content_w - 2) + _BOTTOM_CLOSE,
        style=border,
        no_wrap=True,
        end="",
    )


def _connector(content_w: int) -> list[text.Text]:
    """A single centered ``▼`` between two stacked boxes."""
    pad = max(0, (content_w - 1) // 2)
    return [text.Text(" " * pad + _ARROW, style=theme.TEXT_MUTED, no_wrap=True, end="")]


def _value_divider(content_w: int) -> text.Text:
    """The labeled tap announcing the value head branches off the trunk ``M``."""
    return text.Text(
        "  value head ┄ off trunk M", style=theme.TEXT_MUTED, no_wrap=True, end=""
    )


def _total_row(view: state.ConfiguratorState, content_w: int) -> text.Text:
    """The bottom total-parameter line (``TOTAL ≈ 618k params``)."""
    report = _param_report(view)
    line = text.Text(no_wrap=True, end="")
    line.append("TOTAL ", style=f"bold {theme.BORDER_HEADLINE}")
    line.append(
        f"≈ {text_helpers.human_count(report.total)}", style=f"bold {theme.TEXT_BRIGHT}"
    )
    line.append(" params", style=theme.TEXT_MUTED)
    return line


#### Parameters + focus + formatting ####


def _param_report(view: state.ConfiguratorState) -> architecture.ParamReport:
    """The working config's parameter accounting (per layer / block / total)."""
    cfg = view.working
    return architecture.count_parameters(
        cfg.arch,
        trunk_in=encode.trunk_input_dim(cfg.state_dim, cfg.card_embed_dim),
        choice_in=encode.choice_input_dim(cfg.choice_dim, cfg.card_embed_dim),
        embed_rows=encode.HAND_MULTIHOT_DIM + 1,
        num_families=len(cfg.family_order),
    )


def _box_focused(view: state.ConfiguratorState, section: str) -> bool:
    """Whether the focused field is the one that owns ``section``'s box."""
    return view.selected_attr in _BOX_FOCUS_ATTRS.get(section, set())


def _op_is_focused(view: state.ConfiguratorState, kind: _OpKind) -> bool:
    """Whether the focused field is the shared handle behind op ``kind``."""
    return view.selected_attr == _OP_FIELD.get(kind)


def _op_style(kind: _OpKind, focused: bool) -> str:
    """The style for an op label — brightened when its field is focused."""
    if focused:
        return f"bold {theme.TEXT_BRIGHT}"
    return _OP_COLOR[kind]


def _border_style(focused: bool) -> str:
    """A box's border color: gold when focused, the default rule otherwise."""
    return theme.BORDER_HEADLINE if focused else theme.BORDER_DEFAULT


def _fmt_dropout(dropout: float) -> str:
    """Compact dropout probability (``0.15`` → ``.15``)."""
    return f"{dropout:g}".lstrip("0")
