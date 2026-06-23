"""SVG architecture diagram for the model-summary HTML report.

Builds the self-contained ``<svg>`` drawing of the full network topology that
``wingspan.reporting.html`` embeds in its Architecture section: the single-card and
multi-card encoders on the top row, the state encoder / choice encoder / setup
model on the middle row, and the value / decision heads on the bottom row,
joined by fan-out connectors labelled with how many copies of each encoder's
output the downstream input consumes (e.g. ×33 card embeddings in the state
input — 30 board slots + 3 tray slots).  Every block carries tinted
input/output boxes (descriptive name · element count), centered layer rows,
and exact bare-integer parameter counts overlaid on the borders.

The diagram doubles as the report's navigation: every real input box is wrapped
in an ``arch-click`` group whose ``data-panel`` names the report section it
reveals (the :data:`PANEL_*` contract), and every parameter count in an
``arch-paramclick`` group whose ``data-params-block`` names the parameter-table
block it jumps to — ``wingspan.reporting.html``'s inline script and CSS supply the
behaviour and affordances.

The diagram is data-driven: layer shapes come from the
:class:`wingspan.architecture.ParamReport` (with the hand encoder's shapes
recomputed via :func:`wingspan.architecture.body_layers` when the main net
mean-pools the hand instead), copy counts come from the ``wingspan.encode``
layout constants, and the separate setup net is drawn from its
:class:`~wingspan.architecture.BlockParam` — so the picture stays correct for
any configuration.

The public entry point is :func:`build_arch_svg`.
"""

from __future__ import annotations

import enum
import html as html_lib

import pydantic

from wingspan import architecture, encode, setup_model, state

# ---------------------------------------------------------------------------
# Click contract with ``wingspan.reporting.html``: each clickable input box carries one
# of these panel ids as its ``data-panel`` attribute, and the report gives the
# matching detail section the same id. Parameter counts carry a
# ``data-params-block`` key equal to ``BlockParam.label.lower()`` (the anchor
# suffix of the parameter table's per-block rows), or ``PARAMS_BLOCK_TOTAL``
# for counts with no block rows of their own (the grand total, the separate
# setup net, the mean-pooled hand encoder).

PANEL_CARD = "card"
PANEL_HAND = "hand"
PANEL_STATE = "state"
PANEL_CHOICE = "choice"
PANEL_SETUP = "setup"
PANEL_PARAMS = "params"
PARAMS_BLOCK_TOTAL = "total"

# ---------------------------------------------------------------------------
# Palette.

_SVG_BG = "#f1f5f9"
_SVG_BLOCK_FILL = "#ffffff"
_SVG_BLOCK_STROKE = "#e2e8f0"
_SVG_ARROW = "#94a3b8"
_SVG_TEXT_TITLE = "#1e293b"
_SVG_TEXT_DIM = "#64748b"
_SVG_LINEAR_COLOR = "#3b82f6"
_SVG_ACT_COLOR = "#22c55e"
_SVG_IO_FILL = "#eef2ff"
_SVG_IO_STROKE = "#c7d2fe"
_SVG_IO_TEXT = "#4338ca"
_SVG_TOTAL_COLOR = "#a855f7"

_ACCENT_CARD = "#a855f7"
_ACCENT_HAND = "#d946ef"
_ACCENT_TRUNK = "#3b82f6"
_ACCENT_CHOICE = "#0ea5e9"
_ACCENT_SETUP = "#14b8a6"
_ACCENT_VALUE = "#10b981"
_ACCENT_DECISION = "#f97316"
_ACCENT_DECISION_BADGE_BG = "#fde8d8"
_ACCENT_ATTN = "#f59e0b"

_FONT_MONO = "'Courier New',monospace"
_FONT_SANS = "-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif"

# ---------------------------------------------------------------------------
# Geometry: a 960-wide canvas with three 290-wide columns.  Row 1 holds the
# two shared encoders, row 2 their three consumers, row 3 the output heads;
# the 64px bands between rows carry the elbow connectors on horizontal lanes.

_SVG_W = 960
_SVG_TOP = 22
_SVG_BOTTOM = 18
_SVG_COL_W = 290
_SVG_COL_X: tuple[int, int, int] = (20, 334, 648)
_SVG_COL_CX: tuple[int, int, int] = (165, 479, 793)
_SVG_BAND_H = 64
_SVG_TOTAL_GAP = 30

_SVG_ACCENT_W = 4  # left-border accent bar width
_SVG_RX_BLK = 8  # block corner radius
_SVG_RX_ROW = 4  # mini-row corner radius
_SVG_RX_IO = 5  # input/output box corner radius
_SVG_ROW_H = 28  # mini-row height
_SVG_ROW_STRIDE = 33  # mini-row height + 5px gap
_SVG_IO_H = 26  # input/output box height
_SVG_IO_GAP = 8  # gap between an I/O box and the block border
_SVG_IO_INSET = 24  # I/O box horizontal inset from the block edges

# Block internal layout (relative to the block top): PAD_T | title + optional
# subtitle (HDR_H) | HDR_GAP | rows (ROW_STRIDE each) | PAD_B.
_SVG_BLK_PAD_T = 12
_SVG_BLK_HDR_H = 34
_SVG_BLK_HDR_GAP = 6
_SVG_BLK_PAD_B = 14

# Connector stroke widths scale with how many copies of the source output the
# destination input consumes (the "this output is duplicated" cue).
_STROKE_SINGLE = 1.5
_STROKE_FEW = 2.5
_STROKE_MANY = 4.0
_FEW_COPIES = 2
_MANY_COPIES = 4

# Band-lane y offsets (relative to a band's top) and per-connector attach
# offsets (relative to a column center), chosen so no two vertical connector
# segments share an x where their y-ranges overlap and no two labels collide.
_LANE_HAND = 16
_LANE_CARD_SETUP = 32
_LANE_CARD_CHOICE = 48
_LANE_TRUNK_DECISION = 24

# Lanes relative to the top of the attention band (used when use_board_attention=True).
# Hand→trunk elbow runs below the attention block; tray-3 runs in the col0/col1 gutter.
_LANE_ATTN_HAND_TRUNK = 16
_LANE_ATTN_TRAY = 40

# Gutter x for the tray-3 connector when routing around the col-0 attention block.
_X_ATTN_TRAY_GUTTER = 315

_X_CARD_TRUNK = -40
_X_CARD_CHOICE_SRC = 10
_X_CARD_CHOICE_DST = -20
_X_CARD_SETUP_SRC = 70
_X_CARD_SETUP_DST = -40
_X_HAND_TRUNK_SRC = -40
_X_HAND_TRUNK_DST = 40
_X_HAND_SETUP_SRC = 30
_X_TRUNK_DECISION_SRC = 55
_X_TRUNK_DECISION_DST = -30

# Dual-mode (actor-critic) setup column geometry: the 290px column is split
# into a shared header block at full width plus two 135px sub-blocks with a
# 10px gap between them and 20px of vertical space for the Y-split arrow.
_DUAL_SPLIT_GAP = 20
_DUAL_SUB_W = 135
_DUAL_SUB_GAP = 10


def build_arch_svg(
    arch: architecture.ModelArchitecture,
    param_report: architecture.ParamReport,
    family_order: tuple[str, ...],
    *,
    setup_param: architecture.BlockParam,
    setup_arch: setup_model.SetupArchitecture,
    use_setup_model: bool,
) -> str:
    """Return a self-contained ``<svg>`` string for the architecture diagram.

    The separately-trained setup model is drawn as a third column connected to
    the shared encoders (the copies it carries are frozen syncs of the main
    net's).  It is drawn even when ``use_setup_model`` is False: dashed, with
    an "off" subtitle, so the diagram always shows what the setup net would
    look like.  The hand encoder is likewise always drawn — dashed when the
    main net mean-pools the hand through the card table instead.
    """
    # Assemble the seven blocks, then resolve the row layout from row counts.
    units = _build_units(
        arch,
        param_report,
        family_order,
        setup_param=setup_param,
        setup_arch=setup_arch,
        use_setup_model=use_setup_model,
    )
    geom = _resolve_geometry(units)

    # Root + the three block rows.
    parts = [
        _svg_root(
            geom, arch, param_report, setup_param, len(family_order), use_setup_model
        )
    ]
    placed = (
        (units.card, geom.row1_y, geom.row1_h),
        (units.hand, geom.row1_y, geom.row1_h),
        (units.trunk, geom.row2_y, geom.row2_h),
        (units.choice, geom.row2_y, geom.row2_h),
        (units.setup, geom.row2_y, geom.row2_h),
        (units.value, geom.row3_y, geom.row3_h),
        (units.decision, geom.row3_y, geom.row3_h),
    )
    for unit, top_y, unit_h in placed:
        parts.append(_draw_unit(unit, top_y, unit_h))

    # Attention row sits between row1 and row2 (only when on).
    if units.attention is not None:
        assert geom.attn_row_y is not None and geom.attn_row_h is not None
        parts.append(_draw_unit(units.attention, geom.attn_row_y, geom.attn_row_h))

    # The top-row training note: the shared encoders learn only from in-game
    # decisions; the setup net consumes them as frozen, synced copies. Gated on
    # the distinct hand model — without it the setup net trains its own
    # multi-card encoder, so the blanket note would be wrong.
    if arch.use_distinct_hand_model:
        parts.append(_row1_side_note(geom))

    # Connectors: all bodies first, then all labels, so the white label halos
    # mask any line they cross.
    if units.attention is not None:
        conns = _band1_connectors_attn(geom, arch, use_setup_model) + _band2_connectors(
            geom, arch
        )
    else:
        conns = _band1_connectors(geom, arch, use_setup_model) + _band2_connectors(
            geom, arch
        )
    rendered = [_conn_svg(conn) for conn in conns]
    parts.extend(body for body, _ in rendered)
    parts.extend(label for _, label in rendered if label)

    parts.append(_total_line(geom, param_report, setup_param, use_setup_model))
    parts.append("</svg>")
    return "\n".join(parts)


###### PRIVATE #######

#### Value objects ####


class _OpKind(enum.StrEnum):
    """The two mini-row styles inside a block."""

    LINEAR = "linear"
    ACT = "act"


class _OpRow(pydantic.BaseModel):
    """One mini-row inside a block: a Linear (with its parameter count) or an
    activation."""

    kind: _OpKind
    label: str
    params: int | None = None


class _Unit(pydantic.BaseModel):
    """One drawable block with its input/output boxes, rows, and annotations."""

    x: int
    accent: str
    title: str
    subtitle: str = ""
    rows: tuple[_OpRow, ...]
    # The parameter-total legend overlaid on the bottom-right border: a bare
    # integer count, optionally suffixed ("… each" / "… total").
    sigma_text: str
    in_label: str
    in_count: int
    out_label: str
    out_count: int
    tooltip: str
    dashed: bool = False
    stack: int = 0
    # Click contract (see module constants): the report panel the input box
    # opens (None for the heads, whose inputs are intermediate embeddings) and
    # the parameter-table block the unit's parameter counts jump to.
    panel: str | None = None
    params_key: str | None = None
    # When set, this unit is rendered as a header-only input box that fans out
    # to two side-by-side sub-units (actor-critic setup mode).  The outer unit
    # provides the shared embedder input box; ``dual`` provides the two heads.
    dual: tuple["_Unit", "_Unit"] | None = None


class _Units(pydantic.BaseModel):
    """The eight diagram blocks, named by role (attention is None when off)."""

    card: _Unit
    hand: _Unit
    trunk: _Unit
    choice: _Unit
    setup: _Unit
    value: _Unit
    decision: _Unit
    attention: _Unit | None = None


class _Conn(pydantic.BaseModel):
    """One connector: a straight vertical (``lane_y is None``) or an orthogonal
    elbow routed along a horizontal lane.  ``copies`` drives the stroke width;
    ``label2`` is an optional second label line (used by the card→trunk
    fan-out's breakdown); ``label_left`` puts a straight vertical's labels
    right-aligned to the line's left instead of the default right side (used
    where a long label would overflow onto a neighbouring connector)."""

    src_x: int
    src_y: int
    dst_x: int
    dst_y: int
    lane_y: int | None = None
    copies: int = 1
    label: str = ""
    label2: str = ""
    label_dx: int = 0
    label_left: bool = False
    dashed: bool = False


class _Geom(pydantic.BaseModel):
    """The resolved vertical layout: row tops/heights, band tops, total-line y."""

    row1_y: int
    row1_h: int
    row2_y: int
    row2_h: int
    row3_y: int
    row3_h: int
    band1_y: int
    band2_y: int
    total_y: int
    svg_h: int
    # Present only when use_board_attention=True; None otherwise.
    attn_row_y: int | None = None
    attn_row_h: int | None = None
    band_attn_y: int | None = None


#### Unit assembly ####


def _build_units(
    arch: architecture.ModelArchitecture,
    param_report: architecture.ParamReport,
    family_order: tuple[str, ...],
    *,
    setup_param: architecture.BlockParam,
    setup_arch: setup_model.SetupArchitecture,
    use_setup_model: bool,
) -> _Units:
    attn_block = param_report.board_attention
    return _Units(
        card=_card_unit(arch, param_report),
        hand=_hand_unit(arch, param_report),
        trunk=_trunk_unit(arch, param_report),
        choice=_choice_unit(arch, param_report),
        setup=_build_setup_unit(setup_param, setup_arch, use_setup_model),
        value=_value_unit(arch, param_report),
        decision=_decision_unit(arch, param_report, family_order),
        attention=_attention_unit(arch, attn_block) if attn_block is not None else None,
    )


def _card_unit(
    arch: architecture.ModelArchitecture, param_report: architecture.ParamReport
) -> _Unit:
    block = param_report.embed
    in_dim = block.layers[0].in_features
    return _Unit(
        x=_SVG_COL_X[0],
        accent=_ACCENT_CARD,
        title="SINGLE-CARD ENCODER · per-card MLP",
        rows=_op_rows(
            block.layers,
            arch.card_activation_resolved.value,
            is_trunk=arch.encoder_final_activation,
        ),
        sigma_text=_count_text(block.total),
        in_label="card features",
        in_count=in_dim,
        out_label="card embedding",
        out_count=arch.card_embed_dim,
        tooltip=(
            f"Single-Card Encoder · {_count_text(block.total)} params · "
            f"{in_dim} → {arch.card_embed_dim} · one shared column per card, "
            f"reused across board / tray / hand / choice slots"
        ),
        panel=PANEL_CARD,
        params_key=block.label.lower(),
    )


def _hand_unit(
    arch: architecture.ModelArchitecture, param_report: architecture.ParamReport
) -> _Unit:
    layers = _hand_layers(arch, param_report)
    distinct = param_report.hand is not None
    total = (
        param_report.hand.total
        if param_report.hand is not None
        else sum(layer.params for layer in layers)
    )
    tooltip = (
        f"Multi-Card Encoder · {_count_text(total)} params · "
        f"{layers[0].in_features} → {arch.hand_embed_width} · "
        f"embeds a card set (own hand / setup keep / tray)"
    )
    if not distinct:
        tooltip += " · setup net only — the main net mean-pools the hand through the card table"
    return _Unit(
        x=_SVG_COL_X[1],
        accent=_ACCENT_HAND,
        title="MULTI-CARD ENCODER · card-set MLP",
        subtitle="" if distinct else "setup net only · main net mean-pools",
        rows=_op_rows(
            layers,
            arch.hand_activation_resolved.value,
            is_trunk=arch.encoder_final_activation,
        ),
        sigma_text=_count_text(total),
        in_label="card set + summary",
        in_count=layers[0].in_features,
        out_label="set embedding",
        out_count=arch.hand_embed_width,
        tooltip=tooltip,
        dashed=not distinct,
        panel=PANEL_HAND,
        params_key=(
            param_report.hand.label.lower()
            if param_report.hand is not None
            else PARAMS_BLOCK_TOTAL
        ),
    )


def _trunk_unit(
    arch: architecture.ModelArchitecture, param_report: architecture.ParamReport
) -> _Unit:
    block = param_report.trunk
    in_dim = block.layers[0].in_features
    return _Unit(
        x=_SVG_COL_X[0],
        accent=_ACCENT_TRUNK,
        title="STATE ENCODER",
        rows=_op_rows(
            block.layers, arch.trunk_activation_resolved.value, is_trunk=True
        ),
        sigma_text=_count_text(block.total),
        in_label="state input",
        in_count=in_dim,
        out_label="state embedding",
        out_count=arch.trunk_embed_width,
        tooltip=(
            f"State Encoder · {_count_text(block.total)} params · "
            f"{in_dim} → M={arch.trunk_embed_width}"
        ),
        panel=PANEL_STATE,
        params_key=block.label.lower(),
    )


def _attention_unit(
    arch: architecture.ModelArchitecture, block: architecture.BlockParam
) -> _Unit:
    """Board self-attention block: two single-head MultiheadAttention modules,
    one per seat's 15 board slots, drawn in col 0 between the encoder row and
    the consumer row."""
    token_width = arch.card_embed_dim + encode.SLOT_SCALAR_DIM
    return _Unit(
        x=_SVG_COL_X[0],
        accent=_ACCENT_ATTN,
        title="BOARD ATTENTION · per-seat self-attn",
        rows=(
            _OpRow(kind=_OpKind.LINEAR, label="self-attention ×2 boards", params=None),
            _OpRow(
                kind=_OpKind.LINEAR,
                label=f"{encode.SLOTS_PER_BOARD} tokens · {token_width}-wide",
                params=None,
            ),
        ),
        sigma_text=_count_text(block.total),
        in_label="board tokens",
        in_count=encode.N_BOARD_INDEX_SLOTS,
        out_label="attended tokens",
        out_count=encode.N_BOARD_INDEX_SLOTS,
        tooltip=(
            f"Board Self-Attention · {_count_text(block.total)} params · "
            f"two single-head nn.MultiheadAttention modules, one per seat · "
            f"{encode.SLOTS_PER_BOARD} board-slot tokens × {token_width}-wide · "
            f"attended tokens re-folded into state input (trunk width unchanged)"
        ),
        panel=None,
        params_key=block.label.lower(),
    )


def _choice_unit(
    arch: architecture.ModelArchitecture, param_report: architecture.ParamReport
) -> _Unit:
    block = param_report.choice
    in_dim = block.layers[0].in_features
    return _Unit(
        x=_SVG_COL_X[1],
        accent=_ACCENT_CHOICE,
        title="CHOICE ENCODER",
        rows=_op_rows(
            block.layers,
            arch.choice_activation_resolved.value,
            is_trunk=arch.encoder_final_activation,
        ),
        sigma_text=_count_text(block.total),
        in_label="choice input",
        in_count=in_dim,
        out_label="choice embedding",
        out_count=arch.choice_embed_width,
        tooltip=(
            f"Choice Encoder · {_count_text(block.total)} params · "
            f"{in_dim} → N={arch.choice_embed_width} · run once per offered choice"
        ),
        panel=PANEL_CHOICE,
        params_key=block.label.lower(),
    )


def _setup_unit(
    setup_param: architecture.BlockParam,
    setup_arch: setup_model.SetupArchitecture,
    use_setup_model: bool,
) -> _Unit:
    in_dim = setup_param.layers[0].in_features
    status = "active" if use_setup_model else "off"
    return _Unit(
        x=_SVG_COL_X[2],
        accent=_ACCENT_SETUP,
        title="SETUP MODEL · keep",
        subtitle="" if use_setup_model else "off this run — keep scored in-game",
        rows=_op_rows(setup_param.layers, setup_arch.activation.value, is_trunk=False),
        sigma_text=_count_text(setup_param.total),
        in_label="setup input",
        in_count=in_dim,
        out_label="score margin",
        out_count=1,
        tooltip=(
            f"Setup Model ({status}) · {_count_text(setup_param.total)} params incl. the "
            f"frozen card / hand encoder copies · {in_dim} → 1 "
            f"(predicted end-game score margin)"
        ),
        dashed=not use_setup_model,
        panel=PANEL_SETUP,
        params_key=PARAMS_BLOCK_TOTAL,
    )


def _build_setup_unit(
    setup_param: architecture.BlockParam,
    setup_arch: setup_model.SetupArchitecture,
    use_setup_model: bool,
) -> _Unit:
    """The setup column unit: single block normally, dual-head block when actor-critic.

    When ``setup_arch.use_policy_head`` is False, delegates to ``_setup_unit``
    unchanged.  When True, returns a header ``_Unit`` (shared embedder, no
    layer rows) with a ``dual`` pair of narrow sub-units — SETUP VALUE on the
    left, SETUP POLICY on the right — that ``_draw_dual_unit`` renders side by
    side below the header.
    """
    if not setup_arch.use_policy_head:
        return _setup_unit(setup_param, setup_arch, use_setup_model)

    # Split the doubled layer list back into value/policy halves.
    n_layers = len(setup_arch.hidden_layers) + 1
    value_layers = setup_param.layers[:n_layers]
    policy_layers = setup_param.layers[n_layers:]
    value_params = sum(layer.params for layer in value_layers)
    policy_params = sum(layer.params for layer in policy_layers)
    in_dim = setup_param.layers[0].in_features
    status = "active" if use_setup_model else "off"

    value_unit = _Unit(
        x=_SVG_COL_X[2],
        accent=_ACCENT_SETUP,
        title="SETUP VALUE",
        rows=_op_rows(value_layers, setup_arch.activation.value, is_trunk=False),
        sigma_text=_count_text(value_params),
        in_label="setup input",
        in_count=in_dim,
        out_label="score margin",
        out_count=1,
        tooltip=(
            f"Setup Value Head · {_count_text(value_params)} params · {in_dim} → 1 "
            "(predicted end-game score margin)"
        ),
        dashed=not use_setup_model,
        params_key=PARAMS_BLOCK_TOTAL,
    )
    policy_unit = _Unit(
        x=_SVG_COL_X[2] + _DUAL_SUB_W + _DUAL_SUB_GAP,
        accent=_ACCENT_SETUP,
        title="SETUP POLICY",
        rows=_op_rows(policy_layers, setup_arch.activation.value, is_trunk=False),
        sigma_text=_count_text(policy_params),
        in_label="setup input",
        in_count=in_dim,
        out_label="log policy",
        out_count=1,
        tooltip=(
            f"Setup Policy Head · {_count_text(policy_params)} params · {in_dim} → 1 "
            "(log-probabilities over kept-card subsets)"
        ),
        dashed=not use_setup_model,
        params_key=PARAMS_BLOCK_TOTAL,
    )
    return _Unit(
        x=_SVG_COL_X[2],
        accent=_ACCENT_SETUP,
        title="SETUP INPUT · shared embedder",
        rows=(),
        sigma_text=_count_text(setup_param.extra),
        in_label="setup input",
        in_count=in_dim,
        out_label="",  # not rendered — outputs come from the dual sub-units
        out_count=0,
        tooltip=(
            f"Setup Model ({status}) · {_count_text(setup_param.total)} params incl. "
            "frozen card/hand encoder copies · actor-critic mode"
        ),
        dashed=not use_setup_model,
        panel=PANEL_SETUP,
        params_key=PARAMS_BLOCK_TOTAL,
        dual=(value_unit, policy_unit),
    )


def _value_unit(
    arch: architecture.ModelArchitecture, param_report: architecture.ParamReport
) -> _Unit:
    block = param_report.value
    return _Unit(
        x=_SVG_COL_X[0],
        accent=_ACCENT_VALUE,
        title="VALUE HEAD",
        rows=_op_rows(
            block.layers, arch.value_activation_resolved.value, is_trunk=False
        ),
        sigma_text=_count_text(block.total),
        in_label="state embedding",
        in_count=block.layers[0].in_features,
        out_label="value",
        out_count=1,
        tooltip=(
            f"Value Head · {_count_text(block.total)} params · "
            f"{block.layers[0].in_features} → 1"
        ),
        params_key=block.label.lower(),
    )


def _decision_unit(
    arch: architecture.ModelArchitecture,
    param_report: architecture.ParamReport,
    family_order: tuple[str, ...],
) -> _Unit:
    scorer = param_report.scorer
    mn = arch.trunk_embed_width + arch.choice_embed_width
    num_families = len(family_order)
    if scorer.layers:
        rows = _op_rows(
            scorer.layers, arch.head_activation_resolved.value, is_trunk=False
        )
        per_head = (
            scorer.total // scorer.multiplier if scorer.multiplier > 1 else scorer.total
        )
        sigma_text = f"{_count_text(per_head)} each"
        tooltip = (
            f"Decision Head ×{num_families} · {_count_text(per_head)} params each · "
            f"{_count_text(scorer.total)} total · {mn} → 1 score per offered choice"
        )
    else:
        # Per-family head widths: no shared layer shape to draw — one aggregate row.
        rows = (_OpRow(kind=_OpKind.LINEAR, label="per-family readouts"),)
        sigma_text = f"{_count_text(scorer.total)} total"
        tooltip = (
            f"Decision Head ×{num_families} · per-family layer widths · "
            f"{_count_text(scorer.total)} total · {mn} → 1 score per offered choice"
        )
    return _Unit(
        x=_SVG_COL_X[1],
        accent=_ACCENT_DECISION,
        title="DECISION HEAD",
        rows=rows,
        sigma_text=sigma_text,
        in_label="state ⊕ choice",
        in_count=mn,
        out_label="choice score",
        out_count=1,
        tooltip=tooltip,
        dashed=True,
        stack=num_families,
        params_key=scorer.label.lower(),
    )


def _op_rows(
    layers: tuple[architecture.LayerParam, ...],
    activation: str,
    *,
    is_trunk: bool,
) -> tuple[_OpRow, ...]:
    """The mini-rows for a block: one Linear row per layer, with the activation
    rows the builders interleave (trunk: after every layer; other blocks: after
    every non-final layer)."""
    rows: list[_OpRow] = []
    for idx, layer in enumerate(layers):
        rows.append(
            _OpRow(
                kind=_OpKind.LINEAR,
                label=f"Linear →{layer.out_features}",
                params=layer.linear,
            )
        )
        if _has_act_after(is_trunk, is_final=(idx == len(layers) - 1)):
            rows.append(_OpRow(kind=_OpKind.ACT, label=activation))
    return tuple(rows)


def _has_act_after(is_trunk: bool, is_final: bool) -> bool:
    """Mirror of ``mlp.build_body``: blocks with ``final_activation=True``
    (the trunk always; the encoders when ``arch.encoder_final_activation``)
    get an activation after every layer; other blocks only on non-final layers."""
    if is_trunk:
        return True
    return not is_final


def _hand_layers(
    arch: architecture.ModelArchitecture, param_report: architecture.ParamReport
) -> tuple[architecture.LayerParam, ...]:
    """The hand encoder's per-layer shapes: the main net's HAND block when the
    distinct hand model is active, else the identical stack the setup net owns
    (the same recipe ``setup_model.count_setup_parameters`` prices)."""
    if param_report.hand is not None:
        return param_report.hand.layers
    return architecture.body_layers(
        encode.HAND_ENCODER_INPUT_DIM,
        arch.hand_encoder_layers + (arch.hand_embed_width,),
        arch,
    )


#### Geometry ####


def _block_body_h(num_rows: int) -> int:
    """Pixel height of a block body containing ``num_rows`` mini-rows."""
    return (
        _SVG_BLK_PAD_T
        + _SVG_BLK_HDR_H
        + _SVG_BLK_HDR_GAP
        + num_rows * _SVG_ROW_STRIDE
        + _SVG_BLK_PAD_B
    )


def _unit_h(num_rows: int) -> int:
    """Pixel height of a full unit: input box + block body + output box."""
    return _SVG_IO_H + _SVG_IO_GAP + _block_body_h(num_rows) + _SVG_IO_GAP + _SVG_IO_H


def _setup_col_h(unit: _Unit) -> int:
    """Pixel height of the setup column, accounting for dual (actor-critic) mode.

    In dual mode the column holds a shared-embedder header block at the top, a
    vertical gap for the Y-split arrow, and two narrow sub-blocks side by side
    below — each a full ``_unit_h`` in its own right (no inner input box, but
    an output box is drawn for each head).
    """
    if unit.dual is None:
        return _unit_h(len(unit.rows))
    # Header contributes: input IO box + IO gap + block body (no layer rows).
    header_h = _SVG_IO_H + _SVG_IO_GAP + _block_body_h(0)
    sub_h = max(_unit_h(len(unit.dual[0].rows)), _unit_h(len(unit.dual[1].rows)))
    return header_h + _DUAL_SPLIT_GAP + sub_h


def _resolve_geometry(units: _Units) -> _Geom:
    """Stack the three block rows and two connector bands top to bottom; every
    block in a visual row stretches to the row's tallest unit.

    When board attention is on an extra row is inserted between the encoder row
    and the consumer row, with its own connector bands above and below it.
    """
    row1_h_base = max(_unit_h(len(units.card.rows)), _unit_h(len(units.hand.rows)))
    row2_h = max(
        _unit_h(len(units.trunk.rows)),
        _unit_h(len(units.choice.rows)),
        _setup_col_h(units.setup),
    )
    row3_h = max(_unit_h(len(units.value.rows)), _unit_h(len(units.decision.rows)))
    row1_y = _SVG_TOP

    if units.attention is not None:
        # Guard: attention block contributes to row1_h in case it ever exceeds card/hand.
        attn_row_h = _unit_h(len(units.attention.rows))
        row1_h = row1_h_base
        band1_y = row1_y + row1_h
        attn_row_y = band1_y + _SVG_BAND_H
        band_attn_y = attn_row_y + attn_row_h
        row2_y = band_attn_y + _SVG_BAND_H
    else:
        row1_h = row1_h_base
        band1_y = row1_y + row1_h
        row2_y = band1_y + _SVG_BAND_H
        attn_row_h = None
        attn_row_y = None
        band_attn_y = None

    band2_y = row2_y + row2_h
    row3_y = band2_y + _SVG_BAND_H
    total_y = row3_y + row3_h + _SVG_TOTAL_GAP
    return _Geom(
        row1_y=row1_y,
        row1_h=row1_h,
        row2_y=row2_y,
        row2_h=row2_h,
        row3_y=row3_y,
        row3_h=row3_h,
        band1_y=band1_y,
        band2_y=band2_y,
        total_y=total_y,
        svg_h=total_y + _SVG_BOTTOM,
        attn_row_y=attn_row_y,
        attn_row_h=attn_row_h,
        band_attn_y=band_attn_y,
    )


#### Connectors ####


def _band1_connectors(
    geom: _Geom, arch: architecture.ModelArchitecture, use_setup_model: bool
) -> list[_Conn]:
    """Encoder fan-out: how many copies of each encoder's output land in each
    row-2 input.  Counts come from the encode/state layout constants and the
    architecture flags — never hard-coded."""
    src_y = geom.band1_y
    dst_y = geom.row2_y
    card_cx, hand_cx, setup_cx = _SVG_COL_CX  # col0=card/trunk, col1=hand/choice
    trunk_cx, choice_cx = card_cx, hand_cx
    num_choice_copies = encode.CHOICE_BOARD_IDX_SLOTS + 1  # board slots + the candidate
    conns = [
        _Conn(
            src_x=card_cx + _X_CARD_TRUNK,
            src_y=src_y,
            dst_x=trunk_cx + _X_CARD_TRUNK,
            dst_y=dst_y,
            copies=encode.N_CARD_INDEX_SLOTS,
            label=f"×{encode.N_CARD_INDEX_SLOTS}",
            label2=f"{encode.N_BOARD_INDEX_SLOTS} board + {state.TRAY_SIZE} tray",
            label_left=True,
        ),
        _Conn(
            src_x=card_cx + _X_CARD_CHOICE_SRC,
            src_y=src_y,
            dst_x=choice_cx + _X_CARD_CHOICE_DST,
            dst_y=dst_y,
            lane_y=src_y + _LANE_CARD_CHOICE,
            copies=num_choice_copies,
            label=(
                f"×{num_choice_copies} · {encode.CHOICE_BOARD_IDX_SLOTS} board"
                f" + 1 candidate"
            ),
            label_dx=13,
        ),
        _Conn(
            src_x=card_cx + _X_CARD_SETUP_SRC,
            src_y=src_y,
            dst_x=setup_cx + _X_CARD_SETUP_DST,
            dst_y=dst_y,
            lane_y=src_y + _LANE_CARD_SETUP,
            copies=state.TRAY_SIZE,
            label=f"×{state.TRAY_SIZE} · tray",
            dashed=not use_setup_model,
        ),
        _Conn(
            src_x=hand_cx + _X_HAND_SETUP_SRC,
            src_y=src_y,
            dst_x=setup_cx,
            dst_y=dst_y,
            lane_y=src_y + _LANE_HAND,
            copies=2,
            label="×2 · kept + tray set",
            dashed=not use_setup_model,
        ),
    ]
    if arch.use_distinct_hand_model:
        tray_set = arch.tray_set_embedding
        conns.append(
            _Conn(
                src_x=hand_cx + _X_HAND_TRUNK_SRC,
                src_y=src_y,
                dst_x=trunk_cx + _X_HAND_TRUNK_DST,
                dst_y=dst_y,
                lane_y=src_y + _LANE_HAND,
                copies=2 if tray_set else 1,
                label="×2 · own hand + tray set" if tray_set else "×1 · own hand",
            )
        )
    return conns


def _band1_connectors_attn(
    geom: _Geom, arch: architecture.ModelArchitecture, use_setup_model: bool
) -> list[_Conn]:
    """Encoder fan-out when board attention is on.

    The board path (30 card-index slots) goes CARD → ATTENTION → STATE as two
    straight verticals in col 0.  The three column-clearing fan-outs (card→choice,
    card→setup, hand→setup) are identical to the off-path version.  Hand→trunk is
    re-routed to the band_attn level so its horizontal elbow clears the attention
    block.  The 3 tray slots still land in STATE directly via a gutter elbow.
    """
    assert geom.attn_row_y is not None
    assert geom.band_attn_y is not None

    card_cx, hand_cx, setup_cx = _SVG_COL_CX
    trunk_cx, choice_cx = card_cx, hand_cx
    num_choice_copies = encode.CHOICE_BOARD_IDX_SLOTS + 1

    # Three column-clearing fan-outs — unchanged from the off-path variant.
    conns: list[_Conn] = [
        _Conn(
            src_x=card_cx + _X_CARD_CHOICE_SRC,
            src_y=geom.band1_y,
            dst_x=choice_cx + _X_CARD_CHOICE_DST,
            dst_y=geom.row2_y,
            lane_y=geom.band1_y + _LANE_CARD_CHOICE,
            copies=num_choice_copies,
            label=(
                f"×{num_choice_copies} · {encode.CHOICE_BOARD_IDX_SLOTS} board"
                f" + 1 candidate"
            ),
            label_dx=13,
        ),
        _Conn(
            src_x=card_cx + _X_CARD_SETUP_SRC,
            src_y=geom.band1_y,
            dst_x=setup_cx + _X_CARD_SETUP_DST,
            dst_y=geom.row2_y,
            lane_y=geom.band1_y + _LANE_CARD_SETUP,
            copies=state.TRAY_SIZE,
            label=f"×{state.TRAY_SIZE} · tray",
            dashed=not use_setup_model,
        ),
        _Conn(
            src_x=hand_cx + _X_HAND_SETUP_SRC,
            src_y=geom.band1_y,
            dst_x=setup_cx,
            dst_y=geom.row2_y,
            lane_y=geom.band1_y + _LANE_HAND,
            copies=2,
            label="×2 · kept + tray set",
            dashed=not use_setup_model,
        ),
    ]

    # Board path: CARD → ATTENTION (straight vertical, col 0).
    conns.append(
        _Conn(
            src_x=card_cx + _X_CARD_TRUNK,
            src_y=geom.band1_y,
            dst_x=card_cx + _X_CARD_TRUNK,
            dst_y=geom.attn_row_y,
            copies=encode.N_BOARD_INDEX_SLOTS,
            label=f"×{encode.N_BOARD_INDEX_SLOTS}",
            label2=f"2×{encode.SLOTS_PER_BOARD} board slots",
            label_left=True,
        )
    )

    # Board path: ATTENTION → STATE (straight vertical, col 0).
    conns.append(
        _Conn(
            src_x=trunk_cx + _X_CARD_TRUNK,
            src_y=geom.band_attn_y,
            dst_x=trunk_cx + _X_CARD_TRUNK,
            dst_y=geom.row2_y,
            copies=encode.N_BOARD_INDEX_SLOTS,
            label=f"×{encode.N_BOARD_INDEX_SLOTS} attended",
            label_left=True,
        )
    )

    # Tray-3: 3 card-index slots bypass attention → STATE via gutter elbow.
    conns.append(
        _Conn(
            src_x=_X_ATTN_TRAY_GUTTER,
            src_y=geom.band1_y,
            dst_x=trunk_cx + _X_HAND_TRUNK_DST,
            dst_y=geom.row2_y,
            lane_y=geom.band_attn_y + _LANE_ATTN_TRAY,
            copies=state.TRAY_SIZE,
            label=f"×{state.TRAY_SIZE} tray",
        )
    )

    # Hand → STATE re-routed to the band_attn level.
    if arch.use_distinct_hand_model:
        tray_set = arch.tray_set_embedding
        conns.append(
            _Conn(
                src_x=hand_cx + _X_HAND_TRUNK_SRC,
                src_y=geom.band1_y,
                dst_x=trunk_cx + _X_HAND_TRUNK_DST,
                dst_y=geom.row2_y,
                lane_y=geom.band_attn_y + _LANE_ATTN_HAND_TRUNK,
                copies=2 if tray_set else 1,
                label="×2 · own hand + tray set" if tray_set else "×1 · own hand",
            )
        )

    return conns


def _band2_connectors(geom: _Geom, arch: architecture.ModelArchitecture) -> list[_Conn]:
    """Head merge: the trunk's M feeds both heads; the choice encoder's N joins
    it at the decision head's M+N input."""
    src_y = geom.band2_y
    dst_y = geom.row3_y
    trunk_cx, choice_cx, _ = _SVG_COL_CX
    m_label = f"M={arch.trunk_embed_width}"
    return [
        _Conn(src_x=trunk_cx, src_y=src_y, dst_x=trunk_cx, dst_y=dst_y, label=m_label),
        _Conn(
            src_x=trunk_cx + _X_TRUNK_DECISION_SRC,
            src_y=src_y,
            dst_x=choice_cx + _X_TRUNK_DECISION_DST,
            dst_y=dst_y,
            lane_y=src_y + _LANE_TRUNK_DECISION,
            label=m_label,
        ),
        _Conn(
            src_x=choice_cx,
            src_y=src_y,
            dst_x=choice_cx,
            dst_y=dst_y,
            label=f"N={arch.choice_embed_width}",
        ),
    ]


def _stroke_for(copies: int) -> float:
    """The tiered connector stroke width for an output duplicated ``copies`` times."""
    if copies >= _MANY_COPIES:
        return _STROKE_MANY
    if copies >= _FEW_COPIES:
        return _STROKE_FEW
    return _STROKE_SINGLE


def _conn_svg(conn: _Conn) -> tuple[str, str]:
    """Render one connector as ``(body, labels)`` — bodies are drawn before any
    labels so the labels' white halos mask crossing lines."""
    stroke = _stroke_for(conn.copies)
    dash = ' stroke-dasharray="6,4"' if conn.dashed else ""
    lane_y = conn.lane_y

    # Straight vertical: line with the label(s) stacked beside it — to its
    # right by default, right-aligned to its left under ``label_left``.
    if lane_y is None:
        body = (
            f'<line x1="{conn.src_x}" y1="{conn.src_y}" x2="{conn.dst_x}" y2="{conn.dst_y}" '
            f'stroke="{_SVG_ARROW}" stroke-width="{stroke}"{dash} marker-end="url(#arr)"/>'
        )
        mid_y = (conn.src_y + conn.dst_y) // 2
        anchor, label_x = (
            ("end", conn.src_x - 7) if conn.label_left else ("start", conn.src_x + 7)
        )
        labels: list[str] = []
        if conn.label:
            labels.append(_halo_text(label_x, mid_y - 2, conn.label, anchor=anchor))
        if conn.label2:
            labels.append(_halo_text(label_x, mid_y + 10, conn.label2, anchor=anchor))
        return body, "\n".join(labels)

    # Orthogonal elbow along a horizontal lane, label centered above the run.
    pts = (
        f"{conn.src_x},{conn.src_y} {conn.src_x},{lane_y} "
        f"{conn.dst_x},{lane_y} {conn.dst_x},{conn.dst_y}"
    )
    body = (
        f'<polyline points="{pts}" fill="none" stroke="{_SVG_ARROW}" '
        f'stroke-width="{stroke}"{dash} marker-end="url(#arr)"/>'
    )
    label = ""
    if conn.label:
        label_x = (conn.src_x + conn.dst_x) // 2 + conn.label_dx
        label = _halo_text(label_x, lane_y - 6, conn.label, anchor="middle")
    return body, label


#### Drawing primitives ####


def _draw_unit(unit: _Unit, top_y: int, unit_h: int) -> str:
    """One block with its input box above and output box below, grouped under a
    shared hover tooltip. A unit with a ``panel`` gets its input box wrapped in
    the ``arch-click`` group the report's script opens that panel from.
    A unit with ``dual`` set is rendered as a header-only input box that fans
    out to two side-by-side sub-units below."""
    if unit.dual is not None:
        return _draw_dual_unit(unit, top_y, unit_h)
    body_h = unit_h - 2 * (_SVG_IO_H + _SVG_IO_GAP)
    block_y = top_y + _SVG_IO_H + _SVG_IO_GAP
    in_box = _io_box(
        unit.x,
        top_y,
        f"{unit.in_label} · {_count_text(unit.in_count)}",
        dashed=unit.dashed,
    )
    if unit.panel is not None:
        in_box = (
            f'<g class="arch-click" data-panel="{html_lib.escape(unit.panel)}">'
            f"<title>Click to inspect this vector</title>\n{in_box}\n</g>"
        )
    parts = [
        "<g>",
        f"<title>{html_lib.escape(unit.tooltip)}</title>",
        in_box,
        _draw_block(unit, block_y, body_h),
        _io_box(
            unit.x,
            top_y + unit_h - _SVG_IO_H,
            f"{unit.out_label} · {_count_text(unit.out_count)}",
            dashed=unit.dashed,
        ),
        "</g>",
    ]
    return "\n".join(parts)


def _draw_dual_unit(unit: _Unit, top_y: int, unit_h: int) -> str:
    """Actor-critic setup column: full-width header input block → Y-split arrow
    → two narrow sub-blocks (SETUP VALUE left, SETUP POLICY right) with their
    own output IO boxes.

    Layout (top to bottom within the allocated ``unit_h``):
    * Input IO box at full column width (clickable to the setup panel).
    * Block body at full width, title "SETUP INPUT", sigma = embedder params,
      no layer rows.
    * ``_DUAL_SPLIT_GAP`` pixels of vertical space with a Y-shaped connector.
    * Two ``_DUAL_SUB_W``-wide block bodies (value left, policy right), each
      sharing the remaining height with their output IO box at the bottom.
    """
    assert unit.dual is not None
    value_unit, policy_unit = unit.dual

    # Header block occupies the top of the allocated column height.
    header_body_h = _block_body_h(0)
    header_block_y = top_y + _SVG_IO_H + _SVG_IO_GAP
    header_bottom = header_block_y + header_body_h

    # Sub-units fill the remaining space below the split gap.
    sub_top_y = header_bottom + _DUAL_SPLIT_GAP
    sub_h = top_y + unit_h - sub_top_y

    # Center x-coordinates for the Y-split connector.
    header_cx = unit.x + _SVG_COL_W // 2
    value_cx = value_unit.x + _DUAL_SUB_W // 2
    policy_cx = policy_unit.x + _DUAL_SUB_W // 2
    split_mid_y = header_bottom + _DUAL_SPLIT_GAP // 2

    # Shared embedder input box (full width, clickable to the setup panel).
    in_box = _io_box(
        unit.x,
        top_y,
        f"{unit.in_label} · {_count_text(unit.in_count)}",
        dashed=unit.dashed,
        col_w=_SVG_COL_W,
    )
    if unit.panel is not None:
        in_box = (
            f'<g class="arch-click" data-panel="{html_lib.escape(unit.panel)}">'
            f"<title>Click to inspect this vector</title>\n{in_box}\n</g>"
        )

    # Y-split: two L-shaped polylines sharing the vertical stem, each ending
    # with an arrowhead at the top of its sub-block.
    def _branch(dst_cx: int) -> str:
        pts = (
            f"{header_cx},{header_bottom} {header_cx},{split_mid_y} "
            f"{dst_cx},{split_mid_y} {dst_cx},{sub_top_y}"
        )
        return (
            f'<polyline points="{pts}" fill="none" stroke="{_SVG_ARROW}" '
            f'stroke-width="{_STROKE_SINGLE}" marker-end="url(#arr)"/>'
        )

    y_split = f"{_branch(value_cx)}\n{_branch(policy_cx)}"

    # Sub-block drawing helper: block body + output IO box, no input IO box.
    def _draw_sub(sub: _Unit) -> str:
        sub_body_h = sub_h - _SVG_IO_GAP - _SVG_IO_H
        sub_body_y = sub_top_y
        out_y = sub_top_y + sub_h - _SVG_IO_H
        return "\n".join(
            [
                f"<g><title>{html_lib.escape(sub.tooltip)}</title>",
                _draw_block(sub, sub_body_y, sub_body_h, col_w=_DUAL_SUB_W),
                _io_box(
                    sub.x,
                    out_y,
                    f"{sub.out_label} · {_count_text(sub.out_count)}",
                    dashed=sub.dashed,
                    col_w=_DUAL_SUB_W,
                ),
                "</g>",
            ]
        )

    parts = [
        "<g>",
        f"<title>{html_lib.escape(unit.tooltip)}</title>",
        in_box,
        _draw_block(unit, header_block_y, header_body_h, col_w=_SVG_COL_W),
        y_split,
        _draw_sub(value_unit),
        _draw_sub(policy_unit),
        "</g>",
    ]
    return "\n".join(parts)


def _draw_block(unit: _Unit, y: int, body_h: int, *, col_w: int = _SVG_COL_W) -> str:
    """The block body: bordered rect with accent bar, centered title/subtitle,
    mini-rows, the ×N stack effect, and the parameter-total border legend."""
    x = unit.x
    width = col_w
    parts: list[str] = []

    # Shadow rects for the ×N stacked-card effect (largest offset first).
    if unit.stack > 1:
        for offset in (6, 3):
            parts.append(
                f'<rect x="{x + offset}" y="{y + offset}" width="{width}" height="{body_h}" '
                f'rx="{_SVG_RX_BLK}" fill="{_SVG_BLOCK_FILL}" '
                f'stroke="{_SVG_BLOCK_STROKE}" stroke-width="1"/>'
            )

    # Main rect + left accent bar clipped to the rounded corners.
    clip_id = f"clip{x}-{y}"
    dash = ' stroke-dasharray="5,3"' if unit.dashed else ""
    parts.append(
        f'<defs><clipPath id="{clip_id}">'
        f'<rect x="{x}" y="{y}" width="{width}" height="{body_h}" rx="{_SVG_RX_BLK}"/>'
        f"</clipPath></defs>"
    )
    parts.append(
        f'<rect x="{x}" y="{y}" width="{width}" height="{body_h}" rx="{_SVG_RX_BLK}" '
        f'fill="{_SVG_BLOCK_FILL}" stroke="{_SVG_BLOCK_STROKE}" stroke-width="1"{dash}/>'
    )
    parts.append(
        f'<rect x="{x}" y="{y}" width="{_SVG_ACCENT_W}" height="{body_h}" '
        f'fill="{unit.accent}" clip-path="url(#{clip_id})"/>'
    )

    # Centered title (+ optional italic subtitle) and the ×N badge.
    cx = x + width // 2
    parts.append(
        f'<text x="{cx}" y="{y + _SVG_BLK_PAD_T + 13}" font-family="{_FONT_SANS}" '
        f'font-size="13" font-weight="700" fill="{_SVG_TEXT_TITLE}" text-anchor="middle">'
        f"{html_lib.escape(unit.title)}</text>"
    )
    if unit.subtitle:
        parts.append(
            f'<text x="{cx}" y="{y + _SVG_BLK_PAD_T + 28}" font-family="{_FONT_MONO}" '
            f'font-size="10" font-style="italic" fill="{_SVG_TEXT_DIM}" text-anchor="middle">'
            f"{html_lib.escape(unit.subtitle)}</text>"
        )
    if unit.stack > 1:
        parts.append(_stack_badge(x, y, width, unit.stack))

    parts.append(
        _draw_op_rows(
            unit.rows,
            x,
            y + _SVG_BLK_PAD_T + _SVG_BLK_HDR_H + _SVG_BLK_HDR_GAP,
            params_key=unit.params_key,
            col_w=width,
        )
    )

    # Block parameter total, overlaid on the bottom-right border.
    parts.append(
        _param_click_group(
            _halo_text(
                x + width - 12,
                y + body_h + 4,
                unit.sigma_text,
                anchor="end",
                size=10,
                color=unit.accent,
                bold=True,
            ),
            unit.params_key,
        )
    )
    return "\n".join(parts)


def _draw_op_rows(
    rows: tuple[_OpRow, ...],
    x: int,
    rows_y0: int,
    *,
    params_key: str | None,
    col_w: int = _SVG_COL_W,
) -> str:
    """The mini-rows: centered operation label, with each Linear's parameter
    count overlaid on its bottom-right border (clickable to the parameter
    table's ``params_key`` block)."""
    row_x = x + _SVG_ACCENT_W + 8
    row_w = col_w - _SVG_ACCENT_W - 16
    parts: list[str] = []
    for idx, row in enumerate(rows):
        ry = rows_y0 + idx * _SVG_ROW_STRIDE
        color = _SVG_LINEAR_COLOR if row.kind is _OpKind.LINEAR else _SVG_ACT_COLOR
        parts.append(
            f'<rect x="{row_x}" y="{ry}" width="{row_w}" height="{_SVG_ROW_H}" '
            f'rx="{_SVG_RX_ROW}" fill="{color}" fill-opacity="0.06" '
            f'stroke="{color}" stroke-width="1.5"/>'
        )
        parts.append(
            f'<text x="{row_x + row_w // 2}" y="{ry + 19}" font-family="{_FONT_MONO}" '
            f'font-size="11" font-weight="600" fill="{color}" text-anchor="middle">'
            f"{html_lib.escape(row.label)}</text>"
        )
        if row.params is not None:
            parts.append(
                _param_click_group(
                    _halo_text(
                        row_x + row_w - 8,
                        ry + _SVG_ROW_H + 4,
                        _count_text(row.params),
                        anchor="end",
                        size=9,
                    ),
                    params_key,
                )
            )
    return "\n".join(parts)


def _param_click_group(inner_svg: str, params_key: str | None) -> str:
    """Wrap a parameter-count text in the ``arch-paramclick`` group the report's
    script opens the Parameters panel from, jumped to ``params_key``'s block
    rows. A ``None`` key leaves the text inert."""
    if params_key is None:
        return inner_svg
    return (
        f'<g class="arch-paramclick" data-params-block="{html_lib.escape(params_key)}">'
        f"{inner_svg}</g>"
    )


def _io_box(
    x_col: int, top_y: int, text: str, *, dashed: bool, col_w: int = _SVG_COL_W
) -> str:
    """A tinted input/output vector box with its centered ``name · count`` label."""
    box_x = x_col + _SVG_IO_INSET
    box_w = col_w - 2 * _SVG_IO_INSET
    dash = ' stroke-dasharray="4,3"' if dashed else ""
    rect = (
        f'<rect x="{box_x}" y="{top_y}" width="{box_w}" height="{_SVG_IO_H}" '
        f'rx="{_SVG_RX_IO}" fill="{_SVG_IO_FILL}" stroke="{_SVG_IO_STROKE}" '
        f'stroke-width="1"{dash}/>'
    )
    label = (
        f'<text x="{x_col + col_w // 2}" y="{top_y + 17}" font-family="{_FONT_MONO}" '
        f'font-size="11" font-weight="600" fill="{_SVG_IO_TEXT}" text-anchor="middle">'
        f"{html_lib.escape(text)}</text>"
    )
    return f"{rect}\n{label}"


def _stack_badge(x: int, y: int, width: int, stack: int) -> str:
    """The ×N pill badge in a stacked block's top-right corner."""
    label = f"×{stack}"
    badge_w = len(label) * 7 + 12
    badge_x = x + width - badge_w - 10
    badge_y = y + 10
    return (
        f'<rect x="{badge_x}" y="{badge_y}" width="{badge_w}" height="18" '
        f'rx="9" fill="{_ACCENT_DECISION_BADGE_BG}"/>'
        f'<text x="{badge_x + badge_w // 2}" y="{badge_y + 13}" '
        f'font-family="{_FONT_SANS}" font-size="10" font-weight="700" '
        f'fill="{_ACCENT_DECISION}" text-anchor="middle">{html_lib.escape(label)}</text>'
    )


def _halo_text(
    x: int,
    y: int,
    text: str,
    *,
    anchor: str,
    size: int = 10,
    color: str = _SVG_TEXT_DIM,
    bold: bool = False,
) -> str:
    """Mono text with a white halo (``paint-order: stroke``) so it stays legible
    overlaid on borders and crossing connector lines."""
    weight = ' font-weight="700"' if bold else ""
    return (
        f'<text x="{x}" y="{y}" font-family="{_FONT_MONO}" font-size="{size}"{weight} '
        f'fill="{color}" text-anchor="{anchor}" paint-order="stroke" '
        f'stroke="{_SVG_BLOCK_FILL}" stroke-width="3" stroke-linejoin="round">'
        f"{html_lib.escape(text)}</text>"
    )


#### Root and footer ####


def _svg_root(
    geom: _Geom,
    arch: architecture.ModelArchitecture,
    param_report: architecture.ParamReport,
    setup_param: architecture.BlockParam,
    num_families: int,
    use_setup_model: bool,
) -> str:
    """The opening ``<svg>`` tag with its accessible label, the canvas
    background, and the shared arrowhead marker."""
    trunk_m = arch.trunk_embed_width
    choice_n = arch.choice_embed_width
    status = "active" if use_setup_model else "off"
    total = _count_text(param_report.total)
    setup_total = _count_text(setup_param.total)
    aria = (
        f"PolicyValueNet architecture: Single-Card Encoder (output reused "
        f"×{encode.N_CARD_INDEX_SLOTS} in the state input, "
        f"×{encode.CHOICE_BOARD_IDX_SLOTS + 1} per choice) and Multi-Card Encoder "
        f"feeding State Encoder (M={trunk_m}) and Choice Encoder (N={choice_n}), "
        f"merging into "
        f"Value Head and {num_families} Decision Heads, {total} params total; "
        f"separate Setup Model ({status}, {setup_total} params)"
    )
    if arch.use_board_attention:
        aria += (
            f"; board self-attention over each seat's {encode.SLOTS_PER_BOARD} "
            f"board slots before the State Encoder"
        )
    marker = (
        '<defs><marker id="arr" viewBox="0 0 10 10" refX="8.5" refY="5" '
        'markerWidth="9" markerHeight="9" markerUnits="userSpaceOnUse" orient="auto">'
        f'<path d="M0,1 L8.5,5 L0,9 Z" fill="{_SVG_ARROW}"/></marker></defs>'
    )
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_SVG_W} {geom.svg_h}" '
            f'width="100%" style="display:block;max-width:{_SVG_W}px;" role="img" '
            f'aria-label="{html_lib.escape(aria)}">',
            f"<title>PolicyValueNet · {total} params · M={trunk_m} N={choice_n} "
            f"×{num_families} heads · setup {setup_total} ({status})</title>",
            f'<rect width="{_SVG_W}" height="{geom.svg_h}" fill="{_SVG_BG}"/>',
            marker,
        ]
    )


def _row1_side_note(geom: _Geom) -> str:
    """The annotation beside the top encoder row: both shared encoders are
    trained by in-game decisions only — setup experience never reaches them
    (the setup net carries frozen, synced copies)."""
    note_x = _SVG_COL_X[2] + 12
    mid_y = geom.row1_y + geom.row1_h // 2
    return "\n".join(
        [
            _halo_text(note_x, mid_y - 4, "trained in-game only —", anchor="start"),
            _halo_text(
                note_x, mid_y + 10, "frozen copies in setup net", anchor="start"
            ),
        ]
    )


def _total_line(
    geom: _Geom,
    param_report: architecture.ParamReport,
    setup_param: architecture.BlockParam,
    use_setup_model: bool,
) -> str:
    """The grand-total caption (the separate setup net's count is annotated,
    not summed in), clickable to the parameter table's grand-total row."""
    text = f"TOTAL {_count_text(param_report.total)} params"
    if use_setup_model:
        text += f" · setup {_count_text(setup_param.total)} (separate)"
    return _param_click_group(
        f'<text x="{_SVG_W // 2}" y="{geom.total_y}" font-family="{_FONT_SANS}" '
        f'font-size="13" font-weight="700" fill="{_SVG_TOTAL_COLOR}" text-anchor="middle">'
        f"{html_lib.escape(text)}</text>",
        PARAMS_BLOCK_TOTAL,
    )


def _count_text(value: int) -> str:
    """Exact bare-integer count — no thousands separators, never "123k"."""
    return str(value)
