"""SVG architecture diagram for the model-summary HTML report.

Builds the self-contained ``<svg>`` drawing of the full network topology that
``wingspan.reporting.html`` embeds in its Architecture section: the single-card
encoder alone on the top row producing the shared card embedding; its consumers
on the next row (the multi-card encoder / hand pooling block always, plus board
self-attention when enabled); the state encoder / choice encoder / setup model
on the middle row; and the value / decision heads on the bottom row, joined by
fan-out connectors labelled with how many copies of each encoder's output the
downstream input consumes (e.g. ×33 card embeddings in the state input — 30
board slots + 3 tray slots).  Most blocks carry tinted input/output boxes
(descriptive name · element count), centered layer rows, and exact bare-integer
parameter counts overlaid on the borders; the parameter-less hand-pooling block
is drawn bare (its descriptive rows only, no I/O boxes).

The diagram doubles as the report's navigation: every real input box is wrapped
in an ``arch-click`` group whose ``data-panel`` names the report section it
reveals (the :data:`PANEL_*` contract), and every parameter count in an
``arch-paramclick`` group whose ``data-params-block`` names the parameter-table
block it jumps to — ``wingspan.reporting.html``'s inline script and CSS supply the
behaviour and affordances.

The diagram is data-driven: layer shapes come from the
:class:`wingspan.architecture.ParamReport` (with the hand encoder's shapes
recomputed via :func:`wingspan.architecture.body_layers` when the main net
pools the hand through the card table instead), copy counts come from the ``wingspan.encode``
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
# setup net, the pooled-hand path).

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
_SVG_DROPOUT_COLOR = "#f59e0b"
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

# Single-card encoder is centered on the 960-wide canvas.  The trunk gutter
# (x = _TRUNK_X) runs between the BOARD ATTENTION and MULTI-CARD POOLING
# consumer blocks and carries the three branches down to State / Choice / Setup.
_ENC_X = 335  # SINGLE-CARD ENCODER left edge (centered: (960-290)//2 = 335)
_ENC_CX = 480  # SINGLE-CARD ENCODER center (≈ canvas center)
_TRUNK_X = 479  # trunk stem x — matches the Choice column center exactly

# Consumers row: BOARD ATTENTION stays in col 0 (left-of-center); MULTI-CARD
# POOLING is right-of-center, keeping _TRUNK_X in the gutter between them.
_POOL_X = 510  # MULTI-CARD POOLING left edge
_POOL_W = 200  # MULTI-CARD POOLING block width
_POOL_CX = 610  # MULTI-CARD POOLING center (= _POOL_X + _POOL_W // 2)

# Per-connector attach offsets and band-lane y offsets (relative to each band's
# top), chosen so no two segments share an x/y and no labels collide.
_X_CARD_HAND_SRC = 35  # card embedding → MULTI-CARD POOLING (relative to _ENC_CX)
_X_CARD_ATTN_SRC = -40  # card embedding → BOARD ATTENTION src (relative to _ENC_CX)
_ATTN_VERT_X = 125  # x for card→attention and attention→state verticals
_X_HAND_TRUNK_SRC = -40
_X_HAND_TRUNK_DST = 40
_X_HAND_SETUP_SRC = 30
_X_TRUNK_DECISION_SRC = 55
_X_TRUNK_DECISION_DST = -30

# Band-1 lanes (offset from band-1 top): card→hand and (attention ON) card→attn.
_LANE_CARD_HAND = 16
_LANE_CARD_ATTN = 38

# "cons" band lanes (offset from consumers → row-2 band top).
_LANE_HAND_STATE = 14
_LANE_HAND_CHOICE = 30  # pool → choice encoder (becomes_playable)
_LANE_HAND_SETUP = 42
_LANE_TRUNK_SPLIT = 54  # below both hand-connector lanes; still fits in _SVG_BAND_H=64

_LANE_TRUNK_DECISION = 24  # band-2 lane (trunk → decision elbow)

# The MULTI-CARD POOLING → CHOICE arrow lands at 2/3 the choice column width
# (so it's visually distinct from the trunk→choice arrow that lands at center).
_X_CHOICE_DST_POOL = _SVG_COL_X[1] + _SVG_COL_W * 2 // 3  # = 527

# Dual-mode (actor-critic) setup column geometry: a shared header block at full
# width, then two full-width sub-blocks (SETUP VALUE above SETUP POLICY) each
# separated by _DUAL_SPLIT_GAP pixels of vertical space for the fork arrows.
_DUAL_SPLIT_GAP = 20


def build_arch_svg(
    arch: architecture.ModelArchitecture,
    param_report: architecture.ParamReport,
    family_order: tuple[str, ...],
    *,
    setup_param: setup_model.SetupParamReport,
    setup_arch: setup_model.SetupArchitecture,
    use_setup_model: bool,
) -> str:
    """Return a self-contained ``<svg>`` string for the architecture diagram.

    The separately-trained setup model is drawn as a third column connected to
    the shared encoders (the copies it carries are frozen syncs of the main
    net's).  It is drawn even when ``use_setup_model`` is False: dashed, with
    an "off" subtitle, so the diagram always shows what the setup net would
    look like.  The hand encoder is likewise always drawn — dashed when the
    main net pools the hand through the card table instead.
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

    # Root + the four block rows (single-card encoder · consumers ·
    # state/choice/setup · heads).
    parts = [
        _svg_root(
            geom, arch, param_report, setup_param, len(family_order), use_setup_model
        )
    ]
    placed = (
        (units.card, geom.row1_y, geom.row1_h),
        (units.trunk, geom.row2_y, geom.row2_h),
        (units.choice, geom.row2_y, geom.row2_h),
        (units.setup, geom.row2_y, geom.row2_h),
        (units.value, geom.row3_y, geom.row3_h),
        (units.decision, geom.row3_y, geom.row3_h),
    )
    for unit, top_y, unit_h in placed:
        parts.append(_draw_unit(unit, top_y, unit_h))

    # Consumers row: board attention (col 0, when on) fills the row height; the
    # bare hand-pooling block sits at its own natural height to the right.
    if units.attention is not None:
        parts.append(_draw_unit(units.attention, geom.cons_row_y, geom.cons_row_h))
    parts.append(_draw_unit(units.hand, geom.cons_row_y, _block_outer_h(units.hand)))

    # The training note: the shared encoders learn only from in-game decisions;
    # the setup net consumes them as frozen, synced copies. Gated on the distinct
    # hand model — without it the setup net trains its own multi-card encoder, so
    # the blanket note would be wrong.
    if arch.use_distinct_hand_model:
        parts.append(_row1_side_note(geom))

    # Connectors: all bodies first, then all labels, so the white label halos
    # mask any line they cross.  The trunk (shared card→{state,choice,setup}
    # stem) is emitted separately so its bodies and labels stay in order.
    trunk_bodies, trunk_labels = _trunk_svg(
        geom, arch, use_setup_model, color=_ACCENT_CARD
    )
    conns = _consumer_connectors(
        geom, units, arch, use_setup_model
    ) + _band2_connectors(geom, arch)
    rendered = [_conn_svg(conn) for conn in conns]
    parts.extend(body for body, _ in rendered)
    parts.extend(trunk_bodies)
    parts.extend(label for _, label in rendered if label)
    parts.extend(trunk_labels)

    parts.append(_total_line(geom, param_report, setup_param, use_setup_model))
    parts.append("</svg>")
    return "\n".join(parts)


###### PRIVATE #######

#### Value objects ####


class _OpKind(enum.StrEnum):
    """The three mini-row styles inside a block."""

    LINEAR = "linear"
    ACT = "act"
    DROPOUT = "dropout"


class _OpRow(pydantic.BaseModel):
    """One mini-row inside a block: a Linear (with its parameter count), an
    activation, or a dropout layer."""

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
    # When set, this unit is rendered as a header trunk block that forks via two
    # vertical arrows into two full-width sub-units stacked vertically below
    # (actor-critic setup mode).  The outer unit provides the shared trunk input
    # box; ``dual`` provides the two heads.
    dual: tuple["_Unit", "_Unit"] | None = None
    # A parameter-less block (e.g. card-table hand pooling) drawn as its block
    # body only — no input/output vector boxes — at a custom width.
    bare_block: bool = False
    block_w: int | None = None


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
    stroke: str = _SVG_ARROW
    # Optional dogleg: when both are set, the elbow routes down a vertical
    # corridor at ``corridor_x`` (between the consumer boxes) from ``lane_y`` to
    # ``lane_y2`` before fanning out to ``dst_x`` — a five-segment path.
    corridor_x: int | None = None
    lane_y2: int | None = None


class _Geom(pydantic.BaseModel):
    """The resolved vertical layout: row tops/heights, band tops, total-line y.

    Four block rows stack top to bottom: the single-card encoder (row 1), its
    consumers (the ``cons`` row — hand pooling always, board attention when on),
    the state/choice/setup row (row 2), and the heads (row 3), with a connector
    band between each pair."""

    row1_y: int
    row1_h: int
    cons_row_y: int
    cons_row_h: int
    row2_y: int
    row2_h: int
    row3_y: int
    row3_h: int
    band1_y: int
    band_cons_y: int
    band2_y: int
    total_y: int
    svg_h: int


#### Unit assembly ####


def _build_units(
    arch: architecture.ModelArchitecture,
    param_report: architecture.ParamReport,
    family_order: tuple[str, ...],
    *,
    setup_param: setup_model.SetupParamReport,
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
        x=_ENC_X,
        accent=_ACCENT_CARD,
        title="SINGLE-CARD ENCODER",
        rows=_op_rows(
            block.layers,
            between_activation=arch.card_between_activation_resolved.value,
            final_activation=arch.card_final_activation_resolved.value,
            dropout=arch.card_dropout_resolved,
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


def _hand_pool_rows(arch: architecture.ModelArchitecture) -> tuple[_OpRow, ...]:
    """Descriptive rows for the hand-pooling path (no learnable layers)."""
    desc = {
        architecture.HandPooling.CONCAT_MAX_SUM: "max ⊕ sum ⊕ count",
        architecture.HandPooling.MAX: "elem-wise max + count",
        architecture.HandPooling.SUM: "sum over card table",
        architecture.HandPooling.MEAN: "mean over card table",
    }[arch.hand_pooling]
    return (_OpRow(kind=_OpKind.ACT, label=f"pool: {desc}"),)


def _hand_unit(
    arch: architecture.ModelArchitecture, param_report: architecture.ParamReport
) -> _Unit:
    distinct = param_report.hand is not None

    # When the main net uses a distinct hand model (MLP), draw it as a learned block.
    if distinct:
        layers = _hand_layers(arch, param_report)
        total = param_report.hand.total  # type: ignore[union-attr]
        return _Unit(
            x=_SVG_COL_X[1],
            accent=_ACCENT_HAND,
            title="MULTI-CARD ENCODER",
            rows=_op_rows(
                layers,
                between_activation=arch.hand_between_activation_resolved.value,
                final_activation=arch.hand_final_activation_resolved.value,
                dropout=arch.hand_dropout_resolved,
            ),
            sigma_text=_count_text(total),
            in_label="card set + summary",
            in_count=layers[0].in_features,
            out_label="set embedding",
            out_count=arch.hand_embed_width,
            tooltip=(
                f"Multi-Card Encoder · {_count_text(total)} params · "
                f"{layers[0].in_features} → {arch.hand_embed_width} · "
                f"embeds a card set (own hand / setup keep / tray)"
            ),
            panel=PANEL_HAND,
            params_key=param_report.hand.label.lower(),  # type: ignore[union-attr]
        )

    # Default path: main net pools via the shared card table (no extra params).
    # Drawn bare — its single descriptive row only, no I/O boxes — since it has
    # no learnable weights; sits right-of-center in the consumers row.
    out_count = arch.pooled_hand_width
    return _Unit(
        x=_POOL_X,
        accent=_ACCENT_HAND,
        title="MULTI-CARD POOLING",
        rows=_hand_pool_rows(arch),
        sigma_text="",
        in_label="hand multi-hot",
        in_count=encode.HAND_MULTIHOT_DIM,
        out_label="pooled hand",
        out_count=out_count,
        tooltip=(
            f"Multi-Card Pooling · {encode.HAND_MULTIHOT_DIM} → {out_count} · "
            f"pools hand / playable / egg-blocked multi-hots through the shared card table · "
            f"mode: {arch.hand_pooling.value}"
        ),
        panel=PANEL_HAND,
        params_key=PARAMS_BLOCK_TOTAL,
        bare_block=True,
        block_w=_POOL_W,
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
            block.layers,
            between_activation=arch.trunk_between_activation_resolved.value,
            final_activation=arch.trunk_final_activation_resolved.value,
            dropout=arch.trunk_dropout_resolved,
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
        title="BOARD ATTENTION",
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
            between_activation=arch.choice_between_activation_resolved.value,
            final_activation=arch.choice_final_activation_resolved.value,
            dropout=arch.choice_dropout_resolved,
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
    setup_param: setup_model.SetupParamReport,
    setup_arch: setup_model.SetupArchitecture,
    use_setup_model: bool,
) -> _Unit:
    # Value-only fallback (use_policy_head=False, never used in practice): the
    # state-only value path — value trunk (if present) + value head — as one
    # continuous sequence. The critic is V(s), so it cannot rank keeps.
    all_layers = setup_param.value_trunk + setup_param.value_head
    in_dim = all_layers[0].in_features
    status = "active" if use_setup_model else "off"
    return _Unit(
        x=_SVG_COL_X[2],
        accent=_ACCENT_SETUP,
        title="SETUP TRUNK",
        subtitle="" if use_setup_model else "off this run — keep scored in-game",
        rows=_op_rows(
            all_layers,
            between_activation=setup_arch.between_activation.value,
            final_activation=setup_arch.final_activation.value,
            dropout=setup_arch.dropout,
        ),
        sigma_text=_count_text(setup_param.total),
        in_label="state only",
        in_count=in_dim,
        out_label="V(s)",
        out_count=1,
        tooltip=(
            f"Setup Model ({status}) · {_count_text(setup_param.total)} params incl. the "
            f"frozen card / hand encoder copies · {in_dim} → 1 "
            f"(state-only critic V(s))"
        ),
        dashed=not use_setup_model,
        panel=PANEL_SETUP,
        params_key=PARAMS_BLOCK_TOTAL,
    )


def _build_setup_unit(
    setup_param: setup_model.SetupParamReport,
    setup_arch: setup_model.SetupArchitecture,
    use_setup_model: bool,
) -> _Unit:
    """The setup column unit: single block normally, dual-head block when actor-critic.

    When ``setup_arch.use_policy_head`` is False, delegates to ``_setup_unit``
    unchanged.  When True, returns a header ``_Unit`` (shared embedder + optional
    trunk) with a ``dual`` pair of narrow sub-units — SETUP VALUE on the left,
    SETUP POLICY on the right — that ``_draw_dual_unit`` renders side by side
    below the header.
    """
    if not setup_arch.use_policy_head:
        return _setup_unit(setup_param, setup_arch, use_setup_model)

    # Trunk rows for the header (empty when no trunk).
    trunk_rows = (
        _op_rows(
            setup_param.trunk,
            between_activation=setup_arch.between_activation.value,
            # Trunk's final layer uses between_activation (not NONE) so the
            # output is nonlinear before the heads' first Linear.
            final_activation=setup_arch.between_activation.value,
            dropout=setup_arch.dropout,
        )
        if setup_param.trunk
        else ()
    )
    # The header carries the frozen embedders + the (policy-path) trunk, which
    # reads the fused state ⊕ action candidate; the value sub-unit reads its own
    # narrower state-only embedding (``value_in``).
    embed_in = (
        setup_param.trunk[0].in_features if setup_param.trunk else setup_param.policy_in
    )
    value_in = setup_param.value_in
    policy_in = setup_param.policy_in
    header_sigma = setup_param.embedder_params + setup_param.trunk_params
    status = "active" if use_setup_model else "off"
    header_title = "SETUP TRUNK"

    value_layers = setup_param.value_head
    policy_layers = (
        setup_param.policy_head
        if setup_param.policy_head is not None
        else setup_param.value_head
    )
    value_params = sum(layer.params for layer in value_layers)
    policy_params = sum(layer.params for layer in policy_layers)

    value_unit = _Unit(
        x=_SVG_COL_X[2],
        accent=_ACCENT_SETUP,
        title="SETUP VALUE",
        rows=_op_rows(
            value_layers,
            between_activation=setup_arch.between_activation.value,
            final_activation=setup_arch.final_activation.value,
            dropout=setup_arch.dropout,
        ),
        sigma_text=_count_text(value_params),
        in_label="state only",
        in_count=value_in,
        out_label="V(s)",
        out_count=1,
        tooltip=(
            f"Setup Value Head · {_count_text(value_params)} params · {value_in} → 1 "
            "(state-only critic V(s): tray / feeder / goals / bonus-on-offer, "
            "invariant to the chosen keep)"
        ),
        dashed=not use_setup_model,
        params_key=PARAMS_BLOCK_TOTAL,
    )
    policy_unit = _Unit(
        x=_SVG_COL_X[2],
        accent=_ACCENT_SETUP,
        title="SETUP POLICY",
        rows=_op_rows(
            policy_layers,
            between_activation=setup_arch.between_activation.value,
            final_activation=setup_arch.final_activation.value,
            dropout=setup_arch.dropout,
        ),
        sigma_text=_count_text(policy_params),
        in_label="state ⊕ keep",
        in_count=policy_in,
        out_label="log policy",
        out_count=1,
        tooltip=(
            f"Setup Policy Head · {_count_text(policy_params)} params · {policy_in} → 1 "
            "(log-probabilities over kept-card subsets, from the fused candidate)"
        ),
        dashed=not use_setup_model,
        params_key=PARAMS_BLOCK_TOTAL,
    )
    return _Unit(
        x=_SVG_COL_X[2],
        accent=_ACCENT_SETUP,
        title=header_title,
        rows=trunk_rows,
        sigma_text=_count_text(header_sigma),
        in_label="setup input",
        in_count=embed_in,
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
            block.layers,
            between_activation=arch.value_between_activation_resolved.value,
            final_activation=arch.value_final_activation_resolved.value,
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
            scorer.layers,
            between_activation=arch.head_between_activation_resolved.value,
            final_activation=arch.head_final_activation_resolved.value,
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
        title="POLICY HEAD",
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


def _fmt_dropout(dropout: float) -> str:
    """Format a dropout rate the same way the configurator terminal diagram does."""
    return f"{dropout:g}".lstrip("0")


def _op_rows(
    layers: tuple[architecture.LayerParam, ...],
    *,
    between_activation: str,
    final_activation: str,
    dropout: float = 0.0,
) -> tuple[_OpRow, ...]:
    """The mini-rows for a block: one Linear row per layer, with the activation
    rows the builders interleave. ``between_activation`` applies after every
    non-final layer; ``final_activation`` applies after the last layer. Either
    is skipped when its value is ``'none'``. When ``dropout > 0``, an amber
    Dropout row follows each activation, matching the configurator terminal diagram."""
    rows: list[_OpRow] = []
    for idx, layer in enumerate(layers):
        rows.append(
            _OpRow(
                kind=_OpKind.LINEAR,
                label=f"Linear →{layer.out_features}",
                params=layer.linear,
            )
        )
        is_final = idx == len(layers) - 1
        act_label = final_activation if is_final else between_activation
        if act_label != "none":
            rows.append(_OpRow(kind=_OpKind.ACT, label=act_label))
            if dropout > 0.0:
                rows.append(
                    _OpRow(
                        kind=_OpKind.DROPOUT, label=f"Dropout {_fmt_dropout(dropout)}"
                    )
                )
    return tuple(rows)


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

    In dual mode the column holds a shared trunk header block at the top, then
    two full-width sub-blocks stacked vertically (SETUP VALUE above SETUP POLICY),
    each separated by ``_DUAL_SPLIT_GAP`` pixels for the fork arrows.
    """
    if unit.dual is None:
        return _unit_h(len(unit.rows))
    # Header: input IO box + IO gap + block body (trunk rows).
    header_h = _SVG_IO_H + _SVG_IO_GAP + _block_body_h(len(unit.rows))
    value_h = _unit_h(len(unit.dual[0].rows))
    policy_h = _unit_h(len(unit.dual[1].rows))
    return header_h + _DUAL_SPLIT_GAP + value_h + _DUAL_SPLIT_GAP + policy_h


def _block_outer_h(unit: _Unit) -> int:
    """Pixel height a unit occupies from its top: a full unit (input box + body +
    output box), the dual-column stack, or — for a bare block — the body alone
    plus a top margin the height of an input box, so its body lines up with the
    bodies of neighbouring full blocks."""
    if unit.dual is not None:
        return _setup_col_h(unit)
    if unit.bare_block:
        return _SVG_IO_H + _SVG_IO_GAP + _block_body_h(len(unit.rows))
    return _unit_h(len(unit.rows))


def _resolve_geometry(units: _Units) -> _Geom:
    """Stack the four block rows and three connector bands top to bottom; every
    block in a visual row stretches to the row's tallest unit.

    The single-card encoder sits alone on row 1; its consumers occupy the
    always-present ``cons`` row below it (the hand-pooling / multi-card encoder
    block, plus board attention in col 0 when enabled).
    """
    row1_h = _block_outer_h(units.card)
    cons_row_h = _block_outer_h(units.hand)
    if units.attention is not None:
        cons_row_h = max(cons_row_h, _block_outer_h(units.attention))
    row2_h = max(
        _block_outer_h(units.trunk),
        _block_outer_h(units.choice),
        _block_outer_h(units.setup),
    )
    row3_h = max(_block_outer_h(units.value), _block_outer_h(units.decision))

    row1_y = _SVG_TOP
    band1_y = row1_y + row1_h
    cons_row_y = band1_y + _SVG_BAND_H
    band_cons_y = cons_row_y + cons_row_h
    row2_y = band_cons_y + _SVG_BAND_H
    band2_y = row2_y + row2_h
    row3_y = band2_y + _SVG_BAND_H
    total_y = row3_y + row3_h + _SVG_TOTAL_GAP
    return _Geom(
        row1_y=row1_y,
        row1_h=row1_h,
        cons_row_y=cons_row_y,
        cons_row_h=cons_row_h,
        row2_y=row2_y,
        row2_h=row2_h,
        row3_y=row3_y,
        row3_h=row3_h,
        band1_y=band1_y,
        band_cons_y=band_cons_y,
        band2_y=band2_y,
        total_y=total_y,
        svg_h=total_y + _SVG_BOTTOM,
    )


#### Connectors ####


def _consumer_connectors(
    geom: _Geom,
    units: _Units,
    arch: architecture.ModelArchitecture,
    use_setup_model: bool,
) -> list[_Conn]:
    """Fan-out from the single-card encoder into the consumers row and on to the
    row-2 inputs.  Counts come from the encode/state layout constants and the
    architecture flags — never hard-coded.

    The card embedding feeds board attention (the board path, col 0 when on) and
    — when the hand block pools the card table — the bare MULTI-CARD POOLING
    block.  The three card→{state,choice,setup} feeds share a trunk emitted by
    ``_trunk_svg``; this function only emits the card→hand thick line, the
    pooled-hand feeds, and the board path."""
    setup_cx = _SVG_COL_CX[2]
    band_cons_y, row2_y = geom.band_cons_y, geom.row2_y

    hand = units.hand
    hand_cx = hand.x + (hand.block_w or _SVG_COL_W) // 2
    hand_top_y = geom.cons_row_y + (
        0 if not hand.bare_block else _SVG_IO_H + _SVG_IO_GAP
    )
    hand_bottom_y = geom.cons_row_y + _block_outer_h(hand)

    conns: list[_Conn] = []

    # card embedding → MULTI-CARD POOLING: thick solid line, no label.
    # (The learned MULTI-CARD ENCODER carries its own input box instead.)
    if hand.bare_block:
        conns.append(
            _Conn(
                src_x=_ENC_CX + _X_CARD_HAND_SRC,
                src_y=geom.band1_y,
                dst_x=hand_cx,
                dst_y=hand_top_y,
                lane_y=geom.band1_y + _LANE_CARD_HAND,
                copies=encode.HAND_MULTIHOT_DIM,
                label="",
                stroke=_ACCENT_CARD,
            )
        )

    # pooled hand → choice encoder: the becomes_playable + becomes_unplayable stripes.
    # Lands at 2/3 the choice input box width (trunk→choice lands at center = 1/2).
    if hand.bare_block:
        conns.append(
            _Conn(
                src_x=hand_cx,
                src_y=hand_bottom_y,
                dst_x=_X_CHOICE_DST_POOL,
                dst_y=row2_y,
                lane_y=band_cons_y + _LANE_HAND_CHOICE,
                copies=1,
                label="×2",
                stroke=_ACCENT_HAND,
            )
        )

    # pooled hand → setup: kept set + playable_kept_cards (two pooled embeddings).
    conns.append(
        _Conn(
            src_x=hand_cx + _X_HAND_SETUP_SRC,
            src_y=hand_bottom_y,
            dst_x=setup_cx,
            dst_y=row2_y,
            lane_y=band_cons_y + _LANE_HAND_SETUP,
            copies=2,
            label="×2 · kept + playable",
            dashed=not use_setup_model,
            stroke=_ACCENT_HAND,
        )
    )

    # pooled hand → state encoder, and the board path (via attention, or direct).
    conns.append(_hand_state_conn(geom, arch, hand_cx, hand_bottom_y))
    conns.extend(_board_path_conns(geom, arch))
    return conns


def _hand_state_conn(
    geom: _Geom,
    arch: architecture.ModelArchitecture,
    hand_cx: int,
    src_y: int,
) -> _Conn:
    """The pooled-hand → state-encoder elbow.  The distinct multi-card encoder
    feeds its own-hand (and optional tray) set; the card-table pool feeds the
    hand multi-hot plus the extra playability multi-hots the trunk consumes."""
    trunk_cx = _SVG_COL_CX[0]
    if arch.use_distinct_hand_model:
        tray_set = arch.tray_set_embedding
        copies = 2 if tray_set else 1
        label = "×2 · own hand + tray set" if tray_set else "×1 · own hand"
    else:
        n_sets = 1 + encode.N_HAND_PLAYABLE_MULTIHOTS
        copies = n_sets
        label = f"×{n_sets} · hand + playable"
    return _Conn(
        src_x=hand_cx + _X_HAND_TRUNK_SRC,
        src_y=src_y,
        dst_x=trunk_cx + _X_HAND_TRUNK_DST,
        dst_y=geom.row2_y,
        lane_y=geom.band_cons_y + _LANE_HAND_STATE,
        copies=copies,
        label=label,
        stroke=_ACCENT_HAND,
    )


def _board_path_conns(geom: _Geom, arch: architecture.ModelArchitecture) -> list[_Conn]:
    """The board-slot card embeddings into the state encoder.

    Attention OFF: the trunk's State branch carries all card-index slots; no
    separate connector needed here.  Attention ON: board slots go CARD →
    ATTENTION → STATE as an elbow (band-1) then a straight vertical (band-cons);
    tray slots bypass attention via the trunk's State branch (×TRAY_SIZE)."""
    if not arch.use_board_attention:
        return []

    # card → BOARD ATTENTION: wide elbow in band-1 from encoder center to the
    # attention block's col-0 vertical (at _ATTN_VERT_X).
    return [
        _Conn(
            src_x=_ENC_CX + _X_CARD_ATTN_SRC,
            src_y=geom.band1_y,
            dst_x=_ATTN_VERT_X,
            dst_y=geom.cons_row_y,
            lane_y=geom.band1_y + _LANE_CARD_ATTN,
            copies=encode.N_BOARD_INDEX_SLOTS,
            label=f"×{encode.N_BOARD_INDEX_SLOTS} board",
            stroke=_ACCENT_CARD,
        ),
        # BOARD ATTENTION → STATE: straight vertical.
        _Conn(
            src_x=_ATTN_VERT_X,
            src_y=geom.band_cons_y,
            dst_x=_ATTN_VERT_X,
            dst_y=geom.row2_y,
            copies=encode.N_BOARD_INDEX_SLOTS,
            label=f"×{encode.N_BOARD_INDEX_SLOTS} attended",
            label_left=True,
            stroke=_ACCENT_ATTN,
        ),
    ]


def _trunk_svg(
    geom: _Geom,
    arch: architecture.ModelArchitecture,
    use_setup_model: bool,
    color: str = _ACCENT_CARD,
) -> tuple[list[str], list[str]]:
    """Shared trunk: vertical stem from encoder bottom to a split point in the
    cons band, then three branches to State / Choice / Setup.

    The stem at ``_TRUNK_X`` runs through the gutter between BOARD ATTENTION
    and MULTI-CARD POOLING.  Branch labels appear on the horizontal legs after
    the split.  ``_board_path_conns`` handles the separate attention path when
    board attention is on, so the State branch copies are scaled accordingly."""
    split_y = geom.band_cons_y + _LANE_TRUNK_SPLIT
    state_cx, choice_cx, setup_cx = _SVG_COL_CX

    bodies: list[str] = []
    labels: list[str] = []

    # Vertical stem (no arrowhead): encoder bottom → split point.
    stem_sw = _stroke_for(encode.N_CARD_INDEX_SLOTS)
    bodies.append(
        f'<line x1="{_TRUNK_X}" y1="{geom.band1_y}" x2="{_TRUNK_X}" y2="{split_y}" '
        f'stroke="{color}" stroke-width="{stem_sw}"/>'
    )

    # Left branch: trunk → State encoder.
    if arch.use_board_attention:
        state_copies = state.TRAY_SIZE
        state_label = f"×{state.TRAY_SIZE} tray"
    else:
        state_copies = encode.N_CARD_INDEX_SLOTS
        state_label = f"×{encode.N_CARD_INDEX_SLOTS}"
    state_pts = f"{_TRUNK_X},{split_y} {state_cx},{split_y} {state_cx},{geom.row2_y}"
    bodies.append(
        f'<polyline points="{state_pts}" fill="none" stroke="{color}" '
        f'stroke-width="{_stroke_for(state_copies)}" marker-end="url(#arr)"/>'
    )
    labels.append(
        _halo_text(
            (_TRUNK_X + state_cx) // 2, split_y - 6, state_label, anchor="middle"
        )
    )

    # Middle branch: trunk → Choice encoder (near-vertical, labeled to its right).
    bodies.append(
        f'<line x1="{choice_cx}" y1="{split_y}" x2="{choice_cx}" y2="{geom.row2_y}" '
        f'stroke="{color}" stroke-width="{_stroke_for(1)}" marker-end="url(#arr)"/>'
    )
    labels.append(
        _halo_text(
            choice_cx + 7,
            (split_y + geom.row2_y) // 2,
            "×1 candidate",
            anchor="start",
        )
    )

    # Right branch: trunk → Setup model (dashed when setup is off).
    dash_attr = ' stroke-dasharray="6,4"' if not use_setup_model else ""
    setup_pts = f"{_TRUNK_X},{split_y} {setup_cx},{split_y} {setup_cx},{geom.row2_y}"
    bodies.append(
        f'<polyline points="{setup_pts}" fill="none" stroke="{color}" '
        f'stroke-width="{_stroke_for(state.TRAY_SIZE)}"{dash_attr} marker-end="url(#arr)"/>'
    )
    labels.append(
        _halo_text(
            (_TRUNK_X + setup_cx) // 2,
            split_y - 6,
            f"×{state.TRAY_SIZE} tray",
            anchor="middle",
        )
    )

    return bodies, labels


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
    color = conn.stroke
    if lane_y is None:
        body = (
            f'<line x1="{conn.src_x}" y1="{conn.src_y}" x2="{conn.dst_x}" y2="{conn.dst_y}" '
            f'stroke="{color}" stroke-width="{stroke}"{dash} marker-end="url(#arr)"/>'
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

    # Dogleg: down a vertical corridor between the two consumer boxes, then out
    # to the destination column — a five-segment orthogonal path, labelled on
    # the final horizontal run near the destination.
    if conn.corridor_x is not None and conn.lane_y2 is not None:
        pts = (
            f"{conn.src_x},{conn.src_y} {conn.src_x},{lane_y} "
            f"{conn.corridor_x},{lane_y} {conn.corridor_x},{conn.lane_y2} "
            f"{conn.dst_x},{conn.lane_y2} {conn.dst_x},{conn.dst_y}"
        )
        body = (
            f'<polyline points="{pts}" fill="none" stroke="{color}" '
            f'stroke-width="{stroke}"{dash} marker-end="url(#arr)"/>'
        )
        label = ""
        if conn.label:
            label_x = (conn.corridor_x + conn.dst_x) // 2 + conn.label_dx
            label = _halo_text(label_x, conn.lane_y2 - 6, conn.label, anchor="middle")
        return body, label

    # Orthogonal elbow along a horizontal lane, label centered above the run.
    pts = (
        f"{conn.src_x},{conn.src_y} {conn.src_x},{lane_y} "
        f"{conn.dst_x},{lane_y} {conn.dst_x},{conn.dst_y}"
    )
    body = (
        f'<polyline points="{pts}" fill="none" stroke="{color}" '
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
    A unit with ``dual`` set is rendered as a header trunk block that forks
    into two full-width sub-units stacked vertically below.  A ``bare_block`` unit is drawn as
    its block body alone (no I/O boxes), at its ``block_w`` width."""
    if unit.dual is not None:
        return _draw_dual_unit(unit, top_y, unit_h)
    if unit.bare_block:
        return _draw_bare_unit(unit, top_y, unit_h)
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


def _draw_bare_unit(unit: _Unit, top_y: int, unit_h: int) -> str:
    """A parameter-less block (e.g. card-table hand pooling) drawn as its block
    body only — no input/output vector boxes — at ``block_w`` width, with a top
    margin the height of an input box so its body lines up with the bodies of
    neighbouring full blocks.  Keeps the hover tooltip and the parameter-total
    click; the removed input box drops its detail-panel link."""
    block_y = top_y + _SVG_IO_H + _SVG_IO_GAP
    body_h = unit_h - (_SVG_IO_H + _SVG_IO_GAP)
    col_w = unit.block_w if unit.block_w is not None else _SVG_COL_W
    return "\n".join(
        [
            "<g>",
            f"<title>{html_lib.escape(unit.tooltip)}</title>",
            _draw_block(unit, block_y, body_h, col_w=col_w),
            "</g>",
        ]
    )


def _draw_dual_unit(unit: _Unit, top_y: int, unit_h: int) -> str:
    """Actor-critic setup column: full-width header trunk block → fork arrows →
    two full-width sub-blocks stacked vertically (SETUP VALUE above SETUP POLICY).

    Layout (top to bottom within the allocated ``unit_h``):
    * Input IO box at full column width (clickable to the setup panel).
    * Header block body at full width (trunk rows, if any).
    * ``_DUAL_SPLIT_GAP`` pixels with two vertical fork arrows (one per head).
    * SETUP VALUE — full-width unit with its own input/output IO boxes.
    * ``_DUAL_SPLIT_GAP`` pixels of vertical separation.
    * SETUP POLICY — full-width unit with its own input/output IO boxes.

    Draw order puts the long header→policy arrow behind the value block so the
    value block's white rect naturally masks its mid-section, leaving the arrow
    visible above and below the value block.
    """
    assert unit.dual is not None
    value_unit, policy_unit = unit.dual

    # Header block occupies the top of the allocated column height.
    header_body_h = _block_body_h(len(unit.rows))
    header_block_y = top_y + _SVG_IO_H + _SVG_IO_GAP
    header_bottom = header_block_y + header_body_h

    # Sub-blocks stacked below the header, each at full column width.
    value_h = _unit_h(len(value_unit.rows))
    policy_h = _unit_h(len(policy_unit.rows))
    value_top = header_bottom + _DUAL_SPLIT_GAP
    policy_top = value_top + value_h + _DUAL_SPLIT_GAP

    # Fork x-positions offset slightly off-center to read as two distinct branches.
    cx = unit.x + _SVG_COL_W // 2

    # Shared trunk input box (full width, clickable to the setup panel).
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

    def _fork_arrow(src_x: int, dst_y: int) -> str:
        return (
            f'<line x1="{src_x}" y1="{header_bottom}" x2="{src_x}" y2="{dst_y}" '
            f'fill="none" stroke="{_SVG_ARROW}" '
            f'stroke-width="{_STROKE_SINGLE}" marker-end="url(#arr)"/>'
        )

    parts = [
        "<g>",
        f"<title>{html_lib.escape(unit.tooltip)}</title>",
        in_box,
        _draw_block(unit, header_block_y, header_body_h, col_w=_SVG_COL_W),
        # Long fork arrow to policy drawn first — value block will mask its mid-section.
        _fork_arrow(cx + 15, policy_top),
        _draw_unit(value_unit, value_top, value_h),
        # Short fork arrow to value drawn after value block so arrowhead is on top.
        _fork_arrow(cx - 15, value_top),
        _draw_unit(policy_unit, policy_top, policy_h),
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

    # Block parameter total, overlaid on the bottom-right border (skipped when empty).
    if unit.sigma_text:
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
        color = (
            _SVG_LINEAR_COLOR
            if row.kind is _OpKind.LINEAR
            else _SVG_DROPOUT_COLOR if row.kind is _OpKind.DROPOUT else _SVG_ACT_COLOR
        )
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
    setup_param: setup_model.SetupParamReport,
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
        f"×1 per choice) and Multi-Card Encoder "
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
    """The annotation beside the encoder column: both shared encoders (the
    single-card encoder and the multi-card encoder in the consumers row below
    it) are trained by in-game decisions only — setup experience never reaches
    them (the setup net carries frozen, synced copies)."""
    note_x = _SVG_COL_X[2] + 12
    mid_y = (geom.row1_y + geom.band_cons_y) // 2
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
    setup_param: setup_model.SetupParamReport,
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
