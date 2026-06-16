"""The TRAINING IMPROVEMENT panel body (``GettingBetterChart``): the docked
eval inset plus the two side-by-side convergence line charts, and all the
block- and series-level helpers that draw them onto a braille canvas."""

from __future__ import annotations

import typing

import pydantic
import rich.console as rich_console
from rich import segment, text

from wingspan.training import convergence, metrics, metrics_log, runstate, theme
from wingspan.training.charts import braille, geometry, insets, text_helpers


class GettingBetterChart:
    """The single "TRAINING IMPROVEMENT" panel body: the left-docked EVAL inset
    (the cinematic hero win-rate plus its last/EWMA readouts) followed by two
    side-by-side line charts on their own axes. WIN RATE spans the whole run (no
    sliding window, read from ``metrics.jsonl``) with the yellow opponent-advance
    threshold line and a vertical marker at each challenger upgrade — a single
    EWMA series.  The y-axis top is always 100%; the floor is set dynamically
    from the global minimum EWMA value (floored to a 5% step) so the scale uses
    the available vertical space.  Segments below 50% are drawn in red; a solid
    dim-red baseline marks the 50% level.  FINAL SCORE / MARGIN is a dual-axis
    chart: the EWMA final score on a color-coded left axis and the EWMA eval
    margin on a color-coded right axis, each scaled to its own visible range,
    over a sliding 2000-iteration window pinned to a round left edge. When the
    panel is too narrow the inset drops to a one-line strip beneath the charts.
    The win-rate sawtooths back down each time the reference opponent is advanced
    (it then climbs again vs a stronger self)."""

    def __init__(self, state: runstate.RunState, frame: int):
        self.state = state
        self.frame = frame

    def __rich_console__(
        self, console: rich_console.Console, options: rich_console.ConsoleOptions
    ) -> rich_console.RenderResult:
        width = options.max_width
        height = (
            options.height if options.height is not None else (options.max_height or 16)
        )
        layout = _compute_chart_layout(width, height)
        if not layout.renderable:
            yield text.Text("  collecting data…", style=theme.TEXT_MUTED)
            return

        rows = layout.rows
        beacon_color = theme.BEACON_B if self.frame % 2 else theme.BEACON_A
        # The convergence charts read the full on-disk history (beyond the
        # in-memory cap) so WIN RATE can span the whole run and FINAL SCORE /
        # MARGIN can show a 2000-iteration window; before the first row is
        # flushed (or in tests with no run on disk) we fall back to the live
        # in-memory history.
        full_history = metrics_log.read_iteration_history(
            self.state.config.run.checkpoint_dir
        )
        if not full_history:
            full_history = self.state.history
        left = _winrate_block(
            self.state, full_history, layout.left_w, rows, beacon_color
        )
        right = _score_margin_block(
            self.state, full_history, layout.right_w, rows, beacon_color
        )
        merged = _join_columns(
            [(left, layout.left_w), (right, layout.right_w)], geometry.CHART_GAP
        )

        random_phase = _is_random_phase(self.state)
        if layout.inset:
            box_lines = (
                insets.collect_inset(self.state, len(merged))
                if random_phase
                else insets.eval_inset(self.state, len(merged))
            )
            merged = _merge_columns(box_lines, merged, geometry.INSET_W)
        else:
            merged.extend(
                insets.collect_strip(self.state)
                if random_phase
                else insets.eval_strip(self.state)
            )

        for i, line in enumerate(merged):
            if i:
                yield segment.Segment.line()
            yield line


class _ChartLayout(pydantic.BaseModel):
    """Geometry for one convergence-panel paint: whether the eval inset docks,
    the two chart column widths, the plot-row count, and whether there is room
    to plot at all."""

    inset: bool
    left_w: int
    right_w: int
    rows: int
    renderable: bool


def _compute_chart_layout(width: int, height: int) -> _ChartLayout:
    """Split the panel width into the optional docked inset plus two side-by-side
    charts, and the height into plot rows (both charts and the inset fill the full
    height). ``renderable`` is False when the panel is too small, in which case the
    caller paints a placeholder instead."""
    inset = width >= geometry.INSET_MIN_WIDTH
    inset_w = geometry.INSET_W if inset else 0
    charts_w = width - inset_w - (1 if inset else 0)
    left_w = (charts_w - geometry.CHART_GAP) // 2
    right_w = charts_w - geometry.CHART_GAP - left_w
    plot_rows = height - geometry.TITLE_ROWS - geometry.AXIS_ROWS
    return _ChartLayout(
        inset=inset,
        left_w=left_w,
        right_w=right_w,
        rows=plot_rows,
        renderable=plot_rows >= geometry.MIN_PLOT_ROWS and left_w >= 16,
    )


def _winrate_block(
    state: runstate.RunState,
    history: list[metrics.IterationMetrics],
    width: int,
    rows: int,
    beacon_color: str,
) -> list[text.Text]:
    """The win-rate plot block (title + plot grid + x-axis), exactly
    ``geometry.TITLE_ROWS + rows + geometry.AXIS_ROWS`` lines tall. Plots the
    EWMA win-rate series over the whole run with dynamic y-axis scaling: the top
    is always 100% and the floor is derived from the minimum EWMA value seen
    (rounded to a stable 5% step). Segments above 50% are drawn in green;
    segments below 50% are drawn in red. A solid dim-red horizontal line marks
    the 50% level when it falls within the chart range."""
    plot_cols = max(1, width - geometry.GUTTER_W)
    it_lo, it_hi = convergence.full_range(history)
    ewma = convergence.winrate_ewma_points(
        history, state.config.opponent.eval_ewma_alpha
    )

    # Dynamic floor derived from the global minimum in history; series 0 = red
    # (below 50%, higher canvas priority), series 1 = green (above 50%).
    v_lo = _winrate_v_lo(ewma)
    v_hi = 100.0

    canvas = braille.BrailleCanvas(plot_cols, rows, 2)
    _draw_series_split(
        canvas,
        series_above=1,
        series_below=0,
        points=ewma,
        it_lo=it_lo,
        it_hi=it_hi,
        v_lo=v_lo,
        v_hi=v_hi,
        threshold=50.0,
    )
    beacon = _beacon_cell(canvas, ewma, it_lo, it_hi, v_lo, v_hi)

    threshold = _winrate_threshold_pct(state)
    target_row = (
        round((1.0 - (threshold - v_lo) / (v_hi - v_lo)) * (rows - 1))
        if threshold > v_lo
        else None
    )
    fifty_row = (
        round((1.0 - (50.0 - v_lo) / (v_hi - v_lo)) * (rows - 1))
        if v_lo < 50.0
        else None
    )
    challenger_markers = convergence.marker_columns(
        state.opponent_change_iterations, it_lo, it_hi, plot_cols
    )
    setup_markers = convergence.marker_columns(
        convergence.setup_transition_iterations(history), it_lo, it_hi, plot_cols
    )
    grid = _render_plot_grid(
        canvas,
        {0: theme.WIN_BELOW_50, 1: theme.WIN_COLOR},
        _winrate_label_rows(rows, v_lo),
        beacon,
        beacon_color,
        target_row,
        theme.WIN_THRESHOLD,
        challenger_markers,
        theme.CHALLENGER_MARK,
        setup_markers,
        theme.SETUP_MARK,
        baseline_row=fifty_row,
        baseline_color=theme.FIFTY_PCT_LINE,
    )
    return [_winrate_title(state, width), *grid, *_axis_two(it_lo, it_hi, plot_cols)]


def _score_margin_block(
    state: runstate.RunState,
    history: list[metrics.IterationMetrics],
    width: int,
    rows: int,
    beacon_color: str,
) -> list[text.Text]:
    """The FINAL SCORE / MARGIN dual-axis plot block: the EWMA final score on a
    color-coded left axis and the EWMA eval margin on a color-coded right axis,
    each scaled to its own visible range over the sliding pinned window. The
    margin axis carries a zero line when its range straddles zero."""
    plot_cols = max(1, width - geometry.GUTTER_W * 2)
    it_lo, it_hi = convergence.score_margin_window(history)
    alpha = state.config.opponent.eval_ewma_alpha
    score = [
        pt for pt in convergence.score_ewma_points(history, alpha) if pt[0] >= it_lo
    ]
    margin = [
        pt for pt in convergence.margin_ewma_points(history, alpha) if pt[0] >= it_lo
    ]
    pts_lo, pts_hi = _value_span_padded([value for _, value in score])
    mar_lo, mar_hi = _value_span_padded([value for _, value in margin])

    canvas = braille.BrailleCanvas(plot_cols, rows, 2)
    _draw_series(canvas, 0, score, it_lo, it_hi, pts_lo, pts_hi, dotted=False)
    _draw_series(canvas, 1, margin, it_lo, it_hi, mar_lo, mar_hi, dotted=False)
    beacon = _beacon_cell(canvas, score, it_lo, it_hi, pts_lo, pts_hi)

    zero_row = (
        _value_row(0.0, mar_lo, mar_hi, rows) if mar_lo <= 0.0 <= mar_hi else None
    )
    grid = _render_dual_axis_grid(
        canvas,
        beacon,
        beacon_color,
        _auto_label_rows(rows, pts_lo, pts_hi),
        _auto_label_rows(rows, mar_lo, mar_hi),
        zero_row,
    )
    return [
        _score_margin_title(state, width),
        *grid,
        *_axis_two(it_lo, it_hi, plot_cols),
    ]


def _winrate_title(state: runstate.RunState, width: int) -> text.Text:
    """``WIN RATE vs <opponent>`` — the win-rate value itself is dropped here
    since it is shown large in the docked box (the EVAL win-rate, or the
    collection win-rate during the random-opponent bootstrap phase)."""
    title = text.Text(no_wrap=True, end="", overflow="ellipsis")
    title.append("WIN RATE", style=f"bold {theme.WIN_COLOR}")
    if _is_random_phase(state):
        title.append(" vs random · collect", style=theme.AXIS)
    else:
        gen = state.opponent_generation
        opponent = "random" if gen == 0 else f"self·gen{gen}"
        title.append(f" vs {opponent}", style=theme.AXIS)
    return _pad_to(title, width)


def _is_random_phase(state: runstate.RunState) -> bool:
    return state.training_phase == runstate.TrainingPhase.RANDOM_OPPONENT


def _winrate_threshold_pct(state: runstate.RunState) -> float:
    """The win-rate the dashed threshold line marks: the bootstrap graduation
    bar during the random phase, the opponent-advance bar otherwise."""
    if _is_random_phase(state):
        return state.config.opponent.random_phase_win_rate * 100.0
    return state.config.opponent.opponent_reset_win_rate * 100.0


def _score_margin_title(state: runstate.RunState, width: int) -> text.Text:
    """``FINAL SCORE / MARGIN`` — each word colored to match its line and axis."""
    title = text.Text(no_wrap=True, end="", overflow="ellipsis")
    title.append("FINAL SCORE", style=f"bold {theme.POINTS_COLOR}")
    title.append(" / ", style=theme.AXIS)
    title.append("MARGIN", style=f"bold {theme.MARGIN_COLOR}")
    return _pad_to(title, width)


def _pad_to(line: text.Text, width: int) -> text.Text:
    pad = width - line.cell_len
    if pad > 0:
        line.append(" " * pad)
    return line


def _join_columns(
    blocks: list[tuple[list[text.Text], int]], gap: int
) -> list[text.Text]:
    """Concatenate equal-height line blocks side by side, each padded to its own
    width with ``gap`` blank columns between them (the shared
    :func:`text_helpers.join_columns`)."""
    return text_helpers.join_columns(blocks, gap)


# -- shared plot grid + axis -------------------------------------------------


def _render_plot_grid(
    canvas: braille.BrailleCanvas,
    series_color: dict[int, str],
    label_rows: dict[int, str],
    beacon_cell: tuple[int, int] | None,
    beacon_color: str,
    target_row: int | None,
    target_color: str,
    marker_cols: typing.AbstractSet[int] = frozenset(),
    marker_color: str = theme.AXIS,
    marker2_cols: typing.AbstractSet[int] = frozenset(),
    marker2_color: str = theme.AXIS,
    baseline_row: int | None = None,
    baseline_color: str = "",
) -> list[text.Text]:
    """Paint the braille canvas into colored text rows with a left value gutter,
    an optional dotted gridline (``target_row`` in ``target_color`` — the
    win-rate threshold or the points zero line), an optional solid baseline
    gridline (``baseline_row`` in ``baseline_color`` — the 50% neutral line),
    up to two sets of optional vertical markers (``marker_cols`` /
    ``marker2_cols`` in their respective colors — challenger-upgrade and
    setup-phase-transition lines), and the leading-edge beacon. Adjacent
    same-color cells are batched into one styled run; markers and gridlines
    only fill cells the data line did not. ``marker_cols`` takes priority over
    ``marker2_cols`` when they overlap."""
    lines: list[text.Text] = []
    for row in range(canvas.rows):
        line = text.Text(no_wrap=True, end="")
        line.append(_gutter(row, label_rows), style=theme.AXIS)
        run = ""
        run_color = ""
        for col in range(canvas.cols):
            char, owner = canvas.cell(row, col)
            if beacon_cell == (row, col):
                char, color = "●", beacon_color
            elif owner >= 0:
                color = series_color[owner]
            elif target_row is not None and row == target_row and char == " ":
                char, color = "┄", target_color
            elif baseline_row is not None and row == baseline_row and char == " ":
                char, color = "─", baseline_color
            elif col in marker_cols and char == " ":
                char, color = "┊", marker_color
            elif col in marker2_cols and char == " ":
                char, color = "┊", marker2_color
            else:
                color = theme.AXIS
            if color != run_color:
                if run:
                    line.append(run, style=run_color)
                run, run_color = char, color
            else:
                run += char
        if run:
            line.append(run, style=run_color)
        lines.append(line)
    return lines


def _render_dual_axis_grid(
    canvas: braille.BrailleCanvas,
    beacon_cell: tuple[int, int] | None,
    beacon_color: str,
    left_labels: dict[int, str],
    right_labels: dict[int, str],
    zero_row: int | None,
) -> list[text.Text]:
    """Paint a two-series canvas with a color-coded gutter on *each* side: the
    FINAL SCORE value axis (amber) on the left and the MARGIN value axis (teal)
    on the right, each matching its line's color. The margin zero line is drawn
    faintly where the canvas is otherwise empty."""
    series_color = {0: theme.POINTS_COLOR, 1: theme.MARGIN_COLOR}
    lines: list[text.Text] = []
    for row in range(canvas.rows):
        line = text.Text(no_wrap=True, end="")
        line.append(_gutter(row, left_labels), style=theme.POINTS_COLOR)
        run = ""
        run_color = ""
        for col in range(canvas.cols):
            char, owner = canvas.cell(row, col)
            if beacon_cell == (row, col):
                char, color = "●", beacon_color
            elif owner >= 0:
                color = series_color[owner]
            elif zero_row is not None and row == zero_row and char == " ":
                char, color = "┄", theme.TARGET_GRID
            else:
                color = theme.AXIS
            if color != run_color:
                if run:
                    line.append(run, style=run_color)
                run, run_color = char, color
            else:
                run += char
        if run:
            line.append(run, style=run_color)
        line.append(_gutter_right(row, right_labels), style=theme.MARGIN_COLOR)
        lines.append(line)
    return lines


def _axis_two(it_lo: int, it_hi: int, cols: int) -> list[text.Text]:
    """The shared bottom two rows of a plot: the tick ruler and the iteration
    labels. The labels span the displayed ``[it_lo, it_hi]`` iteration window so
    they track the sliding x-axis. Every label is left-aligned under its tick;
    the tick count is reduced until consecutive labels have room (a number plus at
    least two trailing columns) so they never collide at 5-digit iterations."""
    label_w = len(str(it_hi))  # it_hi is the widest label (the most digits)
    min_spacing = label_w + 2  # a number plus at least two trailing spaces
    # Hold the rightmost tick ``label_w`` columns in from the edge so its
    # left-aligned label still fits on-screen instead of overflowing.
    track = max(1, cols - label_w)
    n_ticks = max(2, min(12, cols // 8))
    while n_ticks > 2 and track / (n_ticks - 1) < min_spacing:
        n_ticks -= 1
    tick_cols = [round(i * track / (n_ticks - 1)) for i in range(n_ticks)]

    ruler = text.Text(no_wrap=True, end="")
    ruler.append(" " * (geometry.GUTTER_W - 1) + "└", style=theme.AXIS)
    ruler.append(
        "".join("┬" if col in tick_cols else "─" for col in range(cols)),
        style=theme.AXIS,
    )

    labels = text.Text(no_wrap=True, end="")
    labels.append(" " * geometry.GUTTER_W, style=theme.AXIS)
    labels.append(_tick_labels(tick_cols, cols, it_lo, it_hi), style=theme.TEXT_MUTED)

    return [ruler, labels]


def _value_row(value: float, v_lo: float, v_hi: float, rows: int) -> int:
    """The plot-row index a value lands on for an auto-scaled ``[v_lo, v_hi]``
    axis (row 0 is the top)."""
    frac = (value - v_lo) / (v_hi - v_lo) if v_hi > v_lo else 0.5
    return round((1.0 - frac) * (rows - 1))


def _value_span_padded(values: list[float]) -> tuple[float, float]:
    """An auto-scaled ``(lo, hi)`` axis range for a value series, with a small
    margin above and below so the topmost / bottommost line is not flush against
    the panel edge."""
    if not values:
        return (0.0, 1.0)
    lo, hi = min(values), max(values)
    if hi <= lo:
        return (lo - 1.0, hi + 1.0)
    pad = (hi - lo) * 0.08
    return (lo - pad, hi + pad)


def _draw_series(
    canvas: braille.BrailleCanvas,
    series: int,
    points: list[tuple[int, float]],
    it_lo: int,
    it_hi: int,
    v_lo: float,
    v_hi: float,
    dotted: bool,
) -> None:
    if not points:
        return
    dots = [
        _to_dot(canvas, it, value, it_lo, it_hi, v_lo, v_hi) for it, value in points
    ]
    if len(dots) == 1:
        canvas.set_dot(dots[0][0], dots[0][1], series)
        return
    for (x0, y0), (x1, y1) in zip(dots, dots[1:]):
        canvas.line(x0, y0, x1, y1, series, dotted=dotted)


def _draw_series_split(
    canvas: braille.BrailleCanvas,
    series_above: int,
    series_below: int,
    points: list[tuple[int, float]],
    it_lo: int,
    it_hi: int,
    v_lo: float,
    v_hi: float,
    threshold: float,
) -> None:
    """Draw a line in two series split at ``threshold``, handling crossings.

    Segments where the value is >= threshold go to ``series_above``; segments
    below go to ``series_below``.  When a segment crosses the threshold the
    crossing pixel is interpolated and each sub-segment is routed to its
    correct series so the color boundary is precise.
    """
    if not points:
        return
    dots = [
        _to_dot(canvas, it, value, it_lo, it_hi, v_lo, v_hi) for it, value in points
    ]

    # Threshold in braille dot-pixel space (higher py = lower value on screen).
    y_thresh = round((1.0 - (threshold - v_lo) / (v_hi - v_lo)) * (canvas.dot_h - 1))

    if len(dots) == 1:
        series = series_above if dots[0][1] <= y_thresh else series_below
        canvas.set_dot(dots[0][0], dots[0][1], series)
        return

    for (x0, y0), (x1, y1) in zip(dots, dots[1:]):
        above0 = y0 <= y_thresh  # lower py means higher value (above threshold)
        above1 = y1 <= y_thresh
        if above0 == above1:
            # Whole segment on one side — no crossing.
            canvas.line(x0, y0, x1, y1, series_above if above0 else series_below)
        else:
            # Segment crosses the threshold; interpolate the crossing pixel.
            dy = y1 - y0
            x_cross = x0 + round((x1 - x0) * (y_thresh - y0) / dy) if dy else x0
            canvas.line(
                x0, y0, x_cross, y_thresh, series_above if above0 else series_below
            )
            canvas.line(
                x_cross, y_thresh, x1, y1, series_above if above1 else series_below
            )


def _to_dot(
    canvas: braille.BrailleCanvas,
    iteration: int,
    value: float,
    it_lo: int,
    it_hi: int,
    v_lo: float,
    v_hi: float,
) -> tuple[int, int]:
    x_frac = (iteration - it_lo) / (it_hi - it_lo) if it_hi > it_lo else 1.0
    y_frac = (value - v_lo) / (v_hi - v_lo) if v_hi > v_lo else 0.5
    px = round(x_frac * (canvas.dot_w - 1))
    py = round((1.0 - y_frac) * (canvas.dot_h - 1))
    return (px, py)


def _beacon_cell(
    canvas: braille.BrailleCanvas,
    points: list[tuple[int, float]],
    it_lo: int,
    it_hi: int,
    v_lo: float,
    v_hi: float,
) -> tuple[int, int] | None:
    """The ``(row, col)`` of the leading edge of ``points`` on ``canvas``'s value
    axis, for the pulsing beacon (None if the series is empty)."""
    if not points:
        return None
    px, py = _to_dot(canvas, points[-1][0], points[-1][1], it_lo, it_hi, v_lo, v_hi)
    return (py // 4, px // 2)


def _winrate_v_lo(ewma: list[tuple[int, float]]) -> float:
    """Dynamic y-axis floor for the win-rate chart.

    Takes the global minimum EWMA value seen in history, floors it to the
    nearest 5%, then subtracts a 5% breathing margin.  Using the global
    minimum means the floor can only move down, never jump upward mid-run.
    """
    if not ewma:
        return 0.0
    min_val = min(val for _, val in ewma)
    floor5 = (min_val // 5) * 5
    return max(0.0, floor5 - 5.0)


def _winrate_label_rows(rows: int, v_lo: float) -> dict[int, str]:
    """Map plot-row index -> percent gutter label for the win-rate axis.

    Places ``geometry.WIN_RATE_TICK_COUNT`` ticks at equidistant row positions
    that never move, with label values recomputed from the current ``v_lo`` so
    the scale adapts without the tick marks jumping around.
    """
    tick_count = geometry.WIN_RATE_TICK_COUNT
    labels: dict[int, str] = {}
    for k in range(tick_count):
        row = round(k * (rows - 1) / (tick_count - 1))
        val = 100.0 - k * (100.0 - v_lo) / (tick_count - 1)
        labels[row] = f"{round(val)}%"
    return labels


def _auto_label_rows(rows: int, lo: float, hi: float) -> dict[int, str]:
    """Map plot-row index -> integer gutter label for an auto-scaled value axis."""
    labels: dict[int, str] = {}
    for tick in range(geometry.POINTS_AXIS_TICKS):
        frac = tick / (geometry.POINTS_AXIS_TICKS - 1)
        value = lo + frac * (hi - lo)
        labels[round((1 - frac) * (rows - 1))] = f"{value:.0f}"
    return labels


def _gutter(row: int, label_rows: dict[int, str]) -> str:
    """The 5-cell left gutter for ``row``: a right-justified value label + tick."""
    return f"{label_rows.get(row, ''):>4}┤"


def _gutter_right(row: int, label_rows: dict[int, str]) -> str:
    """The 5-cell right gutter for ``row`` (the dual-axis chart's MARGIN axis): a
    tick + left-justified value label, mirroring :func:`_gutter`."""
    return f"├{label_rows.get(row, ''):<4}"


def _tick_labels(tick_cols: list[int], cols: int, it_lo: int, it_hi: int) -> str:
    """A row of left-aligned iteration numbers beneath the axis ticks, spanning
    the displayed ``[it_lo, it_hi]`` window. Each label starts at its tick column
    (the caller spaces the ticks far enough apart that they do not collide and
    holds the last tick in from the edge so its number still fits). Values are
    taken by tick fraction so the first and last labels read exactly it_lo / it_hi."""
    chars = [" "] * cols
    span = it_hi - it_lo
    last_index = len(tick_cols) - 1
    for index, col in enumerate(tick_cols):
        iteration = it_lo + round(index / max(last_index, 1) * span)
        label = str(iteration)
        for offset, char in enumerate(label):
            if 0 <= col + offset < cols:
                chars[col + offset] = char
    return "".join(chars)


def _merge_columns(
    left: list[text.Text], right: list[text.Text], left_width: int
) -> list[text.Text]:
    """Place ``right`` lines flush against ``left`` lines padded to ``left_width``
    (one-space gutter), so the chart and the docked inset never misalign."""
    merged: list[text.Text] = []
    rows = max(len(left), len(right))
    for i in range(rows):
        line = text.Text(no_wrap=True, end="")
        if i < len(left):
            line.append_text(left[i])
            pad = left_width - left[i].cell_len
        else:
            pad = left_width
        if pad > 0:
            line.append(" " * pad)
        line.append(" ")
        if i < len(right):
            line.append_text(right[i])
        merged.append(line)
    return merged
