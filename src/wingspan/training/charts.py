"""Custom Rich renderables: the convergence chart, the family histogram, and
the small bar/sparkline helpers the rest of the dashboard reuses.

The charts are real ``__rich_console__`` renderables: each reads
``options.max_width`` / ``options.max_height`` every refresh and rebuilds its
canvas to fill the panel it sits in, so the dashboard reflows with the terminal.
Two drawing primitives do the heavy lifting:

* :class:`BrailleCanvas` — a 2×4-dots-per-cell Unicode-braille bitmap with one
  bit-plane per data series, so several lines share one un-padded canvas and a
  cell touched by more than one series takes the higher-priority series' color.
* the eighth-block ramps (``▏▎▍▌▋▊▉█`` / ``▁▂▃▄▅▆▇█``) used for the score bar,
  the histogram bars, and the sparklines.

The pure series + axis-window math (the EWMA series, the full/sliding x-ranges,
the challenger markers) lives in :mod:`wingspan.training.convergence`; this
module owns only the rendering. Everything degrades gracefully: too-small panels
drop the eval inset, then fall back to a one-line strip.
"""

from __future__ import annotations

import typing

import rich.console as rich_console
from rich import segment, text

from wingspan import decisions
from wingspan.training import convergence, metrics, metrics_log, runstate, theme

# Braille cell geometry: 2 dots wide × 4 tall, base codepoint U+2800.
BRAILLE_BASE = 0x2800
BRAILLE_BITS: dict[tuple[int, int], int] = {
    (0, 0): 0x01,
    (0, 1): 0x02,
    (0, 2): 0x04,
    (0, 3): 0x40,
    (1, 0): 0x08,
    (1, 1): 0x10,
    (1, 2): 0x20,
    (1, 3): 0x80,
}

_GUTTER_W = 5  # "100%" (4) + "┤" tick
_TITLE_ROWS = 1  # the per-chart title row above each plot
_AXIS_ROWS = 2  # axis ruler + iteration labels
_CHART_GAP = 2  # blank columns between the two side-by-side charts
_INSET_W = 28  # docked eval inset width
_INSET_MIN_WIDTH = 96  # below this the inset moves below the side-by-side charts
# The win-rate plot is pinned to exactly 11 rows so its 0..100% axis lands a
# gridline every 2 rows (10% per row → 20% per 2 rows). The points plot matches
# the same row count so the two charts stay vertically aligned for the merge.
_WINRATE_PLOT_ROWS = 11
_Y_LABELS = (100, 80, 60, 40, 20, 0)  # win-rate axis percent gridlines
_POINTS_AXIS_TICKS = 5  # gridline count on the auto-scaled points axis
# Retained for the configurator's history-length hint in
# ``wingspan.training.configure.screen`` — the charts themselves now read the
# full on-disk history (see ``convergence`` for the live window sizes).
CHART_WINDOW = 500


# ---------------------------------------------------------------------------
# Small text helpers (shared by the dashboard panels)


def sparkline(values: typing.Sequence[float], width: int) -> str:
    """An 8-step block sparkline of the trailing ``width`` values."""
    if width <= 0 or not values:
        return ""
    window = list(values[-width:])
    lo, hi = min(window), max(window)
    span = hi - lo
    ramp = theme.SPARK_RAMP
    if span <= 0:
        return ramp[len(ramp) // 2] * len(window)
    return "".join(
        ramp[min(len(ramp) - 1, int((value - lo) / span * (len(ramp) - 1)))]
        for value in window
    )


def eighth_bar(fraction: float, width: int, min_tick: bool = False) -> str:
    """Horizontal eighth-block bar of ``fraction`` of ``width`` cells.

    With ``min_tick`` a positive fraction that rounds to nothing still shows a
    single ``▏`` so a tiny-but-nonzero value never vanishes.
    """
    fraction = max(0.0, min(1.0, fraction))
    eighths = round(fraction * width * 8)
    full, remainder = divmod(eighths, 8)
    bar = "█" * full
    if remainder:
        bar += theme.BAR8_H_RAMP[remainder]
    if not bar and min_tick and fraction > 0:
        bar = "▏"
    return bar


def human_count(value: int) -> str:
    """Compact thousands formatting: ``842`` / ``2.6k`` / ``951k`` / ``1.20M``."""
    if value < 1000:
        return str(value)
    if value < 1_000_000:
        thousands = value / 1000.0
        return f"{thousands:.1f}k" if thousands < 10 else f"{thousands:.0f}k"
    return f"{value / 1_000_000:.2f}M"


# ---------------------------------------------------------------------------
# Braille canvas


class BrailleCanvas:
    """A multi-series braille bitmap. Each series has its own bit-plane; a cell
    is colored by the lowest-id (highest-priority) series that lit any dot in
    it."""

    def __init__(self, cols: int, rows: int, n_series: int):
        self.cols = cols
        self.rows = rows
        self.dot_w = cols * 2
        self.dot_h = rows * 4
        self._planes = [bytearray(cols * rows) for _ in range(n_series)]

    def set_dot(self, px: int, py: int, series: int) -> None:
        if 0 <= px < self.dot_w and 0 <= py < self.dot_h:
            cell = (py // 4) * self.cols + (px // 2)
            self._planes[series][cell] |= BRAILLE_BITS[(px % 2, py % 4)]

    def line(
        self, x0: int, y0: int, x1: int, y1: int, series: int, dotted: bool = False
    ) -> None:
        """Bresenham line; ``dotted`` lights every other dot (for dim series)."""
        for i, (px, py) in enumerate(_bresenham(x0, y0, x1, y1)):
            if dotted and i % 2:
                continue
            self.set_dot(px, py, series)

    def cell(self, row: int, col: int) -> tuple[str, int]:
        """Return ``(char, owner_series)`` for one cell (owner -1 if empty)."""
        bits = 0
        owner = -1
        for series, plane in enumerate(self._planes):
            value = plane[row * self.cols + col]
            if value:
                bits |= value
                if owner < 0:
                    owner = series
        return (chr(BRAILLE_BASE + bits) if bits else " "), owner


# ---------------------------------------------------------------------------
# Convergence charts ("IS IT GETTING BETTER?" — two side-by-side plots + inset)


class GettingBetterChart:
    """The single "TRAINING IMPROVEMENT" panel body: the left-docked EVAL inset
    (the cinematic hero win-rate plus its last/EWMA readouts) followed by two
    side-by-side line charts on their own axes. WIN RATE spans the whole run (no
    sliding window, read from ``metrics.jsonl``) on a fixed 0..100% axis with the
    yellow opponent-advance threshold line and a vertical marker at each
    challenger upgrade — a single EWMA series. FINAL SCORE / MARGIN is a
    dual-axis chart: the EWMA final score on a color-coded left axis and the EWMA
    eval margin on a color-coded right axis, each scaled to its own visible
    range, over a sliding 2000-iteration window pinned to a round left edge. When
    the panel is too narrow the inset drops to a one-line strip beneath the
    charts. The win-rate sawtooths back down each time the reference opponent is
    advanced (it then climbs again vs a stronger self)."""

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
        inset = width >= _INSET_MIN_WIDTH
        inset_w = _INSET_W if inset else 0
        charts_w = width - inset_w - (1 if inset else 0)
        left_w = (charts_w - _CHART_GAP) // 2
        right_w = charts_w - _CHART_GAP - left_w
        plot_rows = height - _TITLE_ROWS - _AXIS_ROWS

        if plot_rows < 5 or left_w < 16:
            yield text.Text("  collecting data…", style=theme.TEXT_MUTED)
            return

        rows = min(plot_rows, _WINRATE_PLOT_ROWS)
        beacon_color = theme.BEACON_B if self.frame % 2 else theme.BEACON_A
        # The convergence charts read the full on-disk history (beyond the
        # in-memory cap) so WIN RATE can span the whole run and FINAL SCORE /
        # MARGIN can show a 2000-iteration window; before the first row is
        # flushed (or in tests with no run on disk) we fall back to the live
        # in-memory history.
        full_history = metrics_log.read_iteration_history(
            self.state.config.checkpoint_dir
        )
        if not full_history:
            full_history = self.state.history
        left = _winrate_block(self.state, full_history, left_w, rows, beacon_color)
        right = _score_margin_block(
            self.state, full_history, right_w, rows, beacon_color
        )
        merged = _join_columns([(left, left_w), (right, right_w)], _CHART_GAP)

        if inset:
            eval_box = _eval_inset(self.state, len(merged))
            merged = _merge_columns(eval_box, merged, _INSET_W)
        else:
            merged.extend(_eval_strip(self.state))

        for i, line in enumerate(merged):
            if i:
                yield segment.Segment.line()
            yield line


def _winrate_block(
    state: runstate.RunState,
    history: list[metrics.IterationMetrics],
    width: int,
    rows: int,
    beacon_color: str,
) -> list[text.Text]:
    """The win-rate plot block (title + plot grid + x-axis), exactly
    ``_TITLE_ROWS + rows + _AXIS_ROWS`` lines tall. Plots a single EWMA win-rate
    series over the *whole* run, with the yellow opponent-advance threshold line
    and a vertical marker at each challenger upgrade."""
    plot_cols = max(1, width - _GUTTER_W)
    it_lo, it_hi = convergence.full_range(history)
    ewma = convergence.winrate_ewma_points(history, state.config.eval_ewma_alpha)

    canvas = BrailleCanvas(plot_cols, rows, 1)
    _draw_series(canvas, 0, ewma, it_lo, it_hi, 0.0, 100.0, dotted=False)
    beacon = _beacon_cell(canvas, ewma, it_lo, it_hi, 0.0, 100.0)

    threshold = state.config.opponent_reset_win_rate * 100.0
    target_row = (
        round((1.0 - threshold / 100.0) * (rows - 1)) if threshold > 0 else None
    )
    markers = convergence.marker_columns(
        state.opponent_change_iterations, it_lo, it_hi, plot_cols
    )
    grid = _render_plot_grid(
        canvas,
        {0: theme.WIN_COLOR},
        _percent_label_rows(rows),
        beacon,
        beacon_color,
        target_row,
        theme.WIN_THRESHOLD,
        markers,
        theme.CHALLENGER_MARK,
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
    plot_cols = max(1, width - _GUTTER_W * 2)
    it_lo, it_hi = convergence.score_margin_window(history)
    alpha = state.config.eval_ewma_alpha
    score = [
        pt for pt in convergence.score_ewma_points(history, alpha) if pt[0] >= it_lo
    ]
    margin = [
        pt for pt in convergence.margin_ewma_points(history, alpha) if pt[0] >= it_lo
    ]
    pts_lo, pts_hi = _value_span_padded([value for _, value in score])
    mar_lo, mar_hi = _value_span_padded([value for _, value in margin])

    canvas = BrailleCanvas(plot_cols, rows, 2)
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
    since it is shown large in the docked EVAL box."""
    gen = state.opponent_generation
    opponent = "random" if gen == 0 else f"self·gen{gen}"
    title = text.Text(no_wrap=True, end="", overflow="ellipsis")
    title.append("WIN RATE", style=f"bold {theme.WIN_COLOR}")
    title.append(f" vs {opponent}", style=theme.AXIS)
    return _pad_to(title, width)


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
    width with ``gap`` blank columns between them."""
    height = max((len(lines) for lines, _ in blocks), default=0)
    out: list[text.Text] = []
    for row in range(height):
        line = text.Text(no_wrap=True, end="")
        for index, (lines, block_w) in enumerate(blocks):
            if index:
                line.append(" " * gap)
            if row < len(lines):
                line.append_text(lines[row])
                pad = block_w - lines[row].cell_len
            else:
                pad = block_w
            if pad > 0:
                line.append(" " * pad)
        out.append(line)
    return out


# -- shared plot grid + axis -------------------------------------------------


def _render_plot_grid(
    canvas: BrailleCanvas,
    series_color: dict[int, str],
    label_rows: dict[int, str],
    beacon_cell: tuple[int, int] | None,
    beacon_color: str,
    target_row: int | None,
    target_color: str,
    marker_cols: typing.AbstractSet[int] = frozenset(),
    marker_color: str = theme.AXIS,
) -> list[text.Text]:
    """Paint the braille canvas into colored text rows with a left value gutter,
    an optional dotted gridline (``target_row`` in ``target_color`` — the
    win-rate threshold or the points zero line), optional vertical markers
    (``marker_cols`` in ``marker_color`` — the challenger-upgrade lines), and the
    leading-edge beacon. Adjacent same-color cells are batched into one styled
    run; markers and the gridline only fill cells the data line did not."""
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
            elif col in marker_cols and char == " ":
                char, color = "┊", marker_color
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
    canvas: BrailleCanvas,
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
    they track the sliding x-axis."""
    n_ticks = max(2, min(12, cols // 8))
    tick_cols = [round(i * (cols - 1) / (n_ticks - 1)) for i in range(n_ticks)]

    ruler = text.Text(no_wrap=True, end="")
    ruler.append(" " * (_GUTTER_W - 1) + "└", style=theme.AXIS)
    ruler.append(
        "".join("┬" if col in tick_cols else "─" for col in range(cols)),
        style=theme.AXIS,
    )

    labels = text.Text(no_wrap=True, end="")
    labels.append(" " * _GUTTER_W, style=theme.AXIS)
    labels.append(_tick_labels(tick_cols, cols, it_lo, it_hi), style=theme.TEXT_MUTED)

    return [ruler, labels]


def _value_row(value: float, v_lo: float, v_hi: float, rows: int) -> int:
    """The plot-row index a value lands on for an auto-scaled ``[v_lo, v_hi]``
    axis (row 0 is the top)."""
    frac = (value - v_lo) / (v_hi - v_lo) if v_hi > v_lo else 0.5
    return round((1.0 - frac) * (rows - 1))


# ---------------------------------------------------------------------------
# Family histogram


class FamilyHistogram:
    """The 13-row "what it's learning to decide" panel: one row per judgment
    family, sorted descending by live count. Each bar is scaled to the busiest
    family — the top row fills the panel width — so the whole panel is used,
    while the trailing percentage stays the honest share of *all* decisions, so
    the ~370× spread between the busiest and rarest family stays legible in both
    the relative bar lengths and the absolute percentages."""

    def __init__(self, counts: metrics.FamilyCounts):
        self.counts = counts

    def __rich_console__(
        self, console: rich_console.Console, options: rich_console.ConsoleOptions
    ) -> rich_console.RenderResult:
        width = options.max_width
        total = max(self.counts.total(), 1)
        peak = max(self.counts.counts, default=0)
        # Wide enough for the longest display label (``commit_to_cost``, 14) so a
        # family name is never truncated.
        label_w = 15
        # label + space + bar + " 100.0%" (7) + "  " (2) + count(6) + margin
        bar_w = max(6, width - label_w - 17)

        rows = sorted(self.counts.items(), key=lambda item: item[1], reverse=True)
        lines: list[text.Text] = []
        for family, count in rows:
            share = count / total
            bar_fraction = count / peak if peak else 0.0
            lines.append(self._row(family, count, share, bar_fraction, label_w, bar_w))

        for i, line in enumerate(lines):
            if i:
                yield segment.Segment.line()
            yield line
        # Trailing newline so a following Group sibling (the total-decisions
        # footer) lands on its own row rather than abutting the last bar.
        yield segment.Segment.line()

    def _row(
        self,
        family: decisions.DecisionFamily,
        count: int,
        share: float,
        bar_fraction: float,
        label_w: int,
        bar_w: int,
    ) -> text.Text:
        color = _tier_color(share, count)
        line = text.Text(no_wrap=True, end="")
        line.append(family.value.ljust(label_w), style=theme.TEXT_DIM2)
        line.append(" ")
        bar = eighth_bar(bar_fraction, bar_w, min_tick=True)
        line.append(bar.ljust(bar_w), style=color)
        line.append(f" {share * 100:>5.1f}%", style=theme.TEXT_PRIMARY)
        line.append("  ")
        line.append(f"{human_count(count):>6}", style=theme.HIST_COUNT)
        return line


# ---------------------------------------------------------------------------
# Series drawing helpers (the pure series + window math lives in ``convergence``)


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
    canvas: BrailleCanvas,
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


def _to_dot(
    canvas: BrailleCanvas,
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
    canvas: BrailleCanvas,
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


def _percent_label_rows(rows: int) -> dict[int, str]:
    """Map plot-row index -> percent gutter label for the fixed 0..100% axis."""
    return {round((1 - pct / 100) * (rows - 1)): f"{pct}%" for pct in _Y_LABELS}


def _auto_label_rows(rows: int, lo: float, hi: float) -> dict[int, str]:
    """Map plot-row index -> integer gutter label for an auto-scaled value axis."""
    labels: dict[int, str] = {}
    for tick in range(_POINTS_AXIS_TICKS):
        frac = tick / (_POINTS_AXIS_TICKS - 1)
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
    """A row of right-spaced iteration numbers beneath the axis ticks, spanning
    the displayed ``[it_lo, it_hi]`` window."""
    chars = [" "] * cols
    span = it_hi - it_lo
    for col in tick_cols:
        iteration = it_lo + round(col / max(cols - 1, 1) * span)
        label = str(iteration)
        start = min(col, cols - len(label))
        for offset, char in enumerate(label):
            if 0 <= start + offset < cols:
                chars[start + offset] = char
    return "".join(chars)


def _tier_color(share: float, count: int) -> str:
    if count <= 0:
        return theme.TEXT_MUTED
    if share >= theme.HIST_TOP_SHARE:
        return theme.HIST_TOP
    if share < theme.HIST_LOW_SHARE:
        return theme.HIST_LOW
    return theme.HIST_MID


def _bresenham(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """Integer line points from (x0,y0) to (x1,y1) inclusive."""
    points: list[tuple[int, int]] = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        points.append((x, y))
        if x == x1 and y == y1:
            break
        err2 = 2 * err
        if err2 >= dy:
            err += dy
            x += sx
        if err2 <= dx:
            err += dx
            y += sy
    return points


# ---------------------------------------------------------------------------
# Eval inset (docked) / strip (fallback when narrow)


def _eval_inset(state: runstate.RunState, height: int) -> list[text.Text]:
    """The left-docked eval box: the cinematic hero win-rate, then the most
    recent win-rate / margin, then an identical EWMA section, then the eval
    sample size, the challenger (``random`` or ``gen N @ iter`` — the frozen
    generation and the iteration it was frozen at), and how many iterations have
    passed since the last upgrade. Padded with blank lines to ``height`` so it
    aligns with the plots."""
    last_eval = _latest_eval(state)
    body: list[text.Text] = [_inset_title(last_eval)]

    if last_eval is None:
        body.append(_inset_text("  awaiting first eval…", theme.TEXT_MUTED))
    else:
        _, result = last_eval
        ewma = state.eval_ewma()
        body.extend(_hero_block(result.win_rate * 100.0, state.best_win_rate))
        # The challenger identity rides right under the hero number so it
        # survives even when a short panel truncates the LAST / EWMA detail.
        body.append(_inset_kv("challenger", _inset_opponent(state), theme.TEXT_DIM2))
        body.append(_inset_kv("since adv", _inset_since(state), theme.TEXT_DIM2))
        body.append(_inset_blank())
        body.append(_inset_section("LAST"))
        body.append(
            _inset_kv("win rate", f"{result.win_rate * 100:.1f}%", theme.WIN_COLOR)
        )
        body.append(
            _inset_kv("margin", f"{result.mean_margin:+.1f} pts", theme.MARGIN_COLOR)
        )
        if ewma is not None:
            body.append(_inset_blank())
            body.append(_inset_section("EWMA"))
            body.append(
                _inset_kv("win rate", f"{ewma.win_rate * 100:.1f}%", theme.WIN_COLOR)
            )
            body.append(
                _inset_kv("margin", f"{ewma.mean_margin:+.1f} pts", theme.MARGIN_COLOR)
            )
        body.append(_inset_blank())
        body.append(_inset_kv("eval games", f"{result.n_games}", theme.TEXT_DIM2))
        if state.best_win_rate is not None:
            body.append(
                _inset_kv(
                    "best so far", f"{state.best_win_rate * 100:.1f}%", theme.HIST_MID
                )
            )

    while len(body) < height - 1:
        body.append(_inset_blank())
    # Reserve the final row for the bottom border so it survives even when the
    # body (hero + both sections + the readouts) would otherwise overflow the
    # available height and push the border off the bottom.
    footer = text.Text(no_wrap=True, end="")
    footer.append("└" + "─" * (_INSET_W - 2) + "┘", style=theme.BORDER_EVAL)
    return body[: height - 1] + [footer]


def _hero_block(win_pct: float, best: float | None) -> list[text.Text]:
    """The oversized, value-recolored win-rate hero number in a heavy box."""
    color = theme.hero_color(win_pct)
    digits = " ".join(f"{win_pct:.1f}%")
    inner = _INSET_W - 4
    top = text.Text(no_wrap=True, end="")
    top.append("│ ", style=theme.BORDER_EVAL)
    top.append("╔" + "═" * (inner - 2) + "╗", style=color)
    top.append(" │", style=theme.BORDER_EVAL)
    mid = text.Text(no_wrap=True, end="")
    mid.append("│ ", style=theme.BORDER_EVAL)
    mid.append("║", style=color)
    mid.append(digits.center(inner - 2), style=f"bold {color}")
    mid.append("║", style=color)
    mid.append(" │", style=theme.BORDER_EVAL)
    bot = text.Text(no_wrap=True, end="")
    bot.append("│ ", style=theme.BORDER_EVAL)
    bot.append("╚" + "═" * (inner - 2) + "╝", style=color)
    bot.append(" │", style=theme.BORDER_EVAL)
    return [top, mid, bot]


def _inset_kv(label: str, value: str, value_color: str) -> text.Text:
    line = text.Text(no_wrap=True, end="")
    line.append("│ ", style=theme.BORDER_EVAL)
    line.append(f"{label:<13}", style=theme.TEXT_MUTED)
    line.append(f"{value:>{_INSET_W - 4 - 13}}", style=value_color)
    line.append(" │", style=theme.BORDER_EVAL)
    return line


def _inset_text(content: str, color: str) -> text.Text:
    line = text.Text(no_wrap=True, end="")
    line.append("│ ", style=theme.BORDER_EVAL)
    line.append(content.ljust(_INSET_W - 4), style=color)
    line.append(" │", style=theme.BORDER_EVAL)
    return line


def _inset_blank() -> text.Text:
    """An empty inset row that keeps the box's side borders (used as a spacer
    between sections and to pad the inset down to the chart height)."""
    return _inset_text("", theme.BORDER_EVAL)


def _inset_title(last_eval: tuple[int, metrics.EvalResult] | None) -> text.Text:
    """The top border of the eval inset, naming the most recent eval iteration."""
    label = "EVAL" if last_eval is None else f"EVAL · iter {last_eval[0]:04d}"
    title = text.Text(no_wrap=True, end="")
    title.append("┌─ ", style=theme.BORDER_EVAL)
    title.append(label + " ", style=theme.BORDER_EVAL)
    title.append(
        "─" * max(0, _INSET_W - title.cell_len - 1) + "┐", style=theme.BORDER_EVAL
    )
    return title


def _inset_section(label: str) -> text.Text:
    """A dim, centered section header (``-- LAST --`` / ``-- EWMA --``) inside the
    eval inset."""
    line = text.Text(no_wrap=True, end="")
    line.append("│ ", style=theme.BORDER_EVAL)
    line.append(f"-- {label} --".center(_INSET_W - 4), style=f"bold {theme.TEXT_MUTED}")
    line.append(" │", style=theme.BORDER_EVAL)
    return line


def _inset_opponent(state: runstate.RunState) -> str:
    """The current reference opponent (the "challenger"): ``random`` while still
    evaluating against the random agent, otherwise the frozen self generation and
    the iteration it was frozen at (``gen N @ iter``)."""
    gen = state.opponent_generation
    if gen == 0:
        return "random"
    return f"gen{gen} @ {state.opponent_since_iteration:04d}"


def _inset_since(state: runstate.RunState) -> str:
    """How many iterations since the frozen self model was last advanced (a dash
    while still evaluating against the random agent, where no frozen self
    exists)."""
    if state.opponent_generation == 0:
        return "—"
    return f"{state.iteration - state.opponent_since_iteration} iters"


def _eval_strip(state: runstate.RunState) -> list[text.Text]:
    """Compact one-line eval readout when the panel is too narrow for the inset."""
    last_eval = _latest_eval(state)
    line = text.Text(no_wrap=True, end="")
    line.append(" " * _GUTTER_W)
    if last_eval is None:
        line.append("eval: awaiting first evaluation…", style=theme.TEXT_MUTED)
        return [line]
    _, result = last_eval
    ewma = state.eval_ewma()
    line.append("eval ", style=theme.TEXT_MUTED)
    line.append(
        f"{result.win_rate * 100:.1f}%", style=theme.hero_color(result.win_rate * 100)
    )
    line.append(f" ±{result.ci95 * 100:.1f}%  ", style=theme.TEXT_DIM2)
    line.append(f"margin {result.mean_margin:+.1f}", style=theme.MARGIN_COLOR)
    if ewma is not None:
        line.append(
            f"  ewma {ewma.win_rate * 100:.1f}% / {ewma.mean_margin:+.1f}",
            style=theme.TEXT_DIM2,
        )
    return [line]


def _latest_eval(
    state: runstate.RunState,
) -> tuple[int, metrics.EvalResult] | None:
    for item in reversed(state.history):
        if item.eval is not None:
            return (item.iteration, item.eval)
    return None


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
