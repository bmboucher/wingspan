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

Everything degrades gracefully: too-small panels drop the eval inset, then the
secondary chart series, then fall back to a scatter — the win-rate curve is the
last thing sacrificed.
"""

from __future__ import annotations

import typing

import rich.console as rich_console
from rich import segment, text

from wingspan import decisions
from wingspan.training import metrics, runstate, theme

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
_BOTTOM_ROWS = 3  # axis ruler + iteration labels + caption/legend
_INSET_W = 28  # docked eval inset width
_INSET_MIN_WIDTH = 96  # below this the inset moves below the axis
_Y_LABELS = (100, 80, 60, 40, 20, 0)  # win-rate axis percent gridlines
_POINTS_AXIS_TICKS = 5  # gridline count on the auto-scaled points axis
# Once a run exceeds this many iterations the convergence charts show only the
# most recent ``CHART_WINDOW`` iterations (a sliding x-axis window) so the
# early, noisy climb stops compressing the live trend. The in-memory history cap
# (``config.history_len``) is comfortably larger, so a resumed run repaints the
# full window from the restored history.
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
# Convergence charts (win-rate hero + self-play points / margin)


class WinRateChart:
    """The left HERO panel body: win-rate-vs-opponent on a fixed 0..100 axis,
    a 100% target gridline, a pulsing leading-edge beacon, and a docked eval
    inset (the cinematic hero win-rate number) that falls back to a one-line
    strip when the panel is too narrow. The win-rate sawtooths back down each
    time the reference opponent is advanced (it climbs vs a stronger self)."""

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
        bottom_extra = 0 if inset else 1  # the narrow-mode one-line eval strip
        plot_cols = width - _GUTTER_W - inset_w - (1 if inset else 0)
        plot_rows = height - _BOTTOM_ROWS - bottom_extra

        if plot_cols < 20 or plot_rows < 5:
            yield text.Text("  collecting data…", style=theme.TEXT_MUTED)
            return

        it_lo, it_hi = _chart_window(self.state.history)
        win = _windowed_series(self.state.history, "win", it_lo)
        canvas = BrailleCanvas(plot_cols, plot_rows, 1)
        _draw_series(canvas, 0, win, it_lo, it_hi, 0.0, 100.0, dotted=False)
        beacon = _beacon_cell(canvas, win, it_lo, it_hi, 0.0, 100.0)

        lines = _render_plot_grid(
            canvas,
            {0: theme.WIN_COLOR},
            _percent_label_rows(plot_rows),
            beacon,
            self._beacon_color(),
            target_row=0,
        )
        lines.extend(_axis_lines(it_lo, it_hi, plot_cols, _winrate_caption()))

        if inset:
            inset_lines = _eval_inset(self.state, len(lines))
            lines = _merge_columns(lines, inset_lines, _GUTTER_W + plot_cols)
        else:
            lines.extend(_eval_strip(self.state))

        for i, line in enumerate(lines):
            if i:
                yield segment.Segment.line()
            yield line

    def _beacon_color(self) -> str:
        return theme.BEACON_B if self.frame % 2 else theme.BEACON_A


class PointsChart:
    """The right HERO panel body: average self-play points per game (solid,
    dominant) and the eval score-margin (dotted) on a shared, auto-scaled point
    axis — the two point-valued series the win-rate panel can't show on its
    0..100% axis. Average points is the headline "how good is the play" signal
    (competitive Wingspan ends north of 100); a one-line readout sits beneath
    the axis."""

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
        plot_cols = width - _GUTTER_W
        plot_rows = height - _BOTTOM_ROWS - 1  # one readout strip line

        it_lo, it_hi = _chart_window(self.state.history)
        points = _windowed_series(self.state.history, "points", it_lo)
        margin = _windowed_series(self.state.history, "margin", it_lo)
        if plot_cols < 20 or plot_rows < 5 or (not points and not margin):
            yield text.Text("  collecting data…", style=theme.TEXT_MUTED)
            return

        v_lo, v_hi = _value_span_padded(
            [value for _, value in points] + [value for _, value in margin]
        )
        canvas = BrailleCanvas(plot_cols, plot_rows, 2)
        _draw_series(canvas, 0, points, it_lo, it_hi, v_lo, v_hi, dotted=False)
        _draw_series(canvas, 1, margin, it_lo, it_hi, v_lo, v_hi, dotted=True)
        beacon = _beacon_cell(canvas, points, it_lo, it_hi, v_lo, v_hi)

        lines = _render_plot_grid(
            canvas,
            {0: theme.POINTS_COLOR, 1: theme.MARGIN_COLOR},
            _auto_label_rows(plot_rows, v_lo, v_hi),
            beacon,
            theme.BEACON_B if self.frame % 2 else theme.BEACON_A,
            target_row=None,
        )
        lines.extend(_axis_lines(it_lo, it_hi, plot_cols, _points_caption()))
        lines.extend(_points_strip(self.state))

        for i, line in enumerate(lines):
            if i:
                yield segment.Segment.line()
            yield line


# -- shared plot grid + axis -------------------------------------------------


def _render_plot_grid(
    canvas: BrailleCanvas,
    series_color: dict[int, str],
    label_rows: dict[int, str],
    beacon_cell: tuple[int, int] | None,
    beacon_color: str,
    target_row: int | None,
) -> list[text.Text]:
    """Paint the braille canvas into colored text rows with a left value gutter,
    an optional dotted target gridline on ``target_row``, and the leading-edge
    beacon. Adjacent same-color cells are batched into one styled run."""
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
        lines.append(line)
    return lines


def _axis_lines(
    it_lo: int, it_hi: int, cols: int, caption: text.Text
) -> list[text.Text]:
    """The shared bottom three rows: the tick ruler, the iteration labels, and
    a chart-specific ``caption`` legend row. The labels span the displayed
    ``[it_lo, it_hi]`` iteration window so they track the sliding x-axis."""
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

    return [ruler, labels, caption]


def _winrate_caption() -> text.Text:
    caption = text.Text(no_wrap=True, end="")
    caption.append(" " * _GUTTER_W)
    caption.append("win% ", style=theme.WIN_COLOR)
    caption.append("· climbs toward the ", style=theme.AXIS)
    caption.append("100% target", style=theme.TARGET_GRID)
    return caption


def _points_caption() -> text.Text:
    caption = text.Text(no_wrap=True, end="")
    caption.append(" " * _GUTTER_W)
    caption.append("self-play pts ", style=theme.POINTS_COLOR)
    caption.append("· ", style=theme.AXIS)
    caption.append("eval margin", style=theme.MARGIN_COLOR)
    return caption


def _points_strip(state: runstate.RunState) -> list[text.Text]:
    """A one-line readout of the latest average self-play points and eval margin."""
    line = text.Text(no_wrap=True, end="")
    line.append(" " * _GUTTER_W)
    if state.last_iter is None:
        line.append("points: awaiting first iteration…", style=theme.TEXT_MUTED)
        return [line]
    line.append("avg ", style=theme.TEXT_MUTED)
    line.append(f"{state.last_iter.avg_self_score:.1f} pts", style=theme.POINTS_COLOR)
    last_eval = _latest_eval(state)
    if last_eval is not None:
        _, result = last_eval
        line.append("   eval margin ", style=theme.TEXT_MUTED)
        line.append(f"{result.mean_margin:+.1f}", style=theme.MARGIN_COLOR)
    return [line]


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
        label_w = 16 if width >= 96 else 13
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
        line.append(family.value[:label_w].ljust(label_w), style=theme.TEXT_DIM2)
        line.append(" ")
        bar = eighth_bar(bar_fraction, bar_w, min_tick=True)
        line.append(bar.ljust(bar_w), style=color)
        line.append(f" {share * 100:>5.1f}%", style=theme.TEXT_PRIMARY)
        line.append("  ")
        line.append(f"{human_count(count):>6}", style=theme.HIST_COUNT)
        return line


# ---------------------------------------------------------------------------
# Series extraction + drawing helpers


def _series(
    history: list[metrics.IterationMetrics], kind: str
) -> list[tuple[int, float]]:
    """``(iteration, value)`` points for one chart series. ``win`` and ``margin``
    only have a point on iterations that ran an eval; ``points`` (average
    self-play score) has one on every iteration."""
    points: list[tuple[int, float]] = []
    for item in history:
        if kind == "win":
            if item.eval is not None:
                points.append((item.iteration, item.eval.win_rate * 100.0))
        elif kind == "margin":
            if item.eval is not None:
                points.append((item.iteration, item.eval.mean_margin))
        else:  # points — average self-play score per game
            points.append((item.iteration, item.avg_self_score))
    return points


def _windowed_series(
    history: list[metrics.IterationMetrics], kind: str, it_lo: int
) -> list[tuple[int, float]]:
    """``_series`` filtered to the sliding window: only points at or after
    ``it_lo`` (the start of the displayed iteration range)."""
    return [(it, value) for it, value in _series(history, kind) if it >= it_lo]


def _chart_window(history: list[metrics.IterationMetrics]) -> tuple[int, int]:
    """The ``(it_lo, it_hi)`` iteration range the convergence charts display: the
    most recent ``CHART_WINDOW`` iterations, so the x-axis slides once the run
    grows past the window instead of compressing the whole history into the
    panel. Both hero charts share this range so they stay aligned."""
    if not history:
        return (0, 1)
    it_hi = history[-1].iteration
    it_lo = max(history[0].iteration, it_hi - CHART_WINDOW + 1)
    return (it_lo, it_hi if it_hi > it_lo else it_lo + 1)


def _value_span_padded(values: list[float]) -> tuple[float, float]:
    """An auto-scaled ``(lo, hi)`` axis range for the points chart, with a small
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
    """The right-docked eval box: the cinematic hero win-rate plus readouts.
    Padded with blank lines to ``height`` so it aligns with the plot."""
    last_eval = _latest_eval(state)
    body: list[text.Text] = []
    title = text.Text(no_wrap=True, end="")
    eval_iter = last_eval[0] if last_eval else None
    label = "EVAL" if eval_iter is None else f"EVAL · iter {eval_iter:04d}"
    title.append("┌─ ", style=theme.BORDER_EVAL)
    title.append(label + " ", style=theme.BORDER_EVAL)
    title.append(
        "─" * max(0, _INSET_W - title.cell_len - 1) + "┐", style=theme.BORDER_EVAL
    )
    body.append(title)

    if last_eval is None:
        body.append(_inset_text("  awaiting first eval…", theme.TEXT_MUTED))
    else:
        _, result = last_eval
        ewma = state.eval_ewma()
        body.extend(_hero_block(result.win_rate * 100.0, state.best_win_rate))
        body.append(
            _inset_kv("win vs random", f"{result.win_rate * 100:.1f}%", theme.WIN_COLOR)
        )
        body.append(
            _inset_text(f"   ±{result.ci95 * 100:.1f}% (95% CI)", theme.TEXT_DIM2)
        )
        if ewma is not None:
            body.append(
                _inset_kv("ewma win", f"{ewma.win_rate * 100:.1f}%", theme.WIN_COLOR)
            )
        body.append(
            _inset_kv(
                "mean margin", f"{result.mean_margin:+.1f} pts", theme.MARGIN_COLOR
            )
        )
        if ewma is not None:
            body.append(
                _inset_kv(
                    "ewma margin", f"{ewma.mean_margin:+.1f} pts", theme.MARGIN_COLOR
                )
            )
        body.append(_inset_kv("eval games", f"{result.n_games}", theme.TEXT_DIM2))
        if state.best_win_rate is not None:
            body.append(
                _inset_kv(
                    "best so far", f"{state.best_win_rate * 100:.1f}%", theme.HIST_MID
                )
            )

    while len(body) < height - 1:
        body.append(_inset_text("", theme.BORDER_EVAL))
    footer = text.Text(no_wrap=True, end="")
    footer.append("└" + "─" * (_INSET_W - 2) + "┘", style=theme.BORDER_EVAL)
    body.append(footer)
    return body[:height]


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
