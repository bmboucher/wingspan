"""Assembles the live dashboard: the Layout tree and the per-region renderers.

``build_layout`` creates the four-band FLIGHT PLAN training skeleton once;
``render`` repaints it from a :class:`runstate.RunState` snapshot each frame.
The bands, top to bottom, read as a guided narrative: WHERE AM I (header) ->
IN-GAME PERFORMANCE / DECISION MODELS (middle) -> TRAINING IMPROVEMENT (the
gold-bordered hero band: the win-rate and self-play-points charts side by side
plus the docked eval inset) -> DIAGNOSTICS (health + events).

Several band-specific micro-renderables live here because they must fill the
full panel width every refresh: the phase-colored header progress bar (which
also carries the two-tone CPU and RAM gauges on the same row) and the stacked
six-component score bar. The hero charts come from :mod:`charts`.
"""

from __future__ import annotations

import rich.console as rich_console
from rich import box, layout, panel, table, text

from wingspan.training import charts, metrics, runstate, theme

_WORDMARK = "🪶 WINGSPAN  FLIGHT PLAN"
_SPARK_CELLS = 8
# RECENT EVENTS lines that fill the footer band: its ``size=8`` minus the
# panel's top and bottom border rows leaves six content rows.
_EVENT_LINES = 6

# IN-GAME PERFORMANCE score table geometry.
_PT_NAME_W = 8  # the per-source name column ("tucked" / "rounds" / "TOTAL")
_PT_PAIR_W = 12  # one "###.#  ##.#%" points / share pair
_Z95 = 1.96  # standard-normal z for the 95% confidence interval

# Header-row gauges — CPU + RAM ride beside the progress bar (no separate band).
_GAUGE_LABEL_W = 5  # "VRAM" + a space — the widest of CPU/RAM/GPU/VRAM
_GAUGE_CAPS_W = 2  # the ▕ ▏ end-caps around the bar
_GAUGE_PCT_W = 5  # the " 100%" / "   0%" trailing percent field (leading space)
_GAUGE_GAP = 3  # blank columns between the three header sections (bar · CPU · RAM)
_GAUGE_MIN_CELLS = 4  # never draw a bar narrower than this
_GAUGE_CHROME_W = _GAUGE_LABEL_W + _GAUGE_CAPS_W + _GAUGE_PCT_W  # non-bar width
# Below this header content width the inline gauges are dropped and the progress
# bar spans the whole row (three sections each need a gauge's minimum footprint).
_HEADER_GAUGE_MIN_WIDTH = 3 * (_GAUGE_CHROME_W + _GAUGE_MIN_CELLS) + 2 * _GAUGE_GAP


def build_layout() -> layout.Layout:
    """Create the empty four-band layout skeleton (populated by :func:`render`)."""
    root = layout.Layout(name="root")
    root.split_column(
        layout.Layout(name="header", size=5),
        # minimum_size fits the DECISION MODELS panel in full: one row per
        # judgment family + a blank + the total line, plus two panel borders.
        # Bump this when ``ALL_DECISION_FAMILIES`` grows past 14 entries.
        layout.Layout(name="middle", ratio=12, minimum_size=18),
        layout.Layout(name="headline", ratio=13, minimum_size=12),
        layout.Layout(name="footer", size=8),
    )
    root["middle"].split_row(
        layout.Layout(name="produce", ratio=58),
        layout.Layout(name="learning", ratio=42, minimum_size=42),
    )
    root["footer"].split_row(
        layout.Layout(name="health", ratio=40),
        layout.Layout(name="timer", size=28),
        layout.Layout(name="events", ratio=60),
    )
    return root


def render(
    root: layout.Layout,
    state: runstate.RunState,
    frame: int,
    pause_buffer: str = "",
) -> None:
    """Repaint every region from the current state."""
    root["header"].update(_header(state))
    root["produce"].update(_produce(state))
    root["learning"].update(_learning(state))
    root["headline"].update(_getting_better_panel(state, frame))
    root["health"].update(_health(state))
    root["timer"].update(_timer(state))
    root["events"].update(_events(state, pause_buffer))


###### PRIVATE #######

#### Header band ####


def _header(state: runstate.RunState) -> panel.Panel:
    body = rich_console.Group(
        _wordmark_row(state), _header_stats_row(state), _HeaderProgress(state)
    )
    return panel.Panel(
        body,
        box=box.ROUNDED,
        border_style=theme.BORDER_DEFAULT,
        padding=(0, 1),
    )


def _wordmark_row(state: runstate.RunState) -> table.Table:
    label = state.phase.value.upper()
    grid = table.Table.grid(expand=True)
    grid.add_column(justify="left", ratio=1)  # absorbs expansion; badge stays fixed
    # A fixed-width, centered badge column carrying the phase color as its cell
    # background. The column style fills the whole cell — including the pad cell
    # on each side of the centered word — so the badge gets symmetric padding;
    # a trailing styled *space* would instead be stripped as line-final
    # whitespace, leaving the right side flush.
    grid.add_column(
        justify="center",
        width=len(label) + 2,
        style=f"bold {theme.CANVAS} on {theme.PHASE_COLOR[state.phase]}",
    )
    grid.add_row(theme.gradient_text(_WORDMARK), text.Text(label, no_wrap=True, end=""))
    return grid


def _header_stats_row(state: runstate.RunState) -> text.Text:
    """The single text status line: iteration, this iteration's progress count,
    cumulative games, the two wall-time chronometers (total time since iter 0
    across restarts, and this process's own session), and this iteration's
    elapsed seconds."""
    label, done, total = _header_progress(state)
    out = text.Text(no_wrap=True)  # default end newline: own its row in the Group
    out.append(f"iter {state.iteration:04d}", style=theme.TEXT_PRIMARY)
    out.append(f"    {label} ", style=theme.TEXT_MUTED)
    out.append(f"{done}/{total}", style=theme.TEXT_PRIMARY)
    out.append("    Σ ", style=theme.TEXT_MUTED)
    out.append(f"{state.total_games:,}", style=theme.TEXT_PRIMARY)
    out.append(" games", style=theme.TEXT_MUTED)
    out.append("    SINCE ITER 0 ", style=theme.TEXT_MUTED)
    out.append(_runtime_clock(state.elapsed()), style=theme.TEXT_DIM2)
    out.append("    THIS RUN ", style=theme.TEXT_MUTED)
    out.append(_runtime_clock(state.session_elapsed()), style=theme.TEXT_DIM2)
    out.append(f"    ({state.iter_elapsed():.1f}s this iter)", style=theme.TEXT_MUTED)
    return out


def _header_progress(state: runstate.RunState) -> tuple[str, int, int]:
    """The ``(label, done, total)`` for the header bar: held-out eval games
    while evaluating, final-eval progress during the target milestone phase,
    otherwise this iteration's self-play collection progress."""
    if state.phase is runstate.Phase.EVALUATING and state.eval_games_in_iter > 0:
        return "eval", state.eval_game_in_iter, state.eval_games_in_iter
    if state.phase is runstate.Phase.FINAL_EVALUATING:
        done, total = state.final_eval_progress
        return "final eval", done, max(total, 1)
    return "game", state.game_in_iter, state.games_in_iter


class _HeaderProgress:
    """The header progress + telemetry row: the phase-colored iteration progress
    bar (the phase word itself is dropped — the upper-right badge already names
    it) sharing its row with the CPU and RAM gauges, each section taking a third
    of the width. On a narrow terminal — or before host telemetry first lands —
    the gauges drop and the bar spans the whole row."""

    def __init__(self, state: runstate.RunState):
        self.state = state

    def __rich_console__(
        self, console: rich_console.Console, options: rich_console.ConsoleOptions
    ) -> rich_console.RenderResult:
        width = options.max_width
        color = theme.PHASE_COLOR[self.state.phase]
        _, done, total = _header_progress(self.state)
        stats = self.state.system
        if stats is None or width < _HEADER_GAUGE_MIN_WIDTH:
            yield _progress_bar(color, done, total, width)
            return

        # Split the row into thirds — progress bar · CPU · RAM — with the bar
        # absorbing the rounding remainder so the three sections plus the two
        # gaps between them exactly fill the width.
        section_w = (width - 2 * _GAUGE_GAP) // 3
        bar_w = width - 2 * _GAUGE_GAP - 2 * section_w
        line = text.Text(no_wrap=True, end="")
        line.append_text(_progress_bar(color, done, total, bar_w))
        line.append(" " * _GAUGE_GAP)
        cpu = _util_gauge("CPU", stats.cpu_percent, section_w)
        line.append_text(cpu)
        line.append(" " * max(0, section_w - cpu.cell_len))
        line.append(" " * _GAUGE_GAP)
        line.append_text(
            _mem_gauge(
                "RAM",
                stats.ram_used_gb,
                stats.ram_total_gb,
                stats.proc_rss_gb,
                stats.ram_percent,
                section_w,
            )
        )
        yield line


def _progress_bar(color: str, done: int, total: int, width: int) -> text.Text:
    """A single phase-colored progress bar ``width`` cells wide (``▕███░░░▏``)."""
    cells = max(1, width - 2)  # leave room for the ▕ ▏ end-caps
    fill = round(cells * done / total) if total else 0
    bar = text.Text(no_wrap=True, end="")
    bar.append("▕", style=theme.TEXT_MUTED)
    bar.append("█" * fill, style=color)
    bar.append("░" * (cells - fill), style=theme.BORDER_DEFAULT)
    bar.append("▏", style=theme.TEXT_MUTED)
    return bar


#### Host-telemetry gauges (shared by the header row) ####


def _util_gauge(label: str, pct: float, width: int) -> text.Text:
    color = (
        theme.GAUGE_UTIL_PEAK if pct >= theme.GAUGE_UTIL_PEAK_PCT else theme.GAUGE_UTIL
    )
    out = _gauge_label(label)
    _append_bar(out, pct / 100.0, color, width - _GAUGE_CHROME_W)
    out.append(f" {pct:>3.0f}%", style=theme.TEXT_PRIMARY)
    return out


def _mem_gauge(
    label: str,
    used_gb: float,
    total_gb: float,
    proc_gb: float,
    pct: float,
    width: int,
) -> text.Text:
    suffix = f"  {used_gb:.1f}/{total_gb:.1f} GB"
    bar_cells = width - _GAUGE_CHROME_W - len(suffix)
    show_suffix = bar_cells >= _GAUGE_MIN_CELLS
    if not show_suffix:
        bar_cells = width - _GAUGE_CHROME_W
    out = _gauge_label(label)
    proc_frac = proc_gb / total_gb if total_gb > 0 else 0.0
    _append_mem_bar(out, pct / 100.0, proc_frac, _mem_color(pct), bar_cells)
    out.append(f" {pct:>3.0f}%", style=theme.TEXT_PRIMARY)
    if show_suffix:
        out.append(suffix, style=theme.TEXT_MUTED)
    return out


def _append_mem_bar(
    out: text.Text, used_frac: float, proc_frac: float, mem_color: str, cells: int
) -> None:
    """A two-tone RAM bar: this process's resident slice (its own color) sits at
    the head of the used region, the rest of the used RAM follows in ``mem_color``,
    then the free remainder as track. Cell-aligned (not eighth-block) so the
    proc / other boundary lands on a whole cell."""
    cells = max(_GAUGE_MIN_CELLS, cells)
    used_cells = min(cells, round(max(0.0, used_frac) * cells))
    proc_cells = min(used_cells, round(max(0.0, proc_frac) * cells))
    out.append("▕", style=theme.GAUGE_BRACKET)
    out.append("█" * proc_cells, style=theme.GAUGE_MEM_PROC)
    out.append("█" * (used_cells - proc_cells), style=mem_color)
    out.append("░" * (cells - used_cells), style=theme.GAUGE_TRACK)
    out.append("▏", style=theme.GAUGE_BRACKET)


def _gauge_label(label: str) -> text.Text:
    out = text.Text(no_wrap=True, end="")
    out.append(f"{label:<{_GAUGE_LABEL_W}}", style=theme.SYSTEM_LABEL)
    return out


def _append_bar(out: text.Text, fraction: float, fill_color: str, cells: int) -> None:
    cells = max(_GAUGE_MIN_CELLS, cells)
    bar = charts.eighth_bar(fraction, cells, min_tick=True)
    out.append("▕", style=theme.GAUGE_BRACKET)
    out.append(bar, style=fill_color)
    out.append("░" * (cells - len(bar)), style=theme.GAUGE_TRACK)
    out.append("▏", style=theme.GAUGE_BRACKET)


def _mem_color(pct: float) -> str:
    if pct >= theme.GAUGE_MEM_FULL_PCT:
        return theme.GAUGE_MEM_FULL
    if pct >= theme.GAUGE_MEM_HIGH_PCT:
        return theme.GAUGE_MEM_HIGH
    return theme.GAUGE_MEM


#### Produce band — in-game performance ####


def _produce(state: runstate.RunState) -> panel.Panel:
    stats = state.produce_stats()
    body: rich_console.RenderableType
    if stats is None:
        body = text.Text("awaiting first game…", style=theme.TEXT_MUTED)
    else:
        body = rich_console.Group(
            _produce_table(stats),
            text.Text(""),
            _StackedBar(stats.breakdown),
            text.Text(""),
            _produce_stats_block(state, stats),
        )
    title = (
        "[b]IN-GAME PERFORMANCE  \\[FINAL][/b]"
        if state.pinned_stats is not None
        else "[b]IN-GAME PERFORMANCE[/b]"
    )
    return panel.Panel(
        body,
        title=title,
        title_align="left",
        box=box.ROUNDED,
        border_style=(
            theme.GOOD if state.pinned_stats is not None else theme.BORDER_DEFAULT
        ),
        padding=(0, 1),
    )


class _StackedBar:
    """One full-width bar split into six segments sized by each score component's
    average points and told apart by color alone (no fill pattern)."""

    def __init__(self, avg: metrics.ScoreBreakdown):
        self.avg = avg

    def __rich_console__(
        self, console: rich_console.Console, options: rich_console.ConsoleOptions
    ) -> rich_console.RenderResult:
        width = options.max_width
        components = self.avg.components()
        total = sum(value for _, value in components) or 1.0
        widths = [round(value / total * width) for _, value in components]
        # absorb rounding drift into the largest (birds) segment
        drift = width - sum(widths)
        if widths:
            widths[0] = max(0, widths[0] + drift)
        bar = text.Text(no_wrap=True)  # default end newline: own its line in the Group
        for (name, _), seg_w in zip(components, widths):
            bar.append("█" * seg_w, style=theme.SCORE_COLOR[name])
        yield bar


def _produce_table(stats: metrics.ProduceStats) -> rich_console.Group:
    """The per-source points / share breakdown: a grouped header (``all`` vs
    ``games won``) over two points / share pairs separated by vertical rules, one
    row per scoring source, and a ruled ``TOTAL`` row at the bottom."""
    overall = stats.breakdown.components()
    winner = dict(stats.winner_breakdown.components())
    overall_total = stats.breakdown.total or 1.0
    winner_total = stats.winner_breakdown.total or 1.0

    lines: list[text.Text] = [_produce_header()]
    for name, value in overall:
        lines.append(
            _produce_row(
                name,
                theme.SCORE_COLOR[name],
                value,
                overall_total,
                winner.get(name, 0.0),
                winner_total,
            )
        )
    lines.append(_produce_rule())
    lines.append(_produce_total(overall_total, winner_total))
    return rich_console.Group(*lines)


def _produce_header() -> text.Text:
    line = text.Text(no_wrap=True)  # default end newline: own its row in the Group
    line.append("  " + " " * _PT_NAME_W)
    line.append(" │ ", style=theme.TEXT_MUTED)
    line.append("all".center(_PT_PAIR_W), style=theme.TEXT_MUTED)
    line.append(" │ ", style=theme.TEXT_MUTED)
    line.append("games won".center(_PT_PAIR_W), style=theme.TEXT_MUTED)
    return line


def _produce_row(
    name: str,
    color: str,
    value: float,
    value_total: float,
    won: float,
    won_total: float,
) -> text.Text:
    line = text.Text(no_wrap=True)  # default end newline: own its row in the Group
    line.append("■ ", style=color)
    line.append(f"{name:<{_PT_NAME_W}}", style=theme.TEXT_PRIMARY)
    line.append(" │ ", style=theme.TEXT_MUTED)
    _append_pair(line, value, value_total, theme.TEXT_PRIMARY)
    line.append(" │ ", style=theme.TEXT_MUTED)
    _append_pair(line, won, won_total, theme.TEXT_DIM2)
    return line


def _produce_rule() -> text.Text:
    line = text.Text(no_wrap=True)  # default end newline: own its row in the Group
    line.append("  ")
    line.append(
        "─" * (_PT_NAME_W + 3 + _PT_PAIR_W + 3 + _PT_PAIR_W), theme.BORDER_DEFAULT
    )
    return line


def _produce_total(value_total: float, won_total: float) -> text.Text:
    bright = f"bold {theme.TEXT_BRIGHT}"
    line = text.Text(no_wrap=True)  # default end newline: own its row in the Group
    line.append("  ")
    line.append(f"{'TOTAL':<{_PT_NAME_W}}", style=bright)
    line.append(" │ ", style=theme.TEXT_MUTED)
    _append_pair(line, value_total, value_total, bright)
    line.append(" │ ", style=theme.TEXT_MUTED)
    _append_pair(line, won_total, won_total, bright)
    return line


def _append_pair(line: text.Text, value: float, total: float, value_style: str) -> None:
    """Append a right-aligned ``###.#  ##.#%`` points / share pair (``_PT_PAIR_W``
    cells wide) to ``line``."""
    pct = value / total * 100.0 if total else 0.0
    line.append(f"{value:>5.1f}", style=value_style)
    line.append(f" {pct:>5.1f}%", style=theme.TEXT_MUTED)


def _produce_stats_block(
    state: runstate.RunState, stats: metrics.ProduceStats
) -> rich_console.Group:
    """The game-length and margin readouts, each with a 95% confidence interval
    (z·σ/√n, n = games per iteration) on the EWMA-smoothed per-cycle σ."""
    games_per_iter = state.config.games_per_iter
    rows = [
        ("Decisions/game", f"{stats.decisions:.1f}", stats.decisions_std),
        ("Score margin", f"{stats.margin:+.1f}", stats.margin_std),
        ("Winning margin", f"{stats.abs_margin:+.1f}", stats.abs_margin_std),
    ]
    lines: list[text.Text] = []
    for label, value, std in rows:
        line = text.Text(no_wrap=True)  # default end newline: own its row in the Group
        line.append(f"{label:<16}", style=theme.TEXT_MUTED)
        line.append(f"{value:>6}", style=theme.TEXT_PRIMARY)
        line.append(f"  ±{_ci95(std, games_per_iter):.1f}", style=theme.TEXT_DIM2)
        lines.append(line)
    return rich_console.Group(*lines)


def _ci95(std: float, n: int) -> float:
    """The 95% confidence-interval half-width ``z·σ/√n`` of a mean (0 for n≤0)."""
    return _Z95 * std / (n**0.5) if n > 0 else 0.0


#### Learning band — decision-model histogram ####


def _learning(state: runstate.RunState) -> panel.Panel:
    # FamilyHistogram owns its own total-decisions footer so the count is never
    # truncated when the panel is short (the bars are clipped first instead).
    return panel.Panel(
        charts.FamilyHistogram(state.cum_family, state.total_decisions),
        title="[b]DECISION MODELS[/b]",
        title_align="left",
        box=box.ROUNDED,
        border_style=theme.BORDER_DEFAULT,
        padding=(0, 1),
    )


#### Headline band — convergence charts ####


def _getting_better_panel(state: runstate.RunState, frame: int) -> panel.Panel:
    """The single gold hero panel: the two side-by-side convergence charts
    (win rate · avg points) plus the docked EVAL inset."""
    return panel.Panel(
        charts.GettingBetterChart(state, frame),
        title="[b]TRAINING IMPROVEMENT[/b]",
        title_align="left",
        box=box.HEAVY,
        border_style=theme.BORDER_HEADLINE,
        padding=(0, 1),
    )


#### Footer band — health + events ####


def _health(state: runstate.RunState) -> panel.Panel:
    grid = table.Table.grid(padding=(0, 1))
    grid.add_column()  # name
    grid.add_column(justify="right")  # value
    grid.add_column()  # sparkline
    grid.add_column()  # verdict
    for row in _health_rows(state):
        grid.add_row(*row)
    return panel.Panel(
        grid,
        title="[b]TRAINING HEALTH[/b]",
        title_align="left",
        box=box.ROUNDED,
        border_style=theme.BORDER_DEFAULT,
        padding=(0, 1),
    )


def _health_rows(state: runstate.RunState) -> list[tuple[text.Text, ...]]:
    last = state.last_iter
    grad_clip = state.config.grad_clip
    specs: list[tuple[str, str, list[float], str]] = []
    if last is not None:
        specs = [
            (
                "policy loss",
                f"{last.policy_loss:.4f}",
                [im.policy_loss for im in state.history],
                "lower",
            ),
            (
                "value loss",
                f"{last.value_loss:.4f}",
                [im.value_loss for im in state.history],
                "lower",
            ),
            (
                "entropy",
                f"{last.entropy:.3f}",
                [im.entropy for im in state.history],
                "entropy",
            ),
            (
                "grad norm",
                f"{last.grad_norm:.2f}",
                [im.grad_norm for im in state.history],
                "grad",
            ),
        ]
    rows: list[tuple[text.Text, ...]] = _perf_health_rows(state)
    for name, value, series, mode in specs:
        verdict_text, verdict_color = _verdict(mode, series, grad_clip)
        rows.append(
            (
                text.Text(name, style=theme.TEXT_MUTED),
                text.Text(value, style=theme.TEXT_PRIMARY),
                text.Text(
                    charts.sparkline(series, _SPARK_CELLS), style=theme.SPARK_COLOR
                ),
                text.Text(verdict_text, style=verdict_color),
            )
        )
    return rows


def _perf_health_rows(state: runstate.RunState) -> list[tuple[text.Text, ...]]:
    """The two throughput readouts, split apart so raw collection speed is not
    conflated with end-to-end progress, and each held *steady between updates*
    rather than recomputed every game (which made both jitter while a cycle was
    still in flight):

    * ``raw`` — the last completed iteration's collection rate
      (games / collect-seconds). It is that iteration's settled figure, so it
      stays fixed while the next iteration's games are still streaming in
      instead of fluctuating with every game that lands.
    * ``overall`` — the true end-to-end rate (games over collect + update + eval
      wall time) across the whole most-recent evaluation cycle: every iteration
      since the previous eval up to and including the one that ran the latest
      eval. Amortizing the eval cost over all the iterations it covers means the
      value no longer dips on the single eval iteration — it advances once per
      eval cycle. Always ``<= raw`` (the denominator only adds overhead).
    """
    last = state.last_iter
    raw = last.games_per_sec if last is not None else 0.0
    raw_series = [im.games_per_sec for im in state.history]
    overall, overall_series = _overall_rates(state.history)
    raw_verdict_text, raw_verdict_color = _verdict("higher", raw_series, 0.0)
    return [
        (
            text.Text("raw perf", style=theme.TEXT_MUTED),
            text.Text(f"{raw:.1f} g/s", style=theme.TEXT_PRIMARY),
            text.Text(
                charts.sparkline(raw_series, _SPARK_CELLS), style=theme.SPARK_COLOR
            ),
            text.Text(raw_verdict_text, style=raw_verdict_color),
        ),
        (
            text.Text("overall", style=theme.TEXT_MUTED),
            text.Text(f"{overall:.2f} g/s", style=theme.TEXT_PRIMARY),
            text.Text(
                charts.sparkline(overall_series, _SPARK_CELLS), style=theme.SPARK_COLOR
            ),
            text.Text("incl update+eval", style=theme.TEXT_DIM2),
        ),
    ]


def _overall_rates(
    history: list[metrics.IterationMetrics],
) -> tuple[float, list[float]]:
    """The end-to-end games/sec of each completed evaluation cycle plus the most
    recent one. An eval cycle runs from just after the previous eval through the
    iteration that ran the next eval; its rate is the cycle's games over its
    total collect + update + eval wall time. The series gains a point only when
    an eval completes, so the live ``overall`` readout holds steady between
    evals. Before the first eval it falls back to the latest iteration's own
    end-to-end rate so the readout is still populated."""
    series: list[float] = []
    cycle_start = 0
    for index, item in enumerate(history):
        if item.eval is None:
            continue
        series.append(_cycle_rate(history[cycle_start : index + 1]))
        cycle_start = index + 1
    if series:
        return series[-1], series
    if history:
        return _cycle_rate(history[-1:]), []
    return 0.0, []


def _cycle_rate(items: list[metrics.IterationMetrics]) -> float:
    """End-to-end games/sec over a span of iterations: their collected games over
    their total collect + update + eval wall time."""
    games = sum(item.games_this_iter for item in items)
    seconds = sum(
        item.collect_seconds + item.update_seconds + item.eval_seconds for item in items
    )
    return games / seconds if seconds > 0.0 else 0.0


def _verdict(mode: str, series: list[float], grad_clip: float) -> tuple[str, str]:
    if len(series) < 2:
        return "—", theme.TEXT_MUTED
    latest, previous = series[-1], series[-2]
    if mode == "higher":
        if latest > previous:
            return "↑ faster", theme.GOOD
        if latest < previous:
            return "↓ slower", theme.CAUTION
        return "= steady", theme.TEXT_MUTED
    if mode == "lower":
        if latest < previous:
            return "↓ good", theme.GOOD
        return "↑ rising", theme.CAUTION
    if mode == "entropy":
        if latest < 0.05:
            return "low — collapsing", theme.BAD
        return "= healthy", theme.GOOD
    # grad norm
    if latest > grad_clip:
        return f"clipped >{grad_clip:.0f}", theme.CAUTION
    return f"= ok <{grad_clip:.0f}", theme.GOOD


def _timer(state: runstate.RunState) -> rich_console.RenderableType:
    """The time-to-target countdown panel between TRAINING HEALTH and RECENT EVENTS.

    Shows an invisible spacer when no target is set, progress during the final
    eval, "TARGET REACHED" at the pause, and a D:HH:MM:SS countdown otherwise.
    """
    target = state.target_iterations
    if target <= 0 or state.phase.is_terminal:
        # No target or run finished: render borderless spacer so health/events
        # expand symmetrically; the layout column still occupies its 28 chars.
        return panel.Panel(
            text.Text(""),
            box=box.SIMPLE,
            border_style=theme.BORDER_DEFAULT,
            padding=(0, 0),
        )

    body_lines: list[text.Text] = []

    if state.phase is runstate.Phase.PAUSED_AT_TARGET:
        line = text.Text(no_wrap=True)
        line.append("TARGET REACHED", style=theme.GOOD)
        body_lines.append(line)
        target_line = text.Text(no_wrap=True)
        target_line.append("iter ", style=theme.TEXT_MUTED)
        target_line.append(f"{target:,}", style=theme.TEXT_DIM2)
        body_lines.append(target_line)
    elif state.phase is runstate.Phase.FINAL_EVALUATING:
        done, total = state.final_eval_progress
        label_line = text.Text("FINAL EVAL", style=theme.TEXT_MUTED, no_wrap=True)
        body_lines.append(label_line)
        progress_line = text.Text(no_wrap=True)
        progress_line.append(f"{done}", style=theme.TEXT_PRIMARY)
        progress_line.append(" / ", style=theme.TEXT_MUTED)
        progress_line.append(f"{total}", style=theme.TEXT_DIM2)
        body_lines.append(progress_line)
    else:
        remaining = state.time_remaining_seconds()
        if remaining is None:
            body_lines.append(text.Text("estimating…", style=theme.TEXT_MUTED))
        elif remaining <= 0:
            body_lines.append(text.Text("arriving…", style=theme.GOOD))
        else:
            clock_line = text.Text(no_wrap=True)
            clock_line.append(_fmt_remaining(remaining), style=theme.TEXT_PRIMARY)
            clock_line.append(" remaining", style=theme.TEXT_MUTED)
            body_lines.append(clock_line)
        target_line = text.Text(no_wrap=True)
        target_line.append("target: ", style=theme.TEXT_MUTED)
        target_line.append(f"{target:,}", style=theme.TEXT_DIM2)
        body_lines.append(target_line)

    return panel.Panel(
        rich_console.Group(*body_lines),
        box=box.ROUNDED,
        border_style=theme.BORDER_DEFAULT,
        padding=(0, 1),
    )


def _fmt_remaining(seconds: float) -> str:
    """Format a remaining-time duration as ``D:HH:MM:SS`` (days omitted when 0)."""
    total = int(seconds)
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days > 0:
        return f"{days:d}:{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:d}:{minutes:02d}:{secs:02d}"


def _events(state: runstate.RunState, pause_buffer: str = "") -> panel.Panel:
    # When the run is paused at the target milestone, replace the event log
    # with the user acknowledgment UI (keys + new-target input).
    if state.phase is runstate.Phase.PAUSED_AT_TARGET:
        return _pause_panel(state, pause_buffer)

    # Oldest of the recent events first so the newest lands at the bottom and the
    # log scrolls upward as fresh events arrive; show as many as fill the band.
    lines: list[text.Text] = []
    for event in state.events[-_EVENT_LINES:]:
        glyph = theme.EVENT_GLYPH[event.kind]
        color = theme.EVENT_COLOR[event.kind]
        line = text.Text(no_wrap=True)  # default end newline: one event per line
        line.append(f"{event.clock}  ", style=theme.TEXT_MUTED)
        line.append(f"{glyph} ", style=color)
        line.append(event.text, style=color)
        lines.append(line)
    body: rich_console.RenderableType = (
        rich_console.Group(*lines)
        if lines
        else text.Text("no events yet", style=theme.TEXT_MUTED)
    )
    return panel.Panel(
        body,
        title="[b]RECENT EVENTS[/b]",
        title_align="left",
        box=box.ROUNDED,
        border_style=theme.BORDER_DEFAULT,
        padding=(0, 1),
    )


def _pause_panel(state: runstate.RunState, pause_buffer: str) -> panel.Panel:
    """Acknowledgment overlay shown in the events panel at PAUSED_AT_TARGET.

    Displays final eval stats and prompts the user with [C]ontinue / [E]nd
    options plus an optional new-target numeric input field.
    """
    eval_stats = state.pinned_stats
    lines: list[text.Text] = [text.Text("")]

    summary_line = text.Text(no_wrap=True)
    summary_line.append(
        f"  Iteration {state.iteration + 1:,} complete.", style=theme.TEXT_PRIMARY
    )
    lines.append(summary_line)

    if eval_stats is not None:
        stats_line = text.Text(no_wrap=True)
        stats_line.append(
            f"  avg {eval_stats.avg_breakdown.total:.1f} pts · "
            f"margin {eval_stats.mean_margin:.1f} pts",
            style=theme.TEXT_DIM2,
        )
        lines.append(stats_line)

    lines.append(text.Text(""))

    key_line = text.Text(no_wrap=True)
    key_line.append("  ", style=theme.TEXT_MUTED)
    key_line.append("[C]", style=theme.GOOD)
    key_line.append(" Continue  ", style=theme.TEXT_PRIMARY)
    key_line.append("[E]", style=theme.CAUTION)
    key_line.append(" End run", style=theme.TEXT_PRIMARY)
    lines.append(key_line)

    lines.append(text.Text(""))

    prompt_line = text.Text(no_wrap=True)
    prompt_line.append("  New target + ENTER to continue:", style=theme.TEXT_MUTED)
    lines.append(prompt_line)

    input_line = text.Text(no_wrap=True)
    input_line.append("  > ", style=theme.TEXT_MUTED)
    input_line.append(
        pause_buffer if pause_buffer else "(or ENTER to skip)", style=theme.TEXT_PRIMARY
    )
    input_line.append("_", style=theme.TEXT_MUTED)
    lines.append(input_line)

    return panel.Panel(
        rich_console.Group(*lines),
        title="[b]TARGET REACHED[/b]",
        title_align="left",
        box=box.ROUNDED,
        border_style=theme.GOOD,
        padding=(0, 1),
    )


def _runtime_clock(seconds: float) -> str:
    """Total runtime as ``D:HH:MM:SS`` (days never zero-padded)."""
    total = int(seconds)
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{days:d}:{hours:02d}:{minutes:02d}:{secs:02d}"
