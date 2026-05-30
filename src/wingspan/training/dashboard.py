"""Assembles the live dashboard: the Layout tree and the per-region renderers.

``build_layout`` creates the five-band "FLYWAY CONTROL" skeleton once;
``render`` repaints it from a :class:`runstate.RunState` snapshot each frame.
The bands, top to bottom, read as a guided narrative: WHERE AM I (header) ->
IN-GAME PERFORMANCE / DECISION MODELS (middle) -> IS IT GETTING BETTER (the
gold-bordered hero band: the win-rate and self-play-points charts side by side
plus the docked eval inset) -> DIAGNOSTICS (health + events).

Several band-specific micro-renderables live here because they must fill the
full panel width every refresh: the phase-colored header progress bar, the
two-tone system RAM gauge, and the stacked six-component score bar. The hero
charts come from :mod:`charts`.
"""

from __future__ import annotations

import rich.console as rich_console
from rich import box, layout, panel, table, text

from wingspan.training import charts, metrics, runstate, theme

_WORDMARK = "🪶 WINGSPAN  FLYWAY CONTROL"
_SPARK_CELLS = 8

# IN-GAME PERFORMANCE score table geometry.
_PT_NAME_W = 8  # the per-source name column ("tucked" / "rounds" / "TOTAL")
_PT_PAIR_W = 12  # one "###.#  ##.#%" points / share pair
_Z95 = 1.96  # standard-normal z for the 95% confidence interval

# System band gauges.
_GAUGE_LABEL_W = 5  # "VRAM" + a space — the widest of CPU/RAM/GPU/VRAM
_GAUGE_CAPS_W = 2  # the ▕ ▏ end-caps around the bar
_GAUGE_PCT_W = 5  # the " 100%" / "   0%" trailing percent field (leading space)
_GAUGE_GAP = 3  # blank columns between the left (compute) and right (memory) halves
_GAUGE_MIN_CELLS = 4  # never draw a bar narrower than this
_GAUGE_CHROME_W = _GAUGE_LABEL_W + _GAUGE_CAPS_W + _GAUGE_PCT_W  # non-bar width


def build_layout() -> layout.Layout:
    """Create the empty five-band layout skeleton (populated by :func:`render`)."""
    root = layout.Layout(name="root")
    root.split_column(
        layout.Layout(name="header", size=5),
        layout.Layout(name="system", size=3),
        layout.Layout(name="middle", ratio=12, minimum_size=14),
        layout.Layout(name="headline", ratio=13, minimum_size=12),
        layout.Layout(name="footer", size=8),
    )
    root["middle"].split_row(
        layout.Layout(name="produce", ratio=58),
        layout.Layout(name="learning", ratio=42, minimum_size=42),
    )
    root["footer"].split_row(
        layout.Layout(name="health", ratio=40),
        layout.Layout(name="events", ratio=60),
    )
    return root


def render(root: layout.Layout, state: runstate.RunState, frame: int) -> None:
    """Repaint every region from the current state."""
    root["header"].update(_header(state))
    root["system"].update(_system(state))
    root["produce"].update(_produce(state))
    root["learning"].update(_learning(state))
    root["headline"].update(_getting_better_panel(state, frame))
    root["health"].update(_health(state))
    root["events"].update(_events(state))


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
    grid = table.Table.grid(expand=True)
    grid.add_column(justify="left")
    grid.add_column(justify="right")
    grid.add_row(_gradient_text(_WORDMARK), _phase_pill(state))
    return grid


def _gradient_text(content: str) -> text.Text:
    colors = theme.gradient_stops(theme.WORDMARK_STOPS, len(content))
    out = text.Text(no_wrap=True, end="")
    for char, color in zip(content, colors):
        out.append(char, style=f"bold {color}")
    return out


def _phase_pill(state: runstate.RunState) -> text.Text:
    """The upper-right phase badge — just the phase word on its accent color."""
    color = theme.PHASE_COLOR[state.phase]
    pill = text.Text(no_wrap=True, end="")
    pill.append(
        f" {state.phase.value.upper()} ", style=f"bold {theme.CANVAS} on {color}"
    )
    return pill


def _header_stats_row(state: runstate.RunState) -> text.Text:
    """The single text status line: iteration, this iteration's progress count,
    cumulative games, total runtime, and this iteration's elapsed seconds."""
    label, done, total = _header_progress(state)
    out = text.Text(no_wrap=True)  # default end newline: own its row in the Group
    out.append(f"iter {state.iteration:04d}", style=theme.TEXT_PRIMARY)
    out.append(f"    {label} ", style=theme.TEXT_MUTED)
    out.append(f"{done}/{total}", style=theme.TEXT_PRIMARY)
    out.append("    Σ ", style=theme.TEXT_MUTED)
    out.append(f"{state.total_games:,}", style=theme.TEXT_PRIMARY)
    out.append(" games", style=theme.TEXT_MUTED)
    out.append("    RUNTIME: ", style=theme.TEXT_MUTED)
    out.append(_runtime_clock(state.elapsed()), style=theme.TEXT_DIM2)
    out.append(
        f"   ({state.iter_elapsed():.1f}s current iteration)", style=theme.TEXT_MUTED
    )
    return out


def _header_progress(state: runstate.RunState) -> tuple[str, int, int]:
    """The ``(label, done, total)`` for the header bar: held-out eval games
    while evaluating, otherwise this iteration's self-play collection progress."""
    if state.phase is runstate.Phase.EVALUATING and state.eval_games_in_iter > 0:
        return "eval", state.eval_game_in_iter, state.eval_games_in_iter
    return "game", state.game_in_iter, state.games_in_iter


class _HeaderProgress:
    """The full-width progress bar row for the current iteration, colored by the
    active phase (the phase word itself is dropped — the upper-right badge already
    names it)."""

    def __init__(self, state: runstate.RunState):
        self.state = state

    def __rich_console__(
        self, console: rich_console.Console, options: rich_console.ConsoleOptions
    ) -> rich_console.RenderResult:
        width = options.max_width
        color = theme.PHASE_COLOR[self.state.phase]
        _, done, total = _header_progress(self.state)
        cells = max(1, width - 2)  # leave room for the ▕ ▏ end-caps
        fill = round(cells * done / total) if total else 0
        bar = text.Text(no_wrap=True, end="")
        bar.append("▕", style=theme.TEXT_MUTED)
        bar.append("█" * fill, style=color)
        bar.append("░" * (cells - fill), style=theme.BORDER_DEFAULT)
        bar.append("▏", style=theme.TEXT_MUTED)
        yield bar


#### System band — host telemetry ####


def _system(state: runstate.RunState) -> panel.Panel:
    return panel.Panel(
        _SystemGauges(state.system),
        title="[b]SYSTEM[/b]",
        title_align="left",
        box=box.ROUNDED,
        border_style=theme.BORDER_DEFAULT,
        padding=(0, 1),
    )


class _SystemGauges:
    """One gauge row — CPU utilization on the left, system RAM on the right —
    sized to fill the panel. A width-aware renderable (like the score bar) so
    each half takes half the width and the bars stretch with the terminal;
    shows a placeholder until the monitor's first snapshot lands. This process's
    resident slice rides inside the RAM bar in its own color."""

    def __init__(self, stats: metrics.SystemStats | None):
        self.stats = stats

    def __rich_console__(
        self, console: rich_console.Console, options: rich_console.ConsoleOptions
    ) -> rich_console.RenderResult:
        if self.stats is None:
            yield text.Text("  sampling host telemetry…", style=theme.TEXT_MUTED)
            return
        left_w, right_w = _split_halves(options.max_width)
        cpu = _util_gauge("CPU", self.stats.cpu_percent, left_w)
        ram = _mem_gauge(
            "RAM",
            self.stats.ram_used_gb,
            self.stats.ram_total_gb,
            self.stats.proc_rss_gb,
            self.stats.ram_percent,
            right_w,
        )
        yield _gauge_row(cpu, ram, left_w)


def _split_halves(width: int) -> tuple[int, int]:
    left = max(0, (width - _GAUGE_GAP) // 2)
    right = max(0, width - _GAUGE_GAP - left)
    return left, right


def _gauge_row(left: text.Text, right: text.Text, left_w: int) -> text.Text:
    line = text.Text(no_wrap=True)  # default end newline: own one row
    line.append_text(left)
    pad = left_w - left.cell_len
    if pad > 0:
        line.append(" " * pad)
    line.append(" " * _GAUGE_GAP)
    line.append_text(right)
    return line


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
            _StackedBar(stats.breakdown),
            text.Text(""),
            _produce_table(stats),
            text.Text(""),
            _produce_stats_block(state, stats),
        )
    return panel.Panel(
        body,
        title="[b]IN-GAME PERFORMANCE[/b]",
        title_align="left",
        box=box.ROUNDED,
        border_style=theme.BORDER_DEFAULT,
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
    body = rich_console.Group(
        charts.FamilyHistogram(state.cum_family),
        _total_decisions_line(state),
    )
    return panel.Panel(
        body,
        title="[b]DECISION MODELS[/b]",
        title_align="left",
        box=box.ROUNDED,
        border_style=theme.BORDER_DEFAULT,
        padding=(0, 1),
    )


def _total_decisions_line(state: runstate.RunState) -> text.Text:
    """The full (un-shortened) cumulative decision count, sitting just above the
    panel's bottom border."""
    line = text.Text(no_wrap=True)  # default end newline: own its row in the Group
    line.append(f"{state.total_decisions:,}", style=theme.TEXT_PRIMARY)
    line.append(" total decisions", style=theme.TEXT_MUTED)
    return line


#### Headline band — convergence charts ####


def _getting_better_panel(state: runstate.RunState, frame: int) -> panel.Panel:
    """The single gold hero panel: the two side-by-side convergence charts
    (win rate · avg points) plus the docked EVAL inset."""
    return panel.Panel(
        charts.GettingBetterChart(state, frame),
        title="[b]IS IT GETTING BETTER?[/b]",
        subtitle="win rate (left) · avg self-play points & eval margin (right)",
        title_align="left",
        subtitle_align="left",
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
    conflated with end-to-end progress — both measured over the *same* recent
    window (the last completed iteration) so ``overall`` is exactly ``raw``
    minus the update/eval/checkpoint overhead:

    * ``raw`` — 1 / (avg wall time per game) during collection, i.e. how fast
      self-play games are produced (live ``games_per_sec``); excludes the
      update and eval phases.
    * ``overall`` — the last iteration's games over its *total* wall time
      (collect + update + eval), the true end-to-end rate. Always ``<= raw``
      because the denominator only adds time. (Deliberately not the lifetime
      ``total_games / elapsed()``: that is a cumulative average distorted by
      resume accounting and by game length drifting over training, and it can
      read *above* the current collection rate — the opposite of intent.)
    """
    raw = state.games_per_sec
    last = state.last_iter
    overall = _iter_overall_rate(last) if last is not None else 0.0
    raw_series = [im.games_per_sec for im in state.history]
    overall_series = [_iter_overall_rate(im) for im in state.history]
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


def _iter_overall_rate(im: metrics.IterationMetrics) -> float:
    """End-to-end games/sec for one iteration: its collected games over its
    whole wall time (collection + update + eval). On an eval-less iteration
    ``eval_seconds`` is 0, so this approaches the raw collection rate."""
    seconds = im.collect_seconds + im.update_seconds + im.eval_seconds
    return im.games_this_iter / seconds if seconds > 0.0 else 0.0


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


def _events(state: runstate.RunState) -> panel.Panel:
    # Oldest of the recent five first so the newest lands at the bottom and the
    # log scrolls upward as fresh events arrive.
    lines: list[text.Text] = []
    for event in state.events[-5:]:
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


def _runtime_clock(seconds: float) -> str:
    """Total runtime as ``D:HH:MM:SS`` (days never zero-padded)."""
    total = int(seconds)
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{days:d}:{hours:02d}:{minutes:02d}:{secs:02d}"
