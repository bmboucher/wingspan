"""Assembles the live dashboard: the Layout tree and the per-region renderers.

``build_layout`` creates the five-band "FLYWAY CONTROL" skeleton once;
``render`` repaints it from a :class:`runstate.RunState` snapshot each frame.
The bands, top to bottom, read as a guided narrative: WHERE AM I (header) ->
WHAT IT'S PRODUCING / WHAT IT'S LEARNING (middle) -> IS IT GETTING BETTER /
HOW STRONG IS THE PLAY (the gold-bordered hero band, split into the win-rate
and the self-play-points charts) -> DIAGNOSTICS (health + events).

Two band-specific micro-renderables live here because they must fill the full
panel width every refresh: the phase-colored status LED rule and the stacked
six-component score bar. The two hero charts come from :mod:`charts`.
"""

from __future__ import annotations

import rich.console as rich_console
from rich import box, layout, panel, table, text

from wingspan.training import charts, metrics, runstate, theme

_WORDMARK = "🪶 WINGSPAN  FLYWAY CONTROL"
_PROGRESS_CELLS = 16
_SPARK_CELLS = 8

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
    # The hero band is two side-by-side charts: win-rate (its own 0..100% axis)
    # and self-play points / eval margin (a shared, auto-scaled point axis).
    root["headline"].split_row(
        layout.Layout(name="winrate", ratio=1),
        layout.Layout(name="points", ratio=1),
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
    root["winrate"].update(_winrate_panel(state, frame))
    root["points"].update(_points_panel(state, frame))
    root["health"].update(_health(state))
    root["events"].update(_events(state))


###### PRIVATE #######

#### Header band ####


def _header(state: runstate.RunState) -> panel.Panel:
    body = rich_console.Group(
        _wordmark_row(state), _status_row(state), _PhaseRule(state)
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
    color = theme.PHASE_COLOR[state.phase]
    pill = text.Text(no_wrap=True, end="")
    pill.append(
        f" {state.phase.value.upper()} ", style=f"bold {theme.CANVAS} on {color}"
    )
    pill.append(f"  {state.config.device}", style=theme.TEXT_DIM2)
    return pill


def _status_row(state: runstate.RunState) -> table.Table:
    grid = table.Table.grid(expand=True)
    grid.add_column(justify="left")
    grid.add_column(justify="right")
    grid.add_row(_left_status(state), _right_status(state))
    return grid


def _left_status(state: runstate.RunState) -> text.Text:
    color = theme.PHASE_COLOR[state.phase]
    label, done, total = _header_progress(state)
    out = text.Text(no_wrap=True, end="")
    out.append(f"iter {state.iteration:04d}", style=theme.TEXT_PRIMARY)
    out.append(f"   {label} ", style=theme.TEXT_MUTED)
    out.append(_progress_bar(done, total, color))
    out.append(f" {done}/{total}", style=theme.TEXT_PRIMARY)
    out.append("   Σ ", style=theme.TEXT_MUTED)
    out.append(f"{state.total_games:,}", style=theme.TEXT_PRIMARY)
    out.append(" games", style=theme.TEXT_MUTED)
    return out


def _header_progress(state: runstate.RunState) -> tuple[str, int, int]:
    """The ``(label, done, total)`` for the header bar: held-out eval games
    while evaluating, otherwise this iteration's self-play collection progress."""
    if state.phase is runstate.Phase.EVALUATING and state.eval_games_in_iter > 0:
        return "eval", state.eval_game_in_iter, state.eval_games_in_iter
    return "game", state.game_in_iter, state.games_in_iter


def _right_status(state: runstate.RunState) -> text.Text:
    out = text.Text(no_wrap=True, end="")
    out.append(f"{state.games_per_sec:.1f}", style=theme.TEXT_PRIMARY)
    out.append(" g/s ", style=theme.TEXT_MUTED)
    spark = charts.sparkline([im.games_per_sec for im in state.history], _SPARK_CELLS)
    out.append(spark, style=theme.SPARK_COLOR)
    out.append("  REINFORCE+baseline ", style=theme.TEXT_MUTED)
    out.append(f"lr {state.config.lr:.0e}", style=theme.TEXT_DIM2)
    return out


def _progress_bar(done: int, total: int, color: str) -> text.Text:
    fill = round(_PROGRESS_CELLS * done / total) if total else 0
    bar = text.Text(no_wrap=True, end="")
    bar.append("▕", style=theme.TEXT_MUTED)
    bar.append("█" * fill, style=color)
    bar.append("░" * (_PROGRESS_CELLS - fill), style=theme.BORDER_DEFAULT)
    bar.append("▏", style=theme.TEXT_MUTED)
    return bar


class _PhaseRule:
    """The full-width phase-colored LED rule with the phase word centered and
    the two live chronometers (since start / since iteration) docked right."""

    def __init__(self, state: runstate.RunState):
        self.state = state

    def __rich_console__(
        self, console: rich_console.Console, options: rich_console.ConsoleOptions
    ) -> rich_console.RenderResult:
        width = options.max_width
        color = theme.PHASE_COLOR[self.state.phase]
        dim = theme.lerp_color(color, theme.CANVAS, 0.5)
        word = f" {self.state.phase.value.upper()} "
        chrono = f"T+ {_clock(self.state.elapsed())}   ⟳ {_clock(self.state.iter_elapsed())} "
        word_start = max(0, (width - len(word)) // 2)
        chrono_start = max(word_start + len(word), width - len(chrono))

        line = text.Text(no_wrap=True, end="")
        line.append("▉" * word_start, style=dim)
        line.append(word, style=f"bold {theme.CANVAS} on {color}")
        line.append("▉" * max(0, chrono_start - word_start - len(word)), style=dim)
        line.append(chrono[: max(0, width - chrono_start)], style=theme.TEXT_DIM2)
        yield line


#### System band — host telemetry ####


def _system(state: runstate.RunState) -> panel.Panel:
    return panel.Panel(
        _SystemGauges(state.system),
        title="[b]SYSTEM[/b]",
        subtitle=_system_subtitle(state.system),
        title_align="left",
        subtitle_align="left",
        box=box.ROUNDED,
        border_style=theme.BORDER_DEFAULT,
        padding=(0, 1),
    )


def _system_subtitle(stats: metrics.SystemStats | None) -> str:
    if stats is None:
        return "host CPU / memory"
    return f"proc {stats.proc_rss_gb:.1f} GB resident"


class _SystemGauges:
    """One gauge row — CPU utilization on the left, system RAM on the right —
    sized to fill the panel. A width-aware renderable (like the score bar) so
    each half takes half the width and the bars stretch with the terminal;
    shows a placeholder until the monitor's first snapshot lands."""

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
    label: str, used_gb: float, total_gb: float, pct: float, width: int
) -> text.Text:
    suffix = f"  {used_gb:.1f}/{total_gb:.1f} GB"
    bar_cells = width - _GAUGE_CHROME_W - len(suffix)
    show_suffix = bar_cells >= _GAUGE_MIN_CELLS
    if not show_suffix:
        bar_cells = width - _GAUGE_CHROME_W
    out = _gauge_label(label)
    _append_bar(out, pct / 100.0, _mem_color(pct), bar_cells)
    out.append(f" {pct:>3.0f}%", style=theme.TEXT_PRIMARY)
    if show_suffix:
        out.append(suffix, style=theme.TEXT_MUTED)
    return out


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


#### Produce band — score breakdown ####


def _produce(state: runstate.RunState) -> panel.Panel:
    stats = state.produce_stats()
    body: rich_console.RenderableType
    if stats is None:
        body = text.Text("awaiting first game…", style=theme.TEXT_MUTED)
    else:
        total_line = table.Table.grid(expand=True)
        total_line.add_column(justify="left")
        total_line.add_column(justify="right")
        total_value = text.Text(no_wrap=True, end="")
        total_value.append("TOTAL  ", style=theme.TEXT_MUTED)
        total_value.append(
            f"{stats.breakdown.total:.1f}", style=f"bold {theme.TEXT_BRIGHT}"
        )
        total_value.append(" pts/game", style=theme.TEXT_MUTED)
        won_value = text.Text(no_wrap=True, end="")
        won_value.append("winners  ", style=theme.TEXT_MUTED)
        won_value.append(
            f"{stats.winner_breakdown.total:.1f}", style=theme.TEXT_PRIMARY
        )
        won_value.append(" pts", style=theme.TEXT_MUTED)
        total_line.add_row(total_value, won_value)

        body = rich_console.Group(
            total_line,
            _StackedBar(stats.breakdown),
            _legend_table(stats),
            _produce_footer(state, stats),
        )
    return panel.Panel(
        body,
        title="[b]WHAT THE AI IS PRODUCING[/b]",
        subtitle="recent (EWMA) score by source · all games vs winners only",
        title_align="left",
        subtitle_align="left",
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


def _legend_table(stats: metrics.ProduceStats) -> table.Table:
    """Per-source legend with two pts / share groups side by side: the all-game
    average (left) and the same split conditioned on just the winning seat
    (right), so the sources that separate winners from losers stand out."""
    overall = stats.breakdown.components()
    winner = dict(stats.winner_breakdown.components())
    overall_total = stats.breakdown.total or 1.0
    winner_total = stats.winner_breakdown.total or 1.0
    grid = table.Table.grid(padding=(0, 1))
    grid.add_column()  # swatch
    grid.add_column()  # name
    grid.add_column(justify="right")  # all-game pts
    grid.add_column(justify="right")  # all-game share
    grid.add_column(justify="right")  # winner pts
    grid.add_column(justify="right")  # winner share
    grid.add_row(
        text.Text(""),
        text.Text(""),
        text.Text("all", style=theme.TEXT_MUTED),
        text.Text(""),
        text.Text("won", style=theme.TEXT_MUTED),
        text.Text(""),
    )
    for name, value in overall:
        color = theme.SCORE_COLOR[name]
        won = winner.get(name, 0.0)
        grid.add_row(
            text.Text("■", style=color),
            text.Text(name, style=theme.TEXT_PRIMARY),
            text.Text(f"{value:>4.1f} pts", style=theme.TEXT_PRIMARY),
            text.Text(f"{value / overall_total * 100:>4.1f}%", style=theme.TEXT_MUTED),
            text.Text(f"{won:>4.1f} pts", style=theme.TEXT_DIM2),
            text.Text(f"{won / winner_total * 100:>4.1f}%", style=theme.TEXT_MUTED),
        )
    return grid


def _produce_footer(state: runstate.RunState, stats: metrics.ProduceStats) -> text.Text:
    lo = state.game_len_min if state.game_len_min is not None else 0
    hi = state.game_len_max if state.game_len_max is not None else 0
    out = text.Text(no_wrap=True, end="")
    out.append("game length ", style=theme.TEXT_MUTED)
    out.append(f"{stats.decisions:.0f}", style=theme.TEXT_PRIMARY)
    out.append(f" dec/game  (range {lo}–{hi})\n", style=theme.TEXT_MUTED)
    out.append("score margin ", style=theme.TEXT_MUTED)
    out.append(f"{stats.margin:+.1f}", style=theme.TEXT_PRIMARY)
    out.append(" self−opp   σ ", style=theme.TEXT_MUTED)
    out.append(f"{stats.margin_std:.1f}\n", style=theme.TEXT_DIM2)
    out.append("winning margin ", style=theme.TEXT_MUTED)
    out.append(f"{stats.abs_margin:.1f}", style=theme.TEXT_PRIMARY)
    out.append(" |self−opp|   σ ", style=theme.TEXT_MUTED)
    out.append(f"{stats.abs_margin_std:.1f}", style=theme.TEXT_DIM2)
    return out


#### Learning band — family histogram ####


def _learning(state: runstate.RunState) -> panel.Panel:
    # The live counts ride in the subtitle so the body is exactly the 13 family
    # rows — they never get squeezed out by an extra header line.
    subtitle = (
        f"{charts.human_count(state.total_decisions)} decisions over "
        f"{state.total_games:,} games · by skill"
    )
    return panel.Panel(
        charts.FamilyHistogram(state.cum_family),
        title="[b]WHAT IT'S LEARNING TO DECIDE[/b]",
        subtitle=subtitle,
        title_align="left",
        subtitle_align="left",
        box=box.ROUNDED,
        border_style=theme.BORDER_DEFAULT,
        padding=(0, 1),
    )


#### Headline band — convergence charts ####


def _winrate_panel(state: runstate.RunState, frame: int) -> panel.Panel:
    return panel.Panel(
        charts.WinRateChart(state, frame),
        title="[b]IS IT GETTING BETTER?[/b]",
        subtitle=f"win rate vs {_opponent_label(state)} · higher is better",
        title_align="left",
        subtitle_align="left",
        box=box.HEAVY,
        border_style=theme.BORDER_HEADLINE,
        padding=(0, 1),
    )


def _points_panel(state: runstate.RunState, frame: int) -> panel.Panel:
    return panel.Panel(
        charts.PointsChart(state, frame),
        title="[b]HOW STRONG IS THE PLAY?[/b]",
        subtitle="avg self-play points & eval margin · climbing toward 100+",
        title_align="left",
        subtitle_align="left",
        box=box.HEAVY,
        border_style=theme.BORDER_HEADLINE,
        padding=(0, 1),
    )


def _opponent_label(state: runstate.RunState) -> str:
    """How the current reference opponent reads in the win-rate subtitle."""
    gen = state.opponent_generation
    return "a random opponent" if gen == 0 else f"a frozen self · gen {gen}"


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
    rows: list[tuple[text.Text, ...]] = []
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
    if not rows:
        rows.append((text.Text("awaiting first iteration…", style=theme.TEXT_MUTED),))
    return rows


def _verdict(mode: str, series: list[float], grad_clip: float) -> tuple[str, str]:
    if len(series) < 2:
        return "—", theme.TEXT_MUTED
    latest, previous = series[-1], series[-2]
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


def _clock(seconds: float) -> str:
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
