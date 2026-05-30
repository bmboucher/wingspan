"""Assembles the live dashboard: the Layout tree and the per-region renderers.

``build_layout`` creates the five-band "FLYWAY CONTROL" skeleton once;
``render`` repaints it from a :class:`runstate.RunState` snapshot each frame.
The bands, top to bottom, read as a guided narrative: WHERE AM I (header) ->
WHAT IT'S PRODUCING / WHAT IT'S LEARNING (middle) -> IS IT GETTING BETTER
(the gold-bordered hero convergence chart) -> DIAGNOSTICS (health + events).

Two band-specific micro-renderables live here because they must fill the full
panel width every refresh: the phase-colored status LED rule and the stacked
six-component score bar. The two big charts come from :mod:`charts`.
"""

from __future__ import annotations

import rich.console as rich_console
from rich import box, layout, panel, table, text

from wingspan.training import charts, metrics, runstate, theme

_WORDMARK = "🪶 WINGSPAN  FLYWAY CONTROL"
_PROGRESS_CELLS = 16
_LEGEND_BAR_CELLS = 16
_SPARK_CELLS = 8


def build_layout() -> layout.Layout:
    """Create the empty five-band layout skeleton (populated by :func:`render`)."""
    root = layout.Layout(name="root")
    root.split_column(
        layout.Layout(name="header", size=5),
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
    root["produce"].update(_produce(state))
    root["learning"].update(_learning(state))
    root["headline"].update(_headline(state, frame))
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
    out = text.Text(no_wrap=True, end="")
    out.append(f"iter {state.iteration:04d}", style=theme.TEXT_PRIMARY)
    out.append("   game ", style=theme.TEXT_MUTED)
    out.append(_progress_bar(state.game_in_iter, state.games_in_iter, color))
    out.append(f" {state.game_in_iter}/{state.games_in_iter}", style=theme.TEXT_PRIMARY)
    out.append("   Σ ", style=theme.TEXT_MUTED)
    out.append(f"{state.total_games:,}", style=theme.TEXT_PRIMARY)
    out.append(" games", style=theme.TEXT_MUTED)
    return out


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


#### Produce band — score breakdown ####


def _produce(state: runstate.RunState) -> panel.Panel:
    avg = state.avg_breakdown()
    total_line = table.Table.grid(expand=True)
    total_line.add_column(justify="left")
    total_line.add_column(justify="right")
    total_value = text.Text(no_wrap=True, end="")
    total_value.append("TOTAL  ", style=theme.TEXT_MUTED)
    total_value.append(f"{avg.total:.1f}", style=f"bold {theme.TEXT_BRIGHT}")
    total_value.append(" pts/game", style=theme.TEXT_MUTED)
    game_len = text.Text(no_wrap=True, end="")
    game_len.append("avg game  ", style=theme.TEXT_MUTED)
    game_len.append(f"{state.avg_decisions():.0f}", style=theme.TEXT_PRIMARY)
    game_len.append(" decisions", style=theme.TEXT_MUTED)
    total_line.add_row(total_value, game_len)

    body = rich_console.Group(
        total_line,
        _StackedBar(avg),
        _legend_table(avg),
        _produce_footer(state),
    )
    return panel.Panel(
        body,
        title="[b]WHAT THE AI IS PRODUCING[/b]",
        subtitle="average final score, split into its six sources",
        title_align="left",
        subtitle_align="left",
        box=box.ROUNDED,
        border_style=theme.BORDER_DEFAULT,
        padding=(0, 1),
    )


class _StackedBar:
    """One full-width bar split into six colored segments proportional to each
    score component's average points."""

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
            bar.append(theme.SCORE_GLYPH[name] * seg_w, style=theme.SCORE_COLOR[name])
        yield bar


def _legend_table(avg: metrics.ScoreBreakdown) -> table.Table:
    components = avg.components()
    max_value = max((value for _, value in components), default=1.0) or 1.0
    grid = table.Table.grid(padding=(0, 1))
    grid.add_column()  # swatch
    grid.add_column()  # name
    grid.add_column(justify="right")  # pts
    grid.add_column()  # mini bar
    grid.add_column(justify="right")  # share
    total = avg.total or 1.0
    for name, value in components:
        color = theme.SCORE_COLOR[name]
        bar = charts.eighth_bar(value / max_value, _LEGEND_BAR_CELLS, min_tick=True)
        grid.add_row(
            text.Text("■", style=color),
            text.Text(name, style=theme.TEXT_PRIMARY),
            text.Text(f"{value:>4.1f} pts", style=theme.TEXT_PRIMARY),
            text.Text(bar.ljust(_LEGEND_BAR_CELLS), style=color),
            text.Text(f"{value / total * 100:>4.1f}%", style=theme.TEXT_MUTED),
        )
    return grid


def _produce_footer(state: runstate.RunState) -> text.Text:
    lo = state.game_len_min if state.game_len_min is not None else 0
    hi = state.game_len_max if state.game_len_max is not None else 0
    out = text.Text(no_wrap=True, end="")
    out.append("game length ", style=theme.TEXT_MUTED)
    out.append(f"{state.avg_decisions():.0f}", style=theme.TEXT_PRIMARY)
    out.append(f" dec/game  (range {lo}–{hi})\n", style=theme.TEXT_MUTED)
    out.append("score margin ", style=theme.TEXT_MUTED)
    out.append(f"{state.avg_margin():+.1f}", style=theme.TEXT_PRIMARY)
    out.append(" self−opp   σ ", style=theme.TEXT_MUTED)
    out.append(f"{state.margin_std():.1f}", style=theme.TEXT_DIM2)
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


#### Headline band — convergence chart ####


def _headline(state: runstate.RunState, frame: int) -> panel.Panel:
    return panel.Panel(
        charts.ConvergenceChart(state, frame),
        title="[b]IS IT GETTING BETTER?[/b]",
        subtitle="win rate vs a random opponent · higher is better",
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
    lines: list[text.Text] = []
    for event in reversed(state.events[-5:]):
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
