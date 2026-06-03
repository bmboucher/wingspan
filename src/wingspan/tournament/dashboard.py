"""The live tournament dashboard: a five-region ``rich`` layout repainted each
frame from :class:`state.TournamentState`.

A header band (wordmark + clock + throughput), a standings table (rank, Elo, an
Elo-trend sparkline, record, win rate, margin — sorted by live Elo), a recent-
events panel, and a progress bar. Visual constants are reused from the training
dashboard's :mod:`theme` and :mod:`charts.text_helpers` so the two UIs match.
"""

from __future__ import annotations

from rich import box, layout, panel, table, text

from wingspan.tournament import state as state_module
from wingspan.training import theme
from wingspan.training.charts import text_helpers

# Elo-trend sparkline width and the count of recent events shown.
_SPARK_WIDTH = 16
_PROGRESS_WIDTH = 48
_EVENTS_SHOWN = 14

_PHASE_COLOR: dict[state_module.TournamentPhase, str] = {
    state_module.TournamentPhase.RUNNING: "#3FB4A6",
    state_module.TournamentPhase.DONE: theme.GOOD,
    state_module.TournamentPhase.STOPPED: theme.CAUTION,
    state_module.TournamentPhase.ERROR: theme.BAD,
}


def build_layout() -> layout.Layout:
    """The fixed five-region layout: header / (standings | events) / progress."""
    root = layout.Layout()
    root.split_column(
        layout.Layout(name="header", size=3),
        layout.Layout(name="body", ratio=1),
        layout.Layout(name="footer", size=3),
    )
    root["body"].split_row(
        layout.Layout(name="standings", ratio=2),
        layout.Layout(name="events", ratio=1),
    )
    return root


def render(root: layout.Layout, live: state_module.TournamentState) -> None:
    """Repaint every region from the current live state."""
    root["header"].update(_build_header(live))
    root["standings"].update(_build_standings(live))
    root["events"].update(_build_events(live))
    root["footer"].update(_build_progress(live))


###### PRIVATE #######


def _build_header(live: state_module.TournamentState) -> panel.Panel:
    grid = table.Table.grid(expand=True)
    grid.add_column(justify="left")
    grid.add_column(justify="right")
    stats = text.Text(no_wrap=True, end="")
    stats.append("● ", style=_PHASE_COLOR.get(live.phase, theme.TEXT_MUTED))
    stats.append(live.phase.value, style=theme.TEXT_DIM2)
    stats.append("   T+", style=theme.TEXT_MUTED)
    stats.append(_clock(live.elapsed()), style=theme.TEXT_DIM2)
    stats.append(
        f"   {live.games_done}/{live.total_games} games", style=theme.TEXT_DIM2
    )
    stats.append(f"   {live.throughput():.1f} g/s", style=theme.TEXT_MUTED)
    grid.add_row(theme.gradient_text("WINGSPAN // TOURNAMENT"), stats)
    return panel.Panel(
        grid, border_style=theme.BORDER_DEFAULT, box=box.ROUNDED, padding=(0, 1)
    )


def _build_standings(live: state_module.TournamentState) -> panel.Panel:
    grid = table.Table(box=box.SIMPLE_HEAD, expand=True, pad_edge=False)
    grid.add_column("#", justify="right", style=theme.TEXT_MUTED, width=2)
    grid.add_column("competitor", justify="left", style=theme.TEXT_PRIMARY)
    grid.add_column("elo", justify="right", style=theme.TEXT_BRIGHT)
    grid.add_column(
        "trend", justify="left", style=theme.SPARK_COLOR, width=_SPARK_WIDTH
    )
    grid.add_column("W·L·T", justify="right", style=theme.TEXT_DIM2)
    grid.add_column("win%", justify="right")
    grid.add_column("margin", justify="right", style=theme.TEXT_DIM2)
    for rank, row in enumerate(live.standings(), start=1):
        grid.add_row(
            str(rank),
            row.display_name,
            f"{row.elo:.0f}",
            text_helpers.sparkline(row.elo_spark, _SPARK_WIDTH),
            f"{row.wins}·{row.losses}·{row.ties}",
            _winrate_text(row.win_rate),
            _margin_text(row.avg_margin),
        )
    return panel.Panel(
        grid,
        title="STANDINGS",
        border_style=theme.BORDER_DEFAULT,
        box=box.ROUNDED,
        padding=(0, 1),
    )


def _build_events(live: state_module.TournamentState) -> panel.Panel:
    body = text.Text(no_wrap=True, end="")
    for index, event in enumerate(live.events[-_EVENTS_SHOWN:]):
        if index:
            body.append("\n")
        body.append(f"{event.clock} ", style=theme.TEXT_MUTED)
        body.append(
            f"{theme.EVENT_GLYPH.get(event.kind, '·')} ",
            style=theme.EVENT_COLOR.get(event.kind, theme.TEXT_PRIMARY),
        )
        body.append(event.text, style=theme.TEXT_PRIMARY)
    return panel.Panel(
        body,
        title="RECENT",
        border_style=theme.BORDER_DEFAULT,
        box=box.ROUNDED,
        padding=(0, 1),
    )


def _build_progress(live: state_module.TournamentState) -> panel.Panel:
    filled = text_helpers.eighth_bar(live.progress(), _PROGRESS_WIDTH, min_tick=True)
    bar = text.Text(no_wrap=True, end="")
    bar.append(filled, style=theme.GOOD)
    bar.append("░" * max(0, _PROGRESS_WIDTH - len(filled)), style=theme.GAUGE_TRACK)
    bar.append(
        f"  {live.games_done}/{live.total_games}  ({live.progress() * 100:.0f}%)",
        style=theme.TEXT_DIM2,
    )
    return panel.Panel(
        bar,
        title="PROGRESS",
        border_style=theme.BORDER_DEFAULT,
        box=box.ROUNDED,
        padding=(0, 1),
    )


def _winrate_text(win_rate: float) -> text.Text:
    """A win-rate percentage colored by the shared eval hero-number ramp."""
    pct = win_rate * 100.0
    return text.Text(f"{pct:.1f}%", style=theme.hero_color(pct))


def _margin_text(avg_margin: float) -> text.Text:
    """A signed average margin, green when positive and clay when negative."""
    if avg_margin > 0:
        color = theme.GOOD
    elif avg_margin < 0:
        color = theme.BAD
    else:
        color = theme.TEXT_MUTED
    return text.Text(f"{avg_margin:+.1f}", style=color)


def _clock(seconds: float) -> str:
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}"
