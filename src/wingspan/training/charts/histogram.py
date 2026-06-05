"""The DECISION MODELS panel body (``FamilyHistogram``): one horizontal bar
per decision family, colored by frequency tier as a gentle data-health cue."""

from __future__ import annotations

import rich.console as rich_console
from rich import segment, text

from wingspan import decisions
from wingspan.training import metrics, theme
from wingspan.training.charts import text_helpers


class FamilyHistogram:
    """The "what it's learning to decide" panel: one row per judgment family,
    sorted descending by live count, then a blank spacer and the cumulative
    total-decisions footer. Each bar is scaled to the busiest family — the top
    row fills the panel width — so the whole panel is used, while the trailing
    percentage stays the honest share of *all* decisions, so the wide spread
    between the busiest and rarest family stays legible in both the relative bar
    lengths and the absolute percentages.

    The label column auto-sizes to the longest family name (so a new, longer
    family never overflows it), and the total footer is reserved at the bottom so
    it is never truncated — the family bars are clipped from the bottom first
    when the panel is too short to hold every row."""

    def __init__(
        self, counts: metrics.FamilyCounts, total_decisions: int | None = None
    ):
        self.counts = counts
        self.total_decisions = total_decisions

    def __rich_console__(
        self, console: rich_console.Console, options: rich_console.ConsoleOptions
    ) -> rich_console.RenderResult:
        width = options.max_width
        total = max(self.counts.total(), 1)
        peak = max(self.counts.counts, default=0)

        rows = sorted(self.counts.items(), key=lambda item: item[1], reverse=True)
        # Size the label column to the longest family name (+1 gap) so a name is
        # never truncated, even as new decision families are added.
        label_w = max((len(family.value) for family, _ in rows), default=14) + 1
        # label + space + bar + " 100.0%" (7) + "  " (2) + count(6) + margin
        bar_w = max(6, width - label_w - 17)

        # Reserve the bottom two rows (a blank spacer + the total line) so the
        # total is never pushed off a short panel; clip family bars to fit.
        footer_rows = 2 if self.total_decisions is not None else 0
        height = (
            options.height if options.height is not None else (options.max_height or 0)
        )
        visible = max(1, height - footer_rows) if height else len(rows)

        lines: list[text.Text] = []
        for family, count in rows[:visible]:
            share = count / total
            bar_fraction = count / peak if peak else 0.0
            lines.append(self._row(family, count, share, bar_fraction, label_w, bar_w))

        for i, line in enumerate(lines):
            if i:
                yield segment.Segment.line()
            yield line

        if self.total_decisions is not None:
            # A blank spacer row, then the full (un-shortened) cumulative count
            # sitting just above the panel's bottom border.
            yield segment.Segment.line()
            yield segment.Segment.line()
            yield _total_decisions_line(self.total_decisions)

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
        bar = text_helpers.eighth_bar(bar_fraction, bar_w, min_tick=True)
        line.append(bar.ljust(bar_w), style=color)
        line.append(f" {share * 100:>5.1f}%", style=theme.TEXT_PRIMARY)
        line.append("  ")
        line.append(f"{text_helpers.human_count(count):>6}", style=theme.HIST_COUNT)
        return line


def _total_decisions_line(total_decisions: int) -> text.Text:
    """The full (un-shortened) cumulative decision count footer line."""
    line = text.Text(no_wrap=True, end="")
    line.append(f"{total_decisions:,}", style=theme.TEXT_PRIMARY)
    line.append(" total decisions", style=theme.TEXT_MUTED)
    return line


def _tier_color(share: float, count: int) -> str:
    if count <= 0:
        return theme.TEXT_MUTED
    if share >= theme.HIST_TOP_SHARE:
        return theme.HIST_TOP
    if share < theme.HIST_LOW_SHARE:
        return theme.HIST_LOW
    return theme.HIST_MID
