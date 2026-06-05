"""Small text-bar helpers reused across the dashboard panels: an 8-step block
sparkline, a horizontal eighth-block bar, and a human-readable count
formatter, plus the chart history-window hint constant."""

from __future__ import annotations

import typing

from rich import text

from wingspan.training import theme

# Retained for the configurator's history-length hint in
# ``wingspan.training.configure.screen`` — the charts themselves now read the
# full on-disk history (see ``convergence`` for the live window sizes).
CHART_WINDOW = 500


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


def join_columns(
    blocks: list[tuple[list[text.Text], int]], gap: int
) -> list[text.Text]:
    """Concatenate line blocks side by side, each padded to its own width with
    ``gap`` blank columns between them. Blocks of unequal height are bottom-padded
    to the tallest so every output row spans every column. Shared by the
    convergence charts and the configurator's architecture diagram."""
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
