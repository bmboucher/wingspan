"""Pure series + axis-window math for the TRAINING IMPROVEMENT convergence charts.

These helpers are split out of :mod:`wingspan.training.charts` (which owns the
rich rendering) so the windowing and EWMA logic stays free of any drawing
dependency and can be unit-tested directly. Everything here is a pure function
over a list of :class:`wingspan.training.metrics.IterationMetrics`.

* WIN RATE spans the whole run (:func:`full_range`) and draws a single EWMA
  series (:func:`winrate_ewma_points`) that resets at each opponent advance.
* FINAL SCORE / MARGIN shows a sliding window pinned to a round left edge
  (:func:`score_margin_window`) with one EWMA series per axis
  (:func:`score_ewma_points`, :func:`margin_ewma_points`).
* Challenger upgrades become vertical markers (:func:`marker_columns`).
"""

from __future__ import annotations

from wingspan.training import metrics

# The FINAL SCORE / MARGIN chart shows the most recent ``SCORE_MARGIN_WINDOW``
# iterations, with the left edge floored to a multiple of ``WINDOW_PIN`` so the
# x-axis steps in round jumps (leaving a gap on the right) rather than scrolling
# every iteration.
SCORE_MARGIN_WINDOW = 2000
WINDOW_PIN = 100


def full_range(history: list[metrics.IterationMetrics]) -> tuple[int, int]:
    """The ``(it_lo, it_hi)`` range spanning the *whole* history — the WIN RATE
    chart's x-axis, which shows the entire run rather than a sliding window."""
    if not history:
        return (0, 1)
    it_lo = history[0].iteration
    it_hi = history[-1].iteration
    return (it_lo, it_hi if it_hi > it_lo else it_lo + 1)


def score_margin_window(history: list[metrics.IterationMetrics]) -> tuple[int, int]:
    """The FINAL SCORE / MARGIN chart's ``(it_lo, it_hi)`` window: the most recent
    ``SCORE_MARGIN_WINDOW`` iterations, with ``it_lo`` floored to a multiple of
    ``WINDOW_PIN`` and a *fixed* right edge ``it_lo + SCORE_MARGIN_WINDOW`` so the
    axis steps in round jumps and leaves a gap on the right rather than scrolling
    every iteration."""
    if not history:
        return (0, SCORE_MARGIN_WINDOW)
    it_hi_data = history[-1].iteration
    raw_lo = it_hi_data - SCORE_MARGIN_WINDOW + 1
    it_lo = max(0, (raw_lo // WINDOW_PIN) * WINDOW_PIN)
    return (it_lo, it_lo + SCORE_MARGIN_WINDOW)


def marker_columns(
    change_iterations: list[int], it_lo: int, it_hi: int, cols: int
) -> set[int]:
    """The canvas columns that fall on a challenger-upgrade iteration within the
    displayed ``[it_lo, it_hi]`` range, for the WIN RATE chart's vertical
    markers. Out-of-range upgrades are dropped."""
    span = it_hi - it_lo
    columns: set[int] = set()
    for iteration in change_iterations:
        if iteration < it_lo or iteration > it_hi:
            continue
        frac = (iteration - it_lo) / span if span > 0 else 1.0
        columns.add(round(frac * (cols - 1)))
    return columns


def winrate_ewma_points(
    history: list[metrics.IterationMetrics], alpha: float
) -> list[tuple[int, float]]:
    """The EWMA-smoothed win-rate (percent) per eval iteration. The EWMA resets
    to the raw value whenever the reference opponent advances, so the curve
    starts a fresh climb after each sawtooth rather than carrying the old
    opponent's saturated rate forward."""
    points: list[tuple[int, float]] = []
    ewma: float | None = None
    generation: int | None = None
    for item in history:
        if item.eval is None:
            continue
        win_pct = item.eval.win_rate * 100.0
        if ewma is None or item.eval.opponent_generation != generation:
            ewma = win_pct
            generation = item.eval.opponent_generation
        else:
            ewma = alpha * win_pct + (1.0 - alpha) * ewma
        points.append((item.iteration, ewma))
    return points


def score_ewma_points(
    history: list[metrics.IterationMetrics], alpha: float
) -> list[tuple[int, float]]:
    """The EWMA-smoothed average self-play final score, one point per iteration
    (the metric is generation-independent, so the EWMA never resets)."""
    points: list[tuple[int, float]] = []
    ewma: float | None = None
    for item in history:
        value = item.avg_self_score
        ewma = value if ewma is None else alpha * value + (1.0 - alpha) * ewma
        points.append((item.iteration, ewma))
    return points


def margin_ewma_points(
    history: list[metrics.IterationMetrics], alpha: float
) -> list[tuple[int, float]]:
    """The EWMA-smoothed eval margin per eval iteration, reset at each opponent
    advance like the win-rate EWMA (the margin is measured vs the reference
    opponent, so it belongs to that generation)."""
    points: list[tuple[int, float]] = []
    ewma: float | None = None
    generation: int | None = None
    for item in history:
        if item.eval is None:
            continue
        value = item.eval.mean_margin
        if ewma is None or item.eval.opponent_generation != generation:
            ewma = value
            generation = item.eval.opponent_generation
        else:
            ewma = alpha * value + (1.0 - alpha) * ewma
        points.append((item.iteration, ewma))
    return points
