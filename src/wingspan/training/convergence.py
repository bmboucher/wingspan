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


def setup_transition_iterations(
    history: list[metrics.IterationMetrics],
) -> list[int]:
    """Iterations where the setup model first entered RANDOM_RECORD (recording
    turned on) or MODEL_DRIVEN (training started). Returns at most two values
    in iteration order; returns an empty list when the setup model was never
    enabled."""
    transitions: list[int] = []
    seen: set[str] = set()
    for item in history:
        phase = item.setup_phase
        if phase in ("RANDOM_RECORD", "MODEL_DRIVEN") and phase not in seen:
            transitions.append(item.iteration)
            seen.add(phase)
    return transitions


def winrate_ewma_points(
    history: list[metrics.IterationMetrics], alpha: float
) -> list[tuple[int, float]]:
    """The EWMA-smoothed win-rate (percent) per win-rate-bearing iteration: the
    eval win-rate where an eval ran, else the random-opponent bootstrap phase's
    collection win-rate vs random. The EWMA resets to the raw value whenever the
    regime changes — each reference opponent advance, and the bootstrap →
    self-play graduation — so the curve starts a fresh climb after each sawtooth
    rather than carrying the old regime's saturated rate forward."""
    points: list[tuple[int, float]] = []
    ewma: float | None = None
    regime: tuple[str, int] | None = None
    for item in history:
        win_pct = _winrate_pct(item)
        if win_pct is None:
            continue
        item_regime = _regime(item)
        if ewma is None or item_regime != regime:
            ewma = win_pct
            regime = item_regime
        else:
            ewma = alpha * win_pct + (1.0 - alpha) * ewma
        points.append((item.iteration, ewma))
    return points


def _winrate_pct(item: metrics.IterationMetrics) -> float | None:
    """The win-rate (percent) one iteration contributes: the eval win-rate, else
    the bootstrap collection win-rate vs random, else None (a self-play
    iteration with no eval)."""
    if item.eval is not None:
        return item.eval.win_rate * 100.0
    if item.collection_win_rate is not None:
        return item.collection_win_rate * 100.0
    return None


def _regime(item: metrics.IterationMetrics) -> tuple[str, int]:
    """A key the win-rate / margin EWMA resets on: eval points are grouped by
    reference opponent generation, and the bootstrap phase's collection points
    form their own ``("collect", 0)`` regime so the EWMA restarts at
    graduation."""
    if item.eval is not None:
        return ("eval", item.eval.opponent_generation)
    return ("collect", 0)


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
    """The EWMA-smoothed margin per win-rate-bearing iteration: the eval margin
    where an eval ran, else (bootstrap phase) the net's average margin over the
    random opponent (``avg_margin`` is seat 0 minus seat 1). Reset at each regime
    change like the win-rate EWMA — the margin belongs to the opponent it was
    measured against."""
    points: list[tuple[int, float]] = []
    ewma: float | None = None
    regime: tuple[str, int] | None = None
    for item in history:
        value = _margin_value(item)
        if value is None:
            continue
        item_regime = _regime(item)
        if ewma is None or item_regime != regime:
            ewma = value
            regime = item_regime
        else:
            ewma = alpha * value + (1.0 - alpha) * ewma
        points.append((item.iteration, ewma))
    return points


def _margin_value(item: metrics.IterationMetrics) -> float | None:
    """The margin one iteration contributes: the eval margin, else the bootstrap
    phase's net-vs-random ``avg_margin``, else None."""
    if item.eval is not None:
        return item.eval.mean_margin
    if item.collection_win_rate is not None:
        return item.avg_margin
    return None
