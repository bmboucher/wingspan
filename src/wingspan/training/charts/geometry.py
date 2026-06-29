"""Layout geometry for the convergence charts and eval inset.

The fixed sizes — gutter width, title/axis row counts, the inset width and the
narrow-panel cutoff — live here so the chart and inset renderers read as
composition over named geometry rather than bare magic numbers. The
``WINDOW_*`` group additionally owns the FINAL SCORE / MARGIN chart's x-axis
window extents (the sliding-window width and its two pin steps), so the
convergence math references named constants instead of literal iteration counts.
The ``WINRATE_`` group owns the WIN RATE chart's y-axis floor parameters.

Constants are kept in global alphabetical order, which — because every name
carries a concern prefix (``CHART_`` / ``INSET_`` / ``TICK_`` / ``WINDOW_`` /
``WINRATE_``) — also keeps each concern grouped together.
"""

from __future__ import annotations

CHART_AXIS_ROWS = 2  # axis ruler + iteration labels
CHART_GAP = 2  # blank columns between the two side-by-side charts
CHART_GUTTER_W = 5  # "100%" (4) + "┤" tick
CHART_MIN_PLOT_ROWS = 5  # below this the panel is too short to plot — placeholder
CHART_TITLE_ROWS = 1  # the per-chart title row above each plot

INSET_MIN_WIDTH = 96  # below this the inset moves below the side-by-side charts
INSET_W = 28  # docked eval inset width

# Both convergence plots fill the panel's full plot height (and so does the
# docked inset beside them). The win-rate gutter labels are placed at
# TICK_WIN_RATE equidistant row positions; their percentage values are computed
# dynamically from the current y-floor so the ticks never move but their labels
# adapt as the scale changes.  The two charts always share the same row count so
# the side-by-side merge stays vertically aligned.
TICK_POINTS = 5  # gridline count on the auto-scaled points axis
TICK_WIN_RATE = 6  # equidistant y-axis tick count for the win-rate chart

# The FINAL SCORE / MARGIN chart uses a two-phase x-axis window:
#   - Growing phase (< WINDOW_SCORE_MARGIN iterations): left edge pinned at 0,
#     right edge = ceil(latest / WINDOW_EARLY_PIN) * WINDOW_EARLY_PIN, so the
#     axis expands in WINDOW_EARLY_PIN-iteration steps as data arrives.
#   - Sliding phase (>= WINDOW_SCORE_MARGIN iterations): a fixed
#     WINDOW_SCORE_MARGIN-wide window; right edge = ceil(latest /
#     WINDOW_SLIDING_PIN) * WINDOW_SLIDING_PIN, stepping in WINDOW_SLIDING_PIN
#     jumps.  WINDOW_SLIDING_PIN is derived from the window so the newest data
#     point can never round off the left edge: the right-edge snap stays within
#     one pin-step of the latest iteration, which is always < WINDOW_SCORE_MARGIN.
WINDOW_EARLY_PIN = 50  # growing-phase right-edge rounding step
WINDOW_SCORE_MARGIN = 200  # sliding-window width, in iterations
WINDOW_SLIDING_DIVISOR = 8  # WINDOW_SCORE_MARGIN / this = the sliding pin step
WINDOW_SLIDING_PIN = WINDOW_SCORE_MARGIN // WINDOW_SLIDING_DIVISOR

# WIN RATE chart y-axis floor parameters: the floor is the visible-window minimum
# rounded down to WINRATE_FLOOR_STEP_PCT, capped at WINRATE_FLOOR_CAP_PCT so the
# 50% baseline remains on screen even when win rates are high.
WINRATE_FLOOR_CAP_PCT = 50.0  # the win-rate y-axis floor never rises above this
WINRATE_FLOOR_STEP_PCT = (
    5.0  # the floor is the window minimum rounded down to this step
)
