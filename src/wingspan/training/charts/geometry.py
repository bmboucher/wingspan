"""Layout geometry for the convergence charts and eval inset.

The fixed sizes — gutter width, title/axis row counts, the inset width and the
narrow-panel cutoff — live here so the chart and inset renderers read as
composition over named geometry rather than bare magic numbers.
"""

from __future__ import annotations

GUTTER_W = 5  # "100%" (4) + "┤" tick
TITLE_ROWS = 1  # the per-chart title row above each plot
AXIS_ROWS = 2  # axis ruler + iteration labels
CHART_GAP = 2  # blank columns between the two side-by-side charts
INSET_W = 28  # docked eval inset width
INSET_MIN_WIDTH = 96  # below this the inset moves below the side-by-side charts
MIN_PLOT_ROWS = 5  # below this the panel is too short to plot — show a placeholder
# Both convergence plots fill the panel's full plot height (and so does the
# docked inset beside them). The win-rate 0..100% gutter labels (every 20%) are
# placed proportionally, so they stay readable at any row count rather than being
# pinned to a fixed height. The two charts always share the same row count so the
# side-by-side merge stays vertically aligned.
Y_LABELS = (100, 80, 60, 40, 20, 0)  # win-rate axis percent gridlines
POINTS_AXIS_TICKS = 5  # gridline count on the auto-scaled points axis
