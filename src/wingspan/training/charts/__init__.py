"""Custom Rich renderables for the FLIGHT PLAN training dashboard.

The convergence chart, the decision-family histogram, and the small bar /
sparkline helpers the rest of the dashboard reuses. Each chart is a real
``__rich_console__`` renderable that rebuilds its canvas to fill its panel every
refresh. The package is split by concern:

- ``geometry``          — the fixed layout sizes (gutter, inset width, ...)
- ``braille``           — the 2x4-dot braille bitmap canvas primitive
- ``text_helpers``      — sparkline / eighth-block bar / human-count formatters
- ``convergence_chart`` — ``GettingBetterChart`` + its block / series drawers
- ``histogram``         — ``FamilyHistogram``
- ``insets``            — the docked eval inset + narrow-panel strip
"""

from wingspan.training.charts.braille import BrailleCanvas
from wingspan.training.charts.convergence_chart import GettingBetterChart
from wingspan.training.charts.histogram import FamilyHistogram
from wingspan.training.charts.text_helpers import (
    CHART_WINDOW,
    eighth_bar,
    human_count,
    sparkline,
)

__all__ = [
    "BrailleCanvas",
    "CHART_WINDOW",
    "FamilyHistogram",
    "GettingBetterChart",
    "eighth_bar",
    "human_count",
    "sparkline",
]
