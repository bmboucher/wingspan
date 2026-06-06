# training.charts — Custom rich renderables

Braille-canvas plots and histogram widgets used by the training dashboard.
All classes implement `rich`'s `__rich_console__` protocol so they can be
placed directly in a `rich.Layout` panel.

## Modules

**`__init__.py`**

**`geometry.py`** — Layout constants shared across chart modules:
`GUTTER`, `INSET_WIDTH`, `MIN_CHART_HEIGHT`, `BAND_*` heights. Centralised
so dashboard layout math doesn't scatter magic numbers.

**`braille.py`** — `BrailleCanvas(width, height)`: a 2×4-dot Unicode braille
bitmap canvas. `plot_line(xs, ys)` maps a float series onto the dot grid;
`plot_bar(xs, ys)` draws filled columns. `render() -> str` returns the canvas
as a block of braille Unicode characters. The foundation for
`convergence_chart.py` and `insets.py`.

**`text_helpers.py`** — Pure string helpers for dashboard text elements:
- `sparkline(values, width) -> str` — Unicode eighth-block bar sparkline.
- `eighth_bar(fraction, width) -> str` — a single fractional bar.
- `human_count(n) -> str` — compact human-readable number ("1.2k", "3.4M").

**`convergence_chart.py`** — `GettingBetterChart(metrics_log, config)`:
the main training-progress renderable. Plots win rate and games-per-second
over time on a braille canvas; overlays the convergence window slope indicator.
`_draw_win_rate_series`, `_draw_gps_series`, `_draw_axis_labels` are private
helpers that write to a shared `BrailleCanvas`.

**`histogram.py`** — `FamilyHistogram(family_counts: FamilyCounts)`:
a `rich` renderable showing the per-decision-family action distribution as
horizontal eighth-block bars. Used in the FLIGHT PLAN band to monitor policy
entropy per family.

**`insets.py`** — Docked panel renderables for the right-side inset column:
- `EvalInset(eval_result)` — compact win-rate + CI display.
- `NarrowPanelStrip(panels)` — stacks multiple narrow renderables vertically
  for the inset column layout.
