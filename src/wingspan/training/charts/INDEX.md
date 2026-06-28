# training.charts — Custom rich renderables

Braille-canvas plots and histogram widgets used by the training dashboard.
All classes implement `rich`'s `__rich_console__` protocol so they can be
placed directly in a `rich.Layout` panel.

## Modules

**`__init__.py`**

**`geometry.py`** — Layout constants shared across chart modules, kept in
global alphabetical order with a concern prefix per name so each concern stays
grouped: `CHART_*` (gutter width, title/axis row counts, the min plottable
height, the inter-chart gap), `INSET_*` (docked-inset width + narrow-panel
cutoff), `TICK_*` (per-axis tick counts), and `WINDOW_*` (the FINAL SCORE /
MARGIN x-axis sliding-window width and its two pin steps, with
`WINDOW_SLIDING_PIN` derived from the window). Centralised so dashboard layout
and convergence-window math don't scatter magic numbers.

**`braille.py`** — `BrailleCanvas(cols, rows, n_series)`: a multi-series 2×4-dot
Unicode braille bitmap canvas; each series has its own bit-plane. Core API:
`set_dot(px, py, series)` lights one dot, `line(x0, y0, x1, y1, series,
dotted)` draws a Bresenham segment, `cell(row, col) -> (char, owner_series)`
reads one cell for rendering. Used by `convergence_chart.py` and `insets.py`.

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

**`insets.py`** — Docked panel builder functions for the right-side inset column:
- `eval_inset(state, height) -> list[text.Text]` — cinematic hero win-rate
  block: hero number, recent and EWMA eval rows, sample size, challenger, and
  iterations since last upgrade; padded to `height`.
- `eval_strip(state) -> list[text.Text]` — one-line narrow-panel eval row.
- `collect_inset(state, height) -> list[text.Text]` — collect-phase inset.
- `collect_strip(state) -> list[text.Text]` — one-line narrow-panel collect row.
