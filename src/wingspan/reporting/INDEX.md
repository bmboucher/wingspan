# reporting — Model introspection + HTML reports

Standalone HTML model-summary report generation and the `wingspan inspect` CLI.
This package depends on `encode.stripes` and `training.runmeta` for the
descriptor seam, but not on PyTorch — reports can be generated without loading
the weights.

## Modules

**`__init__.py`** — re-exports `generate_html_report`, `main_inspect`.

**`html.py`** — `generate_html_report(descriptor: ModelConfig, out_path: Path)`:
produces a self-contained HTML file with a full model summary including:
architecture diagram (via `svg.py`), vector layout table (state + choice stripes
from `encode.stripes`), parameter count breakdown, and training config table.
Also `build_model_summary_html(descriptor, report) -> str` — the pure string
variant consumed by `training.runmeta`'s reporting seam.

**`game_log_html.py`** — the HTML *game*-log viewer (vs `html.py`'s *model*
report). `render_game_log_html(report: GameLogReport) -> str` /
`write_game_log_html(report, out_path)` produce a self-contained, asset-free
page that replays one `wingspan play` game phase-by-phase: a sticky state panel
(3x5 board grids, hands, tray, food, scores, bonus cards, round goals), prev/next
arrows, a `P0 / P1 / both` seat toggle, a collapsible decision log, and a
**Timeline modal** (button opens two stacked SVG panels: top = per-player VP over
game-clock time, bottom = P0-relative future return (per-seat critic prediction vs
discounted-return target) with the realized margin as context). The data model (`GameLogReport`,
`PhaseRecord`, `TimelinePoint`, `PlayerPanel`, `BirdCellInfo`, …) holds
**primitives only** — no engine or torch types — so the page renders from a plain
JSON dump embedded in the document and drawn client-side by an inline script.
`GameLogReport.timeline: list[TimelinePoint] = []` defaults to empty, preserving
backward compatibility for existing callers.

**`game_log_capture.py`** — the engine-aware half of the game-log feature.
`capture_phase(engine, …) -> PhaseRecord` flattens the live `GameState` (both
seats' boards/hands/food/scores, the shared tray/feeder/goals) into the
primitive display models; `build_report(…)` splits the engine's interleaved text
log into one decision-narration block per phase (on `=== ... ===` headers,
dropping each turn's verbose state-summary prefix) and assembles the
`GameLogReport`. `build_timeline(engine, raw_points, seat_configs)` finalizes
provisional per-decision timestamps (via `timestamps.finalize_provisional_timestamps`)
and computes P0-relative future-return chart coordinates for value/target lines,
reusing the `timestamps.discounted_future_returns` kernel. Imported lazily by the
`GameLogHtml` instrumentation handler so its `engine` dependency stays off the
import-time path.

**`svg.py`** — SVG architecture-diagram builder. `build_arch_svg(arch:
ModelArchitecture) -> str` renders the trunk / choice-encoder / scorer-head /
value-head topology as an SVG string, embedded by `html.py`. Widths are drawn
proportional to the hidden-layer sizes.

**`inspect_cli.py`** — `main_inspect(args)`: the `wingspan inspect` CLI handler.
Accepts a run directory or `.pt` path; loads the descriptor via
`training.runmeta.read_model_config` (era-routed, so compat artifacts work);
prints vector layout, architecture topology, and parameter counts. Writes
`model_inspect.json` and `model_summary.html` as side effects.
