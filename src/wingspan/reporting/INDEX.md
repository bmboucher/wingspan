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
per-seat discounted-return target, each seat shown as a separate dashed line)
with the realized margin as context). The decision log
renders four item kinds from `PhaseRecord.log_items: list[LogItem]`: collapsible
`"decision"` boxes (with option bars scaled to max-probability, `+#.#` scores,
and the selected option highlighted); non-collapsible `"forced"` outcome boxes;
muted `"note"` boxes for notifications; and `"group"` collapsible parents whose
body is their `children: list[LogItem]` rendered recursively (used for the setup
food-group "keeps 🌾…" node). `BirdCellInfo.selected` and `BonusCardInfo.selected`
add a green border highlight to the kept hand cards and kept bonus card in the
setup phase. The data model holds **primitives only** — no engine or torch types
— so the page renders from a plain JSON dump embedded in the document and drawn
client-side by an inline script.

**`game_log_csv.py`** — CSV export for the timeline data embedded in a game log.
`timeline_to_csv(report: GameLogReport) -> str` renders `report.timeline` as a
header-plus-rows CSV: one row per `TimelinePoint`, with the critic and training-target
columns sparse (only the moving seat's pair is filled; the other seat's cells are blank),
exactly mirroring the bottom panel of the timeline chart.  All critic / target values are
**P0-relative future-return margins in VP**.  `timeline_csv_data_uri(report) -> str`
wraps the CSV as a `data:text/csv;charset=utf-8;base64,…` URI for use as a download-link
`href`.  The module depends only on the stdlib (`base64`, `csv`, `io`) and references
`GameLogReport` under `TYPE_CHECKING` to avoid a runtime import cycle.

**`game_log_capture.py`** — the engine-aware half of the game-log feature.
`capture_phase(engine, …) -> PhaseRecord` flattens the live `GameState` into
primitive display models. `capture_setup_phase(engine, …, dealt_bonus) ->
PhaseRecord` creates the combined per-player setup phase with bonus options
pre-populated (`pending=True`). `SetupCaptureState` is a transient per-player
bucket filled by `record_setup_decision(capture, engine, decision, choice,
annotation)` for each setup-context decision. `finalize_setup_phase(phase,
capture)` sets `selected` on kept hand cards and the kept bonus, and assembles
the `[keep_item, food_group, bonus_item]` decision log.
`_merge_secondary_setup_segments(segments)` folds CHOOSING BONUS CARD segments
into the preceding segment so the `zip(phases, segments)` count stays 1:1.
`build_decision_item(engine, decision, choice, annotation) -> LogItem` builds a
structured decision box from a `PolicyAnnotation` (up to 5 options by probability,
always including the chosen one). `build_report(…, decision_items)` merges
pre-built decision items with the engine text log (skips `kind=="setup"` phases,
whose log items are already set by `finalize_setup_phase`; converts forced-single
lines to `"forced"` items; humanizes everything else as `"note"` items) and
assembles the report. `build_timeline(engine, raw_points, seat_configs)` finalizes
provisional per-decision timestamps and computes P0-relative future-return chart
coordinates for value/target lines, reusing `timestamps.discounted_future_returns`.
Imported lazily by the `GameLogHtml` instrumentation handler so its `engine`
dependency stays off the import-time path.

**`encode_viewer.py`** — extracts non-zero stripe summaries from raw encoder vectors for
the HTML encoding-viewer modal. `extract_state_stripes(vector, include_setup)` and
`extract_choice_stripes(choice_vec, include_setup)` decode main-net vectors using the
appropriate `stripes.{state,choice}_stripe_layout(spec)` layout. For setup decisions,
`extract_setup_context_stripes(vector, encoding)` and `extract_setup_candidate_stripes(vector,
encoding)` decode a raw setup candidate vector using `setup_model.setup_stripe_layout(encoding)`,
partitioned at `_SETUP_CONTEXT_STRIPES`: context stripes (tray, birdfeeder) go to the state
panel; per-candidate stripes (kept cards, bonus, pricing) go to the choice panel. All functions
skip all-zero and `encoding=="complex"` stripes. Called lazily from `game_log_capture.build_decision_item`
to avoid the `engine` ↔ `reporting` import cycle.

**`humanize.py`** — leaf humanizer module with no engine or torch imports.
`humanize_choice(choice, gs, player_id)` → concise option label per Choice
subclass. `humanize_outcome(decision, choice, gs)` → third-person summary for
the collapsed decision header. `humanize_note(text)` → strips the `[Name]`
prefix and pattern-rewrites common engine notifications (plays, egg lays, card
draws, food gains, power activations, birdfeeder resets). `humanize_forced(label)`
→ rewrites `display_label()` patterns (deck, tray slot, board target) to
human-friendly text.

**`svg.py`** — SVG architecture-diagram builder.
`build_arch_svg(arch, param_report, family_order, *, setup_param, setup_arch, use_setup_model) -> str`
renders the full network topology as a self-contained SVG string embedded by `html.py`.
Three visual rows (encoders / consumers / heads) with `_SVG_BAND_H`-px connector bands between
them hold the seven permanent blocks (SINGLE-CARD ENCODER, MULTI-CARD ENCODER, STATE ENCODER,
CHOICE ENCODER, SETUP MODEL, VALUE HEAD, DECISION HEAD) and, when `arch.use_board_attention` is
True, an additional **BOARD ATTENTION** row inserted between the encoder row and consumer row —
drawn in col 0, with the board-slot path routing CARD→ATTENTION→STATE as straight verticals.
Each block's activation rows use the resolved per-block activation (e.g. `trunk_activation_resolved`)
rather than the global fallback. The diagram doubles as the report's navigation: input boxes carry
`data-panel` attributes and parameter counts carry `data-params-block` attributes (the attention
block uses `panel=None` — it has no vector-layout panel).

**`inspect_cli.py`** — `main_inspect(args)`: the `wingspan inspect` CLI handler.
Accepts a run directory or `.pt` path; loads the descriptor via
`training.runmeta.read_model_config` (era-routed, so compat artifacts work);
prints vector layout, architecture topology, and parameter counts. Writes
`model_inspect.json` and `model_summary.html` as side effects.
