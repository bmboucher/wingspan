# reporting — Model introspection + HTML reports

Standalone HTML model-summary report generation and the `wingspan inspect` CLI.
This package depends on `encode.stripes` and `training.runmeta` for the
descriptor seam, but not on PyTorch — reports can be generated without loading
the weights.

## Modules

**`__init__.py`** — re-exports `generate_html_report`, `main_inspect`.

**`card_view.py`** — shared presentation assets consumed by both `html.py` (Birds
tab) and `game_log_html.py` (play HTML). Holds four canonical string constants:
`CARD_CSS` (`.card-cell` + habitat/egg/power-color rules), `CARD_JS`
(`cardCellHtml` + emoji helpers), `STRIPE_VIEWER_CSS` (`#enc-modal`/`.enc-*` rules),
`STRIPE_VIEWER_JS` (`renderStripes`/`renderSubField`). Also contains
`bird_cell_info(bird: cards.Bird) -> game_log_html.BirdCellInfo` — the static
(no played-state) bird→cell converter used by both the game-log capture and the
Birds tab payload builder. Lazy imports of `game_log_html` inside `bird_cell_info`
(and a `TYPE_CHECKING` guard for the annotation) break the potential circular import.

**`html.py`** — `generate_html_report(descriptor: ModelConfig, out_path: Path)`:
produces a self-contained HTML file with a full model summary including:
architecture diagram (via `svg.py`), vector layout table (state + choice stripes
from `encode.stripes`), parameter count breakdown, training config table, and a
**Birds tab** (Model ↔ Birds toggle in the header; responsive grid of all 180
core cards; click any card to open an `#enc-modal` showing that bird's non-identity
attribute encoding stripes with named, decoded values).  Also
`build_model_summary_html(descriptor, report) -> str` — the pure string
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
pre-populated (`pending=True`).

All log-item content now sources from the structured event tree (see
`gamelog/INDEX.md`) rather than text-parsing engine.log.

`tree_to_log_items(phase: PhaseNode) -> list[LogItem]` converts one tree phase's
events into the `LogItem` list consumed by the HTML viewer: `MainActionEvent` →
a `"decision"` item; `PlayBirdEvent` → a `"group"` headed by its bird-selection
decision, with egg/food sub-events as children and any `WhitePowerEvent` as a
trailing `"note"`; `ActivateBaseEvent` / `ActivateBrownEvent` / `ReactionEvent` /
`RoundGoalEvent` / `FinalScoringEvent` → their sub-events/notes in order.

`_apply_setup_highlights(phase, setup_event)` reads `SetupEvent.kept_card_names`
and `kept_bonus_name` from the recorder's tree to set `selected` on the kept hand
cards and bonus card in a setup phase.

`build_report(*, engine, phases, tree, seed, matchup, timeline)` merges the
handler's phase snapshots with the tree: for setup phases it reads the
`SetupEvent` to apply highlights; for all phases it calls `tree_to_log_items`
to populate `log_items`.

`extract_timeline_points(tree) -> list[RawTimelinePoint]` DFS-walks all
`DecisionSubEvent`s in recording order to collect the timeline scalars
(`value`, `turn_counter`, `setup_slot`, `family_idx`, `score_p0`, `score_p1`,
`margin_before`).

`build_timeline(engine, raw_points, seat_configs)` finalizes provisional
per-decision timestamps and computes P0-relative future-return chart coordinates
for value/target lines, reusing `timestamps.discounted_future_returns`.

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
to avoid the `engine` ↔ `reporting` import cycle. `extract_card_attr_stripes(bird) ->
list[game_log_html.EncodedStripe]` — decodes the non-identity attribute sub-fields from
`state_encode.card_feature_matrix()` for a single bird, producing named decoded labels
(habitats, food_cost, nest, color, bonus_categories, power_exchange, scalar fields); the
`bird_identity` one-hot stripe is intentionally excluded.

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
Four visual rows (single-card encoder / consumers / state-choice-setup / heads) with `_SVG_BAND_H`-px
connector bands between them. The **SINGLE-CARD ENCODER** is centered on the 960-wide canvas
(`_ENC_X`/`_ENC_CX`); the always-present **consumers** row holds the hand block — the bare
**MULTI-CARD POOLING** block (no I/O boxes, no "0 params" legend, drawn right-of-center at `_POOL_X`
via `_draw_bare_unit` when the net pools the card table) or the full **MULTI-CARD ENCODER** (when
`use_distinct_hand_model`) — plus **BOARD ATTENTION** in col 0 when `arch.use_board_attention` is
True. The three card→{state,choice,setup} feeds share a **trunk** (`_trunk_svg`): a vertical stem at
`_TRUNK_X` (gutter between the two consumer blocks), splitting in the cons band into labelled
branches — State gets ×N_CARD_INDEX_SLOTS when attention off or ×TRAY_SIZE tray when on; Choice gets
×1 candidate; Setup gets ×TRAY_SIZE, dashed when inactive. The card→MULTI-CARD POOLING feed is a
thick solid line with no label. When attention is on, `_board_path_conns` emits a wide band-1 elbow
(CARD→ATTENTION) and a col-0 vertical (ATTENTION→STATE); when off it returns [] and the trunk carries
all card slots to State. The pooled hand feeds STATE (`×N · hand + playable`) and SETUP
(`×2 · kept + turn-1 playable`). The STATE / CHOICE / SETUP blocks and VALUE / DECISION heads
occupy the bottom two rows. `_resolve_geometry` stacks all four rows + three bands into `_Geom`;
trunk bodies/labels are emitted alongside the `_conn_svg` renders in bodies-before-labels order so
white halos mask crossing lines. Each block's activation rows use the resolved per-block activation.
The diagram doubles as the report's navigation: input boxes carry `data-panel` attributes and
parameter counts carry `data-params-block` attributes — the attention block uses `panel=None` and the
bare pooling block has no input box, so the card-table-pooling default exposes four clickable panels
(the `hand` panel is reachable only via the distinct multi-card encoder's input box).

**`inspect_cli.py`** — `main_inspect(args)`: the `wingspan inspect` CLI handler.
Accepts a run directory or `.pt` path; loads the descriptor via
`training.runmeta.read_model_config` (era-routed, so compat artifacts work);
prints vector layout, architecture topology, and parameter counts. Writes
`model_inspect.json` and `model_summary.html` as side effects.
