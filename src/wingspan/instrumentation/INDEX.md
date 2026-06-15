# instrumentation — Event-callback instrumentation

General-purpose event router that an `Engine` holds. Configure a set of handlers
via a serializable `InstrumentationConfig`, attach it to an engine, and the
dispatcher fires typed callbacks on game events. Used for training data collection
(decision logging) and analytics (per-bird play tallies).

## Modules

**`__init__.py`**

**`config.py`** — `InstrumentationConfig` and `RunContext`:
- `InstrumentationConfig(handlers: list[HandlerConfig])` — serializable list of
  handler configs; passed at engine construction time. Each entry names a handler
  class and its parameters.
- `RunContext(run_name, checkpoint_dir, iteration)` — per-run metadata threaded
  through to handlers so they can name their output files correctly.
- `HandlerConfig` — abstract base; each handler module defines its own subclass.

**`dispatcher.py`** — `EventDispatcher`: the live event router held by `Engine`.
Built from an `InstrumentationConfig` via `EventDispatcher.from_config(config,
context)`. Key method: `dispatch(event: GameEvent)` — calls all registered
handlers whose `handles(event)` returns `True`. Thread-safe; handlers are called
synchronously in the engine thread.

**`events.py`** — Event taxonomy and handler base classes:
- `GameEvent` — abstract base with `player_id` and `round_idx`.
- Concrete events: `BirdPlayedEvent`, `DecisionMadeEvent`, `RoundEndEvent`,
  `GameEndEvent`, etc. Each carries the typed data relevant to that moment.
- `GameEventHandler` — abstract base; subclasses override `handles(event) -> bool`
  and `handle(event)`.

**`registry.py`** — Config-class-name ↔ handler bijection:
`@register_handler(config_class)` decorator associates a `HandlerConfig`
subclass with its `GameEventHandler` implementation.
`handler_for_config(config) -> GameEventHandler` — lookup used by the dispatcher.

## handlers/ subpackage

Built-in handler implementations.

**`handlers/__init__.py`**

**`handlers/card_visits.py`** — `CardVisitsHandler` / `CardVisitsConfig`: counts
how many times each bird is played per game. Writes a `card_visits.json` summary
at game end. Useful for coverage reports and card-popularity analytics.

**`handlers/decision_logger.py`** — `DecisionLoggerHandler` / `DecisionLoggerConfig`:
appends a JSONL row for every `DecisionMadeEvent`. Each row contains the encoded
state vector, encoded choice matrix, and the chosen index — the primary source
of training data for offline supervised learning.

**`handlers/game_log_html.py`** — `GameLogHtml` (`GameLogHtmlHandler`): records
each game as a navigable, self-contained HTML log viewer (the `wingspan play
--html` flag). Subscribes to `game_start` / `setup_start` / `round_start` /
`turn_start` / `made_decision` / `game_end`. Phase-boundary events fire once per
`=== ... ===` log header so snapshots zip one-to-one with the (merged) text log.
One combined setup phase per player is created at `setup_start`; setup decisions
are routed into per-player `SetupCaptureState` buckets rather than the flat
`_decision_items` list. `finalize_setup_phase` is called at `game_end` to
assemble highlighted hand/bonus and the nested food-group decision log. The
capture layer's `_merge_secondary_setup_segments` folds the secondary
CHOOSING BONUS CARD header into the primary segment so the zip stays aligned.
`made_decision` records a `RawTimelinePoint` for the Timeline chart and, for
non-setup phases with a `PolicyAnnotation`, also builds a `LogItem` (option bars
+ selected highlight) stored in `_decision_items`. At `game_end` calls
`build_timeline` then `build_report`. Config: `output_path` and `index_suffix`.
Call `configure_timeline(seat_configs, probes)` before the first game to inject
per-seat `TrainConfig` and `DecisionProbe` objects; without them the timeline
shows scores only and decision boxes omit option bars. State→model conversion
lives in `reporting.game_log_capture`, imported lazily.
