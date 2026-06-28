# instrumentation — Event-callback instrumentation

General-purpose event router that an `Engine` holds. Configure a set of handlers
via a serializable `InstrumentationConfig`, attach it to an engine, and the
router fires typed callbacks on game events. Used for training data collection
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

**`dispatcher.py`** — `Instrumentation`: the live event router held by `Engine`.
One typed `fire` method per `EventName` (e.g. `game_start(engine=...)`,
`made_decision(engine=..., decision=..., choice=...)`); each iterates that
event's handlers. `open(context)` / `close()` fan out across the *unique*
handler set so a multi-event handler's resources are acquired and released once.
`EMPTY` is the shared no-op router for uninstrumented engines (every event costs
one dict lookup that misses). Built from an `InstrumentationConfig` via
`InstrumentationConfig.build`.

**`registry.py`** — Handler registry: `@register(name)` decorator registers a
`CallbackHandler` subclass under a stable config-key string;
`handler_class_for(name)` and `name_for(handler_class)` provide the forward
and reverse lookups for (de)serialization.

**`events.py`** — Event taxonomy and handler base classes:
- `EventName` — `StrEnum` with one member per game event (`GAME_START`,
  `GAME_END`, `ROUND_START`, `ROUND_END`, `TURN_START`, `TURN_END`,
  `MAKING_DECISION`, `MADE_DECISION`, `BIRD_PLACED`, `FOOD_GAINED`,
  `EGGS_LAID`, `CARDS_DRAWN`, `ROUND_GOAL_SCORED`, `PLAYER_FINAL_SCORED`,
  `SETUP_APPLIED`, `SETUP_START`).
- `CallbackHandler` — abstract base; subclasses implement the methods matching
  the events they subscribe to and declare their `EventName` set in `HANDLES`.
- Concrete handler protocol classes: `GameStartHandler`, `GameEndHandler`,
  `RoundStartHandler`, `RoundEndHandler`, `TurnStartHandler`, `TurnEndHandler`,
  `MakingDecisionHandler`, `MadeDecisionHandler`, `BirdPlacedHandler`,
  `FoodGainedHandler`, `EggsLaidHandler`, `CardsDrawnHandler`,
  `RoundGoalScoredHandler`, `PlayerFinalScoredHandler`, `SetupAppliedHandler`,
  `SetupStartHandler`.

## handlers/ subpackage

Built-in handler implementations.

**`handlers/__init__.py`**

**`handlers/card_visits.py`** — `CardVisitsHandler` / `CardVisitsConfig`: counts
how many times each bird is played per game. Writes a `card_visits.json` summary
at game end. Useful for coverage reports and card-popularity analytics.

**`handlers/decision_logger.py`** — `DecisionLoggerHandler` / `DecisionLoggerConfig`:
appends a JSONL row for every `MADE_DECISION` event. Each row contains the encoded
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
