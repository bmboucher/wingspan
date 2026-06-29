# GAMELOG — Structured game-event tree

The `gamelog/` package is the single source of truth for the detailed game log.
Both the HTML decision-log viewer and the `--log` plaintext file are pure
renderers over the event tree it produces; the raw `engine.log` stream is a
separate independent debug dump (`--debug-log`) and is not parsed by either
renderer.

## Event type taxonomy

Six logical event categories, each with its own Pydantic subclass:

| # | Event class | Typed fields | When emitted |
|---|-------------|--------------|--------------|
| 1 | `PlayBirdEvent` | — | `actions.do_play_bird_action` / `consume_extra_plays` |
| 2 | `ActivateBaseEvent` | `habitat`, `action` | `actions.do_gain_food` / `do_lay_eggs` / `do_draw_cards` |
| 3 | `ActivateBrownEvent` | `bird_name`, `is_brown` | `actions.activate_row_powers` (one per crossed bird) |
| 4 | `MainActionEvent` | — | `engine.core._take_turn` |
| 5 | `SetupEvent` | `kept_card_names`, `kept_bonus_name` | `engine.core._resolve_setup_choice` |
| 6 | `RoundGoalEvent` | `round_idx`, `description`, `counts`, `vps` | `engine.scoring.score_round_goal` |
| 6 | `FinalScoringEvent` | `scores: list[FinalScoreBreakdown]` | `EventRecorder.end_game` |

Additional event types used for nesting:

| Event class | Typed fields | Role |
|-------------|--------------|------|
| `WhitePowerEvent` | `bird_name` | White "when played" power, nested under `PlayBirdEvent` |
| `ReactionEvent` | `bird_name` | Pink reactor firing, attributed to the reacting player |
| `LooseEvent` | — | Auto-wrap bucket for a `record_*` call outside any open bracket |

## Sub-event taxonomy

Every `GameEvent` holds two lists:

- **`sub_events: list[SubEvent]`** — leaf nodes in this event's own scope.
- **`children: list[GameEvent]`** — nested events (e.g. `WhitePowerEvent` under
  `PlayBirdEvent`, `ReactionEvent` nested under a predator event).

Three sub-event shapes:

| Class | Field | Rendered as |
|-------|-------|-------------|
| `DecisionSubEvent` | `outcome_text`, `options`, `state_stripes`, `value`, `turn_counter`, `setup_slot`, `family_idx`, `score_p0`, `score_p1`, `margin_before` | `→ text` (plaintext) / collapsible decision box with option bars (HTML) |
| `ForcedSubEvent` | `text` | `! text` (plaintext) / non-collapsible "forced" box (HTML) |
| `NoteSubEvent` | `text` | bare text (plaintext) / muted "note" box (HTML) |

`DecisionSubEvent` carries all timeline scalars so the timeline chart derives
from the tree (no parallel data structure).  `turn_counter` + `setup_slot`
together give the provisional timestamp (reconstructed in `reporting.game_log_capture`);
`family_idx` identifies the decision type for interpolation.

## Phase structure

The tree top level is a sequence of `PhaseNode(kind, events)` objects whose
positions are **positionally 1-to-1** with the HTML handler's `PhaseRecord`
list — both are populated in the same firing sequence.

| `kind` | When pushed |
|--------|-------------|
| `"game_start"` | `EventRecorder.begin_game` |
| `"setup"` | `EventRecorder.begin_phase("setup")` (once per player) |
| `"round"` | `EventRecorder.begin_phase("round")` (once per round) |
| `"turn"` | `EventRecorder.begin_phase("turn")` (once per player-turn) |
| `"game_end"` | `EventRecorder.end_game` (final scoring) |

## Open-event stack rule

**Single rule, no special-casing:**

- `begin_*` pushes a new `GameEvent` subclass onto the stack.  If the stack is
  non-empty the new event is appended to the top's `children`; if the stack is
  empty it is appended to the current phase's `events`.
- `end_event()` pops the top of the stack.
- `record_decision` / `record_forced` / `note` append to the stack-top's
  `sub_events`.  If the stack is empty, a `LooseEvent` is auto-created and
  appended to the current phase first, then the sub-event is appended to it.

This single rule handles all nesting correctly:
- Pink reactions while a play-bird event is open → nested under `PlayBirdEvent`.
- Pink reactions fired by gain-food/lay-eggs after the base event closed →
  appended at phase level as separate events.
- White power resolution while a play-bird event is open → `WhitePowerEvent`
  child of `PlayBirdEvent`.

## Call-site map

### `engine/core.py`

| Call | Location |
|------|----------|
| `events.begin_game()` | `play_one_game` / `play_one_game_with_setups` before the game loop |
| `events.begin_phase("game_start")` | same, immediately after `begin_game` |
| `events.begin_phase("setup")` | `_resolve_setup_choice` |
| `events.begin_setup(player.id)` / `events.end_event()` | wraps the setup decision asks + deferred resolves |
| `events.begin_phase("round")` | `_play_round` |
| `events.begin_phase("turn")` | `_take_turn` |
| `events.begin_main_action(player.id)` / `events.end_event()` | wraps `_main_action_decision` ask |
| `events.record_forced(self, decision, choice)` | single-choice branch in `Engine.ask` |
| `events.record_decision(self, decision, choice)` | multi-choice branch in `Engine.ask` |
| `events.end_game(engine)` | after the game loop |

### `engine/actions.py`

| Call | Location |
|------|----------|
| `begin_play_bird` / `end_event` | `do_play_bird_action`, `consume_extra_plays` |
| `begin_white_power` / `end_event` | `do_play_bird` around `dispatch_power(…, "play")` |
| `begin_activate_base` / `end_event` | `do_gain_food`, `do_lay_eggs`, `do_draw_cards` |
| `begin_activate_brown` / `end_event` | `activate_row_powers` per crossed bird |

### `engine/reactors.py`

| Call | Location |
|------|----------|
| `begin_reaction(other_player.id, bird_name)` / `end_event` | per-bird body in each `fire_pink_*` function |

### `engine/scoring.py`

| Call | Location |
|------|----------|
| `events.record_round_goal(engine, round_idx, goal, counts, vps)` | `score_round_goal` |

### `end_game` auto-emission

`EventRecorder.end_game(engine)` reads final scores and emits `FinalScoringEvent`
into the `game_end` phase automatically — no explicit call-site needed in scoring.

## Rendering

### HTML (`reporting/game_log_capture.tree_to_log_items`)

Converts one `PhaseNode` to `list[LogItem]` for the HTML viewer:

- `MainActionEvent` → one `"decision"` item.
- `PlayBirdEvent` → `"group"` headed by the bird-selection decision; sub-events
  (egg, food) as children; `WhitePowerEvent` children as trailing `"note"` items.
- `ActivateBaseEvent` / `ActivateBrownEvent` / `ReactionEvent` → sub-events in order.
- `RoundGoalEvent` / `FinalScoringEvent` → sub-events (or a note if none).

### Plaintext (`gamelog/render_text.render_plaintext`)

Each phase: `=== KIND ===`.  Each event: `[label]` where the label is
type-specific (e.g. `[Activate forest (gain food)]`, `[Brown: Elf Owl]`,
`[——: Barn Owl]`, `[White power: Elf Owl]`,
`[Setup (kept: Barn Owl, Elf Owl; bonus: Rodentologist)]`,
`[Round 1 goal — … [P0: 3/4VP, P1: 1/1VP]]`,
`[Final scoring [42, 37]]`).
Sub-events: `→ text` (decision), `! text` (forced), bare text (note).
Children indented two spaces per level.

## Adding a new event type

1. Add a `GameEvent` subclass to `gamelog/models.py` with typed fields.
2. Add a `begin_<name>` method to `EventRecorder` in `gamelog/recorder.py`.
3. Wire the call-site `begin_<name>` / `end_event` brackets in the appropriate
   engine or action module.
4. Handle the new subclass in `game_log_capture.tree_to_log_items` (HTML) and
   `render_text._event_label` / `_render_event` (plaintext).
5. Add tests in `tests/test_gamelog_tree.py`.
6. Update this file.
