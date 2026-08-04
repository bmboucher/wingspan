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

Additional event types used for nesting and for bracketing otherwise-loose asks:

| Event class | Typed fields | Role |
|-------------|--------------|------|
| `WhitePowerEvent` | `bird_name` | White "when played" power, nested under `PlayBirdEvent` |
| `ReactionEvent` | `bird_name` | Pink reactor firing, attributed to the reacting player |
| `ExtraPlayEvent` | `habitat` | One accrued extra play offered; the accepted `PlayBirdEvent` is a child |
| `TurnEndEvent` | — | End-of-turn discard obligations (`DRAW_CARDS_THEN_DISCARD_EOT`) |
| `LooseEvent` | — | Auto-wrap bucket for a `record_*` call outside any open bracket |

Every event also carries `event_id`: a per-game monotonic integer stamped by the
recorder as the event is opened, giving each node a stable address.

## Serialization contract

`sub_events` and `children` (and `PhaseNode.events`) are typed as the
**discriminated unions** `AnySubEvent` / `AnyGameEvent`, keyed on each
subclass's `kind` literal — *not* as their base classes.

This is load-bearing. With base-class annotations, `model_dump_json()` silently
drops every subclass-declared field and the node type is unrecoverable on load:
a `WhitePowerEvent` round-trips to `{"player_id":0,"sub_events":[{"player_id":null}],"children":[]}`.
No exception is raised in either direction, so only an explicit round-trip
catches the regression — `tests/test_gamelog_serialize.py` is that guard.

Adding a `GameEvent` / `SubEvent` subclass therefore means: declare a unique
`kind` literal, add the class to the union, and add a `model_rebuild()` call at
the bottom of `models.py` (each subclass inherits the forward-referenced
`children` field and needs its own rebuild).

## Sub-event taxonomy

Every `GameEvent` holds two lists:

- **`sub_events: list[AnySubEvent]`** — leaf nodes in this event's own scope.
- **`children: list[AnyGameEvent]`** — nested events (e.g. `WhitePowerEvent` under
  `PlayBirdEvent`, `ReactionEvent` nested under a predator event).

Three sub-event shapes. `ForcedSubEvent` and `DecisionSubEvent` share the
abstract base `ResolvedSubEvent`, which holds everything a resolved decision
point carries regardless of whether the agent was actually consulted:

| Class | Fields | Rendered as |
|-------|--------|-------------|
| `ResolvedSubEvent` *(base)* | `outcome_text`, `turn_counter`, `setup_slot`, `family_idx`, `scores`, `margin_before` | — |
| `DecisionSubEvent` | + `options`, `state_stripes`, `value` | `→ text` (plaintext) / collapsible decision box with option bars (HTML) |
| `ForcedSubEvent` | *(base only)* | `! text` (plaintext) / non-collapsible "forced" box (HTML) |
| `NoteSubEvent` | `text` | bare text (plaintext) / muted "note" box (HTML) |

The base carries all timeline scalars so the timeline chart derives from the
tree (no parallel data structure) and forced moves are joinable to it exactly
like genuine decisions.  `turn_counter` + `setup_slot` together give the
provisional timestamp (reconstructed in `reporting.game_log_capture`);
`family_idx` identifies the decision type for interpolation.  `scores` is the
live per-seat score list in seat order (any table size); `margin_before` is
the deciding seat's own margin — its score minus the *best* other seat's
score, reducing to the legacy own-minus-opponent value at 2 seats.  Game logs
are regenerable reporting artifacts, not covered by the model-rehydration
guarantee (`docs/VERSIONING.md`) — this schema can change freely across
versions.

`NoteSubEvent` covers non-decision notifications that would otherwise be
invisible between phase-boundary snapshots. Its first production use is
birdfeeder re-rolls: every reroll — the optional Rule-2 reset, the automatic
Rule-1 refill, and a committed partial-take under `combine_gain_food` — routes
through `actions._reroll_feeder`, the single seam that pairs the reroll with a
note listing the fresh dice faces (`state.format_die_faces`), and the
`ROLL_NOT_IN_FEEDER_CACHE` dice predators (Anhinga and similar), which roll
outside the feeder and note the faces they land on.

## Phase structure

The tree top level is a sequence of `PhaseNode(kind, round_idx, turn_idx, events)`
objects whose positions are **positionally 1-to-1** with the HTML handler's
`PhaseRecord` list — both are populated in the same firing sequence.

| `kind` | When pushed | Coordinates |
|--------|-------------|-------------|
| `"game_start"` | `EventRecorder.begin_game` | — |
| `"setup"` | `begin_phase("setup")` (once per player) | — |
| `"round"` | `begin_phase("round", round_idx=…)` (once per round) | `round_idx` |
| `"turn"` | `begin_phase("turn", round_idx=…, turn_idx=…)` (once per player-turn) | both |
| `"game_end"` | `EventRecorder.end_game` (final scoring) | — |

`round_idx` / `turn_idx` make each phase self-describing, so consumers of the
tree (and of the flat structured log) do not have to recover the coordinates by
positional `zip` against the reporting layer.

> **Alignment invariant.** Every `begin_phase` call must be paired with the
> instrumentation event that makes the HTML handler capture a `PhaseRecord`, and
> vice versa. Adding one without the other misaligns *every* later phase with
> its snapshot — silently, since the `zip` just pairs the wrong items. Both
> setup entry points (`_resolve_setup_choice` and `_setup_phase_fixed`) fire
> `begin_phase("setup")` and `instrumentation.setup_start` together for this
> reason. Guarded by `test_capture_phases_align_one_to_one_with_tree` and
> `test_capture_phases_align_on_the_fixed_setup_path`.
>
> Appending an event to an *earlier* phase is safe — it changes neither the
> count nor the order of phases. `record_round_goal` relies on this to file a
> `RoundGoalEvent` under the round it scores rather than under the round's last
> turn, which is the phase that happens to be current when scoring runs.

## Open-event stack rule

**Single rule, no special-casing:**

- `begin_*` pushes a new `GameEvent` subclass onto the stack.  If the stack is
  non-empty the new event is appended to the top's `children`; if the stack is
  empty it is appended to the current phase's `events`.
- `end_event()` pops the top of the stack.
- `record_decision` / `record_forced` / `note` append to the stack-top's
  `sub_events`.  If the stack is empty, one `LooseEvent` per phase is created on
  first use and appended to the current phase, then reused for the rest of the
  phase — a run of unbracketed calls collects into one bucket, not one event
  apiece.

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
| `events.begin_phase("setup")` | `_resolve_setup_choice` **and** `_setup_phase_fixed` (each paired with `instrumentation.setup_start`) |
| `events.begin_setup(player.id)` / `events.end_event()` | wraps the setup decision asks + deferred resolves |
| `events.begin_phase("round", round_idx=…)` | `_play_round` |
| `events.begin_phase("turn", round_idx=…, turn_idx=…)` | `_take_turn` |
| `events.begin_main_action(player.id)` / `events.end_event()` | wraps `_main_action_decision` ask |
| `events.begin_turn_end(player.id)` / `events.end_event()` | wraps `_resolve_turn_end_discards` |
| `events.record_forced(self, decision, choice)` | single-choice branch in `Engine.ask` |
| `events.record_decision(self, decision, choice)` | multi-choice branch in `Engine.ask` |
| `events.end_game(engine)` | after the game loop |

### `engine/actions.py`

| Call | Location |
|------|----------|
| `begin_play_bird` / `end_event` | `do_play_bird_action`, `consume_extra_plays` |
| `begin_extra_play` / `end_event` | `consume_extra_plays`, around the accept/decline ask |
| `begin_white_power` / `end_event` | `do_play_bird` around `dispatch_power(…, "play")` |
| `begin_activate_base` / `end_event` | `do_gain_food`, `do_lay_eggs`, `do_draw_cards` |
| `begin_activate_brown` / `end_event` | `activate_row_powers` per crossed bird |
| `events.note(text, player_id)` | `_reroll_feeder` — the single seam every birdfeeder reroll (`offer_birdfeeder_reset`, `gain_feeder_die`, `combined_feeder_gain`) routes through |

### `engine/powers/grants.py`

| Call | Location |
|------|----------|
| `events.note(text, player_id)` | `_h_roll_not_in_feeder_cache`, after the `ROLL_NOT_IN_FEEDER_CACHE` dice predator's roll |

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

1. Add a `GameEvent` subclass to `gamelog/models.py` with typed fields and a
   unique `kind: typing.Literal[...]` discriminator.
2. Add it to the `AnyGameEvent` union and add a `model_rebuild()` call at the
   bottom of `models.py` (see **Serialization contract** above — skipping this
   fails silently, not loudly).
3. Add a `begin_<name>` method to `EventRecorder` in `gamelog/recorder.py`, and
   the matching no-op to `_NullRecorder`.
4. Wire the call-site `begin_<name>` / `end_event` brackets in the appropriate
   engine or action module.  If the bracket spans a phase boundary, keep the
   phase/instrumentation pairing intact (see the alignment invariant above).
5. Handle the new subclass in `game_log_capture.tree_to_log_items` (HTML) and
   `render_text._event_label` / `_render_event` (plaintext).
6. Add tests in `tests/test_gamelog_tree.py`; the round-trip in
   `tests/test_gamelog_serialize.py` covers serialization automatically.
7. Update this file.
