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
| `DealEvent` | — | One seat's opening deal (hand, bonus cards, one of each food) |
| `RefillTrayEvent` | — | The face-up tray being restocked; belongs to no player action |
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
| `Effect` *(base)* | *(see the effect ledger below)* | `· kind(fields)` (plaintext) |

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

## The effect ledger

Alongside decisions, `sub_events` carries **effects**: one record per state
mutation, interleaved in true chronological order so `draw_card(Wood Stork)`
sits immediately after the decision that caused it.

> **Naming.** `cards.schema.EffectKind` is a different thing — the *static*
> declaration of what a bird's power is supposed to do. `gamelog.models.Effect`
> records what a game *actually did*. Card data parses into the former;
> executing it emits the latter.

| Class | Fields | Notes |
|-------|--------|-------|
| `GainFoodEffect` | `food`, `amount`, `source` | `source` ∈ feeder / supply / cache / opponent / deal |
| `SpendFoodEffect` | `food`, `amount`, `purpose` | |
| `CacheFoodEffect` | `bird`, `food`, `amount`, `from_supply` | `from_supply` pairs with a `SpendFoodEffect` |
| `UncacheFoodEffect` | `bird`, `food`, `amount` | |
| `LayEggEffect` / `RemoveEggEffect` | `bird`, `habitat`, `slot`, `count`, `purpose` | coordinates are a snapshot — a bird can move |
| `DrawCardEffect` | `card`, `source`, `tray_slot` | **reveal** when `source` is `deck` or `deal` |
| `DiscardCardEffect` | `card`, `purpose` | |
| `TuckCardEffect` | `card`, `bird`, `source` | **reveal** when `source` is `deck` |
| `PlayBirdEffect` | `card`, `habitat`, `slot` | |
| `MoveBirdEffect` | `card`, `from_habitat`, `from_slot`, `to_habitat`, `to_slot` | not a second play — the bird keeps eggs/tucks/cache |
| `PassCardEffect` | `card`, `to_player_id` | |
| `FeederRerollEffect` | `faces` | **reveal** |
| `DiceRollEffect` | `bird`, `faces` | **reveal** — `ROLL_NOT_IN_FEEDER_CACHE` predators |
| `TrayRefillEffect` | `slot`, `card` | **reveal** |

Field naming is consistent: **`card`** is a card moving between zones (hand /
deck / tray / board); **`bird`** is a bird already in play being targeted.

### `engine/ledger.py` is the only producer

Effects are never recorded next to a mutation by convention — every mutation is
*performed by* a `ledger` function that records the effect in the same call.
A power that pokes `pb.eggs` directly would leave the log silently
under-reporting, and nothing about the recording site would look wrong.

`tests/test_gamelog_ledger.py` is what makes this safe: it replays the whole
recorded ledger from an empty game and reconciles it against the final
`GameState` — per-seat food, per-bird eggs / tucks / caches, hand contents, and
board rows rebuilt in order — across several seeds at 2 and 4 seats. A mutation
that bypasses the ledger fails there.

Attribution: an effect's `player_id` is the **owner of the changed resource**,
not the acting seat — a pink reactor mutates the reacting opponent's board and
is attributed to that opponent, so per-seat sums add up. `TrayRefillEffect` is
the one genuinely seatless effect (the end-of-round reset belongs to no player).
This is what lets `summarize` scope an event's header to its own seat.

The **reveal** rows above are the effects the HTML log surfaces individually;
everything else folds into an event header.  See **Summaries** below.

Brant (`DRAW_FROM_TRAY_ALL`) used to bulk-take the tray with a direct
`hand.extend` that bypassed the ledger entirely — a real hole in the "only
producer" invariant. It now calls `take_from_tray` once per slot, so it
records one `DrawCardEffect(source=tray)` per card like every other tray draw.

**Derived state, no new event type.** `Player.known_hand` (the publicly-known
subset of a hand — cards every observer saw arrive face-up) is maintained by
the same `engine/ledger.py` functions that already touch a hand, alongside
their existing effect recording. It introduces no `Effect` subclass of its
own — most of it *is* reconstructable from the effect stream above (a
`DrawCardEffect(source=tray)` marks a card known; `DiscardCardEffect` /
`TuckCardEffect(source=hand)` / `PlayBirdEffect` forget one or all), with one
caveat: `draw_from_deck`'s `face_up` flag (used by the Oystercatcher draft's
initial draw) is not itself part of `DrawCardEffect` — a face-up draft draw
and an ordinary hidden deck draw both record `source=deck`, indistinguishable
in the log. A future reconciliation test mirroring
`tests/test_gamelog_ledger.py` would need to special-case that call site.

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
- `record_decision` / `record_forced` / `record_effect` append to the
  stack-top's `sub_events`.  If the stack is empty, one `LooseEvent` per phase is created on
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
| `events.begin_deal(player.id)` / `events.end_event()` | wraps `_deal_setup_inputs` (the dealt cards are reveals, and the deal precedes the setup phase) |
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

### `engine/ledger.py`

| Call | Location |
|------|----------|
| `events.record_effect(effect)` | every mutation helper — the only producer of effects |
| `begin_refill_tray` / `end_event` | `_record_tray_reveals`, wrapping each tray restock |

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

## Summaries (`gamelog/summarize.py`)

Both renderers head each event with the same line, computed — never stored.
The tree stays pure ground truth; no denormalized header strings live in it.

`summarize(event) -> EventSummary` folds every `Effect` in an event's subtree
into a typed aggregate (food maps, egg counts, card-name lists, played/moved
birds, tray refills, dice faces).  Two rules govern the fold:

- **Descendants count.** A collapsed row must describe everything inside it, so
  a white power firing under a bird play shows up in the play's header.
- **Other seats do not.** A descendant event belonging to another seat — a pink
  reaction nested under the play that triggered it — is skipped with its whole
  subtree.  Rolling it in would credit one player's food gain to another.

`summary_text(event) -> str` templates that aggregate per event type.  It is
derived from **what the ledger says happened**, not from the card's printed
power text, so a power that fizzled reads as having fizzled.  When an event
recorded no effect but did resolve a decision, the header falls back to that
decision's outcome (`California Condor (white): Declines`) rather than claiming
`no effect`.

| Event | Header |
| ----- | ------ |
| `MainActionEvent` | `Main action: Forest (gain food)` — names the row the cube activates |
| `ActivateBaseEvent` | the effect phrase (`Gains fish fish`), else `Lay eggs — no effect` |
| `ActivateBrownEvent` | `Cooper's Hawk (brown): Tucks Bell's Vireo`, or `Turkey Vulture — no brown power` |
| `WhitePowerEvent` / `ReactionEvent` | same shape, tagged `(white)` / `(pink)` |
| `PlayBirdEvent` | `Plays Cooper's Hawk in Forest` |
| `ExtraPlayEvent` | `Extra play (Wetland)` |
| `TurnEndEvent` | `End of turn: Discards Mallard` |
| `RefillTrayEvent` | `Tray refill: Ruddy Duck` |
| `DealEvent` | `Deal (5 cards)` |
| `SetupEvent` | `Setup (kept: …; bonus: …)` |
| `RoundGoalEvent` / `FinalScoringEvent` | unchanged from before |
| `LooseEvent` / anything unhandled | the effect phrase, else `Other` |

Food renders as repeated whole words (`fish fish`) rather than `2fish`, because
the HTML viewer's `applyFoodEmoji` substitutes on word boundaries — `2fish`
would match nothing and print literally.  Past three tokens it switches to
`6x seed`, keeping the space so the substitution still fires.

**Reveals.** `is_reveal(effect)` marks the effects that disclosed hidden
information: a deck draw, a deck tuck, a birdfeeder reroll, a predator's dice,
a tray restock.  These get their own row via `reveal_text(effect)` — they are
the *only* record of what came off the deck.  Every other effect stays silent
and folds into its header.

## Rendering

### HTML (`reporting/game_log_capture.tree_to_log_items`)

Structure-preserving: the tree's nesting survives one-for-one into `LogItem`s,
so a turn's top level reads as one row per logical unit — the main action, the
habitat ability, one per bird crossed — with the detail inside.  Each event
takes one of three shapes, by what it actually contains:

| Contents | Shape |
| -------- | ----- |
| nothing | a muted `"note"` — what gives a bird with no brown power its own row |
| one decision, no child events | that `"decision"` box, retitled with the summary |
| anything else | a `"group"` whose children are its decisions, reveals, and child events |

`RefillTrayEvent` is the one exception (`_UNWRAPPED_EVENTS`): pure bookkeeping
whose header would only re-list its own rows, so its rows go straight into the
log.  `RoundGoalEvent` / `FinalScoringEvent` produce no items at all — they
render as phase snapshots.

`LogItem.power_color` carries `"brown"` / `"white"` / `"pink"` on power events,
tinting the header in the game's own color language.

### Plaintext (`gamelog/render_text.render_plaintext`)

Each phase: `=== KIND ===`.  Each event: `[summary_text]` — the same header the
HTML log collapses to.  Sub-events: `→ text` (decision), `! text` (forced),
`· kind(fields)` (effect).  Children indented two spaces per level.

Unlike the HTML log, this renderer keeps **every** effect row, not just the
reveals: it is the detailed log, where the header says what happened and the
rows below prove it.

### Flat JSONL (`gamelog/render_jsonl.render_rows`)

The third renderer, and the one meant for analysis rather than reading. One
match becomes a `GameMeta` header row followed by **one row per node** — every
event *and* every sub-event — in depth-first order, each a single-line JSON
object. Files concatenate freely: a `--games` series and a whole tournament land
in one file, keyed by `game_id`.

```
wingspan play --p0 best --p1 best --seed 12345 --jsonl game.jsonl
wingspan tournament --jsonl games.jsonl        # every game of the round-robin
```

```python
import pandas
rows = pandas.read_json("game.jsonl", lines=True)
turns = rows[rows.phase == "turn"].groupby("phase_seq")
```

Not wired into training self-play — millions of games, and it would cost
collection throughput for a file nobody reads.

**Common columns**, on every node row:

| Column | Meaning |
| ------ | ------- |
| `row` | `game` / `event` / `sub` — which shape this line is |
| `game_id` | the match; the join key back to the header row |
| `seq` | depth-first position within the game; orders the file |
| `event_id` | the event this row belongs to — *itself* on an `event` row, its **owner** on a `sub` row, so `groupby("event_id")` collects an event with its parts |
| `parent_id` | that event's enclosing event; absent at phase level |
| `depth` | event nesting depth (a sub-event sits one below its event) |
| `phase` | `game_start` / `setup` / `round` / `turn` / `game_end` |
| `phase_seq` | the phase's index in the game — **this is what addresses a turn** |
| `phase_round` / `phase_turn` | the phase's coordinates, namespaced |
| `kind` | the node's `kind` discriminator (`activate_brown`, `draw_card`, …) |
| `player_id` | acting seat on an event, resource owner on an effect |
| `text` | the human-readable line — see below |

> **`phase_turn` is not unique.** It numbers a seat's turns *within a round*, so
> every seat has a turn 3 and `(phase_round, phase_turn)` names one phase *per
> seat*. Group a turn on `phase_seq`, or two seats' turns silently merge.

**Per-node columns.** Each node's own typed fields become its own columns — a
`draw_card` row has `card` / `source`, a `lay_egg` row has `bird` / `habitat` /
`count`. A new `Effect` subclass contributes its columns automatically; nothing
is folded into an opaque payload. Null-valued fields are omitted rather than
written as `null` (pandas fills them as NaN either way).

**Event rows** additionally carry the folded `EventSummary` as `sum_`-prefixed
columns (`sum_eggs_laid`, `sum_food_gained`, `sum_cards_drawn`, …), so the
seat-scoping rule of the fold does not have to be re-derived downstream. Only
non-default entries appear.

**`text`** is the uniform human-readable column: an event's `summary_text`
header (byte-identical to the HTML log's), a resolution's `outcome_text`, a
reveal's disclosure line, and empty for an effect that folds silently into its
event. `outcome_text` is *not* also emitted as its own column — `text` is it.

**What is dropped:** `state_stripes` and each option's `choice_stripes`. Those
are a full feature vector per decision and would dwarf everything else. The
policy distribution itself is kept — `options` holds every offered choice's
`label` / `prob` / `score` / `selected`.

**Two shared column names**, both disambiguated by `kind`: `scores` is a
per-seat `FinalScoreBreakdown` list on a `final_scoring` row and the running
per-seat score on a resolution row; `round_idx` on a `round_goal` row is the
round it scores (equal to that row's `phase_round`).

Size: roughly 120 KB per random game, 210 KB per annotated model game.

**Tournament sharding.** Worker processes cannot share one append handle
without interleaving partial lines, so each writes `<stem>.w<pid><suffix>` and
`runner._merge_shards` concatenates them into the configured path once the pool
has shut down. The caller always ends up with the single file they asked for.

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
5. Add a header branch to `summarize.summary_text`.  All three renderers pick it
   up from there — there is no per-renderer label table.  The fallback returns
   the effect phrase or `"Other"`, so a missing branch degrades quietly rather
   than leaking a class name; `tests/test_gamelog_summarize.py` fails on the leak.
6. Add tests in `tests/test_gamelog_tree.py`; the round-trip in
   `tests/test_gamelog_serialize.py` and the flat-log row schema in
   `tests/test_gamelog_jsonl.py` both cover the new class automatically — the
   JSONL renderer reads `kind` and dumps whatever fields the class declares.
7. Update this file.
