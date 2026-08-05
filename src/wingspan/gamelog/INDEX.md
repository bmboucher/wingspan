# gamelog — Structured game-event tree

Torch-free, engine-free kernel that the engine *produces* and the reporting
layer *consumes*.  The HTML decision log, the `--log` plaintext log, and the
`--jsonl` flat structured log are all pure renderers over this one tree; the raw
`engine.log` stream is an independent debug dump (reachable via `--debug-log`),
not a source.

See [`docs/GAMELOG.md`](../../../docs/GAMELOG.md) for the full design reference:
six event types, sub-event taxonomy, open-event-stack rule, and call-site map.

## Import discipline

All modules in this package depend **only** on `pydantic`, the standard library,
and `wingspan.decisions`.  No engine, state, training, or torch imports at
module load time — heavy deps are lazy-imported inside active-path methods only.
This keeps `gamelog` importable by `reporting` without closing any import cycle.

## Modules

**`models.py`** — Pydantic event-tree node hierarchy (the canonical data models).
Also holds the display primitives formerly in `reporting.game_log_html`
(`EncodedSubField`, `EncodedStripe`, `DecisionOption`) so they can be shared
without importing the reporting layer.

Key classes:

- `GameEventTree(phases: list[PhaseNode])` — the complete tree for one game.
- `PhaseNode(kind, round_idx, turn_idx, events)` — one navigable phase; `kind` is
  one of `"game_start"`, `"setup"`, `"round"`, `"turn"`, `"game_end"`.
  `round_idx` / `turn_idx` make the phase self-describing (both `None` where
  they do not apply).
- Sub-event leaves: the two `ResolvedSubEvent` subclasses — `ForcedSubEvent`
  and `DecisionSubEvent(options, state_stripes, value)`.  `ResolvedSubEvent`
  holds what both resolutions share: `outcome_text` and the clock/score fields
  (`turn_counter`, `setup_slot`, `family_idx`, `scores`, `margin_before`), so a
  forced move sits on the timeline exactly like a genuine decision.
- Effect leaves: the `Effect` subclasses — one per kind of state mutation,
  interleaved with decisions in the same `sub_events` stream so the log reads
  in true chronological order.  `GainFoodEffect`, `SpendFoodEffect`,
  `CacheFoodEffect`, `UncacheFoodEffect`, `LayEggEffect`, `RemoveEggEffect`,
  `DrawCardEffect`, `DiscardCardEffect`, `TuckCardEffect`, `PlayBirdEffect`,
  `MoveBirdEffect`, `PassCardEffect`, `FeederRerollEffect`, `DiceRollEffect`,
  `TrayRefillEffect`, plus the `FoodSource` / `CardSource` / `EffectPurpose`
  StrEnums their fields draw on.  Only `engine.ledger` constructs these — see
  [`docs/GAMELOG.md`](../../../docs/GAMELOG.md) for the ledger contract.
- Top-level event types (one subclass per logical action, each carrying typed
  fields — no opaque payloads), all sharing `event_id`:
  `MainActionEvent(action)`, `PlayBirdEvent`, `WhitePowerEvent(bird_name)`,
  `ReactionEvent(bird_name)`, `ActivateBaseEvent(habitat, action)`,
  `ActivateBrownEvent(bird_name, is_brown)`, `ExtraPlayEvent(habitat)`,
  `TurnEndEvent`, `RefillTrayEvent`, `DealEvent`,
  `SetupEvent(kept_card_names, kept_bonus_name)`,
  `RoundGoalEvent(round_idx, description, counts, vps)`,
  `FinalScoringEvent(scores)`, `LooseEvent`.
- `FinalScoreBreakdown(birds, eggs, tucked, cached, bonus, goals, total)`.
- Flat-log rows (consumed by `render_jsonl.py`): `LogSource`, `RowKind`,
  `GameMeta` (one match's identity, setup, and outcome — the header row) and
  `NodeRow` (the columns every node row carries; a node's own typed fields ride
  along as pydantic extras, so a new `Effect` class contributes its own columns
  with no schema edit).
- `AnySubEvent`, `AnyGameEvent` — **discriminated unions** keyed on each
  subclass's `kind` literal, and the declared type of `sub_events` / `children` /
  `PhaseNode.events`.  Annotating those containers with the *base* classes makes
  `model_dump_json` silently drop every subclass field, so these aliases are
  load-bearing, not cosmetic.  `tests/test_gamelog_serialize.py` guards the
  round trip.

**`recorder.py`** — `EventRecorder` + the `null_recorder()` no-op factory.

`EventRecorder(probes)` maintains an open-event stack and a current phase.  The
engine and action modules call `begin_*/end_event` brackets and `record_*` at
each logical decision or mutation point; the recorder builds the
`GameEventTree` in-place, stamping each event with a per-game monotonic
`event_id` as it is opened.

Key public methods:

- `begin_game()` / `end_game(engine)` — reset and finalize the tree.
- `begin_phase(kind, round_idx=None, turn_idx=None)` — push a new `PhaseNode`.
- `begin_main_action(player_id)` / `begin_play_bird(player_id)` /
  `begin_white_power(player_id, bird_name)` / `begin_reaction(player_id, bird_name)` /
  `begin_activate_base(player_id, habitat, action)` /
  `begin_activate_brown(player_id, bird_name, is_brown)` /
  `begin_extra_play(player_id, habitat)` / `begin_turn_end(player_id)` /
  `begin_refill_tray(player_id)` / `begin_deal(player_id)` /
  `begin_setup(player_id)` — open a typed event.
- `end_event()` — close the most-recently-opened event.
- `record_decision(engine, decision, choice)` — reads the seat's `DecisionProbe`,
  builds a fully-annotated `DecisionSubEvent`, and appends it to the stack-top.
  Stripe panels decode the annotation's recorded vectors with the annotation's
  own `state_layout` / `choice_layout` (the producing net's era geometry);
  the live layout is only a fallback for annotations that carry none.
- `record_forced(engine, decision, choice)` — appends a `ForcedSubEvent`.

  Both resolution paths first call `_stamp_open_events`, which copies a resolved
  choice's outcome onto the open event that summarizes it — the setup's kept
  cards and bonus, and `MainActionEvent.action`.  These cannot be passed to
  `begin_*` because at bracket-open time the answer is not yet known.
- `record_round_goal(engine, round_idx, goal, counts, vps)` — appends a
  `RoundGoalEvent` to the **round phase it scores** (scoring runs after the
  round's last turn, so the current phase is that turn).
- `record_effect(effect)` — appends one recorded state mutation to the
  stack-top.  Its sole producer is `engine.ledger`, which performs each
  mutation and records its effect in the same call so the two cannot drift;
  `tests/test_gamelog_ledger.py` reconciles the replayed ledger against the
  final `GameState` to prove the seam is complete.

Unbracketed `record_*` calls collect into a single `LooseEvent` per phase,
created on first use, rather than one event apiece.

`null_recorder()` builds a fresh `_NullRecorder` whose every method is a no-op;
held by uninstrumented engines so call-site code needs no
`if recorder is not None` guards.  It is a factory rather than a shared
constant so no two engines alias one tree.

`AnyRecorder = EventRecorder | _NullRecorder` — the type used by `engine.core`
for the `events` field.

**`summarize.py`** — The shared header text both renderers use.  Pure:
`models` + `pydantic` + stdlib only.

- `EventSummary` — a typed aggregate of every `Effect` in one event's subtree
  (food maps keyed by type, egg counts, card-name lists, played/moved birds,
  tray refills, dice faces), plus an `is_empty` property.
- `summarize(event) -> EventSummary` — the fold.  Descendants count (a
  collapsed row must describe everything inside it); other seats do not (a pink
  reaction nested under the play that triggered it is skipped with its whole
  subtree, so one player's gain is never credited to another).
- `summary_text(event) -> str` — the one-line header per event type, e.g.
  `Main action: Forest (gain food)`, `Gains fish fish`,
  `Cooper's Hawk (brown): Tucks Bell's Vireo`, `Turkey Vulture — no brown
  power`, `Plays Cooper's Hawk in Forest`, `Tray refill: Ruddy Duck`.  Derived
  from what the ledger recorded, never from the card's printed power text; an
  event with no effects but a resolved decision falls back to that decision's
  outcome rather than claiming `no effect`.
- `effect_phrase(summary) -> str` / `food_words(counts) -> str` — the prose
  builders.  Food is repeated whole words (`fish fish`, not `2fish`) because
  the HTML viewer's emoji substitution matches on word boundaries; past three
  tokens it switches to `6x seed`, keeping the space.
- `is_reveal(effect)` / `reveal_text(effect)` — the hidden-information effects
  (deck draw, deck tuck, feeder reroll, predator dice, tray restock) that earn
  their own row in the HTML log instead of folding into a header.

**`render_text.py`** — `render_plaintext(tree: GameEventTree) -> str`.

Pure renderer: `models` + `summarize` + stdlib only.  Each phase opens with
`=== KIND ===`; each event with `[summary_text]` — the same header the HTML log
collapses to.  Sub-event lines follow: `→ text` for decisions, `! text` for
forced moves, `· kind(fields)` for effects (a mechanical field dump, so a new
effect class renders truthfully without a hand-written template).
Children (nested events) are indented two spaces per level.

Unlike the HTML log, this keeps **every** effect row rather than only the
reveals: it is the detailed log, where the header says what happened and the
rows below it prove it.

**`render_jsonl.py`** — the flat structured log, for analysis rather than
reading.  Pure: `models` + `summarize` + stdlib only.

- `render_rows(tree, meta) -> list[str]` — the game's `GameMeta` header row,
  then one `NodeRow` per node (event *and* sub-event) in depth-first order.
- `render_jsonl(tree, meta) -> str` / `append_game(path, tree, meta)` — the
  newline-terminated block and the append that writes it.  Files concatenate,
  so a `--games` series, a tournament worker's shard, and a single game all go
  through the same call.

Every row carries the tree links as columns (`event_id` / `parent_id` /
`depth` / `seq` / `phase_seq`), so the nesting reconstructs from the `event`
rows alone.  Event rows also carry the folded `EventSummary` as `sum_`-prefixed
columns and the same `text` header the HTML log shows.  The encoding-viewer
stripes are dropped (a full feature vector per decision); the policy
distribution is kept.  See [`docs/GAMELOG.md`](../../../docs/GAMELOG.md) for the
full row schema.
