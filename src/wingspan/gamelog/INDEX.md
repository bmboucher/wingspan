# gamelog — Structured game-event tree

Torch-free, engine-free kernel that the engine *produces* and the reporting
layer *consumes*.  Both the HTML decision log and the `--log` plaintext log
are pure renderers over this one tree; the raw `engine.log` stream is an
independent debug dump (reachable via `--debug-log`), not a source.

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
- Sub-event leaves: `NoteSubEvent(text)` plus the two `ResolvedSubEvent`
  subclasses — `ForcedSubEvent` and
  `DecisionSubEvent(options, state_stripes, value)`.  `ResolvedSubEvent` holds
  what both resolutions share: `outcome_text` and the clock/score fields
  (`turn_counter`, `setup_slot`, `family_idx`, `scores`, `margin_before`), so a
  forced move sits on the timeline exactly like a genuine decision.
- Top-level event types (one subclass per logical action, each carrying typed
  fields — no opaque payloads), all sharing `event_id`:
  `MainActionEvent`, `PlayBirdEvent`, `WhitePowerEvent(bird_name)`,
  `ReactionEvent(bird_name)`, `ActivateBaseEvent(habitat, action)`,
  `ActivateBrownEvent(bird_name, is_brown)`, `ExtraPlayEvent(habitat)`,
  `TurnEndEvent`, `SetupEvent(kept_card_names, kept_bonus_name)`,
  `RoundGoalEvent(round_idx, description, counts, vps)`,
  `FinalScoringEvent(scores)`, `LooseEvent`.
- `FinalScoreBreakdown(birds, eggs, tucked, cached, bonus, goals, total)`.
- `AnySubEvent`, `AnyGameEvent` — **discriminated unions** keyed on each
  subclass's `kind` literal, and the declared type of `sub_events` / `children` /
  `PhaseNode.events`.  Annotating those containers with the *base* classes makes
  `model_dump_json` silently drop every subclass field, so these aliases are
  load-bearing, not cosmetic.  `tests/test_gamelog_serialize.py` guards the
  round trip.

**`recorder.py`** — `EventRecorder` + the `null_recorder()` no-op factory.

`EventRecorder(probes)` maintains an open-event stack and a current phase.  The
engine and action modules call `begin_*/end_event` brackets and `record_*` /
`note` at each logical decision or notification point; the recorder builds the
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
  `begin_setup(player_id)` — open a typed event.
- `end_event()` — close the most-recently-opened event.
- `record_decision(engine, decision, choice)` — reads the seat's `DecisionProbe`,
  builds a fully-annotated `DecisionSubEvent`, and appends it to the stack-top.
  Stripe panels decode the annotation's recorded vectors with the annotation's
  own `state_layout` / `choice_layout` (the producing net's era geometry);
  the live layout is only a fallback for annotations that carry none.
- `record_forced(engine, decision, choice)` — appends a `ForcedSubEvent`.
- `record_round_goal(engine, round_idx, goal, counts, vps)` — appends a
  `RoundGoalEvent` to the **round phase it scores** (scoring runs after the
  round's last turn, so the current phase is that turn).
- `note(text, player_id)` — appends a `NoteSubEvent` to the stack-top. Its
  producers: `engine.actions._reroll_feeder` (every birdfeeder reroll) and
  `engine.powers.grants._h_roll_not_in_feeder_cache` (the dice predators'
  off-feeder rolls).

Unbracketed `record_*` calls collect into a single `LooseEvent` per phase,
created on first use, rather than one event apiece.

`null_recorder()` builds a fresh `_NullRecorder` whose every method is a no-op;
held by uninstrumented engines so call-site code needs no
`if recorder is not None` guards.  It is a factory rather than a shared
constant so no two engines alias one tree.

`AnyRecorder = EventRecorder | _NullRecorder` — the type used by `engine.core`
for the `events` field.

**`render_text.py`** — `render_plaintext(tree: GameEventTree) -> str`.

Pure renderer: `models` + stdlib only.  Each phase opens with
`=== KIND ===`.  Events render with a type-specific bracket label
(`[Activate forest (gain food)]`, `[Brown: Elf Owl]`, `[——: Barn Owl]`,
`[White power: Elf Owl]`, `[Extra play in forest]`, `[End of turn]`,
`[Setup (kept: Barn Owl; bonus: Rodentologist)]`,
`[Round 1 goal — ... [P0: 3/4VP, P1: 1/1VP]]`, `[Final scoring [42, 37]]`,
etc.) followed by sub-event lines:
`→ text` for decisions, `! text` for forced moves, bare text for notes.
Children (nested events) are indented two spaces per level.
