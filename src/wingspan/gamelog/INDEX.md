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
- `PhaseNode(kind, events)` — one navigable phase; `kind` is one of
  `"game_start"`, `"setup"`, `"round"`, `"turn"`, `"game_end"`.
- Sub-event leaves: `NoteSubEvent(text)`, `ForcedSubEvent(text)`,
  `DecisionSubEvent(outcome_text, options, state_stripes, value, turn_counter, …)`.
- Top-level event types (one subclass per logical action, each carrying typed
  fields — no opaque payloads):
  `MainActionEvent`, `PlayBirdEvent`, `WhitePowerEvent(bird_name)`,
  `ReactionEvent(bird_name)`, `ActivateBaseEvent(habitat, action)`,
  `ActivateBrownEvent(bird_name, is_brown)`, `SetupEvent(kept_card_names, kept_bonus_name)`,
  `RoundGoalEvent(round_idx, description, counts, vps)`,
  `FinalScoringEvent(scores)`, `LooseEvent`.
- `FinalScoreBreakdown(birds, eggs, tucked, cached, bonus, goals, total)`.
- `AnySubEvent`, `AnyGameEvent` — type-union aliases for renderers.

**`recorder.py`** — `EventRecorder` + `EMPTY` null-recorder singleton.

`EventRecorder(probes, seat_configs)` maintains an open-event stack and a
current phase.  The engine and action modules call `begin_*/end_event` brackets
and `record_*` / `note` at each logical decision or notification point; the
recorder builds the `GameEventTree` in-place.

Key public methods:

- `begin_game()` / `end_game(engine)` — reset and finalize the tree.
- `begin_phase(kind)` — push a new `PhaseNode`.
- `begin_main_action(player_id)` / `begin_play_bird(player_id)` /
  `begin_white_power(player_id, bird_name)` / `begin_reaction(player_id, bird_name)` /
  `begin_activate_base(player_id, habitat, action)` /
  `begin_activate_brown(player_id, bird_name, is_brown)` /
  `begin_setup(player_id)` — open a typed event.
- `end_event()` — close the most-recently-opened event.
- `record_decision(engine, decision, choice)` — reads the seat's `DecisionProbe`,
  builds a fully-annotated `DecisionSubEvent`, and appends it to the stack-top.
- `record_forced(engine, decision, choice)` — appends a `ForcedSubEvent`.
- `record_round_goal(engine, round_idx, goal, counts, vps)` — appends a
  `RoundGoalEvent` to the current phase.
- `note(text, player_id)` — appends a `NoteSubEvent` to the stack-top.

`EMPTY = _NullRecorder()` — every method is a no-op; held by uninstrumented
engines so call-site code needs no `if recorder is not None` guards.

`AnyRecorder = EventRecorder | _NullRecorder` — the type used by `engine.core`
for the `events` field.

**`render_text.py`** — `render_plaintext(tree: GameEventTree) -> str`.

Pure renderer: `models` + stdlib only.  Each phase opens with
`=== KIND ===`.  Events render with a type-specific bracket label
(`[Activate forest (gain food)]`, `[Brown: Elf Owl]`, `[——: Barn Owl]`,
`[White power: Elf Owl]`, `[Setup (kept: Barn Owl; bonus: Rodentologist)]`,
`[Round 1 goal — ... [P0: 3/4VP, P1: 1/1VP]]`, `[Final scoring [42, 37]]`,
etc.) followed by sub-event lines:
`→ text` for decisions, `! text` for forced moves, bare text for notes.
Children (nested events) are indented two spaces per level.
