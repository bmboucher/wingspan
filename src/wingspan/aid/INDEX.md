# aid — Live physical-game assistant

Drives the real `Engine` for a physical 2-player Wingspan game: every hidden
or random outcome (card reveals, dice rolls, feeder rolls, opponent hand
contents) is supplied by the human at the table instead of `random.Random`.
Landing stage by stage (see the plan doc for the full architecture); this
file tracks the module map as each stage merges.

## Modules

**`__init__.py`** — package docstring only.

**`models.py`** — Pydantic data shapes: `FeederEntry` (the 5 birdfeeder
dice-face counts, aligned to `cards.ALL_FOODS`, plus the choice-die count;
cross-validated so the total equals `state.BIRDFEEDER_DICE`; `to_food_pool()`
converts to a `state.FoodPool`), `SetupEntry` (the full physical deal: hand,
bonus pair, four round goals, tray, feeder, start seat), `OpponentPlayNote` /
`TurnNotes` (mid-turn scratch consumed by the relay/advisor/hooks trio below —
`TurnNotes.clear()` resets at the start of every turn), `SetupPreview` (the
advisor's combined setup-keep recommendation under a split-setup regime —
kept cards, resolved bonus card, kept food pool; `format_line()` renders
`model recommends: keep [...] + bonus [...] + foods [...]`, produced by
`preview.py`), `SessionReport` (stage 4's end-of-session summary),
`TypeaheadOutcome` / `TypeaheadState` / `CountsState` (the `widgets.py`
input-widget editable state: what a typeahead keystroke resolved to, the
typeahead query + highlighted match index, and the counts widget's per-field
values + focused index).

**`console.py`** — `Console`: the injectable `read`/`write` pair every aid
prompt flows through (`say`/`ask`/`menu`/`confirm`), so a full session can
run headlessly in tests. `supports_interactive()` reports whether both
`read`/`write` are the builtin defaults AND stdin/stdout are real ttys —
`widgets.py`'s tty shells check this before taking over stdio; a scripted
test console (injected `read`/`write`) always reports `False`. `LogEcho`:
mirrors new `GameState.log` lines (ANSI-stripped via `agents.display.strip_ansi`)
to the console as session narration; `game_state` is attached after
`oracle_state.build_state` constructs the state, and `flush()` is a no-op
before that.

**`widgets.py`** — Interactive input widgets (typeahead card lookup, numeric
per-field counts entry), layered for headless testability. Pure
reducers/renderers: `step_typeahead(state, n_matches, event, *, allow_blank)`
and `step_counts(state, target_total, event, *, field_caps)` apply one
decoded `training.configure.keys.KeyEvent` to a `models.TypeaheadState` /
`models.CountsState` and return the next state (never mutating the input)
plus a `models.TypeaheadOutcome` / an `accepted` bool; `render_typeahead_frame`
/ `render_counts_frame` render a state to plain-ASCII frame lines (no ANSI —
the in-place redraw is what matters). `MAX_VISIBLE_MATCHES` caps how many
typeahead matches a frame shows before truncating. Drivers:
`run_typeahead(events, prompt, find, render, draw, *, allow_blank, initial)`
and `run_counts(events, prompt, labels, target_total, draw, *, field_caps)`
thread an injectable `keys.KeyEvent` iterator and a `DrawFn` (the
`agents.interactive.draw_frame` contract) through the reducers to completion,
so a full widget session is scriptable and assertable headlessly; both raise
`RuntimeError` if `events` is exhausted without a selection. Tty shells
`typeahead_pick(con, prompt, find, render, *, allow_blank, initial)` and
`counts_entry(con, prompt, labels, target_total, *, field_caps)` are thin
wrappers opening a real `keys.KeyReader` and feeding an endless poll loop
into the drivers, echoing the result through `con.say`; callers branch on
`console.Console.supports_interactive()` before reaching these, so a
scripted test console never does. `typeahead_pick` is wired into `entry.py`'s
identify/setup-entry dialogs and `oracle.py`'s reveal prompts as of stage A2;
`counts_entry` is wired into the feeder/dice-roll entry points (`entry.py`'s
`_collect_feeder`, `oracle.py`'s `feeder_roll`/`dice_roll`) and `relay.py`'s
opponent distinct-foods prompt as of stage A3.

**`placeholders.py`** — `PlaceholderRegistry`: mints identity-distinct
`model_copy(deep=True)` clones of one fixed catalog bird/bonus card
(`catalog.birds_ordered()[0]` / `catalog.bonus_cards_ordered()[0]`) as
stand-ins for physically hidden cards (opponent hand, face-down draws).
`is_placeholder` answers by Python object identity, not value equality — a
genuine catalog card that happens to equal the source card by value is not a
placeholder. `count_birds_in` / `first_bird_in` / `swap_bird` are the hand
helpers the advisor sweep uses; `swap_bonus` mirrors `swap_bird` for a
player's `bonus_cards` list (used by `hooks.AidHandler.round_end`).

**`oracle.py`** — `SessionOracle`: the interactive authority for every
hidden/random outcome. Setup deals are pre-queued (`queue_bird_reveals` /
`queue_bonus_reveals` / `queue_feeder_roll`) and drained silently in FIFO
order (`None` entries mint placeholders, no prompt); anything unqueued
flushes the log, then `reveal_bird`/`reveal_bonus` branch once on
`console.supports_interactive()` — interactive consoles get a
`widgets.typeahead_pick` picker (`allow_blank=True`, so a blank Enter ->
`None` -> mints a placeholder, same as the fallback's blank answer);
scripted/non-tty consoles keep the original free-text loop, resolving
answers via `wingspan.cards.lookup` with menu-based disambiguation on
multiple candidates and a retry loop on no match. `feeder_roll` / `dice_roll`
branch the same way (stage A3): interactive consoles get a
`widgets.counts_entry` entry over the five food fields plus a choice-face
field (`_die_face_labels()`), split back into a `models.FeederEntry`/
`state.FoodPool` at the `cards.N_FOODS` boundary; scripted/non-tty consoles
keep the space-separated die-face grammar, `parse_die_faces` (public — also
reused by `entry.run_setup_entry` for the initial feeder-roll entry): each
token is a food name/alias or the literal `CHOICE_FACE_TOKEN` (`"choice"`,
public as of stage A3). On a parse failure the fallback's retry message
names any unrecognized token(s) via `invalid_die_face_tokens` (public), else
restates the count/grammar requirement unchanged — `die_face_retry_message`
(public) builds this message and is shared by every die-face text-mode
fallback (`feeder_roll`, `dice_roll`, `entry._collect_feeder`) so the
wording stays consistent.

**`oracle_state.py`** — `OracleBirdfeeder` / `OracleGameState`: pydantic
subclasses of `state.Birdfeeder` / `state.GameState` that route every
reveal/roll through a `SessionOracle` field instead of `rng` (`reroll` /
`roll_out_of_feeder` / `draw_bird` / `draw_bonus` overrides — the latter two
still pop a filler card first so `len(bird_deck)`/`len(bonus_deck)` stay
truthful for the encoder, which only reads lengths). `build_state` mirrors
`state.new_game`, replacing the shuffle/roll with the entered `SetupEntry`
facts; see its docstring for the numbered construction sequence.

**`entry.py`** — Shared "must-identify" dialogs (no face-down escape, unlike
`SessionOracle`'s reveal prompts): `identify_bird` / `identify_bonus` take a
keyword-only `exclude` collection and branch once on
`con.supports_interactive()` — interactive consoles get a
`widgets.typeahead_pick` picker whose `find` is narrowed to drop `exclude`
(an interactive-only dedup aid for already-picked cards, with no effect in
text mode); scripted/non-tty consoles fall back to the original loop via
`wingspan.cards.lookup.find_birds`/`find_bonus_cards` until a query
resolves, opening a disambiguation menu on multiple matches. `pick_habitat`
menus over `bird.habitats` (auto-resolves when only one is legal).
`run_setup_entry` is the pre-game dialog: start seat, then the 5 dealt
birds, the 2 dealt bonus cards, the 4 round goals, the 3 tray cards
left-to-right (passed the already-entered hand so its exclusion set folds
hand cards in), and the initial feeder roll — each block loops on a
"Correct?" confirm-echo before the next one starts. The per-block collectors
(`_collect_hand` / `_collect_bonus_pair` / `_collect_goals` / `_collect_tray`)
each branch internally on the same interactive check: interactive consoles
drive one typeahead pick per slot with a running `exclude` set built up as
picks are made (goals go straight through `widgets.typeahead_pick`, since
goals have no `identify_*` helper — `find`/`render`/`initial` built from
`wingspan.cards.lookup.find_goals`, `goal.description`, and
`cards.load_all()`'s goal list respectively); scripted consoles keep the
pre-A2 comma-separated-line / free-text loops verbatim. `_collect_feeder`
(stage A3) branches the same way: interactive consoles get a
`widgets.counts_entry` entry over the five food fields plus a choice-face
field (labels built from `cards.ALL_FOODS` + the promoted
`oracle.CHOICE_FACE_TOKEN`), split into a `models.FeederEntry` at the
`cards.N_FOODS` boundary; scripted consoles keep the
`oracle.parse_die_faces` text grammar, with `oracle.die_face_retry_message`
naming any unrecognized token in the retry message.

**`preview.py`** — `preview_setup(engine, inner, probe, decision, preferred)
-> models.SetupPreview`: replays a preferred `SetupChoice` through the real
deferred-resolution steps (`engine.setup_flow.apply_setup_choice` /
`resolve_deferred_setup_bonus` / `resolve_deferred_setup_food`) on a
`copy.deepcopy` of the live `GameState`, wrapped in a throwaway `Engine` so
nothing touches the real state or console (at most ~3 extra `inner` forward
passes: one bonus pick, up to two food picks). Drains `probe` once at the
end so the preview's own `inner` calls don't leak into the caller's
decision annotation.

**`advisor.py`** — `advisor_agent(inner, probe, con, echo, registry,
score_norm)`: the seat-0 `Agent`. Per decision: flushes the log, sweeps any
placeholder out of the deciding seat's hand (`entry.identify_bird` +
`registry.swap_bird`, rewriting any offered `BirdChoice`/`PlayBirdChoice`
still pointing at the swapped placeholder in place), calls `inner` and reads
back its `DecisionProbe` value/policy annotation (discarding `inner`'s own
pick), shows the model's ranked top-`_AID_TOP_K` recommendation (setup
decisions via the setup net's per-candidate `display_label`, or — under a
split-setup regime, detected via `agents.cli.setup_dialog_axes` — a compact
`keep:[...]` label plus a combined `preview.preview_setup(...).format_line()`
recommendation line after the ranking; other decisions via the promoted
`agents.cli.format_choice_line`, plus a `model eval: ±N.N VP expected
margin` line scaled by `score_norm`), then asks what was actually played
(setup via the promoted `agents.cli.resolve_setup_choice_dialog`; everything
else via an Enter-defaults-to-model-pick index prompt, prefixed with a
`(still setup — this pick completes your opening)` framing line whenever
`engine.state.turn_counter == 0` — true for the deferred bonus/food picks
too) and writes the corrected `chosen_idx` back onto the probe so a recorder
captures the real play.

**`relay.py`** — `relay_agent(con, echo, registry, notes)`: the seat-1
`Agent`. Auto-answers a `MainActionDecision`/`PlayBirdDecision` from an
unconsumed `TurnNotes.plays` entry (populated by
`hooks.AidHandler.turn_start`) without prompting; auto-picks index 0 when
every offered choice carries a placeholder bird (draft piles, unseen
discards — an identity the user cannot know either); infers the opponent's
`SetupDecision` keep from what is physically visible (how many cards kept,
plus which foods when the regime's `SetupChoice` carries a food axis) and
returns the first offered choice matching both, leaving any axis the regime
omits (food, and always bonus — the opponent's kept bonus is never visible)
unconstrained; otherwise falls back to a plain "what did they do?" menu,
rendering any placeholder-bird choice as `(face-down card)`. The food axis
is entered via `_ask_distinct_foods`, which branches on
`con.supports_interactive()` (stage A3): interactive consoles get a
`widgets.counts_entry` entry over the five food fields, each capped at 1
(`field_caps=[1] * len(cards.ALL_FOODS)`, distinctness by construction) —
the foods with a nonzero value are returned; scripted consoles keep the
comma-separated free-text loop, re-asking until exactly the requested count
of distinct, resolvable foods is entered.

**`hooks.py`** — `AidHandler` (a `pydantic` `events.CallbackHandler` mixing
in `GameStart`/`GameEnd`/`RoundStart`/`RoundEnd`/`TurnStart`/`TurnEnd`
handlers): every method flushes the log first. `turn_start` on the
opponent's seat loops "did they play (another) bird?", identifying each via
`entry.identify_bird` + `entry.pick_habitat`, swapping the placeholder in
their tracked hand (`registry.swap_bird`, skipped with a warning if none is
present) and appending a `models.OpponentPlayNote` the relay consumes; a
no-op on our own seat (the advisor sweeps per-decision instead). `round_end`
on the final round (`round_num == len(state.ROUND_CUBES) - 1`) offers to
enter each remaining placeholder opponent bonus card for exact scoring
(`entry.identify_bonus` + `registry.swap_bonus`), stopping at the first
decline and recording `opponent_bonus_entered`. `build_instrumentation`
wires one `AidHandler` instance into a fresh `dispatcher.Instrumentation`
across all six events it implements.

**`app.py`** — `wingspan aid` CLI wiring. `_build_parser()` (`prog="wingspan
aid"`): a positional checkpoint spec (default `last`) plus
`--checkpoint-dir`/`--device`/`--seed`/`--log`/`--jsonl` (no `--html` — the
navigable HTML viewer needs the training-config timeline plumbing
`cli._open_instrumentation` carries, which does not compose cleanly with
`AidHandler`'s own event router). `main(argv)`: resolves the model spec via
`players.parse_player_spec` (refusing `human`/`random`), builds the inner
agent via `players.build_agent(..., greedy=True, value_probe=probe)`, derives
the opening regime from the one loaded `TrainConfig`
(`resolve_split_setup_bonus`/`_food`, `resolve_combine_gain_food`,
`resolve_num_players((cfg,), 2)`), then runs `entry.run_setup_entry` ->
`oracle_state.build_state` -> wires `hooks.AidHandler` +
`advisor.advisor_agent` + `relay.relay_agent` -> `Engine.play_one_game`. The
whole interactive session (setup dialog through game end) is wrapped in
`try/except KeyboardInterrupt` so a Ctrl-C aborts cleanly rather than leaving
a half-drawn board. The final report (`models.SessionReport`) prints each
seat's score, the winner, and — when `AidHandler.opponent_bonus_entered` is
`False` — a reminder that the opponent's bonus VP is a placeholder to count
manually. Excluded from the coverage gate (`pyproject.toml`'s
`[tool.coverage.run] omit`, alongside `cli.py`): it is argparse/interactive
wiring exercised by hand, not unit tests — `tests/test_aid_session.py`
exercises the real advisor/relay/hooks stack it wires together, headlessly,
via `Engine.play_one_game` directly rather than through `main()`.
