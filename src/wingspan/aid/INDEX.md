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
`TurnNotes.clear()` resets at the start of every turn), `SessionReport`
(stage 4's end-of-session summary).

**`console.py`** — `Console`: the injectable `read`/`write` pair every aid
prompt flows through (`say`/`ask`/`menu`/`confirm`), so a full session can
run headlessly in tests. `LogEcho`: mirrors new `GameState.log` lines
(ANSI-stripped via `agents.display.strip_ansi`) to the console as session
narration; `game_state` is attached after `oracle_state.build_state`
constructs the state, and `flush()` is a no-op before that.

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
prompts interactively through `console.py`, resolving free-text answers via
`wingspan.cards.lookup` with menu-based disambiguation on multiple
candidates and a retry loop on no match. `feeder_roll` / `dice_roll` share a
space-separated die-face grammar, `parse_die_faces` (public — also reused by
`entry.run_setup_entry` for the initial feeder-roll entry): each token is a
food name/alias or the literal `"choice"`.

**`oracle_state.py`** — `OracleBirdfeeder` / `OracleGameState`: pydantic
subclasses of `state.Birdfeeder` / `state.GameState` that route every
reveal/roll through a `SessionOracle` field instead of `rng` (`reroll` /
`roll_out_of_feeder` / `draw_bird` / `draw_bonus` overrides — the latter two
still pop a filler card first so `len(bird_deck)`/`len(bonus_deck)` stay
truthful for the encoder, which only reads lengths). `build_state` mirrors
`state.new_game`, replacing the shuffle/roll with the entered `SetupEntry`
facts; see its docstring for the numbered construction sequence.

**`entry.py`** — Shared "must-identify" dialogs (no face-down escape, unlike
`SessionOracle`'s reveal prompts): `identify_bird` / `identify_bonus` loop
via `wingspan.cards.lookup.find_birds`/`find_bonus_cards` until a query
resolves, opening a disambiguation menu on multiple matches; `pick_habitat`
menus over `bird.habitats` (auto-resolves when only one is legal).
`run_setup_entry` is the pre-game dialog: start seat, then the 5 dealt birds
(one comma-separated line via `lookup.parse_bird_list`, each unresolved
token re-asked individually), the 2 dealt bonus cards, the 4 round goals
(via `lookup.find_goals`), the 3 tray cards left-to-right, and the initial
feeder roll (via `oracle.parse_die_faces`) — each block loops on a
"Correct?" confirm-echo before the next one starts.

**`advisor.py`** — `advisor_agent(inner, probe, con, echo, registry,
score_norm)`: the seat-0 `Agent`. Per decision: flushes the log, sweeps any
placeholder out of the deciding seat's hand (`entry.identify_bird` +
`registry.swap_bird`, rewriting any offered `BirdChoice`/`PlayBirdChoice`
still pointing at the swapped placeholder in place), calls `inner` and reads
back its `DecisionProbe` value/policy annotation (discarding `inner`'s own
pick), shows the model's ranked top-`_AID_TOP_K` recommendation (setup
decisions via the setup net's per-candidate `display_label`s; other
decisions via the promoted `agents.cli.format_choice_line`, plus a
`model eval: ±N.N VP expected margin` line scaled by `score_norm`), then
asks what was actually played (setup via the promoted
`agents.cli.resolve_setup_choice_dialog`; everything else via an
Enter-defaults-to-model-pick index prompt) and writes the corrected
`chosen_idx` back onto the probe so a recorder captures the real play.

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
rendering any placeholder-bird choice as `(face-down card)`.

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
