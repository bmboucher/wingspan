# engine — Game engine

Turn loop, setup phase, action dispatch, pink reactors, and final scoring.
`core.Engine` is the orchestrator; everything else (actions, powers, reactors,
scoring) lives in sibling modules as free functions whose first argument is the
`Engine`. Sibling modules break the import cycle with
`if typing.TYPE_CHECKING: from wingspan.engine import core`.

## Modules

**`__init__.py`** — re-exports `Engine`, `Agent`, `print_coverage_report`.

**`core.py`** — The central engine class and Agent protocol:
- `Agent` — `typing.Protocol` with generic `__call__[C: Choice](self, engine,
  decision: Decision[C], /) -> C`. Non-generic at use sites (`list[Agent]`
  typechecks); each call's return type tracks the Decision's parameterization.
- `Engine(gs: GameState, agents, instrumentation, *, combine_gain_food=False)` —
  constructor. `combine_gain_food` is the engine-behavior flag (default off) read
  by `actions.do_gain_food` / the raven supply handler / the setup-food branch to
  collapse multi-food gains into one combined subset decision.
- `Engine.create(seed) -> (Engine, birds, bonuses, goals)` — static factory that
  instantiates a fresh game from a seed.
- `Engine.play_one_game(gs, agents: Sequence[Agent], instrumentation, split_setup_bonus) -> Engine`
  — static entry point for a complete game. `agents` is indexed by `Player.id`
  (length must match `len(gs.players)`), not fixed at 2.
- `Engine.ask[C](agent, decision) -> C` — validates the agent's answer against
  `decision.choices`; auto-picks single-choice decisions; fires instrumentation
  callbacks. Never bypass `ask` — constructing a `Choice` directly skips validation.
- `Engine.agent_for(player) -> Agent` — returns the agent assigned to a seat.
- `Engine.state` — the live `GameState`.
- `Engine.log(msg, player_id=None)` — appends to both `state.log` (plain `str`
  list for backward compat) and `state.log_entries` (structured `LogEntry`
  list). Omitting `player_id` defaults to `state.current_player`. Pass
  `player_id=None` explicitly (or use `log_global`) for truly global lines.
- `Engine.log_global(msg)` — appends a global line (no player attribution) to
  both logs. Use for round headers, game start/end banners.
- `Engine.log_section(msg, global_line=False)` — section header with blank-line
  guarantee. Pass `global_line=True` for banners that belong to no single player.

**`state.LogEntry`** — Pydantic model (`player_id: int | None`, `text: str`).
Parallel structured log in `GameState.log_entries`; consumed by `cli._write_split_logs`
to produce per-player log files (`FILE_p0.log` / `FILE_p1.log`). `player_id=None`
marks global lines that appear in both per-player files.

**`actions.py`** — The four main actions as free functions:
`do_gain_food(engine, agent)`, `do_lay_eggs(engine, agent)`,
`do_draw_cards(engine, agent)`, `do_play_bird(engine, agent)`. Each mutates
`engine.state` and calls `engine.ask` for any decisions required by the action.
Under `engine.combine_gain_food`, multi-food gains route through the combined
builders: `combined_feeder_gain(engine, agent, player, n)` (the path-dependent
Forest feeder gain — reset folded in, partial subset → committed reroll →
recurse, `n==1` delegates to `take_one_from_feeder`), `_apply_subset(engine,
player, choice)` (moves a chosen `FoodSubsetChoice`'s dice out of the feeder,
bypassing `gain_feeder_die`'s mid-take reroll), and `combined_supply_gain(engine,
agent, player, n, *, per_food_capacity, prompt)` (the ravens' supply gain and the
setup keep — multisets within a per-food capacity).

**`reactors.py`** — Pink (between-turns) reactor hooks, each taking the
triggering player by reference (not id) and firing every OTHER player's
matching pink birds in clockwise order from the trigger: `trigger_pink_lay_eggs_reactors(engine,
active_player: state.Player)`, `trigger_pink_play_bird_reactors(engine,
active_player: state.Player, played_habitat)`, `trigger_pink_gain_food_reactors(engine,
active_player: state.Player, gained_foods)`, `trigger_pink_predator_success(engine,
hunter_player: state.Player)` (fires opposing `PINK_PREDATOR_FEEDER` birds after
a successful `PREDATOR_HUNT` / `ROLL_NOT_IN_FEEDER_CACHE`). See `docs/BIRDS.md`
for the reactive power taxonomy.

**`scoring.py`** — `score_round_goal(engine, round_idx) -> None` awards each
round's goal VP via `placement_payouts(counts, payouts) -> list[int]` (the
N-player placement kernel: ranks counts descending, a count of 0 never places,
ties split the floor of their combined places' payouts) against
`state.ROUND_GOAL_PAYOUTS[round_idx]`. `winners(players) -> list[int]` /
`determine_winner(players) -> int` are the shared game-winner kernel (highest
`final_score`, ties broken by most unused supply food; `determine_winner`
returns -1 for a genuine shared victory). `final_scoring(engine) -> None` sets
each `Player.final_score`. Bonus-card scoring lives here too; each
`BonusCard.scoring_rule` is dispatched through a registry.

**`helpers.py`** — Pure utility functions with no side effects:
`cost_meets(food_pool, cost) -> bool` and
`enumerate_payments(food_pool, cost) -> list[FoodPool]` (all valid payment
combinations). Used by both `actions.py` and the encoder.

**`playability.py`** — Pure playability predicates over `state.Player`:
`classify_hand_playability(player) -> (playable_now, egg_blocked)` (the two
hand multi-hot sources), `newly_playable_after_food`, `newly_playable_after_egg`,
`gainable_feeder_foods`, `newly_playable_after_feeder_food`,
`min_food_to_unlock(player, candidates) -> list[int]` (per-food smallest count
that would newly unlock a candidate bird — source of the v1.4
`hand_food_unlock_me` / `tray_food_unlock_me` state stripes), and
`setup_turn1_playable`. Imported **locally** inside encoder functions to keep
`encode` engine-free at import time.

**`log_format.py`** — Formatting helpers for the game log: `format_bird_log`,
`format_food_log`, etc. Pure string functions; no engine state.

## Subpackage

**`powers/`** — Bird-power dispatch: registry, dispatcher, and handler modules
grouped by `EffectKind` family.
See [`powers/INDEX.md`](powers/INDEX.md).
