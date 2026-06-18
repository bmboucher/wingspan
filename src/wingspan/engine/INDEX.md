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
- `Engine(gs: GameState, agents, instrumentation)` — constructor.
- `Engine.create(seed) -> (Engine, birds, bonuses, goals)` — static factory that
  instantiates a fresh game from a seed.
- `Engine.play_one_game(gs, agents, instrumentation, split_setup_bonus) -> Engine`
  — static entry point for a complete game.
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

**`reactors.py`** — Pink (between-turns) reactor hooks: `fire_pink_reactors(engine,
trigger_player_id, event)`. Iterates all players' boards and calls the pink
handler for any bird whose power matches the trigger event. See `docs/BIRDS.md`
for the reactive power taxonomy.

**`scoring.py`** — `score_round_goal(gs, goal) -> list[int]` (2-player payouts)
and `final_scoring(gs) -> dict[str, ScoreBreakdown]`. Bonus-card scoring lives
here too; each `BonusCard.scoring_rule` is dispatched through a registry.

**`helpers.py`** — Pure utility functions with no side effects:
`cost_meets(food_pool, cost) -> bool` and
`enumerate_payments(food_pool, cost) -> list[FoodPool]` (all valid payment
combinations). Used by both `actions.py` and the encoder.

**`playability.py`** — Pure playability predicates over `state.Player`:
`classify_hand_playability(player) -> (playable_now, egg_blocked)` (the two
hand multi-hot sources), `newly_playable_after_food`, `newly_playable_after_egg`,
`gainable_feeder_foods`, `newly_playable_after_feeder_food`, and
`setup_turn1_playable`. Imported **locally** inside encoder functions to keep
`encode` engine-free at import time.

**`log_format.py`** — Formatting helpers for the game log: `format_bird_log`,
`format_food_log`, etc. Pure string functions; no engine state.

## Subpackage

**`powers/`** — Bird-power dispatch: registry, dispatcher, and handler modules
grouped by `EffectKind` family.
See [`powers/INDEX.md`](powers/INDEX.md).
