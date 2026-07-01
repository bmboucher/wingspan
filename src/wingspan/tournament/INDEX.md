# tournament — Round-robin tournament runner

Pits trained AIs against each other, computes ELO ratings, and renders a live
rich dashboard with results. Entry point: `wingspan tournament`.

## Modules

**`__init__.py`**

**`models.py`** — Pydantic models for tournament parameters, results, and live
state: `TournamentConfig`, `RegimeFlags`, `ParticipantSpec`, `RunOption`,
`GameTask`, `GameResult`, `MatchupResult`, `ParticipantResult`,
`TournamentReport`, `EloTable`, `LiveRecord`, `StandingRow`. Also
`ParticipantKind`, `Orientation`, `TournamentPhase` enums. `RegimeFlags` is the
frozen `split_setup_bonus`/`split_setup_food`/`combine_gain_food` carrier every
game runs under, shipped to workers as pool `initargs`.

**`participants.py`** — Competitor discovery and loading:
- `random_spec() -> models.ParticipantSpec` — the built-in random-agent spec.
- `discover_runs(base_dir) -> list[models.RunOption]` — enumerates active +
  archived run directories under `base_dir` using the configurator's
  `runs.inspect_run`; filters to loadable, encoding-compatible runs.
- `spec_from_dir(checkpoint_dir) -> models.ParticipantSpec` — single-run spec.
- `with_unique_ids(specs) -> list[models.ParticipantSpec]` — deduplicates specs
  from identical run names by appending a numeric suffix.
- `load_player(spec, config) -> engine.Agent` — resolves a spec to a live
  `Agent` (random agent, or model loaded from its checkpoint).
- `resolve_regime_flags(specs) -> models.RegimeFlags` — the setup/food engine
  regime the games run under, derived from every model competitor's stored
  `RunConfig` (via the same `players.resolve_*` functions `wingspan play` uses)
  so games mirror how the nets were trained; random competitors express no
  preference. Raises `ValueError` when two competitors were trained under
  different regimes (they cannot share a faithful game).

**`schedule.py`** — `build_schedule(config, specs) -> list[models.GameTask]`:
generates the complete list of head-to-head pairings (each ordered pair plays
`config.games_per_pair` games with deterministic per-pair seeds).

**`runner.py`** — Plays scheduled games:
- `run_tournament(cfg, on_result=None, should_stop=None, *, in_process=False,
  regime=None) -> models.TournamentReport` — fans games across a
  `ProcessPoolExecutor` (or runs them in-process for tests), streaming each
  finished `GameResult` to `on_result`. Resolves `regime` from the competitors'
  configs via `participants.resolve_regime_flags` when not supplied, so every
  caller runs games under the trained regime by construction.
- `play_tournament_game(specs_by_id, model_agents, task, device, regime) ->
  models.GameResult` — the pure per-game unit shared by the pool and in-process
  paths; passes `regime`'s flags into `Engine.play_one_game`. The `RegimeFlags`
  ride to each worker inside `_WorkerRoster`.

**`state.py`** — `TournamentState(pydantic.BaseModel)`: live shared snapshot
read by the dashboard renderer. Fields: `config`, `phase`, `total_games`,
`games_done`, `live_table`, `records`, `elo_history`, `events`, `error`.
`record_game(result)` folds a finished game into the live Elo, records, and
sparklines. `new_tournament_state(cfg) -> TournamentState` constructs the
initial empty state.

**`elo.py`** — `replay(results, competitors, k=32, initial=1000) -> EloTable`:
replays all match results in order to compute ELO ratings from scratch.

**`results.py`** — `aggregate(cfg, games) -> TournamentReport`: rolls finished
games into the full report: deterministic final Elo, per-pair first/second/overall
splits, and per-competitor records. Data shapes live in `models.py`.

**`dashboard.py`** — Live `rich` dashboard renderer. `TournamentDashboard`
registers as a `TournamentState` observer and re-renders the ELO table and
win-rate matrix on each update.

**`picker.py`** — Interactive competitor-selection UI shown before the
tournament starts: lets the user deselect runs from the discovered list.

**`app.py`** — Entry point: wires config → picker → schedule → runner →
dashboard → results report.
