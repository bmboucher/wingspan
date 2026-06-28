# tournament — Round-robin tournament runner

Pits trained AIs against each other, computes ELO ratings, and renders a live
rich dashboard with results. Entry point: `wingspan tournament`.

## Modules

**`__init__.py`**

**`models.py`** — Pydantic models for tournament parameters, results, and live
state: `TournamentConfig`, `ParticipantSpec`, `RunOption`, `GameTask`,
`GameResult`, `MatchupResult`, `ParticipantResult`, `TournamentReport`,
`EloTable`, `LiveRecord`, `StandingRow`. Also `ParticipantKind`, `Orientation`,
`TournamentPhase` enums.

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

**`schedule.py`** — `build_schedule(config, specs) -> list[models.GameTask]`:
generates the complete list of head-to-head pairings (each ordered pair plays
`config.games_per_pair` games with deterministic per-pair seeds).

**`runner.py`** — Plays scheduled games:
- `run_tournament(schedule, config) -> TournamentState` — process-parallel
  execution when `config.parallel > 1`, sequential fallback otherwise. Each
  worker calls `Engine.play_one_game` and posts a `GameOutcome` to the shared
  state.

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
