# tournament — Round-robin tournament runner

Pits trained AIs against each other, computes ELO ratings, and renders a live
rich dashboard with results. Entry point: `wingspan tournament`.

## Modules

**`__init__.py`**

**`config.py`** — `TournamentConfig(competitors, games_per_pair, seed,
checkpoint_dir, parallel)` — Pydantic model for tournament parameters. Loaded
from CLI flags; validated on parse.

**`participants.py`** — Competitor resolution:
- `Competitor(spec: PlayerSpec, run_name, agent)` — a resolved tournament
  entrant.
- `load_competitors(config) -> list[Competitor]` — discovers runs in the
  checkpoint directory, resolves each spec via `players.factory`, and loads the
  agent. On-disk run discovery keys off each subdirectory's `last.pt` and reads
  its config descriptor via the dispatching `runmeta.read_model_config`
  (`run_config_<stamp>.json` for ≥0.5, legacy `model_config.json` otherwise).

**`schedule.py`** — `RoundRobinSchedule(competitors) -> list[Match]`: generates
the complete list of head-to-head pairings (each ordered pair plays `games_per_pair`
games). `Match(player_a, player_b, game_idx)` — one scheduled game unit.

**`runner.py`** — Plays scheduled games:
- `run_tournament(schedule, config) -> TournamentState` — process-parallel
  execution when `config.parallel > 1`, sequential fallback otherwise. Each
  worker calls `Engine.play_one_game` and posts a `GameOutcome` to the shared
  state.

**`state.py`** — `TournamentState`: live shared snapshot of results. Holds
`outcomes: list[GameOutcome]` (append-only), exposes `win_matrix()`,
`games_played()`, and fires update callbacks consumed by the dashboard.

**`elo.py`** — `compute_elo(outcomes, competitors, k=32, initial=1000)
-> dict[str, float]`: standard ELO rating computation from a list of
`GameOutcome` records.

**`results.py`** — `TournamentSummary` Pydantic model aggregating win rates,
ELO ratings, and head-to-head matrices. `build_summary(state, competitors)
-> TournamentSummary`. Also `write_report(summary, path)` — writes a JSON
results file.

**`dashboard.py`** — Live `rich` dashboard renderer. `TournamentDashboard`
registers as a `TournamentState` observer and re-renders the ELO table and
win-rate matrix on each update.

**`picker.py`** — Interactive competitor-selection UI shown before the
tournament starts: lets the user deselect runs from the discovered list.

**`app.py`** — Entry point: wires config → picker → schedule → runner →
dashboard → results report.
