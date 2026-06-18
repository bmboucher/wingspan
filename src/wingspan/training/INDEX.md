# training — Live training + monitoring dashboard

The "FLIGHT PLAN" training loop: self-play collection → REINFORCE learner →
paired-game evaluation → checkpoint. Entry point: `python -m wingspan.training`
(or `wingspan train`). See `docs/TRAINING.md` for the training program,
hyperparameter guidance, and Phase 0–3 roadmap.

## Entry points

**`__main__.py` / `app.py`** — Argparse + `--config` flag → instantiates
`TrainingLoop` on a background thread, starts `rich.Live` with the dashboard,
and blocks until the loop stops or the user hits Ctrl-C.

## Config and metadata

**`config.py`** — `RunConfig`: the self-describing hyperparameter object,
organized into six nested **section sub-models** (`TrainConfig` is a kept alias
for `RunConfig`). Raw scalar reads are nested; the heavily-used derivations stay
top-level computed properties so call sites don't churn.
- `architecture: ArchitectureConfig` — topology + encoding-shape toggles +
  era-synced dims. `main: MainNetArchitecture` (trunk/choice/head widths, card +
  hand encoders), `setup: SetupNetArchitecture`, the `use_setup_model` /
  `split_setup_*` toggles, and the era-synced `encoding_version` / `state_dim` /
  `choice_dim` / `family_order`.
- `run: RunSettings` — `games_per_iter`, `max_iterations`, `target_iterations`,
  `eval_every`, `eval_games`, `checkpoint_dir`, `run_name`, `resume`, `history_len`.
- `training: TrainingConfig` — `lr`, `value_coef`, `entropy_coef`, `grad_clip`,
  `score_norm`, `reward_mode` (`terminal_margin` | `decision_delta`),
  `reward_discount` (both REGIME), and `setup: SetupTrainingConfig` (setup-net
  `lr`, schedule `record_start_iter` / `train_iter`, offline-fit + actor-critic knobs).
- `opponent: OpponentConfig` — `bootstrap_opponent` (`"none"` | `"random"` |
  ckpt path; a path is CPU-only), `random_phase_win_rate`,
  `opponent_reset_win_rate`, `opponent_max_iterations`, `eval_ewma_alpha`.
- `engine: EngineConfig` — documented placeholder for future
  encoding-independent game-variant knobs (empty today).
- `misc: MiscConfig` — `seed`, `device`, `produce_ewma_alpha`, `instrumentation`.
- `dagger: DaggerConfig` — `expert_checkpoint` (`.pt` path or `"none"`),
  `clone_iters` (pure imitation iters before RL). Cross-section validators:
  expert set ⟹ `device='cpu'`, `clone_iters >= 1`; `clone_iters > 0` ⟹
  `bootstrap_opponent == 'none'`. REGIME (see `docs/TRAINING.md §6.7`).
- Top-level computed properties (delegating into sections): `arch:
  ModelArchitecture`, `setup_arch`, `setup_encoding`, `architecture_key`,
  `setup_architecture_key`, `encoding_spec`, `encoding_version`, `state_dim`,
  `choice_dim`, `family_order`, `eval_pairs`, `initial_vs_random`,
  `bootstrap_opponent_checkpoint`, `dagger_expert_checkpoint`,
  `dagger_active_at(iteration: int) -> bool`, `split_setup_*_active`, `trunk/choice_hidden`.
  `encoding_version` is the artifact era the run trains at (adopted from the run
  dir on resume, never user-edited); `state_dim` / `choice_dim` are era-routed
  from it. See "Training resume: era pinning" in `docs/VERSIONING.md`.
- `RunConfigFile` — the dated on-disk wrapper (`version`, `saved_at`,
  `started_at`, `git_sha`, `resumed`, `resumed_from_iteration`, `config`).
- Module functions: `run_config_from_artifact(raw, artifact_version)`
  (`train_config_from_artifact` alias) — validate a payload's embedded config at
  its own era; a ≥0.5 dict is nested and passes through, a ≤0.4 dict is *flat*
  and reshaped into the six sections (legacy `bootstrap_opponent` migration
  preserved); pre-field configs derive `encoding_version` from the `version`
  stamp. `with_encoding_version(cfg, era)` — validated era-pinned copy.

**`artifacts.py`** — `ArtifactPaths(checkpoint_dir)`: canonical on-disk
filenames. Constants: `LAST_CKPT`, `BEST_CKPT`, `OPPONENT_CKPT`,
`METRICS_LOG`, `GAMES_LOG`, `RUN_CONFIG_PREFIX` / `RUN_CONFIG_GLOB` (the unified
≥0.5 file), and the legacy `MODEL_CONFIG` / `PROCESS_JSON` / `PROCESS_GLOB`
(read for ≤0.4 dirs). Used everywhere that writes or reads from a run directory.

**`runmeta.py`** — The unified config file, the legacy sidecars (read-only for
≤0.4 dirs), and the era-routed descriptor reporting seam:
- `write_run_config(...)` / `read_run_config(dir) -> RunConfigFile` — write/read
  the dated `run_config_<stamp>.json` (≥0.5); the writer replaces the three
  legacy writers, and `read_run_config` raises `FileNotFoundError` on ≤0.4 dirs.
- `ModelConfig` — the in-memory weight-compat descriptor; carries `run_name`,
  `state_dim`, `choice_dim`, `family_order`, `architecture`, `include_setup`,
  `version`. `read_model_config(dir) -> ModelConfig` **dispatches on presence**:
  derived from `run_config_<stamp>.json` when one exists, else read from the
  legacy `model_config.json` (with compat shims by version).
- Reporting seam: `choice_layout_for(descriptor)`, `param_report_for(descriptor)`,
  `build_model_summary_html(descriptor, ...)` — all route by the descriptor's
  version so compat-era reports are correct without touching the live encoder.

**`setup_runmeta.py`** — `SetupConfig` for the setup model; `read_setup_config`
dispatches the same way (derived from the unified file for ≥0.5, legacy
`setup_config.json` for ≤0.4).

## Training loop orchestrator

**`loop.py`** — `TrainingLoop(cfg, pause_at_target)`: the top-level class.
Key members:
- `run()` — main entry (runs on background thread).
- `request_stop()`, `stopped` — graceful shutdown signal.
- `signal_target_response(choice, new_target)` — unblock from a target pause.
- `self.net`, `self.optimizer`, `self.state (RunState)`, `self.lock (RLock)`.

**`loop_resume.py`** — `maybe_resume(loop)`: loads `LAST_CKPT` if present,
validates `architecture_key` (alarm + fresh start on mismatch, including when
the weights themselves fail to load), initializes phase and target.
`write_run_metadata(loop)` drops this startup's `run_config_<stamp>.json` (plus
the inspect report + summary HTML); `reset_history_logs_if_fresh(loop)` clears a
prior run's logs and stale session records (both `run_config_*.json` and legacy
`process_*.json`) on a non-resumed start. `adopt_checkpoint_era(cfg)` — called by
`TrainingLoop.__init__` before the net is built: pins the config to the
resumable checkpoint's era when that adoption is exactly what makes the keys
agree (era-pinned resume across a FRESH change), and re-keys any *fresh*
launch at the live `MODEL_VERSION` so a new run never inherits a stale era
(`docs/VERSIONING.md`). `validate_dagger_expert(loop)` — called on `__init__`
after `validate_bootstrap_opponent`; loads the DAgger expert checkpoint to
fail-fast on a bad path (no-op when expert is `None`).

**`loop_collect.py`** — `run_collection(loop, iteration) -> CollectResult`:
dispatches to `mp_collect.ProcessCollector` (CPU) or `batched_collect` (CUDA)
based on `config.device`. Returns accumulated steps and score breakdowns.

**`loop_setup.py`** — Setup-model lifecycle:
`fit_setup_model(loop)` (offline fit on stored samples),
`update_setup_model(loop, steps)` (on-policy MSE),
`sync_setup_net(loop)` (copies weights to the opponent net).

**`loop_eval.py`** — `run_eval(loop, iteration) -> EvalResult`: plays
`config.eval_games` paired games vs the opponent net; checks win-rate threshold
for graduation. `load_opponent(loop)` — loads the opponent checkpoint with
graceful FRESH restart on architecture mismatch.

**`loop_target.py`** — `check_target(loop, iteration)`: milestone sequencing
(checkpoint → eval → pause at user-configured target iteration).

**`loop_checkpoint.py`** — `commit_checkpoint(loop, iteration, result)`:
atomic checkpoint write (LAST then BEST on improvement), games log append,
seed advancement. `finish(loop)` — final checkpoint + cleanup.

**`loop_metrics.py`** — Pure metrics aggregation: `aggregate_metrics(steps,
outcomes) -> IterationMetrics`. No loop state; easy to test in isolation.

## Collection

**`collect.py`** — `collect_game(net, opponent_net, config) -> CollectResult`:
baseline single-game collector. Calls `Engine.play_one_game` with a sampling
policy agent and a greedy opponent; returns `(steps, score_breakdown)`.

**`mp_collect.py`** — `ProcessCollector`: process-parallel collection for the
CPU path. Manages a `multiprocessing.Pool`; each worker runs `collect_game`
and returns steps via IPC (fp16-compressed for bandwidth). The pool is kept
alive across iterations to amortize process-spawn cost. `_WorkerArch` carries
`encoding_version` so workers rebuild the era's net class
(`PolicyValueNet.class_for_version`) for both their own net and the eval
opponent.

**`batched_collect.py`** — `BatchedCollector`: batched-forward collection for
the CUDA path. Runs multiple game environments in lockstep, forwarding a batch
of state tensors through the net in one call. Game threads encode via the
server's `encode_state` / `encode_choices` (delegating to the served net), so
an era-pinned net's frozen encoders are honored here too.

**`policy.py`** — `sampling_agent(net, spec) -> Agent` (stochastic, for
collection) and `greedy_agent(net, spec) -> Agent` (argmax, for eval). Both
call `net.encode_state` / `net.encode_choices` — never the raw encoder.

**`steps.py`** — `Step(state, choices, chosen_idx, player_id, family_idx,
margin_before, timestamp, expert_probs=None)` — the recorded self-play transition
consumed by the learner. `margin_before` is the deciding player's running margin
(own − opponent) before the decision, differenced into the `decision_delta`
return; `timestamp` is the decision's game-clock time (see `timestamps.py`).
`expert_probs: np.ndarray | None` — shape `(n_choices,)`, the DAgger expert's
soft policy distribution, set at collection time when `dagger_active=True`; `None`
in RL mode or for SETUP steps when the expert lacks the SETUP head. IPC-only
(not persisted to `games.jsonl`). All arrays stored as fp16 during IPC.

**`timestamps.py`** — The game clock for time-based discounting: setup
decisions at 0 / 1/3 / 2/3, turn N's main action at exactly N (the engine's
`GameState.turn_counter`), mid-turn decisions interpolated into (N, N+1).
`provisional_timestamp(decision, turn_counter)` at record time,
`finalize_timestamps(recorded)` once the game is complete (the interpolation
needs each turn's full decision count), `final_timestamp(turn_counter)` for the
terminal checkpoint (`GameRecord.final_timestamp`). Torch-free.

## Learning

**`learner.py`** — `update(net, optimizer, records, cfg, device, imitation_phase=False)`:
- Length-bucketed REINFORCE with advantage normalisation (normal RL mode).
- In the DAgger imitation phase (`imitation_phase=True`): loss is
  `CE(student, expert) + value_coef * value_MSE`; policy-gradient and entropy
  terms are zero. The `has_expert` mask (1.0 when `Step.expert_probs` is not
  `None`) weights the CE mean; `clamp(min=1)` guards the all-unlabeled bucket.
  Returns `UpdateStats` with `imitation_loss: float` (0.0 in RL mode).
- `_flatten` pairs each step with its return per `cfg.reward_mode`:
  `_terminal_margin_returns` broadcasts the end-of-game margin; for
  `decision_delta`, `_decision_delta_returns` discounts per-decision
  `margin_before` deltas by γ^Δt of game-clock time between checkpoints
  (γ = `reward_discount`, Δt from `Step.timestamp` /
  `GameRecord.final_timestamp`) into each step's return.

**`setup_net.py`** — `SetupNet(arch: SetupArchitecture, input_dim)`: the setup
model's MLP value-regressor. Single scalar output (predicted final score).

**`setup_learner.py`** — `SetupLearner(net, optimizer)`:
`offline_fit(store, epochs)` and `online_update(samples)` (MSE on recent
on-policy samples).

## Evaluation + metrics

**`evaluate.py`** — `evaluate_vs_opponent(net, opponent_net, config) -> EvalResult`:
plays `n` paired games (each pair swaps seats) and computes win rate + 95% CI.

**`convergence.py`** — `series_slope(values, window)` and
`axis_window(values, window_frac)`: windowed math used by the convergence
charts to determine whether training has plateaued.

**`metrics.py`** — Pydantic models: `ScoreBreakdown`, `FamilyCounts`,
`EvalResult`, `IterationMetrics`, `GameOutcome`. `GameOutcome` is the JSONL
row format for `games.jsonl`.

**`metrics_log.py`** — `MetricsLog(path)`: cached reader for the append-only
`metrics.jsonl` history. `load() -> list[IterationMetrics]`; re-reads only
new lines on subsequent calls.

**`runstate.py`** — `RunState`: the shared live snapshot the dashboard reads.
Fields: `phase`, `iteration`, `best_win_rate`, `games_per_sec`, `recent_metrics`,
`target_event` (for the pause prompt). Protected by `TrainingLoop.lock`.

## Dashboard + theme

**`dashboard.py`** — `TrainingDashboard`: the five-band `rich` Layout
(SYSTEM / FLIGHT PLAN / PROGRESS / CONVERGENCE / LOG) + per-region renderers.
Reads from `RunState` on each refresh tick.

**`theme.py`** — Palette (`WETLAND_*` color constants) and glyph constants
("wetland dawn" aesthetic). Imported by dashboard, charts, and configure.

**`sysmon.py`** — `sample_system() -> SystemSnapshot`: CPU %, RAM %, per-core
load. Polled on a side thread by `loop._monitor_loop`.

## Subpackages

**`charts/`** — Custom `rich` renderables (braille canvas, convergence chart,
histogram, eval inset).
See [`charts/INDEX.md`](charts/INDEX.md).

**`configure/`** — Interactive "FLIGHT PLAN" configurator
(`python -m wingspan.training --config`).
See [`configure/INDEX.md`](configure/INDEX.md).
