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

**`config.py`** — `TrainConfig`: the self-describing hyperparameter object.
Key fields:
- Loop shape: `games_per_iter`, `max_iterations`, `target_iterations`.
- Optimization: `lr`, `value_coef`, `entropy_coef`, `grad_clip`, `score_norm`,
  `reward_mode` (`RewardMode`: `terminal_margin` | `decision_delta`) and
  `reward_discount` (γ for the decision-delta return). Both REGIME.
- Evaluation: `eval_every`, `eval_games`, `opponent_reset_win_rate`.
- Bootstrap: `initial_vs_random`, `random_phase_win_rate`,
  `bootstrap_opponent_checkpoint` (optional path to a `.pt` checkpoint used
  as the bootstrap-phase opponent instead of the random agent; CPU-only,
  requires `initial_vs_random=True`).
- Architecture: `arch: ModelArchitecture` (assembled via the `arch` property
  from flat fields so `TrainConfig` serializes flat).
- Era: `encoding_version` — the artifact era the run trains at (adopted from
  the run dir on resume, never user-edited); `state_dim` / `choice_dim` are
  era-routed from it (`compat.encoding_dims_for_era`), and it leads
  `architecture_key`. See "Training resume: era pinning" in
  `docs/VERSIONING.md`.
- Derived: `architecture_key`, `state_dim`, `choice_dim`, `encoding_spec`.
- Module functions: `train_config_from_artifact(raw, artifact_version)` —
  validate a payload's embedded config at the payload's own era (pre-field
  configs derive `encoding_version` from the `version` stamp);
  `with_encoding_version(cfg, era)` — validated era-pinned copy.

**`artifacts.py`** — `ArtifactPaths(checkpoint_dir)`: canonical on-disk
filenames. Constants: `LAST_CKPT`, `BEST_CKPT`, `OPPONENT_CKPT`,
`METRICS_LOG`, `GAMES_LOG`, `MODEL_CONFIG`, `PROCESS_JSON`. Used everywhere
that writes or reads from a run directory.

**`runmeta.py`** — Sidecar JSON files and the era-routed descriptor reporting
seam:
- `ModelConfig` — written to `model_config.json`; carries `run_name`,
  `state_dim`, `choice_dim`, `family_order`, `architecture`, `include_setup`,
  `version`.
- `write_model_config(...)`, `read_model_config(path) -> ModelConfig` — the
  sanctioned write/read pair; `read_model_config` applies compat shims by version.
- Reporting seam: `choice_layout_for(descriptor)`, `param_report_for(descriptor,
  net)`, `build_model_summary_html(descriptor, report)` — all route by the
  descriptor's version so compat-era reports are correct without touching the
  live encoder.

**`setup_runmeta.py`** — Analogous sidecar for the setup model:
`SetupConfig`, `write_setup_config`, `read_setup_config`.

## Training loop orchestrator

**`loop.py`** — `TrainingLoop(cfg, pause_at_target)`: the top-level class.
Key members:
- `run()` — main entry (runs on background thread).
- `request_stop()`, `stopped` — graceful shutdown signal.
- `signal_target_response(choice, new_target)` — unblock from a target pause.
- `self.net`, `self.optimizer`, `self.state (RunState)`, `self.lock (RLock)`.

**`loop_resume.py`** — `maybe_resume(loop)`: loads `LAST_CKPT` if present,
validates `architecture_key` (alarm + fresh start on mismatch, including when
the weights themselves fail to load), initializes phase and target, writes the
`process_<stamp>.json` sidecar. `adopt_checkpoint_era(cfg)` — called by
`TrainingLoop.__init__` before the net is built: pins the config to the
resumable checkpoint's era when that adoption is exactly what makes the keys
agree (era-pinned resume across a FRESH change; `docs/VERSIONING.md`).

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
margin_before)` — the recorded self-play transition consumed by the learner.
`margin_before` is the deciding player's running margin (own − opponent) before
the decision, differenced into the `decision_delta` return. Stored as fp16
arrays during IPC.

## Learning

**`learner.py`** — `Learner(net, optimizer, config)`:
- `update(steps: list[Step]) -> LearnerResult` — length-bucketed REINFORCE with
  advantage normalisation. Groups steps by game length, computes returns,
  normalizes advantages, runs one Adam step per bucket.
- `_flatten` pairs each step with its return per `cfg.reward_mode`:
  `_terminal_margin_returns` broadcasts the end-of-game margin; for
  `decision_delta`, `_decision_delta_returns` discounts per-decision
  `margin_before` deltas (γ = `reward_discount`) into each step's return.

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
