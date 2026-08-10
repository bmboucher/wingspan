# cloud — Containerized training runs + monitor

Headless supervisor (`wingspan cloud`) and the read-only "FLOCK WATCH" monitor
(`wingspan monitor`). Wraps the training loop with S3 persistence so runs can
live on remote compute and be observed locally.

## Modules

**`__init__.py`**, **`__main__.py`** — package entry points.

**`runfile.py`** — `CloudRunFile` Pydantic model: a single YAML file that fully
specifies one cloud run: `run_name`, `s3: S3Config | None` (`bucket`, `prefix`,
`region`, `endpoint_url` — required for cloud runs, omitted for local ones;
carries no credentials), `train: TrainConfig` (verbatim training
hyperparameters; `checkpoint_dir` lives at `train.run.checkpoint_dir`, not as a
flat field), and `sync: SyncConfig` (upload cadence: `status_interval_seconds`
default 30s, `checkpoint_upload_iters` default 10, `games_chunk_iters` default
25, `games_chunk_mb` default 8.0, `download_on_start` default `True`). A
model validator aligns `train.run.run_name` / `train.run.checkpoint_dir` to the
top-level `run_name`. Loaded by the supervisor at startup via `parse_run_file`;
validated on parse so config errors surface before any training begins.

**`runner.py`** — `HeadlessRunner`: the headless supervisor (`wingspan cloud`).
Reads the `CloudRunFile`, starts the `TrainingLoop` in a background thread, and
on the main thread periodically runs the `S3Sync` sidecar: status snapshots
publish on a wall-clock `sync.status_interval_seconds` tick, checkpoint sets
upload every `sync.checkpoint_upload_iters` completed iterations, and game-log
chunks offload once either `sync.games_chunk_mb` of unsent bytes accumulates or
`sync.games_chunk_iters` iterations pass — none of these are per-iteration.
Handles graceful shutdown on SIGTERM/SIGINT (a final sync before exit).

**`s3sync.py`** — `S3Sync`: the S3 persistence sidecar. `upload_file(local_path,
suffix)` uploads one file; `upload_bytes(data, suffix)` uploads raw bytes;
`upload_checkpoint_set(local_dir)` uploads the full checkpoint set in one call;
`offload_game_chunk(...)` streams game-log chunks. `download_run(local_dir)`
syncs the latest checkpoint from S3 to a local directory (used on resume).
`iter_run_statuses(s3_config)` — module-level function that polls all known run
prefixes and yields `RunStatus` snapshots (used by the monitor).
Credentials via the ambient AWS environment (IAM role or `~/.aws/`).

**`status.py`** — `RunStatus` Pydantic model: the compact monitoring snapshot
written to `status.json`, refreshed on the `sync.status_interval_seconds`
wall-clock tick (not per iteration). A tiny projection of the live
`training.runstate.RunState` — ~27 fields including `run_name`, `iteration`,
`phase`, `training_phase`, `pct_complete`, `total_games`, `avg_score`,
`win_rate`, `win_rate_ci95`, `games_per_sec`, `eta_seconds`, `error`,
`finished`, `final_eval`, and the `updated_at` heartbeat (ISO-8601 UTC). Read
by the monitor without needing the full metrics log.

**`monitor.py`** — "FLOCK WATCH" read-only roster (`wingspan monitor`). Reads
`status.json` from each configured S3 prefix and renders a live `rich` table of
all known cloud runs. Refreshes on a configurable interval; no local training
state required.
