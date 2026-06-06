# cloud — Containerized training runs + monitor

Headless supervisor (`wingspan cloud`) and the read-only "FLOCK WATCH" monitor
(`wingspan monitor`). Wraps the training loop with S3 persistence so runs can
live on remote compute and be observed locally.

## Modules

**`__init__.py`**, **`__main__.py`** — package entry points.

**`runfile.py`** — `RunFile` Pydantic model: a single YAML file that fully
specifies one cloud run (`run_name`, `checkpoint_dir`, `s3_bucket`, `s3_prefix`,
`train_config`, …). Loaded by the supervisor at startup; validated on parse so
config errors surface before any training begins.

**`runner.py`** — `CloudRunner`: the headless supervisor (`wingspan cloud`).
Reads the `RunFile`, starts the `TrainingLoop` in a background thread, and runs
the `S3Sync` sidecar that pushes checkpoints and metrics to S3 on each
iteration. Handles graceful shutdown on SIGTERM/SIGINT.

**`s3sync.py`** — `S3Sync`: the S3 persistence sidecar. `push(paths)` uploads
the given local paths to `s3://{bucket}/{prefix}/`. `pull_latest(dest)` syncs
the latest checkpoint from S3 to a local directory (used on resume).
Credentials via the ambient AWS environment (IAM role or `~/.aws/`).

**`status.py`** — `RunStatus` Pydantic model: the compact monitoring snapshot
written to `status.json` each iteration (`run_name`, `iteration`, `phase`,
`win_rate`, `games_per_sec`, `last_updated`). Read by the monitor without
needing the full metrics log.

**`monitor.py`** — "FLOCK WATCH" read-only roster (`wingspan monitor`). Reads
`status.json` from each configured S3 prefix and renders a live `rich` table of
all known cloud runs. Refreshes on a configurable interval; no local training
state required.
