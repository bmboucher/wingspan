"""Shared on-disk artifact names for a training run's checkpoint directory.

These filenames are written by :mod:`wingspan.training.loop` and read /
relocated by :mod:`wingspan.training.configure.runs`, so they live in one
torch-free module rather than being duplicated as private literals in each
(the house rule: public constants shared across modules go in a single file).
They name *artifacts*, not hyperparameters, so they deliberately do not live on
``TrainConfig``.
"""

from __future__ import annotations

# The three checkpoint payloads and the two history logs a run writes, all
# relative to ``TrainConfig.checkpoint_dir``.
LAST_CKPT = "last.pt"  # resumable head: model + optimizer + run progress
BEST_CKPT = "best.pt"  # best eval win-rate snapshot (per opponent generation)
OPPONENT_CKPT = "opponent.pt"  # the frozen "player to beat" (TRAINING.md §7)
METRICS_LOG = "metrics.jsonl"  # one IterationMetrics row per line, appended
GAMES_LOG = "games.jsonl"  # one GameOutcome row per finished game, appended

# Human-readable JSON sidecars (``wingspan.training.runmeta``). The model
# descriptor is one-per-run (rewritten each startup); the process records are
# one-per-session, dated, and accumulate across restarts.
MODEL_CONFIG_JSON = "model_config.json"  # weight-compatibility descriptor
PROCESS_PREFIX = "process_"  # session record name stem -> ``process_<stamp>.json``
PROCESS_GLOB = "process_*.json"  # the dated per-session process records

# The subfolder under ``checkpoint_dir`` where a finished run's artifacts are
# moved when the configurator archives it before a fresh run.
ARCHIVE_SUBDIR = "archive"

# Glob for the per-run log files ``app`` writes (``{run_name}.log``) — swept
# alongside the checkpoints when a run is archived.
LOG_GLOB = "*.log"

# Suffix ``loop._atomic_save`` appends for its write-then-replace temp file; a
# crash mid-write can leave one behind, so the archive sweep clears them too.
TMP_GLOB = "*.tmp"
