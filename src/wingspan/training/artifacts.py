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

# Human-readable JSON sidecars (``wingspan.training.runmeta``).
#
# ≥0.5 runs write a single dated unified file per session:
RUN_CONFIG_PREFIX = (
    "run_config_"  # unified config name stem → ``run_config_<stamp>.json``
)
RUN_CONFIG_GLOB = "run_config_*.json"  # the dated per-session unified config records
#
# ≤0.4 legacy files (still read for backward compat; no longer written):
MODEL_CONFIG_JSON = "model_config.json"  # weight-compatibility descriptor
SETUP_CONFIG_JSON_LEGACY = "setup_config.json"  # (alias: same as SETUP_CONFIG_JSON)
PROCESS_PREFIX = "process_"  # session record name stem -> ``process_<stamp>.json``
PROCESS_GLOB = "process_*.json"  # the dated per-session process records
#
# Written for all versions:
INSPECT_REPORT_JSON = "model_inspect.json"  # encoding + parameter breakdown sidecar
MODEL_SUMMARY_HTML = "model_summary.html"  # standalone browser-readable model report

# Compact monitoring snapshot (``wingspan.cloud.status``): a tiny JSON the cloud
# runner refreshes frequently and the monitor reads, so a run's progress is
# legible without torch-loading ``last.pt``. The per-game log's S3 chunks live
# under this subfolder of the checkpoint dir (mirroring the S3 ``games/`` prefix).
STATUS_JSON = "status.json"  # one RunStatus snapshot (overwritten each refresh)
GAMES_SUBDIR = "games"  # immutable per-game-log chunks, grouped by session

# Setup-model artifacts (only written when ``RunConfig.architecture.use_setup_model``):
# the setup net's resumable checkpoint, its weight-compatibility descriptor, and the
SETUP_CKPT = "setup.pt"  # setup net + optimizer state
SETUP_CONFIG_JSON = "setup_config.json"  # setup-net shape descriptor (legacy ≤0.4)

# The subfolder under ``checkpoint_dir`` where a finished run's artifacts are
# moved when the configurator archives it before a fresh run.
ARCHIVE_SUBDIR = "archive"

# Glob for the per-run log files ``app`` writes (``{run_name}.log``) — swept
# alongside the checkpoints when a run is archived.
LOG_GLOB = "*.log"

# Suffix ``loop._atomic_save`` appends for its write-then-replace temp file; a
# crash mid-write can leave one behind, so the archive sweep clears them too.
TMP_GLOB = "*.tmp"


def final_ckpt_name(iteration: int) -> str:
    """Return the filename for a target-milestone final checkpoint.

    Uses Python's underscore thousands-separator so the iteration count is
    immediately legible: ``final_1_000_000.pt`` for iteration 1 000 000.
    """
    return f"final_{iteration:_}.pt"


def final_eval_name(iteration: int) -> str:
    """Return the filename for a target-milestone final-eval result.

    The :class:`~wingspan.training.metrics.FinalEvalStats` JSON written beside
    the ``final_<n>.pt`` checkpoint, so the large fixed-model evaluation a run
    lands on is a durable artifact (uploaded to its own S3 object) rather than a
    dashboard-only readout. Same underscore-separated naming as
    :func:`final_ckpt_name`: ``final_eval_1_000_000.json``.
    """
    return f"final_eval_{iteration:_}.json"
