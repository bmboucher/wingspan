"""Cloud training: containerized, S3-persisted, interruptible runs + a monitor.

This package wraps :class:`~wingspan.training.loop.TrainingLoop` (it does not
replace it) so a run can execute unattended in a container: configured by a
single YAML run-file (:mod:`runfile`), persisted to S3 (:mod:`s3sync`), surfaced
through a compact status snapshot (:mod:`status`), driven headlessly with
graceful interruption + resume (:mod:`runner`), and watched across all runs by a
read-only roster (:mod:`monitor`).
"""

from __future__ import annotations

from wingspan.cloud import monitor, runfile, runner, s3sync, status
from wingspan.cloud.runfile import CloudRunFile, S3Config, SyncConfig, parse_run_file
from wingspan.cloud.runner import HeadlessRunner
from wingspan.cloud.s3sync import S3Sync
from wingspan.cloud.status import RunStatus, build_status

__all__ = [
    "CloudRunFile",
    "HeadlessRunner",
    "RunStatus",
    "S3Config",
    "S3Sync",
    "SyncConfig",
    "build_status",
    "monitor",
    "parse_run_file",
    "runfile",
    "runner",
    "s3sync",
    "status",
]
