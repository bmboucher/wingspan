"""The single YAML run-file that configures one training run.

A run is fully described by one YAML document, parsed here into
``CloudRunFile``: the training hyperparameters (``train``, a verbatim
:class:`~wingspan.training.config.TrainConfig`), optionally where artifacts are
persisted in S3 (``s3``), and how often they are offloaded (``sync``). The
``s3`` block is required for cloud (Fargate) runs and omitted for local runs.
The run-file carries no credentials — the container authenticates to S3 through
the standard AWS chain (the Fargate task role in the cloud, environment /
``~/.aws`` locally).

:func:`parse_run_file` turns YAML text into a validated ``CloudRunFile``; reading
that text from a local path or an ``s3://`` URI is the entry point's job
(:mod:`wingspan.cloud.__main__`), so this module stays free of any S3 dependency.
"""

from __future__ import annotations

import pathlib
import typing

import pydantic
import yaml

from wingspan.training import config

# Local scratch layout used when the YAML does not override ``checkpoint_dir``:
# a per-run directory under the container workdir. The durable copy is in S3, so
# this is throwaway space the run reconstructs on startup.
_DEFAULT_WORKDIR = "/work"
_DEFAULT_CHECKPOINT_DIR = "checkpoints"  # mirrors TrainConfig.checkpoint_dir's default


class S3Config(pydantic.BaseModel):
    """Where a run's artifacts live in S3 and how to reach the service.

    No credentials live here by design — they come from the AWS chain (the task
    role on Fargate), so nothing secret is ever written into a run-file.
    ``endpoint_url`` is for pointing at an S3-compatible service such as a local
    MinIO during testing; leave it unset for real AWS.
    """

    bucket: typing.Annotated[str, pydantic.Field(min_length=1)]
    prefix: str = "runs"  # objects live under ``<prefix>/<run_name>/`` in the bucket
    region: str | None = None
    endpoint_url: str | None = None


class SyncConfig(pydantic.BaseModel):
    """How often local artifacts are offloaded to S3.

    Tuned so S3 is never spammed: the tiny status snapshot refreshes on a
    wall-clock interval, the checkpoint set every few iterations, and the
    high-volume per-game log as size-bounded immutable chunks. The training
    loop's per-iteration *local* writes are unchanged — only the upload cadence
    is governed here.
    """

    status_interval_seconds: typing.Annotated[float, pydantic.Field(gt=0.0)] = 30.0
    checkpoint_upload_iters: typing.Annotated[int, pydantic.Field(ge=1)] = 10
    games_chunk_iters: typing.Annotated[int, pydantic.Field(ge=1)] = 25
    games_chunk_mb: typing.Annotated[float, pydantic.Field(gt=0.0)] = 8.0
    download_on_start: bool = True


class CloudRunFile(pydantic.BaseModel):
    """One training run, exactly as the YAML run-file describes it.

    ``s3`` is required for cloud (Fargate) runs.  Omit it for local runs where
    artifacts stay on disk only.
    """

    run_name: typing.Annotated[str, pydantic.Field(min_length=1)]
    s3: S3Config | None = None
    train: config.TrainConfig = pydantic.Field(default_factory=config.TrainConfig)
    sync: SyncConfig = pydantic.Field(default_factory=SyncConfig)

    @pydantic.model_validator(mode="after")
    def _align_train_to_run(self) -> CloudRunFile:
        """Make the top-level ``run_name`` authoritative over the ``train`` block.

        The training loop keys its on-disk artifacts off ``train.run_name`` /
        ``train.checkpoint_dir``; rather than make the user repeat them, derive
        both from the single top-level ``run_name`` (and, for cloud runs, a
        per-run scratch dir under the container workdir when the YAML left the
        default), so one ``run_name`` drives the S3 prefix, the local directory,
        and the run label together.  For local runs (``s3`` absent) the
        ``checkpoint_dir`` default is kept as-is.
        """
        updates: dict[str, str] = {"run_name": self.run_name}
        if self.s3 is not None and self.train.checkpoint_dir == _DEFAULT_CHECKPOINT_DIR:
            scratch = pathlib.PurePosixPath(_DEFAULT_WORKDIR) / self.run_name
            updates["checkpoint_dir"] = str(scratch)
        self.train = self.train.model_copy(update=updates)
        return self


def parse_run_file(text: str) -> CloudRunFile:
    """Parse YAML run-file text into a validated :class:`CloudRunFile`."""
    return CloudRunFile.model_validate(yaml.safe_load(text))
