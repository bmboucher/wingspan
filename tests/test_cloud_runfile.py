"""Tests for the cloud run-file (the single-YAML config) and artifact naming.

Covers that ``parse_run_file`` validates a YAML document into a ``CloudRunFile``
whose top-level ``run_name`` is made authoritative over the embedded
``TrainConfig`` (driving the local scratch dir), and that the new artifact-name
helpers / S3-URI helper behave as the runner and monitor rely on.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("boto3")

from wingspan.cloud import runfile, s3sync
from wingspan.training import artifacts

_SAMPLE_YAML = """
run_name: testrun
s3:
  bucket: my-bucket
  prefix: runs
  region: us-east-1
sync:
  status_interval_seconds: 5
  checkpoint_upload_iters: 2
train:
  run:
    games_per_iter: 16
    max_iterations: 4
    target_iterations: 4
    eval_games: 8
"""


def test_parse_run_file_injects_run_name_and_scratch_dir() -> None:
    run = runfile.parse_run_file(_SAMPLE_YAML)
    assert run.run_name == "testrun"
    assert run.s3 is not None
    assert run.s3.bucket == "my-bucket"
    assert run.sync.checkpoint_upload_iters == 2
    # run_name is authoritative over the train block; checkpoint_dir defaults to
    # a per-run scratch dir under the container workdir.
    assert run.train.run.run_name == "testrun"
    assert run.train.run.checkpoint_dir == "/work/testrun"
    assert run.train.run.games_per_iter == 16
    assert run.train.run.target_iterations == 4


def test_parse_run_file_preserves_explicit_checkpoint_dir() -> None:
    # 4-space indent nests under the train.run section (a RunSettings field).
    text = _SAMPLE_YAML + "    checkpoint_dir: /data/custom\n"
    run = runfile.parse_run_file(text)
    assert run.train.run.checkpoint_dir == "/data/custom"


def test_parse_run_file_requires_bucket() -> None:
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        runfile.parse_run_file("run_name: x\ns3:\n  prefix: runs\n")


def test_is_s3_uri() -> None:
    assert s3sync.is_s3_uri("s3://bucket/key.yaml")
    assert not s3sync.is_s3_uri("/local/path/run.yaml")
    assert not s3sync.is_s3_uri("run.yaml")


def test_final_artifact_names() -> None:
    assert artifacts.final_ckpt_name(1_000_000) == "final_1_000_000.pt"
    assert artifacts.final_eval_name(1_000_000) == "final_eval_1_000_000.json"
    assert artifacts.STATUS_JSON == "status.json"
    assert artifacts.GAMES_SUBDIR == "games"
