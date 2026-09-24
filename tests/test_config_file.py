"""Tests for ``wingspan.training.config_file`` — loading a full ``RunConfig``
from a standalone ``--config FILE``.

* The four accepted shapes (artifact, defaults envelope, cloud run-file, bare
  dump) each reduce to the expected ``RunConfig``.
* An older-era artifact still validates, at the live ``MODEL_VERSION``.
* Every failure mode (missing file, invalid JSON, non-mapping top level,
  pydantic-invalid config) raises ``ConfigFileError`` naming the path.
"""

from __future__ import annotations

import json
import pathlib

import pytest

pytest.importorskip("torch")
pytest.importorskip("rich")

from wingspan import version
from wingspan.training import config, config_file
from wingspan.training.configure import user_defaults

# --------------------------------------------------------------------------- #
# The four accepted shapes                                                    #
# --------------------------------------------------------------------------- #


def test_load_run_config_bare_dump(tmp_path: pathlib.Path):
    cfg = config.RunConfig(
        misc=config.MiscConfig(collect_device="cpu", train_device="cuda"),
        training=config.TrainingConfig(lr=1e-3),
    )
    path = tmp_path / "bare.json"
    path.write_text(json.dumps(cfg.model_dump(mode="json")), encoding="utf-8")

    loaded = config_file.load_run_config(path)

    assert loaded.training.lr == 1e-3
    assert loaded.misc.collect_device == "cpu" and loaded.misc.train_device == "cuda"


def test_load_run_config_artifact_shape_validates_at_live_version(
    tmp_path: pathlib.Path,
):
    cfg = config.RunConfig(training=config.TrainingConfig(lr=5e-4))
    raw_config = cfg.model_dump(mode="json")
    # A stale prior era stamped into the embedded config — must be stripped and
    # re-derived at the live MODEL_VERSION rather than rejected or preserved.
    raw_config["architecture"]["encoding_version"] = "1.4"
    payload = {
        "version": "1.4",
        "saved_at": "2026-01-01T00:00:00",
        "started_at": "2026-01-01T00:00:00",
        "git_sha": None,
        "resumed": False,
        "resumed_from_iteration": 0,
        "config": raw_config,
    }
    path = tmp_path / "run_config_20260101-000000.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = config_file.load_run_config(path)

    assert loaded.encoding_version == version.MODEL_VERSION
    assert loaded.training.lr == 5e-4


def test_load_run_config_defaults_envelope_shape(tmp_path: pathlib.Path):
    cfg = config.RunConfig(
        misc=config.MiscConfig(collect_device="cuda", train_device="cuda"),
        run=config.RunSettings(checkpoint_dir="somewhere", run_name="saved-run"),
        training=config.TrainingConfig(entropy_coef=0.05),
    )
    path = user_defaults.save_defaults(cfg, directory=tmp_path)

    loaded = config_file.load_run_config(path)

    # Reusable hyperparameters travel...
    assert loaded.training.entropy_coef == 0.05
    # ...but the defaults envelope never carries identity fields (by design —
    # user_defaults strips them), so they read as RunConfig's own defaults.
    default_run = config.RunSettings()
    assert loaded.run.checkpoint_dir == default_run.checkpoint_dir
    assert loaded.run.run_name == default_run.run_name
    assert loaded.misc.collect_device == "cpu" and loaded.misc.train_device == "cpu"


def test_load_run_config_cloud_shape_yaml(tmp_path: pathlib.Path):
    yaml_text = """
run_name: cloudrun
train:
  run:
    games_per_iter: 64
  training:
    lr: 0.001
  misc:
    collect_device: cpu
    train_device: cpu
s3:
  bucket: some-bucket
sync:
  status_interval_seconds: 10
"""
    path = tmp_path / "cloud.yaml"
    path.write_text(yaml_text, encoding="utf-8")

    loaded = config_file.load_run_config(path)

    # The top-level run_name is authoritative over train.run.run_name.
    assert loaded.run.run_name == "cloudrun"
    assert loaded.run.games_per_iter == 64
    assert loaded.training.lr == 0.001


def test_load_run_config_cloud_shape_yml_suffix_and_no_top_level_run_name(
    tmp_path: pathlib.Path,
):
    yaml_text = """
train:
  run:
    run_name: from-train-block
"""
    path = tmp_path / "cloud.yml"
    path.write_text(yaml_text, encoding="utf-8")

    loaded = config_file.load_run_config(path)

    assert loaded.run.run_name == "from-train-block"


# --------------------------------------------------------------------------- #
# Failure modes                                                               #
# --------------------------------------------------------------------------- #


def test_load_run_config_missing_file(tmp_path: pathlib.Path):
    path = tmp_path / "nope.json"
    with pytest.raises(config_file.ConfigFileError) as excinfo:
        config_file.load_run_config(path)
    assert str(path) in str(excinfo.value)


def test_load_run_config_invalid_json(tmp_path: pathlib.Path):
    path = tmp_path / "broken.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(config_file.ConfigFileError) as excinfo:
        config_file.load_run_config(path)
    assert str(path) in str(excinfo.value)


def test_load_run_config_non_mapping_top_level(tmp_path: pathlib.Path):
    path = tmp_path / "list.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(config_file.ConfigFileError) as excinfo:
        config_file.load_run_config(path)
    assert str(path) in str(excinfo.value)


def test_load_run_config_pydantic_invalid(tmp_path: pathlib.Path):
    path = tmp_path / "invalid.json"
    # lr must be strictly positive (gt=0.0).
    path.write_text(
        json.dumps({"training": {"lr": -1.0}}),
        encoding="utf-8",
    )
    with pytest.raises(config_file.ConfigFileError) as excinfo:
        config_file.load_run_config(path)
    assert str(path) in str(excinfo.value)
