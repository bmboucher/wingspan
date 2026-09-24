# pyright: reportPrivateUsage=false
# (exercises app._explicit_dests / app._first_disallowed_flag directly —
# deliberate intra-module coupling, same pattern as test_device_split.py)
"""Tests for ``wingspan dashboard --config FILE [--start]`` wiring in
``wingspan.training.app``:

* Explicit-flag detection (``app._explicit_dests`` / ``_first_disallowed_flag``)
  and the identity-override precedence with ``--config``.
* ``main``'s dispatch: config-screen seeding, headless launch via
  ``configure.controller.prepare_headless_launch``, and every error path
  (usage errors exit 2, launch refusals exit 1).

All tests are CPU-only and monkeypatch ``torch.cuda.is_available`` where the
code under test consults it, and stub out ``configure.run_configurator`` /
``app._run_training`` so no training loop or TUI is ever constructed.
"""

from __future__ import annotations

import json
import pathlib

import pytest

pytest.importorskip("torch")
pytest.importorskip("rich")

import rich.console as rich_console
import torch

from wingspan import version
from wingspan.training import app, artifacts, config, configure, runstate


def _write_checkpoint(directory: pathlib.Path, cfg: config.RunConfig) -> None:
    """A minimal ``last.pt`` — just enough for ``runs.inspect_run`` to read a
    saved config and progress snapshot back out, mirroring the fixture pattern
    in ``tests/test_training_configurator.py``."""
    directory.mkdir(parents=True, exist_ok=True)
    progress = runstate.RunProgress(
        iteration=2, total_games=50, best_win_rate=0.6, opponent_generation=0
    )
    payload = {
        "config": cfg.model_dump(),
        "progress": progress.model_dump(),
        "git_sha": "abc1234",
        "version": cfg.encoding_version,
    }
    torch.save(payload, directory / artifacts.LAST_CKPT)


def _write_config_file(path: pathlib.Path, cfg: config.RunConfig) -> None:
    path.write_text(json.dumps(cfg.model_dump(mode="json")), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Explicit-flag detection                                                     #
# --------------------------------------------------------------------------- #


def test_explicit_dests_only_contains_typed_flags():
    explicit = app._explicit_dests(["--lr", "0.001", "--checkpoint-dir", "x"])
    assert explicit == frozenset({"lr", "checkpoint_dir"})


def test_explicit_dests_empty_for_no_args():
    assert app._explicit_dests([]) == frozenset()


def test_first_disallowed_flag_identifies_non_identity_flag():
    explicit = app._explicit_dests(["--config", "f.json", "--lr", "0.001"])
    assert app._first_disallowed_flag(explicit) == "--lr"


def test_first_disallowed_flag_allows_identity_and_config_flags():
    explicit = app._explicit_dests(
        [
            "--config",
            "f.json",
            "--start",
            "--checkpoint-dir",
            "x",
            "--run-name",
            "y",
            "--collect-device",
            "cpu",
            "--train-device",
            "cuda",
            "--resume",
        ]
    )
    assert app._first_disallowed_flag(explicit) is None


# --------------------------------------------------------------------------- #
# main(): usage errors                                                        #
# --------------------------------------------------------------------------- #


def test_start_without_config_is_a_usage_error(capsys: pytest.CaptureFixture[str]):
    assert app.main(["--start"]) == 2
    err = capsys.readouterr().err
    assert "--start" in err and "--config" in err


def test_config_with_disallowed_flag_is_a_usage_error(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
):
    path = tmp_path / "file.json"
    _write_config_file(path, config.RunConfig())

    exit_code = app.main(["--config", str(path), "--lr", "0.001"])

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "--config" in err and "--lr" in err


def test_config_missing_file_reports_config_file_error(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
):
    missing = tmp_path / "nope.json"
    assert app.main(["--config", str(missing)]) == 2
    err = capsys.readouterr().err
    assert str(missing) in err


# --------------------------------------------------------------------------- #
# main(): --config alone opens the config screen seeded from the file         #
# --------------------------------------------------------------------------- #


def test_config_alone_seeds_configurator_from_file(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    file_cfg = config.RunConfig(training=config.TrainingConfig(lr=2e-4))
    path = tmp_path / "file.json"
    _write_config_file(path, file_cfg)

    calls: list[tuple[config.RunConfig, str | None]] = []

    def fake_run_configurator(
        passed_cfg: config.RunConfig,
        term: rich_console.Console,
        cuda_available: bool,
        seed_file: str | None = None,
    ) -> config.RunConfig | None:
        calls.append((passed_cfg, seed_file))
        return None  # user quits without launching

    monkeypatch.setattr(configure, "run_configurator", fake_run_configurator)

    assert app.main(["--config", str(path)]) == 0
    assert len(calls) == 1
    passed_cfg, seed_file = calls[0]
    assert passed_cfg.training.lr == 2e-4
    assert seed_file == path.name


def test_config_with_identity_overrides(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    file_cfg = config.RunConfig(
        run=config.RunSettings(checkpoint_dir="orig", run_name="orig-name"),
        training=config.TrainingConfig(lr=3e-4),
    )
    path = tmp_path / "file.json"
    _write_config_file(path, file_cfg)

    calls: list[config.RunConfig] = []

    def fake_run_configurator(
        passed_cfg: config.RunConfig,
        term: rich_console.Console,
        cuda_available: bool,
        seed_file: str | None = None,
    ) -> config.RunConfig | None:
        calls.append(passed_cfg)
        return None

    monkeypatch.setattr(configure, "run_configurator", fake_run_configurator)

    exit_code = app.main(
        [
            "--config",
            str(path),
            "--checkpoint-dir",
            "overridden",
            "--run-name",
            "overridden-name",
        ]
    )

    assert exit_code == 0
    passed_cfg = calls[0]
    assert passed_cfg.run.checkpoint_dir == "overridden"
    assert passed_cfg.run.run_name == "overridden-name"
    assert passed_cfg.training.lr == 3e-4  # untouched, from the file


# --------------------------------------------------------------------------- #
# main(): --config --start headless launch                                   #
# --------------------------------------------------------------------------- #


def test_config_start_empty_dir_launches_fresh(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    checkpoint_dir = tmp_path / "run"
    file_cfg = config.RunConfig(
        run=config.RunSettings(checkpoint_dir=str(checkpoint_dir), resume=True)
    )
    path = tmp_path / "file.json"
    _write_config_file(path, file_cfg)

    calls: list[config.RunConfig] = []

    def fake_run_training(
        passed_cfg: config.RunConfig, term: rich_console.Console
    ) -> bool:
        calls.append(passed_cfg)
        return False

    monkeypatch.setattr(app, "_run_training", fake_run_training)

    assert app.main(["--config", str(path), "--start"]) == 0
    assert len(calls) == 1
    launched = calls[0]
    assert launched.run.resume is False
    assert launched.encoding_version == version.MODEL_VERSION


def test_config_start_resumable_dir_resumes(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    checkpoint_dir = tmp_path / "run"
    saved_cfg = config.RunConfig(
        run=config.RunSettings(checkpoint_dir=str(checkpoint_dir))
    )
    _write_checkpoint(checkpoint_dir, saved_cfg)

    file_cfg = config.RunConfig(
        run=config.RunSettings(checkpoint_dir=str(checkpoint_dir), resume=True)
    )
    path = tmp_path / "file.json"
    _write_config_file(path, file_cfg)

    calls: list[config.RunConfig] = []

    def fake_run_training(
        passed_cfg: config.RunConfig, term: rich_console.Console
    ) -> bool:
        calls.append(passed_cfg)
        return False

    monkeypatch.setattr(app, "_run_training", fake_run_training)

    assert app.main(["--config", str(path), "--start"]) == 0
    assert calls[0].run.resume is True


def test_config_start_incompatible_dir_refuses(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    checkpoint_dir = tmp_path / "run"
    saved_cfg = config.RunConfig(
        run=config.RunSettings(checkpoint_dir=str(checkpoint_dir)),
        architecture=config.ArchitectureConfig(
            main=config.MainNetArchitecture(
                trunk_layers=(256, 256), choice_layers=(256, 256)
            )
        ),
    )
    _write_checkpoint(checkpoint_dir, saved_cfg)

    file_cfg = config.RunConfig(
        run=config.RunSettings(checkpoint_dir=str(checkpoint_dir), resume=True)
    )
    path = tmp_path / "file.json"
    _write_config_file(path, file_cfg)

    exit_code = app.main(["--config", str(path), "--start"])

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "archive" in err.lower()


def test_config_start_resumable_dir_with_resume_off_refuses(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    checkpoint_dir = tmp_path / "run"
    saved_cfg = config.RunConfig(
        run=config.RunSettings(checkpoint_dir=str(checkpoint_dir))
    )
    _write_checkpoint(checkpoint_dir, saved_cfg)

    file_cfg = config.RunConfig(
        run=config.RunSettings(checkpoint_dir=str(checkpoint_dir), resume=False)
    )
    path = tmp_path / "file.json"
    _write_config_file(path, file_cfg)

    exit_code = app.main(["--config", str(path), "--start"])

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "archive" in err.lower()


def test_config_start_validate_launchable_failure(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    checkpoint_dir = tmp_path / "run"  # never created — an empty target
    file_cfg = config.RunConfig(
        run=config.RunSettings(checkpoint_dir=str(checkpoint_dir)),
        misc=config.MiscConfig(collect_device="cuda", train_device="cuda"),
        opponent=config.OpponentConfig(bootstrap_opponent="some/path.pt"),
    )
    path = tmp_path / "file.json"
    _write_config_file(path, file_cfg)

    exit_code = app.main(["--config", str(path), "--start"])

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "collect_device='cpu'" in err
