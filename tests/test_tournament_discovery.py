"""Tests for on-disk run discovery.

``discover_runs`` must find the active run and its archived runs, skip dirs
without a loadable checkpoint (no ``last.pt`` or no ``model_config.json``), and
never raise on an unreadable checkpoint.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import torch

from wingspan.tournament import participants
from wingspan.training import artifacts


def _make_run(directory: pathlib.Path, iteration: int) -> None:
    """Write a minimal but inspectable run (a readable last.pt + a descriptor)."""
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": {},
        "progress": {"iteration": iteration, "total_games": iteration * 5},
    }
    torch.save(payload, directory / artifacts.LAST_CKPT)
    (directory / artifacts.MODEL_CONFIG_JSON).write_text("{}", encoding="utf-8")


def test_discover_finds_active_and_archived_runs(tmp_path: pathlib.Path) -> None:
    base = tmp_path / "checkpoints"
    _make_run(base, iteration=10)
    _make_run(base / artifacts.ARCHIVE_SUBDIR / "old", iteration=3)

    # An archive with a checkpoint but no descriptor is not loadable -> skipped.
    no_descriptor = base / artifacts.ARCHIVE_SUBDIR / "nodesc"
    no_descriptor.mkdir(parents=True)
    torch.save({"model": {}}, no_descriptor / artifacts.LAST_CKPT)

    found = {option.checkpoint_dir for option in participants.discover_runs(str(base))}
    assert str(base) in found
    assert str(base / artifacts.ARCHIVE_SUBDIR / "old") in found
    assert str(no_descriptor) not in found


def test_discover_never_raises_on_unreadable_checkpoint(tmp_path: pathlib.Path) -> None:
    base = tmp_path / "checkpoints"
    bad = base / artifacts.ARCHIVE_SUBDIR / "corrupt"
    bad.mkdir(parents=True)
    (bad / artifacts.LAST_CKPT).write_bytes(b"not a real checkpoint")
    (bad / artifacts.MODEL_CONFIG_JSON).write_text("{}", encoding="utf-8")

    # Does not raise, and the corrupt run is excluded.
    found = {option.checkpoint_dir for option in participants.discover_runs(str(base))}
    assert str(bad) not in found


def test_discover_empty_dir_returns_nothing(tmp_path: pathlib.Path) -> None:
    assert participants.discover_runs(str(tmp_path / "missing")) == []
