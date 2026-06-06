"""Tests for on-disk run discovery.

``discover_runs`` must find the active run and its archived runs, skip dirs
without a loadable checkpoint (no ``last.pt``, no ``model_config.json``, or a
descriptor whose encoding no longer matches the live encoder), and never raise
on an unreadable checkpoint.
"""

from __future__ import annotations

import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import pytest
import torch

from wingspan import version
from wingspan.tournament import participants
from wingspan.training import artifacts, config, runmeta


def _make_run(directory: pathlib.Path, iteration: int) -> None:
    """Write a minimal but inspectable run (a readable last.pt + a descriptor)."""
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": {},
        "progress": {"iteration": iteration, "total_games": iteration * 5},
        "version": version.MODEL_VERSION,
    }
    torch.save(payload, directory / artifacts.LAST_CKPT)
    _write_descriptor(directory)


def _write_descriptor(directory: pathlib.Path, *, choice_dim_offset: int = 0) -> None:
    """Write a ``model_config.json`` matching the live encoder, optionally with
    a deliberately stale ``choice_dim`` (offset by ``choice_dim_offset``).

    Stamped with the current artifact version, exactly like the production
    writer (``runmeta.write_model_config``) — a version-less descriptor would
    read as the pre-0.1 era and be expected to carry the frozen ``compat.v0_0``
    dims instead of the live ones."""
    cfg = config.TrainConfig()
    descriptor = runmeta.ModelConfig(
        run_name=cfg.run_name,
        state_dim=cfg.state_dim,
        choice_dim=cfg.choice_dim + choice_dim_offset,
        family_order=cfg.family_order,
        architecture=cfg.arch,
        include_setup=cfg.encoding_spec.include_setup,
        version=version.MODEL_VERSION,
    )
    (directory / artifacts.MODEL_CONFIG_JSON).write_text(
        descriptor.model_dump_json(), encoding="utf-8"
    )


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


def test_stale_encoding_not_discovered_and_fails_seating(
    tmp_path: pathlib.Path,
) -> None:
    """A run whose descriptor was written against a different encoding layout
    (here ``choice_dim`` off by 3) is not offered by discovery, and naming it
    directly fails with a clear ``ValueError`` at seating instead of a
    mid-game tensor-shape error."""
    base = tmp_path / "checkpoints"
    _make_run(base, iteration=2)

    stale = base / artifacts.ARCHIVE_SUBDIR / "stale"
    stale.mkdir(parents=True)
    torch.save({"model": {}, "progress": {"iteration": 1}}, stale / artifacts.LAST_CKPT)
    _write_descriptor(stale, choice_dim_offset=-3)

    found = {option.checkpoint_dir for option in participants.discover_runs(str(base))}
    assert str(base) in found
    assert str(stale) not in found

    spec = participants.spec_from_dir(str(stale))
    with pytest.raises(ValueError, match="encoding layout"):
        participants.load_player(spec, torch.device("cpu"), random.Random(0))
