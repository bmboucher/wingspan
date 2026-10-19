"""Tests for the bundled-release machinery: ``wingspan.release``,
``wingspan.release_prep``, and the ``release`` / fallback player specs.

The bundled-artifact tests run against the real files shipped under
``src/wingspan/data/release/``; the strip tests build a tiny checkpoint in
``tmp_path``.
"""

from __future__ import annotations

import json
import pathlib
import typing

import pytest
import torch

from wingspan import engine, model, players, release, release_prep, version
from wingspan.training import artifacts, collect, config, policy, runstate

_DEVICE = torch.device("cpu")
_SMALL_LAYERS = (32, 32)
_SMALL_HEAD_LAYERS = (16,)
_SMALL_CARD_EMBED_DIM = 8
_SMOKE_SEED = 7


###### HELPERS #######


def _tiny_config() -> config.RunConfig:
    return config.RunConfig(
        architecture=config.ArchitectureConfig(
            main=config.MainNetArchitecture(
                trunk_layers=_SMALL_LAYERS,
                choice_layers=_SMALL_LAYERS,
                head_layers=_SMALL_HEAD_LAYERS,
                card_embed_dim=_SMALL_CARD_EMBED_DIM,
            ),
        ),
    )


def _write_tiny_checkpoint(directory: pathlib.Path) -> config.RunConfig:
    """Write a ``best.pt`` with the full training payload shape (including the
    keys the release tool strips)."""
    cfg = _tiny_config()
    net = model.PolicyValueNet(
        state_dim=cfg.state_dim,
        choice_dim=cfg.choice_dim,
        num_families=len(cfg.family_order),
        arch=cfg.arch,
        spec=cfg.encoding_spec,
    )
    optimizer = torch.optim.Adam(net.parameters())
    progress = runstate.RunProgress(iteration=12)
    payload: dict[str, typing.Any] = {
        "config": cfg.model_dump(),
        "model": net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "progress": progress.model_dump(),
        "git_sha": "abc1234",
        "version": cfg.encoding_version,
        "metrics": {"anything": 1},
    }
    torch.save(payload, directory / artifacts.BEST_CKPT)
    return cfg


###### TESTS #######


def test_strip_round_trip(tmp_path: pathlib.Path) -> None:
    src = tmp_path / "2026.01.01-run"
    src.mkdir()
    cfg = _write_tiny_checkpoint(src)
    out = tmp_path / "out"
    out.mkdir()

    stripped = release_prep.strip_payload(
        src / artifacts.BEST_CKPT,
        out / artifacts.LAST_CKPT,
        ("optimizer", "metrics"),
    )
    assert "optimizer" not in stripped and "metrics" not in stripped

    reloaded = torch.load(out / artifacts.LAST_CKPT, weights_only=False)
    assert "optimizer" not in reloaded and "metrics" not in reloaded
    assert {"config", "model", "version", "git_sha", "progress"} <= set(reloaded)

    release_prep.verify(out)  # no setup.pt: only the policy net round-trips
    net, loaded_cfg = players.load_policy_net(out / artifacts.LAST_CKPT, _DEVICE)
    assert loaded_cfg.state_dim == cfg.state_dim
    assert isinstance(net, model.PolicyValueNet)

    manifest = release_prep.build_manifest(stripped, src, artifacts.BEST_CKPT)
    assert manifest.era == cfg.encoding_version
    assert manifest.run_name == "2026.01.01-run"
    assert manifest.iteration == 12
    assert manifest.git_sha == "abc1234"
    parsed = release.ReleaseManifest.model_validate_json(manifest.model_dump_json())
    assert parsed == manifest


def test_build_manifest_without_progress(tmp_path: pathlib.Path) -> None:
    manifest = release_prep.build_manifest(
        {"version": "1.5"}, tmp_path / "run", artifacts.LAST_CKPT
    )
    assert manifest.iteration is None
    assert manifest.git_sha is None


def test_newest_run_config(tmp_path: pathlib.Path) -> None:
    with pytest.raises(FileNotFoundError):
        release_prep.newest_run_config(tmp_path)
    (tmp_path / "run_config_20260101-000000.json").write_text("{}")
    (tmp_path / "run_config_20260201-000000.json").write_text("{}")
    assert (
        release_prep.newest_run_config(tmp_path).name
        == "run_config_20260201-000000.json"
    )


def test_prepare_end_to_end_no_setup(tmp_path: pathlib.Path) -> None:
    src = tmp_path / "run"
    src.mkdir()
    cfg = _write_tiny_checkpoint(src)
    (src / "run_config_20260101-000000.json").write_text("{}")
    out = tmp_path / "out"

    manifest = release_prep.prepare(src, "best", out)

    assert manifest.era == cfg.encoding_version
    assert (out / "run_config_20260101-000000.json").is_file()
    written = json.loads((out / release.MANIFEST_FILE).read_text())
    assert written["run_name"] == "run"
    assert not (out / artifacts.SETUP_CKPT).exists()


def test_release_spec_points_into_release_dir(tmp_path: pathlib.Path) -> None:
    spec = players.parse_player_spec("release", tmp_path)
    assert spec.kind is players.PlayerKind.MODEL
    assert spec.checkpoint_path == release.release_dir() / "last.pt"
    assert spec.run_dir == release.release_dir()
    assert not spec.bundled_fallback


def test_named_spec_falls_back_to_bundle(tmp_path: pathlib.Path) -> None:
    assert release.has_release()
    for raw in ("last", "best", "opponent"):
        spec = players.parse_player_spec(raw, tmp_path)
        assert spec.bundled_fallback
        assert spec.checkpoint_path == release.release_dir() / "last.pt"
        assert spec.run_dir == release.release_dir()


def test_no_fallback_when_checkpoint_present(tmp_path: pathlib.Path) -> None:
    (tmp_path / artifacts.LAST_CKPT).write_bytes(b"dummy")
    spec = players.parse_player_spec("last", tmp_path)
    assert not spec.bundled_fallback
    assert spec.checkpoint_path == tmp_path / artifacts.LAST_CKPT
    assert spec.run_dir == tmp_path


def test_load_manifest_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path):
    monkeypatch.setattr(release, "release_dir", lambda: tmp_path)
    assert release.load_manifest() is None
    assert not release.has_release()


def test_release_path_not_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """A non-directory at the release path is a hard error (zipped installs
    are unsupported)."""
    bogus = tmp_path / "wingspan"
    (bogus / "data").mkdir(parents=True)
    (bogus / "data" / "release").write_text("not a dir")

    def fake_files(_name: str) -> pathlib.Path:
        return bogus

    monkeypatch.setattr(release.importlib.resources, "files", fake_files)
    with pytest.raises(RuntimeError):
        release.release_dir()


def test_bundled_release_plays_a_game() -> None:
    assert release.has_release()
    directory = release.release_dir()
    net, cfg = players.load_policy_net(directory / artifacts.LAST_CKPT, _DEVICE)
    assert players.load_setup_net(directory, _DEVICE) is not None

    manifest = release.load_manifest()
    assert manifest is not None
    payload = torch.load(directory / artifacts.LAST_CKPT, weights_only=False)
    assert manifest.era == payload["version"]
    assert manifest.era == version.MODEL_VERSION

    num_players = cfg.num_players
    greedy = policy.greedy_agent(net, _DEVICE)
    eng = collect.new_engine(_SMOKE_SEED, num_players)
    engine.Engine.play_one_game(
        eng.state,
        [greedy] * num_players,
        split_setup_bonus=players.resolve_split_setup_bonus([cfg]),
        split_setup_food=players.resolve_split_setup_food([cfg]),
        combine_gain_food=players.resolve_combine_gain_food([cfg]),
    )
    assert all(player.final_score is not None for player in eng.state.players)
