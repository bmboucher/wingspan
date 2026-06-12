"""Tests for the bootstrap-opponent feature.

Covers:

1. ``TrainConfig`` validation — constraint cases and property derivations.
2. An in-process worker game vs a fresh tiny checkpoint.
3. A cross-version worker game using the pinned v0.1 fixture.
4. ``BootstrapField`` parse / format round-trip and ``visible_when`` gate.
5. Fail-fast validation on a missing path.
"""

from __future__ import annotations

import gzip
import pathlib
import sys
import typing

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from wingspan import model  # noqa: E402
from wingspan.training import collect, config, loop_resume, mp_collect  # noqa: E402
from wingspan.training.configure import fields  # noqa: E402

# Pinned v0.1 compat fixture used for cross-version test.
_FIXTURE_DIR = pathlib.Path(__file__).parent / "data" / "compat" / "v0.1"

# Small network dims shared by the in-process bootstrap-opponent tests so
# worker spawn/broadcast stays cheap.
_SMALL_LAYERS = (32, 32)
_SMALL_CARD_EMBED_DIM = 16
_SMALL_CARD_ENCODER_LAYERS = (32,)


# ---------------------------------------------------------------------------
# Helpers


def _small_config(tmp_path: pathlib.Path) -> config.TrainConfig:
    return config.TrainConfig(
        device="cpu",
        checkpoint_dir=str(tmp_path),
        trunk_layers=_SMALL_LAYERS,
        choice_layers=_SMALL_LAYERS,
        card_embed_dim=_SMALL_CARD_EMBED_DIM,
        card_encoder_layers=_SMALL_CARD_ENCODER_LAYERS,
    )


def _small_net(cfg: config.TrainConfig) -> model.PolicyValueNet:
    net = model.PolicyValueNet(arch=cfg.arch, spec=cfg.encoding_spec)
    net.eval()
    return net


def _save_checkpoint(
    net: model.PolicyValueNet, cfg: config.TrainConfig, path: pathlib.Path
) -> None:
    """Save a minimal self-describing checkpoint that ``loaders.load_policy_net`` accepts."""
    import wingspan.version as version_module

    payload: dict[str, typing.Any] = {
        "version": version_module.MODEL_VERSION,
        "config": cfg.model_dump(),
        "model": net.state_dict(),
    }
    torch.save(payload, path)


def _gunzip_to_temp(gz_path: pathlib.Path, dest: pathlib.Path) -> None:
    """Decompress ``gz_path`` (gzip) into ``dest``."""
    dest.write_bytes(gzip.decompress(gz_path.read_bytes()))


# ---------------------------------------------------------------------------
# 1. Config validators and property derivations


def test_bootstrap_opponent_defaults_to_random() -> None:
    cfg = config.TrainConfig()
    assert cfg.bootstrap_opponent == "random"
    assert cfg.initial_vs_random is True
    assert cfg.bootstrap_opponent_checkpoint is None


def test_bootstrap_opponent_none_disables_bootstrap() -> None:
    cfg = config.TrainConfig(bootstrap_opponent="none")
    assert cfg.initial_vs_random is False
    assert cfg.bootstrap_opponent_checkpoint is None


def test_bootstrap_opponent_path_sets_checkpoint() -> None:
    cfg = config.TrainConfig(bootstrap_opponent="some/path.pt")
    assert cfg.initial_vs_random is True
    assert cfg.bootstrap_opponent_checkpoint == "some/path.pt"


def test_config_rejects_bootstrap_path_on_cuda() -> None:
    with pytest.raises(Exception, match="requires device='cpu'"):
        config.TrainConfig(bootstrap_opponent="some/path.pt", device="cuda")


def test_config_accepts_none_and_random_on_any_device() -> None:
    # "none" and "random" never require a specific device.
    cfg_none = config.TrainConfig(bootstrap_opponent="none", device="cuda")
    assert cfg_none.bootstrap_opponent == "none"

    cfg_random = config.TrainConfig(bootstrap_opponent="random", device="cuda")
    assert cfg_random.bootstrap_opponent == "random"


# ---------------------------------------------------------------------------
# 2. In-process worker game vs a tiny saved checkpoint


def test_worker_game_vs_bootstrap_opponent(tmp_path: pathlib.Path) -> None:
    """A game via mp_collect.ProcessCollector completes when a bootstrap
    checkpoint is configured — the workers load the opponent and play seat 1."""
    device = torch.device("cpu")
    cfg = _small_config(tmp_path)
    net = _small_net(cfg)

    # Save a self-describing checkpoint the workers can load.
    ckpt_path = tmp_path / "opponent.pt"
    _save_checkpoint(net, cfg, ckpt_path)

    bootstrap_cfg = config.TrainConfig(
        device="cpu",
        checkpoint_dir=str(tmp_path),
        trunk_layers=_SMALL_LAYERS,
        choice_layers=_SMALL_LAYERS,
        card_embed_dim=_SMALL_CARD_EMBED_DIM,
        card_encoder_layers=_SMALL_CARD_ENCODER_LAYERS,
        bootstrap_opponent=str(ckpt_path),
    )
    collector = mp_collect.ProcessCollector(bootstrap_cfg, num_workers=2)
    try:
        records = collector.collect_games(net, device, [201, 202], vs_random=True)
    finally:
        collector.close()

    assert len(records) == 2
    assert all(isinstance(record, collect.GameRecord) for record in records)
    assert all(record.winner in (-1, 0, 1) for record in records)


# ---------------------------------------------------------------------------
# 3. Cross-version worker game using the v0.1 compat fixture


def test_worker_game_vs_v0_1_bootstrap_opponent(tmp_path: pathlib.Path) -> None:
    """Workers can load a v0.1 checkpoint (via the loaders shim dispatch)
    as the bootstrap opponent and complete a full game."""
    gz_path = _FIXTURE_DIR / "last.pt.gz"
    if not gz_path.exists():
        pytest.skip("v0.1 fixture not present (LFS not pulled)")

    # Gunzip the fixture to a plain .pt file the loader can open.
    opponent_pt = tmp_path / "v0_1_last.pt"
    _gunzip_to_temp(gz_path, opponent_pt)

    # The training config for the current run does NOT need to match the
    # v0.1 checkpoint's architecture — load_policy_net builds the net from the
    # checkpoint's own topology and the version-routing compat shim handles the
    # card-encoder shape mismatch.
    device = torch.device("cpu")
    cfg = _small_config(tmp_path)
    net = _small_net(cfg)

    bootstrap_cfg = config.TrainConfig(
        device="cpu",
        checkpoint_dir=str(tmp_path),
        trunk_layers=_SMALL_LAYERS,
        choice_layers=_SMALL_LAYERS,
        card_embed_dim=_SMALL_CARD_EMBED_DIM,
        card_encoder_layers=_SMALL_CARD_ENCODER_LAYERS,
        bootstrap_opponent=str(opponent_pt),
    )
    collector = mp_collect.ProcessCollector(bootstrap_cfg, num_workers=2)
    try:
        records = collector.collect_games(net, device, [301], vs_random=True)
    finally:
        collector.close()

    assert len(records) == 1
    assert records[0].winner in (-1, 0, 1)


# ---------------------------------------------------------------------------
# 4. BootstrapField parse / format round-trip and visible_when gate

_BOOTSTRAP_FIELD_SPEC = fields.BootstrapField(
    attr="bootstrap_opponent",
    label="bootstrap opponent",
    section=fields.ConfigSection.EVAL,
    group="bootstrap",
    help="Bootstrap phase opponent.",
)


def test_bootstrap_field_formats_fixed_values() -> None:
    """'none' and 'random' pass through format_value unchanged."""
    cfg_none = config.TrainConfig(bootstrap_opponent="none")
    assert fields.format_value(cfg_none, _BOOTSTRAP_FIELD_SPEC) == "none"

    cfg_random = config.TrainConfig(bootstrap_opponent="random")
    assert fields.format_value(cfg_random, _BOOTSTRAP_FIELD_SPEC) == "random"


def test_bootstrap_field_formats_path_as_last_two_parts() -> None:
    """A path value is displayed as its last two components."""
    cfg = config.TrainConfig(bootstrap_opponent="some/archive/run_iter1000/last.pt")
    formatted = fields.format_value(cfg, _BOOTSTRAP_FIELD_SPEC)
    assert formatted == "run_iter1000/last.pt"


def test_bootstrap_field_commit_roundtrip(tmp_path: pathlib.Path) -> None:
    """Committing a path string stores it verbatim in bootstrap_opponent."""
    cfg = config.TrainConfig(device="cpu", checkpoint_dir=str(tmp_path))
    new_cfg, error = fields.commit(
        cfg, _BOOTSTRAP_FIELD_SPEC, "path/to/checkpoint.pt.gz"
    )
    assert error is None
    assert new_cfg.bootstrap_opponent == "path/to/checkpoint.pt.gz"


def test_bootstrap_field_commit_empty_rejects(tmp_path: pathlib.Path) -> None:
    """Committing an empty string should return an error."""
    cfg = config.TrainConfig(device="cpu", checkpoint_dir=str(tmp_path))
    _, error = fields.commit(cfg, _BOOTSTRAP_FIELD_SPEC, "")
    assert error is not None


def test_bootstrap_field_commit_none_string(tmp_path: pathlib.Path) -> None:
    """Committing 'none' sets bootstrap_opponent to 'none'."""
    cfg = config.TrainConfig(
        device="cpu", checkpoint_dir=str(tmp_path), bootstrap_opponent="random"
    )
    new_cfg, error = fields.commit(cfg, _BOOTSTRAP_FIELD_SPEC, "none")
    assert error is None
    assert new_cfg.bootstrap_opponent == "none"


def test_graduate_hidden_when_bootstrap_none(tmp_path: pathlib.Path) -> None:
    """The 'graduate @' field is hidden when bootstrap_opponent is 'none'."""
    cfg = config.TrainConfig(
        device="cpu", checkpoint_dir=str(tmp_path), bootstrap_opponent="none"
    )
    visible_attrs = fields.editable_attrs(cfg)
    assert "random_phase_win_rate" not in visible_attrs


def test_graduate_visible_when_bootstrap_active(tmp_path: pathlib.Path) -> None:
    """The 'graduate @' field is visible when bootstrap_opponent is 'random' or a path."""
    cfg_random = config.TrainConfig(
        device="cpu", checkpoint_dir=str(tmp_path), bootstrap_opponent="random"
    )
    assert "random_phase_win_rate" in fields.editable_attrs(cfg_random)

    cfg_path = config.TrainConfig(
        device="cpu", checkpoint_dir=str(tmp_path), bootstrap_opponent="some/path.pt"
    )
    assert "random_phase_win_rate" in fields.editable_attrs(cfg_path)


def test_bootstrap_opponent_always_in_editable_attrs(tmp_path: pathlib.Path) -> None:
    """The bootstrap_opponent field is always navigable regardless of its value."""
    for value in ("none", "random", "some/path.pt"):
        cfg = config.TrainConfig(
            device="cpu", checkpoint_dir=str(tmp_path), bootstrap_opponent=value
        )
        assert "bootstrap_opponent" in fields.editable_attrs(cfg)


# ---------------------------------------------------------------------------
# 5. Fail-fast validation on missing path


def test_validate_bootstrap_opponent_raises_on_missing_file(
    tmp_path: pathlib.Path,
) -> None:
    """``validate_bootstrap_opponent`` propagates the FileNotFoundError from
    ``loaders.load_policy_net`` when the path does not exist."""

    class _FakeLoop:
        config = config.TrainConfig(
            device="cpu",
            checkpoint_dir=str(tmp_path),
            bootstrap_opponent=str(tmp_path / "nonexistent.pt"),
        )

    with pytest.raises(FileNotFoundError):
        loop_resume.validate_bootstrap_opponent(_FakeLoop())  # type: ignore[arg-type]


def test_validate_bootstrap_opponent_noop_when_random(tmp_path: pathlib.Path) -> None:
    """``validate_bootstrap_opponent`` is a no-op when bootstrap_opponent is 'random'."""

    class _FakeLoop:
        config = config.TrainConfig(
            device="cpu",
            checkpoint_dir=str(tmp_path),
            bootstrap_opponent="random",
        )

    loop_resume.validate_bootstrap_opponent(_FakeLoop())  # type: ignore[arg-type]


def test_validate_bootstrap_opponent_noop_when_none(tmp_path: pathlib.Path) -> None:
    """``validate_bootstrap_opponent`` is a no-op when bootstrap_opponent is 'none'."""

    class _FakeLoop:
        config = config.TrainConfig(
            device="cpu",
            checkpoint_dir=str(tmp_path),
            bootstrap_opponent="none",
        )

    loop_resume.validate_bootstrap_opponent(_FakeLoop())  # type: ignore[arg-type]
