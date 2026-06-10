"""Tests for the bootstrap-opponent feature (Plan 03).

Covers:

1. ``TrainConfig`` validation — the three constraint cases.
2. An in-process worker game vs a fresh tiny checkpoint.
3. A cross-version worker game using the pinned v0.1 fixture.
4. ``OptionalPathField`` parse / format round-trip and ``visible_when`` gate.
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
# 1. Config validators


def test_config_rejects_bootstrap_without_initial_vs_random() -> None:
    with pytest.raises(Exception, match="requires initial_vs_random"):
        config.TrainConfig(
            bootstrap_opponent_checkpoint="some/path.pt",
            initial_vs_random=False,
        )


def test_config_rejects_bootstrap_on_cuda() -> None:
    with pytest.raises(Exception, match="requires device='cpu'"):
        config.TrainConfig(
            bootstrap_opponent_checkpoint="some/path.pt",
            initial_vs_random=True,
            device="cuda",
        )


def test_config_accepts_none_bootstrap_with_any_settings() -> None:
    # None bootstrap_opponent_checkpoint is always valid regardless of other flags.
    cfg_default = config.TrainConfig(bootstrap_opponent_checkpoint=None)
    assert cfg_default.bootstrap_opponent_checkpoint is None

    cfg_no_random = config.TrainConfig(
        bootstrap_opponent_checkpoint=None, initial_vs_random=False
    )
    assert cfg_no_random.bootstrap_opponent_checkpoint is None


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
        bootstrap_opponent_checkpoint=str(ckpt_path),
        initial_vs_random=True,
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
        bootstrap_opponent_checkpoint=str(opponent_pt),
        initial_vs_random=True,
    )
    collector = mp_collect.ProcessCollector(bootstrap_cfg, num_workers=2)
    try:
        records = collector.collect_games(net, device, [301], vs_random=True)
    finally:
        collector.close()

    assert len(records) == 1
    assert records[0].winner in (-1, 0, 1)


# ---------------------------------------------------------------------------
# 4. OptionalPathField round-trip and visible_when


_BOOTSTRAP_FIELD_SPEC = fields.OptionalPathField(
    attr="bootstrap_opponent_checkpoint",
    label="bootstrap opponent checkpoint",
    section=fields.ConfigSection.EVAL,
    group="bootstrap",
    none_label="random agent",
    help="Path to a .pt.gz checkpoint used as the bootstrap opponent.",
)


def test_optional_path_field_roundtrip(tmp_path: pathlib.Path) -> None:
    """Parsing a path string and formatting it back yields the original value."""
    cfg = config.TrainConfig(
        device="cpu",
        checkpoint_dir=str(tmp_path),
        bootstrap_opponent_checkpoint="path/to/checkpoint.pt.gz",
        initial_vs_random=True,
    )
    formatted = fields.format_value(cfg, _BOOTSTRAP_FIELD_SPEC)
    assert formatted == "path/to/checkpoint.pt.gz"


def test_optional_path_field_none_displays_as_label(tmp_path: pathlib.Path) -> None:
    """When the field is ``None`` it displays as the configured none_label."""
    cfg = config.TrainConfig(
        device="cpu",
        checkpoint_dir=str(tmp_path),
        bootstrap_opponent_checkpoint=None,
    )
    formatted = fields.format_value(cfg, _BOOTSTRAP_FIELD_SPEC)
    assert formatted == "random agent"


def test_optional_path_field_none_text_parses_to_none(tmp_path: pathlib.Path) -> None:
    """Typing 'none' or clearing the field resets the config field to ``None``."""
    # Start with a checkpoint path set.
    cfg_with_path = config.TrainConfig(
        device="cpu",
        checkpoint_dir=str(tmp_path),
        bootstrap_opponent_checkpoint="some/path.pt",
        initial_vs_random=True,
    )
    # Committing "none" should clear the field.
    new_cfg, error = fields.commit(cfg_with_path, _BOOTSTRAP_FIELD_SPEC, "none")
    assert error is None
    assert new_cfg.bootstrap_opponent_checkpoint is None

    # Committing an empty string also clears the field.
    new_cfg2, error2 = fields.commit(cfg_with_path, _BOOTSTRAP_FIELD_SPEC, "")
    assert error2 is None
    assert new_cfg2.bootstrap_opponent_checkpoint is None


def test_bootstrap_opponent_hidden_when_initial_vs_random_false(
    tmp_path: pathlib.Path,
) -> None:
    """The bootstrap opponent checkpoint field is hidden when initial_vs_random
    is False — the field's ``visible_when`` predicate returns False."""
    cfg_no_random = config.TrainConfig(
        device="cpu",
        checkpoint_dir=str(tmp_path),
        initial_vs_random=False,
    )
    visible_attrs = fields.editable_attrs(cfg_no_random)
    assert "bootstrap_opponent_checkpoint" not in visible_attrs


def test_bootstrap_opponent_visible_when_initial_vs_random_true(
    tmp_path: pathlib.Path,
) -> None:
    """The bootstrap opponent checkpoint field is visible when initial_vs_random
    is True."""
    cfg_random = config.TrainConfig(
        device="cpu",
        checkpoint_dir=str(tmp_path),
        initial_vs_random=True,
    )
    visible_attrs = fields.editable_attrs(cfg_random)
    assert "bootstrap_opponent_checkpoint" in visible_attrs


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
            bootstrap_opponent_checkpoint=str(tmp_path / "nonexistent.pt"),
            initial_vs_random=True,
        )

    with pytest.raises(FileNotFoundError):
        loop_resume.validate_bootstrap_opponent(_FakeLoop())  # type: ignore[arg-type]


def test_validate_bootstrap_opponent_noop_when_none(tmp_path: pathlib.Path) -> None:
    """``validate_bootstrap_opponent`` is a no-op when the checkpoint is None."""

    class _FakeLoop:
        config = config.TrainConfig(
            device="cpu",
            checkpoint_dir=str(tmp_path),
            bootstrap_opponent_checkpoint=None,
        )

    loop_resume.validate_bootstrap_opponent(_FakeLoop())  # type: ignore[arg-type]
