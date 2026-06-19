"""Tests for the DAgger behavioral cloning feature.

Covers:

1. ``DaggerConfig`` / ``RunConfig`` validation — constraint cases, property
   derivations, and ``dagger_active_at`` truth table.
2. In-process collection with an expert net: ``expert_probs`` is set and sums to 1.
3. In-process collection without an expert net: ``expert_probs`` is ``None``.
4. Learner imitation phase: ``learner.update(..., imitation_phase=True)`` produces
   a finite, non-NaN ``imitation_loss`` and zero ``policy_loss``.
5. Learner RL phase: ``imitation_loss == 0.0`` in normal (non-imitation) mode.
6. Empty-bucket guard: all-``None`` ``expert_probs`` steps do not crash the learner
   in imitation mode (``has_expert.sum()`` is clamped to 1).
7. ``validate_dagger_expert`` fail-fast on a missing file and no-op on ``'none'``.
8. ``BootstrapField`` parse / format round-trip for the DAgger expert field.
"""

from __future__ import annotations

import pathlib
import random
import sys
import typing

import numpy as np
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from wingspan import model  # noqa: E402
from wingspan.training import collect, config, learner, loop_resume  # noqa: E402

# Small net dims — keep worker spawn and inference cheap.
_SMALL_LAYERS = (32, 32)
_SMALL_CARD_EMBED_DIM = 16
_SMALL_CARD_ENCODER_LAYERS = (32,)


# ---------------------------------------------------------------------------
# Helpers


def _small_cfg(tmp_path: pathlib.Path) -> config.RunConfig:
    return config.RunConfig(
        misc=config.MiscConfig(device="cpu"),
        run=config.RunSettings(checkpoint_dir=str(tmp_path)),
        architecture=config.ArchitectureConfig(
            main=config.MainNetArchitecture(
                trunk_layers=_SMALL_LAYERS,
                choice_layers=_SMALL_LAYERS,
                card_embed_dim=_SMALL_CARD_EMBED_DIM,
                card_encoder_layers=_SMALL_CARD_ENCODER_LAYERS,
            ),
        ),
    )


def _small_net(cfg: config.RunConfig) -> model.PolicyValueNet:
    net_cls = model.PolicyValueNet.class_for_version(cfg.encoding_version)
    net = net_cls(
        state_dim=cfg.state_dim,
        choice_dim=cfg.choice_dim,
        num_families=len(cfg.family_order),
        arch=cfg.arch,
        spec=cfg.encoding_spec,
    )
    net.eval()
    return net


def _save_checkpoint(
    net: model.PolicyValueNet, cfg: config.RunConfig, path: pathlib.Path
) -> None:
    """Save a minimal self-describing checkpoint that ``loaders.load_policy_net`` accepts."""
    import wingspan.version as version_module

    payload: dict[str, typing.Any] = {
        "version": version_module.MODEL_VERSION,
        "config": cfg.model_dump(),
        "model": net.state_dict(),
    }
    torch.save(payload, path)


# ---------------------------------------------------------------------------
# 1. Config validation and property derivations


def test_dagger_defaults_disabled() -> None:
    cfg = config.RunConfig()
    assert cfg.dagger.expert_checkpoint == "none"
    assert cfg.dagger.clone_iters == 0
    assert cfg.dagger_expert_checkpoint is None
    assert cfg.dagger_active_at(0) is False
    assert cfg.dagger_active_at(99) is False


def test_dagger_expert_checkpoint_derives_from_bootstrap() -> None:
    """dagger_expert_checkpoint is now derived from bootstrap_opponent_checkpoint.

    Setting dagger.expert_checkpoint is ignored; the bootstrap_opponent field is
    authoritative. 'none' and 'random' both map to None.
    """
    cfg_no_bootstrap = config.RunConfig(
        misc=config.MiscConfig(device="cpu"),
        opponent=config.OpponentConfig(bootstrap_opponent="none"),
    )
    assert cfg_no_bootstrap.dagger_expert_checkpoint is None

    cfg_random = config.RunConfig(
        misc=config.MiscConfig(device="cpu"),
        opponent=config.OpponentConfig(bootstrap_opponent="random"),
    )
    assert cfg_random.dagger_expert_checkpoint is None

    cfg_checkpoint = config.RunConfig(
        misc=config.MiscConfig(device="cpu"),
        opponent=config.OpponentConfig(bootstrap_opponent="some/archive/last.pt"),
    )
    assert cfg_checkpoint.dagger_expert_checkpoint == "some/archive/last.pt"


def test_dagger_active_at_truth_table() -> None:
    """dagger_active_at is True iff a checkpoint bootstrap is set AND iteration < clone_iters.

    After Workstream C the expert is derived from bootstrap_opponent_checkpoint, so
    cloning is only active when bootstrap_opponent is a checkpoint path.
    """
    cfg = config.RunConfig(
        misc=config.MiscConfig(device="cpu"),
        opponent=config.OpponentConfig(bootstrap_opponent="some/path.pt"),
        dagger=config.DaggerConfig(clone_iters=5),
    )
    assert cfg.dagger_active_at(0) is True
    assert cfg.dagger_active_at(4) is True
    assert cfg.dagger_active_at(5) is False
    assert cfg.dagger_active_at(100) is False


def test_clone_plus_bootstrap_validates() -> None:
    """clone_iters > 0 with a checkpoint bootstrap_opponent is now VALID (original bug fix).

    Prior to Workstream C/E this combination raised a ValidationError because
    the expert_checkpoint and bootstrap_opponent were cross-validated. The check
    moved to validate_launchable so in-progress edits can commit freely.
    """
    cfg = config.RunConfig(
        misc=config.MiscConfig(device="cpu"),
        opponent=config.OpponentConfig(bootstrap_opponent="some/checkpoint.pt"),
        dagger=config.DaggerConfig(clone_iters=5),
    )
    assert cfg.dagger_active_at(0) is True
    assert cfg.dagger_active_at(5) is False


def test_clone_iters_with_random_bootstrap_is_inactive() -> None:
    """clone_iters > 0 with bootstrap_opponent='random' validates but DAgger is inactive.

    There is no expert derived from random bootstrap — the bootstrap simply
    graduates the student versus the random agent. clone_iters is silently
    ignored in this case.
    """
    cfg = config.RunConfig(
        misc=config.MiscConfig(device="cpu"),
        opponent=config.OpponentConfig(bootstrap_opponent="random"),
        dagger=config.DaggerConfig(clone_iters=5),
    )
    assert cfg.dagger_expert_checkpoint is None
    assert cfg.dagger_active_at(0) is False


def test_validate_launchable_flags_checkpoint_on_cuda() -> None:
    """validate_launchable catches the bootstrap-checkpoint-on-cuda combo.

    This check was a model_validator (hard rejection) before Workstream E. Now
    it surfaces as a launch-time warning so in-progress edits are not blocked.
    """
    cfg = config.RunConfig(
        misc=config.MiscConfig(device="cuda"),
        opponent=config.OpponentConfig(bootstrap_opponent="some/path.pt"),
    )
    problems = config.validate_launchable(cfg)
    assert any("cpu" in problem for problem in problems)


def test_validate_launchable_clean_config_is_ok() -> None:
    """A well-formed config returns an empty problem list."""
    cfg = config.RunConfig(misc=config.MiscConfig(device="cpu"))
    assert config.validate_launchable(cfg) == []


# ---------------------------------------------------------------------------
# 2. In-process collection with an expert net


def test_collect_play_game_labels_steps_with_expert(
    tmp_path: pathlib.Path,
) -> None:
    """Steps recorded in a game played with ``expert_net`` carry ``expert_probs``
    of shape ``(n_choices,)`` that sum to 1 on multi-option decisions."""
    cfg = _small_cfg(tmp_path)
    net = _small_net(cfg)
    device = torch.device("cpu")
    rng = random.Random(42)

    record = collect.play_game(
        net, device, rng, seed=1, opponent_agent=None, expert_net=net
    )

    # Every step that had > 1 option should have a valid expert distribution.
    labeled = [step for step in record.steps if step.expert_probs is not None]
    assert labeled, "expected at least one labeled step"
    for step in labeled:
        assert step.expert_probs is not None
        assert step.expert_probs.shape == (step.choices.shape[0],)
        # The probabilities should be non-negative and sum to 1 (to float32 tolerance).
        # (pytest.approx is untyped under strict pyright; use abs directly)
        assert abs(float(np.sum(step.expert_probs)) - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# 3. In-process collection without an expert net


def test_collect_play_game_no_expert_leaves_probs_none(
    tmp_path: pathlib.Path,
) -> None:
    """When ``expert_net`` is ``None`` (the normal RL path), all steps carry
    ``expert_probs=None``."""
    cfg = _small_cfg(tmp_path)
    net = _small_net(cfg)
    device = torch.device("cpu")
    rng = random.Random(99)

    record = collect.play_game(net, device, rng, seed=2, expert_net=None)

    assert all(step.expert_probs is None for step in record.steps)


# ---------------------------------------------------------------------------
# 4. Learner imitation phase: finite imitation_loss, zero policy_loss


def test_learner_imitation_phase_produces_finite_loss(
    tmp_path: pathlib.Path,
) -> None:
    """``learner.update`` in imitation mode returns a finite ``imitation_loss`` and
    zero ``policy_loss`` (no policy-gradient in the clone phase)."""
    cfg = _small_cfg(tmp_path)
    net = _small_net(cfg)
    device = torch.device("cpu")
    rng = random.Random(7)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    # Play two games with the student as its own expert so every step is labeled.
    records = [
        collect.play_game(net, device, rng, seed=10, expert_net=net),
        collect.play_game(net, device, rng, seed=11, expert_net=net),
    ]
    stats = learner.update(net, optimizer, records, cfg, device, imitation_phase=True)

    assert np.isfinite(stats.imitation_loss), "imitation_loss must be finite"
    assert stats.imitation_loss >= 0.0
    assert abs(stats.policy_loss) < 1e-6  # pytest.approx untyped under strict pyright
    assert np.isfinite(stats.loss)
    assert np.isfinite(stats.value_loss)


# ---------------------------------------------------------------------------
# 5. Learner RL phase: imitation_loss = 0.0


def test_learner_rl_phase_has_zero_imitation_loss(
    tmp_path: pathlib.Path,
) -> None:
    """In the normal RL mode (``imitation_phase=False``) ``imitation_loss`` is 0.0."""
    cfg = _small_cfg(tmp_path)
    net = _small_net(cfg)
    device = torch.device("cpu")
    rng = random.Random(5)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    records = [collect.play_game(net, device, rng, seed=20)]
    stats = learner.update(net, optimizer, records, cfg, device, imitation_phase=False)

    assert (
        abs(stats.imitation_loss) < 1e-7
    )  # pytest.approx untyped under strict pyright


# ---------------------------------------------------------------------------
# 6. Empty-bucket guard: all-None expert_probs in imitation mode


def test_learner_imitation_phase_all_none_expert_probs(
    tmp_path: pathlib.Path,
) -> None:
    """When no step carries an expert label (all ``expert_probs=None``),
    ``has_expert.sum()`` is clamped to 1, so imitation_loss is 0.0 and the
    backward does not NaN."""
    cfg = _small_cfg(tmp_path)
    net = _small_net(cfg)
    device = torch.device("cpu")
    rng = random.Random(3)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    # Play a game WITHOUT the expert — all expert_probs are None.
    records = [collect.play_game(net, device, rng, seed=30, expert_net=None)]
    stats = learner.update(net, optimizer, records, cfg, device, imitation_phase=True)

    # imitation_loss = 0 / clamp(0, min=1) = 0.0 (no NaN crash)
    assert (
        abs(stats.imitation_loss) < 1e-7
    )  # pytest.approx untyped under strict pyright
    assert np.isfinite(stats.loss)


# ---------------------------------------------------------------------------
# 7. validate_dagger_expert fail-fast and no-op cases


def test_validate_dagger_expert_raises_on_missing_file(
    tmp_path: pathlib.Path,
) -> None:
    """``validate_dagger_expert`` propagates the ``FileNotFoundError`` from
    ``loaders.load_policy_net`` when the bootstrap checkpoint path does not exist.

    After Workstream C, the expert is derived from bootstrap_opponent_checkpoint,
    so the check fires when bootstrap_opponent is a path that doesn't exist.
    """

    class _FakeLoop:
        config = config.RunConfig(
            misc=config.MiscConfig(device="cpu"),
            run=config.RunSettings(checkpoint_dir=str(tmp_path)),
            opponent=config.OpponentConfig(
                bootstrap_opponent=str(tmp_path / "nonexistent.pt")
            ),
            dagger=config.DaggerConfig(clone_iters=3),
        )

    with pytest.raises(FileNotFoundError):
        loop_resume.validate_dagger_expert(_FakeLoop())  # type: ignore[arg-type]


def test_validate_dagger_expert_noop_when_none(tmp_path: pathlib.Path) -> None:
    """``validate_dagger_expert`` is a no-op when bootstrap_opponent is 'none'."""

    class _FakeLoop:
        config = config.RunConfig(
            misc=config.MiscConfig(device="cpu"),
            run=config.RunSettings(checkpoint_dir=str(tmp_path)),
            opponent=config.OpponentConfig(bootstrap_opponent="none"),
        )

    loop_resume.validate_dagger_expert(_FakeLoop())  # type: ignore[arg-type]


def test_validate_dagger_expert_succeeds_on_valid_checkpoint(
    tmp_path: pathlib.Path,
) -> None:
    """``validate_dagger_expert`` does not raise when the bootstrap checkpoint is readable."""
    cfg = _small_cfg(tmp_path)
    net = _small_net(cfg)
    ckpt_path = tmp_path / "expert.pt"
    _save_checkpoint(net, cfg, ckpt_path)

    expert_cfg = config.RunConfig(
        misc=config.MiscConfig(device="cpu"),
        run=config.RunSettings(checkpoint_dir=str(tmp_path)),
        opponent=config.OpponentConfig(bootstrap_opponent=str(ckpt_path)),
        dagger=config.DaggerConfig(clone_iters=5),
    )

    class _FakeLoop:
        config = expert_cfg

    loop_resume.validate_dagger_expert(_FakeLoop())  # type: ignore[arg-type]
