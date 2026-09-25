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
7. ``validate_dagger_expert`` fail-fast on a missing file, a seat-count
   mismatch, and no-op on ``'none'``.
8. Clone-phase optimisation schedule: per-minibatch SGD (``clone_epochs`` x
   ``clone_minibatch_steps``) actually moves the imitation loss, steps the
   optimizer the expected number of times, and orders its minibatches
   deterministically per ``(misc.seed, iteration)``.
9. ``BootstrapField`` parse / format round-trip for the DAgger expert field.
"""

from __future__ import annotations

import math
import pathlib
import random
import typing

import numpy as np
import pytest
import torch

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
        misc=config.MiscConfig(collect_device="cpu", train_device="cpu"),
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
        misc=config.MiscConfig(collect_device="cpu", train_device="cpu"),
        opponent=config.OpponentConfig(bootstrap_opponent="none"),
    )
    assert cfg_no_bootstrap.dagger_expert_checkpoint is None

    cfg_random = config.RunConfig(
        misc=config.MiscConfig(collect_device="cpu", train_device="cpu"),
        opponent=config.OpponentConfig(bootstrap_opponent="random"),
    )
    assert cfg_random.dagger_expert_checkpoint is None

    cfg_checkpoint = config.RunConfig(
        misc=config.MiscConfig(collect_device="cpu", train_device="cpu"),
        opponent=config.OpponentConfig(bootstrap_opponent="some/archive/last.pt"),
    )
    assert cfg_checkpoint.dagger_expert_checkpoint == "some/archive/last.pt"


def test_dagger_active_at_truth_table() -> None:
    """dagger_active_at is True iff a checkpoint bootstrap is set AND iteration < clone_iters.

    After Workstream C the expert is derived from bootstrap_opponent_checkpoint, so
    cloning is only active when bootstrap_opponent is a checkpoint path.
    """
    cfg = config.RunConfig(
        misc=config.MiscConfig(collect_device="cpu", train_device="cpu"),
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
        misc=config.MiscConfig(collect_device="cpu", train_device="cpu"),
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
        misc=config.MiscConfig(collect_device="cpu", train_device="cpu"),
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
        misc=config.MiscConfig(collect_device="cuda", train_device="cuda"),
        opponent=config.OpponentConfig(bootstrap_opponent="some/path.pt"),
    )
    problems = config.validate_launchable(cfg)
    assert any("cpu" in problem for problem in problems)


def test_validate_launchable_clean_config_is_ok() -> None:
    """A well-formed config returns an empty problem list."""
    cfg = config.RunConfig(
        misc=config.MiscConfig(collect_device="cpu", train_device="cpu")
    )
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
            misc=config.MiscConfig(collect_device="cpu", train_device="cpu"),
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
            misc=config.MiscConfig(collect_device="cpu", train_device="cpu"),
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
        misc=config.MiscConfig(collect_device="cpu", train_device="cpu"),
        run=config.RunSettings(checkpoint_dir=str(tmp_path)),
        opponent=config.OpponentConfig(bootstrap_opponent=str(ckpt_path)),
        dagger=config.DaggerConfig(clone_iters=5),
    )

    class _FakeLoop:
        config = expert_cfg

    loop_resume.validate_dagger_expert(_FakeLoop())  # type: ignore[arg-type]


def test_validate_dagger_expert_raises_on_num_players_mismatch(
    tmp_path: pathlib.Path,
) -> None:
    """``validate_dagger_expert`` raises when the expert checkpoint was trained
    at a different seat count than this run, naming both counts.

    The expert net's architecture is internally self-consistent, so it loads
    without error on its own — nothing else catches the mismatch, and an
    unguarded expert would silently misencode a live N-seat ``GameState``
    instead of crashing.
    """
    expert_cfg = _small_cfg(tmp_path)  # num_players=2 (default)
    expert_net = _small_net(expert_cfg)
    ckpt_path = tmp_path / "expert.pt"
    _save_checkpoint(expert_net, expert_cfg, ckpt_path)

    run_cfg = config.RunConfig(
        misc=config.MiscConfig(collect_device="cpu", train_device="cpu"),
        run=config.RunSettings(checkpoint_dir=str(tmp_path)),
        architecture=config.ArchitectureConfig(num_players=3),
        opponent=config.OpponentConfig(bootstrap_opponent=str(ckpt_path)),
        dagger=config.DaggerConfig(clone_iters=5),
    )

    class _FakeLoop:
        config = run_cfg

    with pytest.raises(ValueError, match=r"num_players=2.*num_players=3"):
        loop_resume.validate_dagger_expert(_FakeLoop())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 8. Clone-phase optimisation schedule


def test_imitation_loss_decreases_on_fixed_batch(tmp_path: pathlib.Path) -> None:
    """Per-minibatch SGD (``clone_epochs`` shuffled passes, one optimizer step
    per ``clone_minibatch_steps`` chunk) actually drives the imitation
    cross-entropy down on a fixed batch of games — the bug this feature fixes
    is that the RL path's single accumulated step per iteration left the loss
    flat (docs/TRAINING.md §6.8 "Why cloning steps per minibatch").

    The expert's targets are sharpened to a hard one-hot (on each step's first
    legal candidate) rather than left as a second freshly-initialized net's raw
    softmax: cross-entropy is bounded below by the target's own entropy, and two
    fresh nets are both near-uniform, so the raw-softmax loss already sits at
    its floor and cannot fall. A one-hot target has zero entropy, giving the
    optimizer real room to move the loss so the test exercises the stepping
    mechanics.
    """
    cfg = _small_cfg(tmp_path).model_copy(
        update={"dagger": config.DaggerConfig(clone_epochs=2, clone_minibatch_steps=64)}
    )
    device = torch.device("cpu")

    torch.manual_seed(1)  # pyright: ignore[reportUnknownMemberType]
    student = _small_net(cfg)
    torch.manual_seed(2)  # pyright: ignore[reportUnknownMemberType]
    expert = _small_net(cfg)  # different init; only used to shape real self-play games

    rng = random.Random(123)
    records = [
        collect.play_game(student, device, rng, seed=seed, expert_net=expert)
        for seed in (201, 202, 203)
    ]
    for record in records:
        for step in record.steps:
            if step.expert_probs is not None:
                hard_target = np.zeros_like(step.expert_probs)
                hard_target[0] = 1.0
                step.expert_probs = hard_target

    optimizer = torch.optim.Adam(student.parameters(), lr=2e-2)
    losses: list[float] = []
    for iteration in range(5):
        stats = learner.update(
            student,
            optimizer,
            records,
            cfg,
            device,
            imitation_phase=True,
            iteration=iteration,
        )
        losses.append(stats.imitation_loss)
        assert stats.policy_loss == 0.0
        assert stats.entropy == 0.0

    assert losses[0] > 0.0
    assert losses[-1] < 0.8 * losses[0], f"loss did not drop enough: {losses}"


def test_imitation_update_steps_once_per_minibatch(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cloning takes one ``optimizer.step()`` per minibatch per epoch — unlike
    the RL update, which accumulates gradients and steps once per epoch."""
    clone_epochs = 2
    clone_minibatch_steps = 64
    cfg = _small_cfg(tmp_path).model_copy(
        update={
            "dagger": config.DaggerConfig(
                clone_epochs=clone_epochs, clone_minibatch_steps=clone_minibatch_steps
            )
        }
    )
    device = torch.device("cpu")

    torch.manual_seed(1)  # pyright: ignore[reportUnknownMemberType]
    student = _small_net(cfg)
    torch.manual_seed(2)  # pyright: ignore[reportUnknownMemberType]
    expert = _small_net(cfg)

    rng = random.Random(321)
    records = [
        collect.play_game(student, device, rng, seed=seed, expert_net=expert)
        for seed in (301, 302, 303)
    ]
    n_steps = sum(len(record.steps) for record in records)

    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    step_calls = 0

    def _counting_step() -> None:
        nonlocal step_calls
        step_calls += 1
        torch.optim.Adam.step(optimizer)  # pyright: ignore[reportUnknownMemberType]

    monkeypatch.setattr(optimizer, "step", _counting_step)

    learner.update(
        student, optimizer, records, cfg, device, imitation_phase=True, iteration=0
    )

    expected_steps = clone_epochs * math.ceil(n_steps / clone_minibatch_steps)
    assert step_calls == expected_steps
    assert step_calls > 1


def test_imitation_update_is_deterministic(tmp_path: pathlib.Path) -> None:
    """Same seed + same records + same iteration produces byte-identical
    post-update weights: the minibatch order is seeded by ``(misc.seed,
    iteration)`` and the small test net runs in eval mode, so no other
    randomness enters the update.

    ``clone_minibatch_steps`` is kept small relative to the batch so a game's
    decisions span several minibatches per epoch: with only one minibatch,
    reshuffling can't change which steps get averaged together, and the
    follow-up "different iteration diverges" check below would be testing
    float summation-order noise instead of a real behavioral difference.
    """
    cfg = _small_cfg(tmp_path).model_copy(
        update={"dagger": config.DaggerConfig(clone_epochs=1, clone_minibatch_steps=16)}
    )
    device = torch.device("cpu")

    torch.manual_seed(2)  # pyright: ignore[reportUnknownMemberType]
    expert = _small_net(cfg)
    torch.manual_seed(9)  # pyright: ignore[reportUnknownMemberType]
    labeler = _small_net(cfg)
    rng = random.Random(555)
    records = [
        collect.play_game(labeler, device, rng, seed=seed, expert_net=expert)
        for seed in (401, 402, 403)
    ]

    def _run_update(iteration: int) -> dict[str, torch.Tensor]:
        torch.manual_seed(1)  # pyright: ignore[reportUnknownMemberType]
        student = _small_net(cfg)
        optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
        learner.update(
            student,
            optimizer,
            records,
            cfg,
            device,
            imitation_phase=True,
            iteration=iteration,
        )
        return dict(student.state_dict())

    state_a = _run_update(3)
    state_b = _run_update(3)
    for key in state_a:
        assert torch.equal(
            state_a[key], state_b[key]
        ), f"{key} differs across identical runs"

    # A different iteration reshuffles into different minibatches, so the
    # gradient trajectory (and final weights) should diverge.
    state_c = _run_update(4)
    any_diff = any(not torch.equal(state_a[key], state_c[key]) for key in state_a)
    assert any_diff, "a different iteration should reshuffle and diverge weights"
