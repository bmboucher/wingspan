"""Tests for the gradient-accumulation minibatching path in ``learner``.

The core property: when ``update_minibatch_steps > 0``, one optimizer.step()
per epoch is taken — same count as today — and the accumulated gradient
reproduces the full-batch gradient up to float summation order.  The
equivalence tests below verify this by running ``update`` from identical
initial weights with ``mb=0`` and ``mb=small``, then asserting the
post-update parameters agree within a tight tolerance:

- REINFORCE (single-pass): ``atol=5e-4`` — the prepass reproduces the full-batch
  advantage normalization in the same bucket order, so mean/std are bitwise
  identical; remaining differences come from per-minibatch ``.mean()`` reduction
  ordering that varies with xdist worker memory layout.
- PPO+GAE / GAE-only (reuse path): ``atol=5e-4`` — both paths use numpy std
  (N denominator) for advantage normalization so those match, but float
  summation order across minibatch means are enough to exceed 1e-4 for some
  parameters when policy and value gradients nearly cancel.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wingspan import model
from wingspan.training import collect, config, learner

# ---------------------------------------------------------------------------
# Default config: minibatching is off
# ---------------------------------------------------------------------------


def test_default_minibatch_steps_is_zero():
    """``update_minibatch_steps`` defaults to 0 (whole-batch path unchanged)."""
    cfg = config.RunConfig()
    assert cfg.training.update_minibatch_steps == 0


def test_config_without_minibatch_field_loads_with_default():
    """A config dict missing ``update_minibatch_steps`` loads with default 0."""
    cfg = config.RunConfig.model_validate(
        {"training": {"lr": 1e-3, "ppo_reuse_epochs": 4}}
    )
    assert cfg.training.update_minibatch_steps == 0


# ---------------------------------------------------------------------------
# REINFORCE (single-pass) equivalence: mb=0 vs mb=small
# ---------------------------------------------------------------------------


def test_reinforce_minibatch_equivalent_to_full_batch():
    """Gradient-accumulation REINFORCE reproduces the full-batch update.

    Both runs start from identical weights and a fresh Adam; the post-update
    parameters must agree within atol=5e-4.

    The prepass computes advantages in the same bucket order as the full-batch
    path so the mean/std normalization is numerically identical.  Remaining
    differences come from float summation order in the per-minibatch ``.mean()``
    reductions (PyTorch does not guarantee reduction order across calls), which
    can vary between runs because xdist workers have different memory layouts.
    """
    net = model.PolicyValueNet()
    device = torch.device("cpu")
    rng = random.Random(42)
    records = [collect.play_game(net, device, rng, seed=s) for s in (1, 2, 3)]
    flat_count = sum(len(rec.steps) for rec in records)
    mb_size = max(1, flat_count // 4)

    initial_params = [p.detach().clone() for p in net.parameters()]

    # Full-batch run.
    cfg_full = config.RunConfig(
        training=config.TrainingConfig(update_minibatch_steps=0)
    )
    optimizer_full = torch.optim.Adam(net.parameters(), lr=1e-3)
    learner.update(net, optimizer_full, records, cfg_full, device)
    params_full = [p.detach().clone() for p in net.parameters()]

    # Minibatch run from the same initial weights.
    with torch.no_grad():
        for param, init_val in zip(net.parameters(), initial_params):
            param.copy_(init_val)
    cfg_mb = config.RunConfig(
        training=config.TrainingConfig(update_minibatch_steps=mb_size)
    )
    optimizer_mb = torch.optim.Adam(net.parameters(), lr=1e-3)
    learner.update(net, optimizer_mb, records, cfg_mb, device)
    params_mb = [p.detach().clone() for p in net.parameters()]

    for idx, (pa, pb) in enumerate(zip(params_full, params_mb)):
        max_diff = float((pa - pb).abs().max().item())
        assert (
            max_diff <= 5e-4
        ), f"REINFORCE param[{idx}] max diff {max_diff:.2e} > 5e-4"


# ---------------------------------------------------------------------------
# PPO + GAE reuse equivalence: mb=0 vs mb=small
# ---------------------------------------------------------------------------


def test_ppo_gae_minibatch_equivalent_to_full_batch():
    """Gradient-accumulation PPO+GAE (1 epoch) reproduces the full-batch update.

    The reuse path uses numpy std for advantage normalization (same in both
    full-batch and minibatch), so the only source of divergence is float
    summation order across minibatches.  atol=5e-4 accommodates that.
    """
    net = model.PolicyValueNet()
    device = torch.device("cpu")
    rng = random.Random(7)
    records = [collect.play_game(net, device, rng, seed=s) for s in (10, 11, 12)]
    flat_count = sum(len(rec.steps) for rec in records)
    mb_size = max(1, flat_count // 4)

    initial_params = [p.detach().clone() for p in net.parameters()]

    base_training = config.TrainingConfig(
        policy_loss=config.PolicyLoss.PPO,
        reward_mode=config.RewardMode.GAE,
        ppo_reuse_epochs=1,
        ppo_clip_eps=0.2,
        gae_lambda=0.95,
    )

    # Full-batch run.
    cfg_full = config.RunConfig(training=base_training)
    optimizer_full = torch.optim.Adam(net.parameters(), lr=1e-3)
    learner.update(net, optimizer_full, records, cfg_full, device)
    params_full = [p.detach().clone() for p in net.parameters()]

    # Minibatch run from the same initial weights.
    with torch.no_grad():
        for param, init_val in zip(net.parameters(), initial_params):
            param.copy_(init_val)
    cfg_mb = config.RunConfig(
        training=config.TrainingConfig(
            policy_loss=config.PolicyLoss.PPO,
            reward_mode=config.RewardMode.GAE,
            ppo_reuse_epochs=1,
            ppo_clip_eps=0.2,
            gae_lambda=0.95,
            update_minibatch_steps=mb_size,
        )
    )
    optimizer_mb = torch.optim.Adam(net.parameters(), lr=1e-3)
    learner.update(net, optimizer_mb, records, cfg_mb, device)
    params_mb = [p.detach().clone() for p in net.parameters()]

    for idx, (pa, pb) in enumerate(zip(params_full, params_mb)):
        max_diff = float((pa - pb).abs().max().item())
        assert max_diff <= 5e-4, f"PPO+GAE param[{idx}] max diff {max_diff:.2e} > 5e-4"


# ---------------------------------------------------------------------------
# GAE-only (REINFORCE + GAE, no PPO clip) equivalence
# ---------------------------------------------------------------------------


def test_gae_only_minibatch_equivalent_to_full_batch():
    """GAE with REINFORCE loss also reproduces the full-batch update.

    Same float summation order caveat as PPO+GAE; atol=5e-4.
    """
    net = model.PolicyValueNet()
    device = torch.device("cpu")
    rng = random.Random(99)
    records = [collect.play_game(net, device, rng, seed=s) for s in (20, 21)]
    flat_count = sum(len(rec.steps) for rec in records)
    mb_size = max(1, flat_count // 3)

    initial_params = [p.detach().clone() for p in net.parameters()]

    # Full-batch run.
    cfg_full = config.RunConfig(
        training=config.TrainingConfig(
            policy_loss=config.PolicyLoss.REINFORCE,
            reward_mode=config.RewardMode.GAE,
            gae_lambda=0.95,
        )
    )
    optimizer_full = torch.optim.Adam(net.parameters(), lr=1e-3)
    learner.update(net, optimizer_full, records, cfg_full, device)
    params_full = [p.detach().clone() for p in net.parameters()]

    # Minibatch run from same start.
    with torch.no_grad():
        for param, init_val in zip(net.parameters(), initial_params):
            param.copy_(init_val)
    cfg_mb = config.RunConfig(
        training=config.TrainingConfig(
            policy_loss=config.PolicyLoss.REINFORCE,
            reward_mode=config.RewardMode.GAE,
            gae_lambda=0.95,
            update_minibatch_steps=mb_size,
        )
    )
    optimizer_mb = torch.optim.Adam(net.parameters(), lr=1e-3)
    learner.update(net, optimizer_mb, records, cfg_mb, device)
    params_mb = [p.detach().clone() for p in net.parameters()]

    for idx, (pa, pb) in enumerate(zip(params_full, params_mb)):
        max_diff = float((pa - pb).abs().max().item())
        assert max_diff <= 5e-4, f"GAE-only param[{idx}] max diff {max_diff:.2e} > 5e-4"


# ---------------------------------------------------------------------------
# Imitation mode: runs, returns finite stats, changes weights
# ---------------------------------------------------------------------------


def test_imitation_minibatch_runs_and_changes_weights():
    """Minibatched imitation update completes and changes network weights."""
    net = model.PolicyValueNet()
    device = torch.device("cpu")
    rng = random.Random(5)
    records = [collect.play_game(net, device, rng, seed=s) for s in (30, 31)]

    # Inject uniform expert_probs on every step so the imitation loss is active.
    for rec in records:
        for step in rec.steps:
            n_choices = step.choices.shape[0]
            step.expert_probs = np.full(n_choices, 1.0 / n_choices, dtype=np.float32)

    flat_count = sum(len(rec.steps) for rec in records)
    mb_size = max(1, flat_count // 4)
    cfg = config.RunConfig(
        training=config.TrainingConfig(update_minibatch_steps=mb_size)
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    params_before = [p.detach().clone() for p in net.parameters()]

    stats = learner.update(net, optimizer, records, cfg, device, imitation_phase=True)

    assert stats.n_steps > 0
    assert np.isfinite(stats.loss)
    assert np.isfinite(stats.imitation_loss)
    assert np.isfinite(stats.value_loss)
    assert np.isfinite(stats.grad_norm)

    any_changed = any(
        not torch.equal(before, after)
        for before, after in zip(params_before, list(net.parameters()))
    )
    assert any_changed, "expected weights to change after minibatched imitation update"


# ---------------------------------------------------------------------------
# Stats are finite for minibatched paths
# ---------------------------------------------------------------------------


def test_reinforce_minibatch_stats_finite():
    """Minibatched REINFORCE returns finite stats."""
    net = model.PolicyValueNet()
    device = torch.device("cpu")
    rng = random.Random(11)
    records = [collect.play_game(net, device, rng, seed=40)]
    flat_count = sum(len(rec.steps) for rec in records)
    mb_size = max(1, flat_count // 2)

    cfg = config.RunConfig(
        training=config.TrainingConfig(update_minibatch_steps=mb_size)
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    stats = learner.update(net, optimizer, records, cfg, device)

    assert stats.n_steps > 0
    for name, val in (
        ("loss", stats.loss),
        ("policy_loss", stats.policy_loss),
        ("value_loss", stats.value_loss),
        ("entropy", stats.entropy),
        ("grad_norm", stats.grad_norm),
        ("advantage_mean", stats.advantage_mean),
        ("advantage_std", stats.advantage_std),
    ):
        assert np.isfinite(val), f"stats.{name} should be finite, got {val}"


def test_ppo_minibatch_stats_finite():
    """Minibatched PPO returns finite stats including clip_fraction and approx_kl."""
    net = model.PolicyValueNet()
    device = torch.device("cpu")
    rng = random.Random(22)
    records = [collect.play_game(net, device, rng, seed=s) for s in (50, 51)]
    flat_count = sum(len(rec.steps) for rec in records)
    mb_size = max(1, flat_count // 3)

    cfg = config.RunConfig(
        training=config.TrainingConfig(
            policy_loss=config.PolicyLoss.PPO,
            reward_mode=config.RewardMode.GAE,
            ppo_reuse_epochs=2,
            update_minibatch_steps=mb_size,
        )
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    stats = learner.update(net, optimizer, records, cfg, device)

    assert stats.n_steps > 0
    for name, val in (
        ("loss", stats.loss),
        ("policy_loss", stats.policy_loss),
        ("value_loss", stats.value_loss),
        ("entropy", stats.entropy),
        ("grad_norm", stats.grad_norm),
        ("clip_fraction", stats.clip_fraction),
        ("approx_kl", stats.approx_kl),
    ):
        assert np.isfinite(val), f"stats.{name} should be finite, got {val}"
