"""Tests for the PPO and GAE algorithm extensions.

Covers:
* ``timestamps.gae_advantages`` kernel correctness (torch-free, hand examples).
* PPO clipped-surrogate loss math (ratio=1 recovers REINFORCE; clipping fires).
* Default config dispatches to the single-pass REINFORCE path (no behavioural change).
* CPU collector (``collect.play_game``) captures finite ``behavior_logp`` and ``value_pred``.
* End-to-end smoke: ``update`` with ``policy_loss=ppo`` + ``reward_mode=gae`` runs one
  reuse loop, weights change, and all stats are finite.
"""

from __future__ import annotations

import math
import os
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

torch = pytest.importorskip("torch")
F = pytest.importorskip("torch.nn.functional")

from wingspan import model
from wingspan.training import collect, config, learner, timestamps

# ---------------------------------------------------------------------------
# GAE kernel
# ---------------------------------------------------------------------------


def test_gae_lambda1_gamma1_reduces_to_decision_delta():
    """With λ=1 and γ=1 GAE reduces to G/score_norm − V (advantage) and
    G/score_norm (value target) — the decision_delta correctness check."""
    # One player, N=3 decisions.  Checkpoints: margins before each decision plus
    # terminal.  Values: the critic's estimate at each step.
    checkpoints = [10.0, 12.0, 15.0, 20.0]  # terminal = 20.0
    times = [1.0, 2.0, 3.0, 4.0]
    values = [0.1, 0.2, 0.3]  # N=3, in normalized units
    score_norm = 50.0
    discount = 1.0
    lam = 1.0

    adv, vt = timestamps.gae_advantages(
        checkpoints, times, values, score_norm, discount, lam
    )

    # Expected G for each step (decision_delta telescope at γ=1):
    # G_0 = (20 - 10) / 50 = 0.2
    # G_1 = (20 - 12) / 50 = 0.16
    # G_2 = (20 - 15) / 50 = 0.10
    expected_returns = [
        (checkpoints[-1] - checkpoints[i]) / score_norm for i in range(3)
    ]
    expected_advantages = [expected_returns[i] - values[i] for i in range(3)]

    assert len(adv) == 3
    assert len(vt) == 3
    for i in range(3):
        assert math.isclose(
            adv[i], expected_advantages[i], rel_tol=1e-6
        ), f"adv[{i}]: {adv[i]!r} != {expected_advantages[i]!r}"
        assert math.isclose(
            vt[i], expected_returns[i], rel_tol=1e-6
        ), f"vt[{i}]: {vt[i]!r} != {expected_returns[i]!r}"


def test_gae_lambda0_reduces_to_one_step_td():
    """With λ=0 GAE reduces to one-step TD advantage δ = r + γV' − V."""
    checkpoints = [0.0, 5.0, 10.0]  # N=2
    times = [1.0, 2.0, 3.0]
    values = [0.05, 0.15]
    score_norm = 50.0
    discount = 0.9
    lam = 0.0

    adv, vt = timestamps.gae_advantages(
        checkpoints, times, values, score_norm, discount, lam
    )

    # Step 1 (position 1): next_v = 0.0 (terminal), dt = 1.0
    # r_1 = (10 - 5) / 50 = 0.1; delta_1 = 0.1 + 0 - 0.15 = -0.05
    r1 = (checkpoints[2] - checkpoints[1]) / score_norm
    delta1 = r1 + 0.0 - values[1]
    # Step 0 (position 0): next_v = values[1] = 0.15, dt = 1.0
    # r_0 = (5 - 0) / 50 = 0.1; delta_0 = 0.1 + 0.9 * 0.15 - 0.05 = 0.185
    r0 = (checkpoints[1] - checkpoints[0]) / score_norm
    delta0 = r0 + discount**1.0 * values[1] - values[0]

    assert math.isclose(adv[1], delta1, rel_tol=1e-6)
    assert math.isclose(adv[0], delta0, rel_tol=1e-6)
    assert math.isclose(vt[1], delta1 + values[1], rel_tol=1e-6)
    assert math.isclose(vt[0], delta0 + values[0], rel_tol=1e-6)


def test_gae_finite_on_hand_example():
    """GAE with intermediate λ and γ returns finite, reasonable values."""
    score_norm = 50.0
    checkpoints = [5.0, 8.0, 12.0, 18.0]
    times = [1.0, 2.0, 3.0, 4.0]
    values = [0.1, 0.15, 0.2]

    adv, vt = timestamps.gae_advantages(
        checkpoints, times, values, score_norm, 0.95, 0.95
    )

    assert all(math.isfinite(a) for a in adv)
    assert all(math.isfinite(v) for v in vt)
    assert len(adv) == len(values)
    assert len(vt) == len(values)


# ---------------------------------------------------------------------------
# PPO loss math
# ---------------------------------------------------------------------------


def test_ppo_ratio_one_equals_neg_adv_mean():
    """When old_logp == logp_new (ratio=1), the PPO surrogate reduces to −adv.mean().

    Both ``surr1 = ratio·adv`` and ``surr2 = clip(ratio)·adv`` equal ``adv``
    when ratio=1 (it's inside the clip radius), so ``min(surr1, surr2) = adv``
    and the loss is ``−adv.mean()``.  The gradient of the PPO surrogate w.r.t. θ
    at ratio=1 equals the REINFORCE gradient ``−A·∂logπ/∂θ`` by the chain rule,
    but the *values* differ by logp — this test checks the value identity.
    """
    torch_module = torch
    logp = torch_module.tensor([-1.0, -2.0, -0.5], dtype=torch.float32)
    adv = torch_module.tensor([0.5, -0.3, 1.2], dtype=torch.float32)
    eps = 0.2

    old_logp = logp.clone()
    ratio = (logp - old_logp).exp()  # all 1.0
    surr1 = ratio * adv
    surr2 = ratio.clamp(1.0 - eps, 1.0 + eps) * adv
    ppo_loss = -torch_module.min(surr1, surr2).mean()

    expected = -adv.mean()
    assert torch_module.isclose(ppo_loss, expected, atol=1e-6)


def test_ppo_clipping_fires():
    """A ratio outside [1−ε, 1+ε] produces a smaller gradient than unclipped."""
    torch_module = torch
    adv = torch_module.tensor([1.0])  # positive advantage
    old_logp = torch_module.tensor([0.0])
    logp = torch_module.tensor([1.5])  # ratio = exp(1.5) >> 1 + ε
    eps = 0.2

    ratio = (logp - old_logp).exp()
    surr1 = ratio * adv
    surr2 = ratio.clamp(1.0 - eps, 1.0 + eps) * adv
    ppo_loss = -torch_module.min(surr1, surr2).mean()

    # When ratio > 1+ε with positive advantage, the clipped value is (1+ε)*adv.
    clipped_loss = -((1.0 + eps) * adv).mean()
    assert torch_module.isclose(ppo_loss, clipped_loss, atol=1e-6).all()


# ---------------------------------------------------------------------------
# Default config dispatches to single-pass (backward-compat)
# ---------------------------------------------------------------------------


def test_default_config_dispatches_to_single_pass():
    """With default config (REINFORCE, terminal_margin) update dispatches to
    ``_update_single_pass``; PPO-specific diagnostics are 0.0."""
    net = model.PolicyValueNet()
    device = torch.device("cpu")
    rng = random.Random(0)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    cfg = config.RunConfig(misc=config.MiscConfig(device="cpu"))

    records = [collect.play_game(net, device, rng, seed=seed) for seed in (42, 43)]
    stats = learner.update(net, optimizer, records, cfg, device)

    assert stats.n_steps > 0
    assert np.isfinite(stats.loss)
    # Single-pass path never sets PPO diagnostics.
    assert stats.clip_fraction == 0.0
    assert stats.approx_kl == 0.0


def test_step_without_new_fields_validates():
    """A Step dict without behavior_logp / value_pred should parse with defaults."""
    import numpy as np

    from wingspan.training import steps

    bare = steps.Step(
        state=np.zeros(10, dtype=np.float32),
        choices=np.zeros((2, 5), dtype=np.float32),
        chosen_idx=0,
        player_id=0,
        family_idx=0,
    )
    assert bare.behavior_logp == 0.0
    assert bare.value_pred == 0.0


def test_config_without_ppo_gae_fields_loads_with_defaults():
    """A config dict missing the new PPO/GAE fields should load via model_validate."""
    cfg = config.RunConfig.model_validate(
        {
            "training": {
                "lr": 1e-3,
                "value_coef": 0.5,
                "entropy_coef": 0.01,
            }
        }
    )
    assert cfg.training.policy_loss is config.PolicyLoss.REINFORCE
    assert cfg.training.ppo_clip_eps == 0.2
    assert cfg.training.ppo_reuse_epochs == 4
    assert cfg.training.gae_lambda == 0.95


# ---------------------------------------------------------------------------
# Collector capture
# ---------------------------------------------------------------------------


def test_cpu_collector_captures_behavior_logp_and_value_pred():
    """collect.play_game populates behavior_logp (finite, ≤ 0) and value_pred
    (finite) on every recorded Step."""
    net = model.PolicyValueNet()
    device = torch.device("cpu")
    rng = random.Random(7)
    record = collect.play_game(net, device, rng, seed=99)

    assert record.steps, "expected at least one recorded step"
    for i, step in enumerate(record.steps):
        assert math.isfinite(step.behavior_logp), f"step {i}: behavior_logp not finite"
        assert (
            step.behavior_logp <= 0.0
        ), f"step {i}: log-prob > 0 ({step.behavior_logp!r})"
        assert math.isfinite(step.value_pred), f"step {i}: value_pred not finite"


# ---------------------------------------------------------------------------
# End-to-end PPO + GAE smoke
# ---------------------------------------------------------------------------


def test_ppo_gae_update_runs_and_changes_weights():
    """One update with policy_loss=ppo and reward_mode=gae completes the reuse
    loop, returns finite stats including clip_fraction and approx_kl, and
    changes the network weights."""
    net = model.PolicyValueNet()
    device = torch.device("cpu")
    rng = random.Random(0)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    cfg = config.RunConfig(
        misc=config.MiscConfig(device="cpu"),
        training=config.TrainingConfig(
            policy_loss=config.PolicyLoss.PPO,
            reward_mode=config.RewardMode.GAE,
            ppo_reuse_epochs=2,
            ppo_clip_eps=0.2,
            gae_lambda=0.95,
        ),
    )

    records = [collect.play_game(net, device, rng, seed=seed) for seed in (1, 2, 3)]

    # Snapshot weights before the update.
    params_before = [p.detach().clone() for p in net.parameters()]

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
        ("clip_fraction", stats.clip_fraction),
        ("approx_kl", stats.approx_kl),
    ):
        assert np.isfinite(val), f"{name} should be finite, got {val}"

    # At least one parameter tensor must have changed.
    params_after = list(net.parameters())
    any_changed = any(
        not torch.equal(before, after)
        for before, after in zip(params_before, params_after)
    )
    assert any_changed, "expected weights to change after the PPO+GAE update"


def test_ppo_only_update_runs():
    """PPO with MC returns (no GAE) also runs cleanly."""
    net = model.PolicyValueNet()
    device = torch.device("cpu")
    rng = random.Random(0)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    cfg = config.RunConfig(
        misc=config.MiscConfig(device="cpu"),
        training=config.TrainingConfig(
            policy_loss=config.PolicyLoss.PPO,
            reward_mode=config.RewardMode.TERMINAL_MARGIN,
            ppo_reuse_epochs=3,
        ),
    )

    records = [collect.play_game(net, device, rng, seed=seed) for seed in (10, 11)]
    stats = learner.update(net, optimizer, records, cfg, device)

    assert stats.n_steps > 0
    assert np.isfinite(stats.loss)
    assert np.isfinite(stats.clip_fraction)
    assert np.isfinite(stats.approx_kl)


def test_gae_only_update_runs():
    """GAE with REINFORCE policy loss (no PPO) runs with 1 epoch and zero
    clip_fraction / approx_kl."""
    net = model.PolicyValueNet()
    device = torch.device("cpu")
    rng = random.Random(0)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    cfg = config.RunConfig(
        misc=config.MiscConfig(device="cpu"),
        training=config.TrainingConfig(
            policy_loss=config.PolicyLoss.REINFORCE,
            reward_mode=config.RewardMode.GAE,
            gae_lambda=0.95,
        ),
    )

    records = [collect.play_game(net, device, rng, seed=seed) for seed in (20, 21)]
    stats = learner.update(net, optimizer, records, cfg, device)

    assert stats.n_steps > 0
    assert np.isfinite(stats.loss)
    # REINFORCE+GAE path sets clip_fraction/approx_kl to 0.0.
    assert stats.clip_fraction == 0.0
    assert stats.approx_kl == 0.0
