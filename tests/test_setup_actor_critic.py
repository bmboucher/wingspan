"""Tests for the actor-critic setup training path.

Covers: SetupNet policy head construction/forwarding, actor_critic_update loss
and gradient flow, and play_game_with_setup populating chosen_idx/all_candidates.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import setup_model  # noqa: E402
from wingspan.model import core as model_core  # noqa: E402
from wingspan.setup_model import record  # noqa: E402
from wingspan.training import collect, config, setup_learner, setup_net  # noqa: E402

# ---------------------------------------------------------------------------
# SetupNet: policy head construction


def _make_net(use_policy_head: bool) -> setup_net.SetupNet:
    arch = setup_model.SetupArchitecture(
        hidden_layers=(16, 8), use_policy_head=use_policy_head
    )
    return setup_net.SetupNet(arch=arch)


def test_policy_mlp_absent_by_default():
    net = _make_net(use_policy_head=False)
    assert net.policy_mlp is None


def test_policy_mlp_present_when_enabled():
    net = _make_net(use_policy_head=True)
    assert net.policy_mlp is not None


def test_forward_unchanged_shape_with_policy_head():
    """forward() still returns (B,) value scalars regardless of policy head."""
    net = _make_net(use_policy_head=True)
    batch = torch.zeros((5, setup_model.SETUP_FEATURE_DIM), dtype=torch.float32)
    out = net(batch)
    assert out.shape == (5,)


def test_policy_and_value_returns_two_tensors():
    net = _make_net(use_policy_head=True)
    batch = torch.zeros((5, setup_model.SETUP_FEATURE_DIM), dtype=torch.float32)
    policy_logits, value_preds = net.policy_and_value(batch)
    assert policy_logits.shape == (5,)
    assert value_preds.shape == (5,)


def test_policy_and_value_heads_differ():
    """Policy and value heads have independent weights; their outputs should diverge
    after the first non-trivial input."""
    net = _make_net(use_policy_head=True)
    rng = torch.Generator()
    rng.manual_seed(42)
    batch = torch.randn(3, setup_model.SETUP_FEATURE_DIM, generator=rng)
    policy_logits, value_preds = net.policy_and_value(batch)
    # They share the same embedding but have separate final MLPs initialized
    # with different random seeds, so their outputs should differ.
    assert not torch.allclose(policy_logits, value_preds)


def test_policy_and_value_raises_without_policy_head():
    net = _make_net(use_policy_head=False)
    batch = torch.zeros((2, setup_model.SETUP_FEATURE_DIM), dtype=torch.float32)
    with pytest.raises(RuntimeError, match="use_policy_head"):
        net.policy_and_value(batch)


# ---------------------------------------------------------------------------
# actor_critic_update: loss computation and gradient flow


def _make_config() -> config.TrainConfig:
    return config.RunConfig(
        architecture=config.ArchitectureConfig(
            setup=config.SetupNetArchitecture(
                hidden_layers=(16, 8),
            ),
        ),
    )


def _make_ac_sample(k: int = 10, chosen_idx: int = 0) -> record.SetupSample:
    """One synthetic SetupSample with all_candidates populated."""
    rng = np.random.default_rng(0)
    features = rng.standard_normal(setup_model.SETUP_FEATURE_DIM).astype(np.float32)
    all_candidates = rng.standard_normal((k, setup_model.SETUP_FEATURE_DIM)).astype(
        np.float32
    )
    return record.SetupSample(
        features=features,
        margin=5.0,
        iteration=2001,
        chosen_idx=chosen_idx,
        all_candidates=all_candidates,
    )


def test_actor_critic_update_returns_nonzero_loss():
    cfg = _make_config()
    net = _make_net(use_policy_head=True)
    optimizer = torch.optim.Adam(
        [p for p in net.parameters() if p.requires_grad], lr=1e-3
    )
    samples = [_make_ac_sample() for _ in range(4)]
    stats = setup_learner.actor_critic_update(
        net, optimizer, samples, cfg, torch.device("cpu")
    )
    assert stats.n_samples == 4
    assert np.isfinite(stats.loss)
    assert stats.loss != 0.0


def test_actor_critic_update_empty_samples():
    cfg = _make_config()
    net = _make_net(use_policy_head=True)
    optimizer = torch.optim.Adam(
        [p for p in net.parameters() if p.requires_grad], lr=1e-3
    )
    stats = setup_learner.actor_critic_update(
        net, optimizer, [], cfg, torch.device("cpu")
    )
    assert stats.n_samples == 0
    assert stats.loss == 0.0


def test_actor_critic_update_skips_samples_without_ac_data():
    """Samples with chosen_idx=None are silently skipped."""
    cfg = _make_config()
    net = _make_net(use_policy_head=True)
    optimizer = torch.optim.Adam(
        [p for p in net.parameters() if p.requires_grad], lr=1e-3
    )
    # One valid + two without ac data
    rng = np.random.default_rng(1)
    no_ac = record.SetupSample(
        features=rng.standard_normal(setup_model.SETUP_FEATURE_DIM).astype(np.float32),
        margin=2.0,
        iteration=2001,
    )
    samples = [_make_ac_sample(), no_ac, no_ac]
    stats = setup_learner.actor_critic_update(
        net, optimizer, samples, cfg, torch.device("cpu")
    )
    assert stats.n_samples == 1


def test_actor_critic_update_gradient_flows_to_policy_head():
    """Gradient reaches policy_mlp parameters but not the frozen card encoder."""
    cfg = _make_config()
    net = _make_net(use_policy_head=True)
    optimizer = torch.optim.Adam(
        [p for p in net.parameters() if p.requires_grad], lr=1e-3
    )
    samples = [_make_ac_sample(k=8, chosen_idx=3) for _ in range(2)]

    # Zero all grads first, then run one update.
    optimizer.zero_grad()
    net.train()
    loss_tensor = setup_learner._ac_group_loss(  # pyright: ignore[reportPrivateUsage]
        samples, net, cfg, torch.device("cpu")
    )
    loss_tensor.backward()  # pyright: ignore[reportUnknownMemberType]

    # Policy MLP parameters must have nonzero grad.
    assert net.policy_mlp is not None
    policy_params = list(net.policy_mlp.parameters())
    assert any(p.grad is not None and p.grad.abs().max() > 0 for p in policy_params)

    # Frozen card encoder parameters must have no grad.
    for param in net.card_encoder.parameters():
        assert param.grad is None or param.grad.abs().max() == 0


def test_actor_critic_update_groups_by_candidate_count():
    """Samples with different K values (504 vs 252) are processed separately."""
    cfg = _make_config()
    net = _make_net(use_policy_head=True)
    optimizer = torch.optim.Adam(
        [p for p in net.parameters() if p.requires_grad], lr=1e-3
    )
    samples_504 = [_make_ac_sample(k=504) for _ in range(2)]
    samples_252 = [_make_ac_sample(k=252) for _ in range(2)]
    stats = setup_learner.actor_critic_update(
        net, optimizer, samples_504 + samples_252, cfg, torch.device("cpu")
    )
    assert stats.n_samples == 4
    assert np.isfinite(stats.loss)


# ---------------------------------------------------------------------------
# shape_key includes use_policy_head


def test_shape_key_differs_by_policy_head():
    key_off = setup_model.SetupArchitecture(
        hidden_layers=(64, 32), use_policy_head=False
    ).shape_key
    key_on = setup_model.SetupArchitecture(
        hidden_layers=(64, 32), use_policy_head=True
    ).shape_key
    assert key_off != key_on


def test_shape_key_same_layers_different_policy_head():
    key_a = setup_model.SetupArchitecture(
        hidden_layers=(128, 64), use_policy_head=False
    ).shape_key
    key_b = setup_model.SetupArchitecture(
        hidden_layers=(128, 64), use_policy_head=True
    ).shape_key
    assert key_a[0] == key_b[0]  # hidden_layers same
    assert key_a[1] != key_b[1]  # use_policy_head differs


# ---------------------------------------------------------------------------
# play_game_with_setup: actor-critic data populated in MODEL_DRIVEN phase


def test_play_game_with_setup_ac_data_in_model_driven():
    """MODEL_DRIVEN + use_actor_critic=True → SetupSamples carry chosen_idx and
    all_candidates."""
    from wingspan import setup_model as sm

    net_cfg = config.RunConfig(
        architecture=config.ArchitectureConfig(
            setup=config.SetupNetArchitecture(
                hidden_layers=(16, 8),
            )
        ),
    )
    device = torch.device("cpu")

    # The setup net always trains actor-critic.
    net = setup_net.SetupNet(arch=net_cfg.setup_arch)

    main_net = model_core.PolicyValueNet(
        state_dim=net_cfg.state_dim,
        choice_dim=net_cfg.choice_dim,
        arch=net_cfg.arch,
        spec=net_cfg.encoding_spec,
    )

    generator = sm.RandomSetupGenerator(hand_combos=2, food_sets=1)
    spec = collect.SetupGameSpec(
        deal_seed=42,
        continuation_seed=42,
        iteration=2001,
    )

    game_record = collect.play_game_with_setup(
        main_net,
        device,
        spec,
        generator,
        net,
        setup_temperature=1.0,
    )

    assert len(game_record.setup_samples) == 2  # both seats in self-play
    for sample in game_record.setup_samples:
        assert sample.chosen_idx is not None
        assert sample.all_candidates is not None
        assert sample.all_candidates.shape[0] > 0
        assert sample.all_candidates.shape[1] == sm.SETUP_FEATURE_DIM
