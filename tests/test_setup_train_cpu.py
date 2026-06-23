"""CPU smoke tests for the setup-model actor-critic learner."""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
from torch import optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import setup_model  # noqa: E402
from wingspan.training import config, setup_learner, setup_net  # noqa: E402

_CONSTANT_MARGIN = 7.0


def _config() -> config.TrainConfig:
    return config.RunConfig(
        architecture=config.ArchitectureConfig(
            use_setup_model=True,
            setup=config.SetupNetArchitecture(hidden_layers=(32, 16)),
        ),
        training=config.TrainingConfig(
            score_norm=50.0,
            setup=config.SetupTrainingConfig(lr=1e-2),
        ),
        misc=config.MiscConfig(seed=0),
    )


def _ac_samples(count: int) -> list[setup_model.SetupSample]:
    rng = np.random.default_rng(0)
    k = 10
    return [
        setup_model.SetupSample(
            features=(rng.random(setup_model.SETUP_FEATURE_DIM) < 0.2).astype(
                np.float32
            ),
            margin=_CONSTANT_MARGIN,
            iteration=1500,
            chosen_idx=0,
            all_candidates=rng.random((k, setup_model.SETUP_FEATURE_DIM)).astype(
                np.float16
            ),
        )
        for _ in range(count)
    ]


def test_actor_critic_update_returns_finite_loss():
    cfg = _config()
    device = torch.device("cpu")
    net = setup_net.SetupNet(arch=cfg.setup_arch).to(device)
    optimizer = optim.Adam(net.parameters(), lr=cfg.training.setup.lr)

    stats = setup_learner.actor_critic_update(
        net, optimizer, _ac_samples(32), cfg, device
    )
    assert stats.n_samples == 32
    assert np.isfinite(stats.loss)


def test_empty_actor_critic_update_is_noop():
    cfg = _config()
    device = torch.device("cpu")
    net = setup_net.SetupNet(arch=cfg.setup_arch).to(device)
    optimizer = optim.Adam(net.parameters(), lr=cfg.training.setup.lr)

    stats = setup_learner.actor_critic_update(net, optimizer, [], cfg, device)
    assert stats.n_samples == 0
