"""CPU smoke tests for the setup-model learner (offline fit + on-policy step)."""

from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
import torch
from torch import optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import setup_model  # noqa: E402
from wingspan.training import config, setup_learner, setup_net  # noqa: E402

_CONSTANT_MARGIN = 7.0


def _config() -> config.TrainConfig:
    return config.TrainConfig(
        use_setup_model=True,
        setup_hidden_layers=(32, 16),
        setup_lr=1e-2,
        setup_offline_epochs=80,
        setup_offline_batch_size=32,
        score_norm=50.0,
        seed=0,
    )


def _samples(count: int) -> list[setup_model.SetupSample]:
    rng = np.random.default_rng(0)
    return [
        setup_model.SetupSample(
            features=(rng.random(setup_model.SETUP_FEATURE_DIM) < 0.2).astype(
                np.float32
            ),
            margin=_CONSTANT_MARGIN,
            iteration=1500,
        )
        for _ in range(count)
    ]


def test_offline_fit_learns_constant_margin(tmp_path: pathlib.Path):
    cfg = _config()
    device = torch.device("cpu")
    net = setup_net.SetupNet(arch=cfg.setup_arch).to(device)
    optimizer = optim.Adam(net.parameters(), lr=cfg.setup_lr)
    store = setup_model.SetupDataStore(tmp_path / "setup_data.jsonl")
    store.append(_samples(256))

    stats = setup_learner.offline_fit(net, optimizer, store, cfg, device)

    assert stats.n_samples == 256
    assert stats.n_epochs == cfg.setup_offline_epochs
    assert np.isfinite(stats.loss)
    # The net should learn to predict the constant margin (within a loose band).
    assert abs(stats.pred_margin_mean - _CONSTANT_MARGIN) < 2.0
    assert abs(stats.realized_margin_mean - _CONSTANT_MARGIN) < 1e-6


def test_online_update_runs_one_epoch():
    cfg = _config()
    device = torch.device("cpu")
    net = setup_net.SetupNet(arch=cfg.setup_arch).to(device)
    optimizer = optim.Adam(net.parameters(), lr=cfg.setup_lr)

    stats = setup_learner.online_update(net, optimizer, _samples(64), cfg, device)
    assert stats.n_samples == 64
    assert stats.n_epochs == 1
    assert np.isfinite(stats.loss)


def test_empty_offline_fit_is_noop(tmp_path: pathlib.Path):
    cfg = _config()
    device = torch.device("cpu")
    net = setup_net.SetupNet(arch=cfg.setup_arch).to(device)
    optimizer = optim.Adam(net.parameters(), lr=cfg.setup_lr)
    store = setup_model.SetupDataStore(tmp_path / "empty.jsonl")
    stats = setup_learner.offline_fit(net, optimizer, store, cfg, device)
    assert stats.n_samples == 0
