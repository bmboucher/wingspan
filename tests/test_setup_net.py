"""Tests for the setup network and its config descriptor round-trip."""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import setup_model  # noqa: E402
from wingspan.training import setup_net, setup_runmeta  # noqa: E402


def test_forward_shape():
    arch = setup_model.SetupArchitecture(hidden_layers=(32, 16))
    net = setup_net.SetupNet(arch=arch)
    batch = torch.zeros((7, setup_model.SETUP_FEATURE_DIM), dtype=torch.float32)
    out = net(batch)
    assert out.shape == (7,)


def test_from_setup_config_round_trip():
    descriptor = setup_runmeta.SetupConfig(
        run_name="t",
        setup_feature_dim=setup_model.SETUP_FEATURE_DIM,
        setup_arch=setup_model.SetupArchitecture(hidden_layers=(64,)),
    )
    net = setup_net.SetupNet.from_setup_config(descriptor)
    assert net.feature_dim == setup_model.SETUP_FEATURE_DIM
    assert net.arch.hidden_layers == (64,)
    # State dict loads into a net built independently from the same descriptor.
    twin = setup_net.SetupNet.from_setup_config(descriptor)
    twin.load_state_dict(net.state_dict())
