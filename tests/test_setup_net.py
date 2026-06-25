"""Tests for the setup network and its config descriptor round-trip."""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import architecture, setup_model  # noqa: E402
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
        setup_encoding=setup_model.SetupEncoding(),
        setup_arch=setup_model.SetupArchitecture(hidden_layers=(64,)),
        main_arch=architecture.ModelArchitecture(
            card_embed_dim=16, use_distinct_hand_model=True, hand_embed_dim=24
        ),
    )
    net = setup_net.SetupNet.from_setup_config(descriptor)
    assert net.feature_dim == setup_model.SETUP_FEATURE_DIM
    assert net.arch.hidden_layers == (64,)
    assert net.main_arch.card_embed_dim == 16
    assert net.main_arch.hand_embed_width == 24
    # State dict loads into a net built independently from the same descriptor.
    twin = setup_net.SetupNet.from_setup_config(descriptor)
    twin.load_state_dict(net.state_dict())


def test_old_descriptor_without_main_arch_still_parses():
    # setup_config.json files written before the shared embedders carry no
    # main_arch — the field's default must fill in so the JSON deserializes.
    descriptor = setup_runmeta.SetupConfig.model_validate_json(
        '{"run_name": "t", "setup_feature_dim": 477,'
        ' "setup_arch": {"hidden_layers": [64]}}'
    )
    assert descriptor.main_arch == architecture.ModelArchitecture()


def test_forward_shape_with_playable_kept_cards():
    """A net built with include_playable_kept_cards=True accepts the larger vector."""
    encoding = setup_model.SetupEncoding(include_playable_kept_cards=True)
    arch = setup_model.SetupArchitecture(hidden_layers=(32, 16))
    net = setup_net.SetupNet(encoding=encoding, arch=arch)
    batch = torch.zeros((5, encoding.total_dim), dtype=torch.float32)
    out = net(batch)
    assert out.shape == (5,)


def test_policy_and_value_with_playable_kept_cards():
    """Both heads return the right shapes when playable_kept_cards stripe is active."""
    encoding = setup_model.SetupEncoding(include_playable_kept_cards=True)
    arch = setup_model.SetupArchitecture(hidden_layers=(32,), use_policy_head=True)
    net = setup_net.SetupNet(encoding=encoding, arch=arch)
    batch = torch.zeros((4, encoding.total_dim), dtype=torch.float32)
    policy_logits, value_preds = net.policy_and_value(batch)
    assert policy_logits.shape == (4,)
    assert value_preds.shape == (4,)


def test_state_dict_syncs_with_playable_kept_cards():
    """Two nets with the same encoding load each other's state dict without errors."""
    encoding = setup_model.SetupEncoding(include_playable_kept_cards=True)
    arch = setup_model.SetupArchitecture(hidden_layers=(32,))
    net = setup_net.SetupNet(encoding=encoding, arch=arch)
    twin = setup_net.SetupNet(encoding=encoding, arch=arch)
    twin.load_state_dict(net.state_dict())
