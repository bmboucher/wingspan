# pyright: reportPrivateUsage=false
# (white-box tests access _embed and _embed_card_set to verify pooling paths)
"""Tests for the setup network and its config descriptor round-trip."""

from __future__ import annotations

import torch

from wingspan import architecture, setup_model  # noqa: E402
from wingspan.training import setup_net, setup_runmeta  # noqa: E402


def test_forward_shape():
    arch = setup_model.SetupArchitecture(hidden_layers=(32, 16))
    net = setup_net.SetupNet(arch=arch)
    batch = torch.zeros((7, net.feature_dim), dtype=torch.float32)
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
    assert net.feature_dim == descriptor.setup_encoding.total_dim
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


def test_pooling_path_matches_main_net_hand_pooling():
    """The setup net's kept-set embedding equals the main net's hand pooling for
    the same multi-hot and synced card table — the core consistency guarantee."""
    from wingspan.model import core as model_core
    from wingspan.model import hand_model

    main_arch = architecture.ModelArchitecture(card_embed_dim=16)
    main_net = model_core.PolicyValueNet(arch=main_arch)
    setup_net_inst = setup_net.SetupNet(main_arch=main_arch)
    # Sync card encoder weights so both nets share the same card table.
    setup_net_inst.card_encoder.load_state_dict(main_net.card_encoder.state_dict())
    main_net.eval()
    setup_net_inst.eval()

    # Construct a batch of kept-card multi-hots (5 birds each).
    batch_size = 4
    kept_multihot = torch.zeros(batch_size, setup_model.SetupEncoding().kept_cards_dim)
    for row in range(batch_size):
        kept_multihot[row, row * 5 : row * 5 + 5] = 1.0

    with torch.no_grad():
        # Main net's hand pooling for the same multihot.
        main_card_table = main_net.card_table()
        expected = hand_model.pool_card_set(
            kept_multihot, main_card_table[1:], main_arch.hand_pooling
        )
        # Setup net's kept-set embedding via _embed_card_set (pooling path).
        setup_card_table = setup_net_inst.card_table()
        actual = setup_net_inst._embed_card_set(kept_multihot, setup_card_table)

    assert (
        actual.shape == expected.shape
    ), f"shape mismatch: setup={actual.shape} vs main={expected.shape}"
    assert torch.allclose(
        actual, expected
    ), "setup net's pooled kept-set embedding diverges from main net's hand pooling"


def test_forward_with_trunk():
    """SetupNet with a shared trunk still produces the correct (B,) output shape."""
    arch = setup_model.SetupArchitecture(trunk_layers=(64,), hidden_layers=(32,))
    net = setup_net.SetupNet(arch=arch)
    batch = torch.zeros((5, net.feature_dim), dtype=torch.float32)
    out = net(batch)
    assert out.shape == (5,)


def test_policy_and_value_with_trunk():
    """Both heads accept the trunk output and return the correct shapes."""
    arch = setup_model.SetupArchitecture(
        trunk_layers=(64,), hidden_layers=(32,), use_policy_head=True
    )
    net = setup_net.SetupNet(arch=arch)
    batch = torch.zeros((4, net.feature_dim), dtype=torch.float32)
    policy_logits, value_preds = net.policy_and_value(batch)
    assert policy_logits.shape == (4,)
    assert value_preds.shape == (4,)


def test_empty_trunk_matches_no_trunk_output():
    """trunk_layers=() produces byte-identical output to the pre-trunk architecture.

    This is the backward-compat guarantee: old configs that lack trunk_layers
    deserialize to () and reconstruct the net in exactly the same shape."""
    arch_no_trunk = setup_model.SetupArchitecture(hidden_layers=(32, 16))
    arch_empty_trunk = setup_model.SetupArchitecture(
        trunk_layers=(), hidden_layers=(32, 16)
    )
    net_a = setup_net.SetupNet(arch=arch_no_trunk)
    net_b = setup_net.SetupNet(arch=arch_empty_trunk)
    # State dicts must be identical in structure.
    assert set(net_a.state_dict().keys()) == set(net_b.state_dict().keys())
    # Load net_a's weights into net_b and verify identical forward pass.
    net_b.load_state_dict(net_a.state_dict())
    batch = torch.randn(
        3, net_a.feature_dim, generator=torch.Generator().manual_seed(0)
    )
    with torch.no_grad():
        net_a.eval()
        net_b.eval()
        assert torch.allclose(net_a(batch), net_b(batch))


def test_trunk_adds_state_dict_keys():
    """A non-empty trunk adds keys under 'trunk.' in the state dict."""
    arch_no_trunk = setup_model.SetupArchitecture(hidden_layers=(32,))
    arch_with_trunk = setup_model.SetupArchitecture(
        trunk_layers=(64,), hidden_layers=(32,)
    )
    net_no_trunk = setup_net.SetupNet(arch=arch_no_trunk)
    net_with_trunk = setup_net.SetupNet(arch=arch_with_trunk)
    trunk_keys = {k for k in net_with_trunk.state_dict() if k.startswith("trunk.")}
    no_trunk_keys = {k for k in net_no_trunk.state_dict() if k.startswith("trunk.")}
    assert len(trunk_keys) > 0
    assert len(no_trunk_keys) == 0


def test_trunk_shape_key_differs():
    """trunk_layers affects the shape_key so changed configs force a fresh net."""
    key_no_trunk = setup_model.SetupArchitecture(hidden_layers=(32,)).shape_key
    key_with_trunk = setup_model.SetupArchitecture(
        trunk_layers=(64,), hidden_layers=(32,)
    ).shape_key
    assert key_no_trunk != key_with_trunk


def test_tray_embeds_as_slot_rows_only():
    """The tray stripe contributes exactly TRAY_SIZE card-table rows to the readout
    input — no tray-set embedding appended."""
    from wingspan import state

    main_arch = architecture.ModelArchitecture(card_embed_dim=16)
    encoding = setup_model.SetupEncoding()
    net = setup_net.SetupNet(main_arch=main_arch, encoding=encoding)
    net.eval()

    # Build a feature vector with known tray indices (birds 1, 2, 3 = indices 1+1, 2+1, 3+1).
    features = torch.zeros(1, encoding.total_dim)
    features[0, encoding.off_tray : encoding.off_feeder] = torch.tensor([2.0, 3.0, 4.0])

    with torch.no_grad():
        card_table = net.card_table()
        tray_idx = (
            features[..., encoding.off_tray : encoding.off_feeder].long().clamp_(0, 180)
        )
        expected_tray_emb = card_table[tray_idx].reshape(1, -1)

        # Extract the tray block from the embedded vector: it immediately follows
        # kept_emb (set_width) and passthrough (kept_cards_dim..off_tray - kept_cards_dim).
        embedded = net._embed(features)

    set_width = main_arch.pooled_hand_width
    passthrough_width = encoding.off_tray - encoding.kept_cards_dim
    tray_start = set_width + passthrough_width
    tray_end = tray_start + state.TRAY_SIZE * main_arch.card_embed_dim
    actual_tray_block = embedded[:, tray_start:tray_end]

    assert actual_tray_block.shape == expected_tray_emb.shape
    assert torch.allclose(
        actual_tray_block, expected_tray_emb
    ), "tray block in embedded output does not match card-table lookup rows"
