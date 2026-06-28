# pyright: reportPrivateUsage=false
# (white-box tests of the setup net's frozen embedder copies and the training
# loop's private sync helper — the mechanism under test is internal by design)
"""The setup net's shared (frozen, per-iteration-synced) embedder copies.

``SetupNet`` carries copies of the main net's card encoder and hand (set)
encoder: the card copy is always frozen and synced; the hand copy is frozen +
synced only when the main architecture actually has a distinct hand model, and
is the setup net's own trainable block otherwise. These tests pin the freezing,
the optimizer exclusion, the loop's sync helper, the analytic parameter count,
and the inference card-table cache.
"""

from __future__ import annotations

import pathlib

import pytest

torch = pytest.importorskip("torch")

from wingspan import architecture, setup_model  # noqa: E402
from wingspan.setup_model import encode as setup_encode  # noqa: E402
from wingspan.training import config, loop, loop_setup, setup_net  # noqa: E402

_DISTINCT_MAIN = architecture.ModelArchitecture(
    trunk_layers=(32, 32),
    choice_layers=(32, 32),
    card_embed_dim=8,
    use_distinct_hand_model=True,
)
_MEANPOOL_MAIN = _DISTINCT_MAIN.model_copy(
    update={"use_distinct_hand_model": False, "tray_set_embedding": False}
)


def _loop_config(tmp_path: pathlib.Path) -> config.TrainConfig:
    return config.RunConfig(
        misc=config.MiscConfig(device="cpu"),
        run=config.RunSettings(
            checkpoint_dir=str(tmp_path),
            resume=False,
        ),
        architecture=config.ArchitectureConfig(
            use_setup_model=True,
            main=config.MainNetArchitecture(
                use_distinct_hand_model=True,
                trunk_layers=(32, 32),
                choice_layers=(32, 32),
                card_embed_dim=8,
            ),
            setup=config.SetupNetArchitecture(hidden_layers=(16,)),
        ),
    )


def test_setup_feature_dim_is_308_with_candidate_pricing():
    # 180 kept cards + 5 kept foods + 26 kept bonus + 3 tray indices + 6 feeder
    # counts + 80 goal one-hots + 4 kept-bonus pricing + 4 goal affinities.
    assert setup_model.SETUP_FEATURE_DIM == 308


def test_card_block_frozen_regardless_hand_block_per_main_arch():
    distinct = setup_net.SetupNet(main_arch=_DISTINCT_MAIN)
    assert all(not p.requires_grad for p in distinct.card_encoder.parameters())
    assert all(not p.requires_grad for p in distinct.hand_encoder.parameters())
    assert all(p.requires_grad for p in distinct.mlp.parameters())

    meanpool = setup_net.SetupNet(main_arch=_MEANPOOL_MAIN)
    assert all(not p.requires_grad for p in meanpool.card_encoder.parameters())
    # No distinct hand model on the main net -> the pooling path is used in
    # _embed_card_set; the hand encoder is built for load_state_dict compat but
    # receives no gradients. Its requires_grad stays True (untethered copy).
    assert all(p.requires_grad for p in meanpool.hand_encoder.parameters())


def test_optimizer_excludes_frozen_parameters(tmp_path: pathlib.Path):
    training = loop.TrainingLoop(_loop_config(tmp_path))
    assert training._setup_net is not None and training._setup_optimizer is not None
    optimized = {
        id(param)
        for group in training._setup_optimizer.param_groups
        for param in group["params"]
    }
    frozen = [p for p in training._setup_net.parameters() if not p.requires_grad]
    trainable = [p for p in training._setup_net.parameters() if p.requires_grad]
    assert frozen, "expected frozen embedder copies under the distinct hand model"
    assert all(id(param) not in optimized for param in frozen)
    assert all(id(param) in optimized for param in trainable)


def test_sync_copies_main_weights_and_resyncs_after_update(tmp_path: pathlib.Path):
    training = loop.TrainingLoop(_loop_config(tmp_path))
    setup_policy_net = training._setup_net
    assert setup_policy_net is not None

    # __init__ already synced once: the copies match the main net exactly, so
    # the setup net's tray-slot embeddings ARE the main net's card-table rows.
    assert torch.equal(setup_policy_net.card_table(), training.net.card_table())

    # Perturb the main embedders (stand-in for an optimizer step), re-sync, and
    # the copies must follow.
    with torch.no_grad():
        for param in training.net.card_encoder.parameters():
            param.add_(1.0)
        for param in training.net.hand_encoder.parameters():
            param.add_(1.0)
    training.net.eval()
    assert not torch.equal(setup_policy_net.card_table(), training.net.card_table())
    loop_setup.sync_setup_embedders(training)
    assert torch.equal(setup_policy_net.card_table(), training.net.card_table())
    for name, tensor in setup_policy_net.hand_encoder.state_dict().items():
        assert torch.equal(tensor, training.net.hand_encoder.state_dict()[name])
    # The sync helper leaves the setup net in eval (the cache contract).
    assert not setup_policy_net.training


def test_count_setup_parameters_matches_numel_both_modes():
    setup_arch = setup_model.SetupArchitecture(hidden_layers=(32, 16))
    for main_arch in (_DISTINCT_MAIN, _MEANPOOL_MAIN):
        net = setup_net.SetupNet(arch=setup_arch, main_arch=main_arch)
        block = setup_model.count_setup_parameters(
            setup_arch,
            feature_dim=net.feature_dim,
            main_arch=main_arch,
            encoding=net.encoding,
        )
        assert block.total == sum(param.numel() for param in net.parameters())


def test_card_table_cache_populates_and_invalidates():
    net = setup_net.SetupNet(main_arch=_DISTINCT_MAIN)
    net.eval()
    assert net._inference_card_table is None
    features = torch.zeros(2, net.feature_dim)
    with torch.no_grad():
        net(features)
    cached = net._inference_card_table
    assert cached is not None
    assert torch.equal(cached, net.card_table().detach())
    net.train()
    assert net._inference_card_table is None
    net.eval()
    assert net._inference_card_table is None


def test_train_mode_pins_frozen_submodules_to_eval():
    distinct = setup_net.SetupNet(main_arch=_DISTINCT_MAIN)
    distinct.train()
    assert distinct.training
    assert not distinct.card_encoder.training
    assert not distinct.hand_encoder.training
    assert distinct.mlp.training

    meanpool = setup_net.SetupNet(main_arch=_MEANPOOL_MAIN)
    meanpool.train()
    assert not meanpool.card_encoder.training
    assert meanpool.hand_encoder.training  # trainable copy follows train mode


def test_setup_state_dict_round_trips_through_worker_load():
    """The broadcast path: a worker-built twin strict-loads the main process's
    setup state_dict, frozen embedder weights included."""
    source = setup_net.SetupNet(main_arch=_DISTINCT_MAIN)
    twin = setup_net.SetupNet(main_arch=_DISTINCT_MAIN)
    twin.load_state_dict(source.state_dict())
    features = torch.randn(3, source.feature_dim)
    # Tray index columns must hold valid indices, not random floats.
    features[:, setup_encode.OFF_TRAY : setup_encode.OFF_FEEDER] = 0.0
    source.eval()
    twin.eval()
    with torch.no_grad():
        assert torch.allclose(source(features), twin(features))
