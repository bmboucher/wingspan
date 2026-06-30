# pyright: reportPrivateUsage=false
# (white-box tests of the loop's private setup-resume gate — the mechanism
# under test is internal by design)
"""The setup net's resume gate under the shared-embedder layout.

``setup_architecture_key`` must register a reshape of either frozen embedder
copy, and a ``setup.pt`` recorded under an older encoding layout (a mismatched
persisted ``setup_feature_dim``) must start the setup net fresh — clearing the
recorded sample store — without crashing the run or disturbing the main net's
resume.
"""

from __future__ import annotations

import pathlib

import pytest

torch = pytest.importorskip("torch")

from wingspan import version  # noqa: E402
from wingspan.training import (  # noqa: E402
    artifacts,
    config,
    loop,
    loop_setup,
    runstate,
)

_OLD_FEATURE_DIM = 477  # the pre-shared-embedder layout's width


def _cfg(tmp_path: pathlib.Path, **overrides: object) -> config.TrainConfig:
    base: dict[str, object] = {
        "device": "cpu",
        "checkpoint_dir": str(tmp_path),
        "use_setup_model": True,
        "trunk_layers": (32, 32),
        "choice_layers": (32, 32),
        "card_embed_dim": 8,
        "setup_head_layers": (16,),
    }
    base.update(overrides)
    # The flat keys above are routed to their nested sections by the same
    # migration the loaders use for ≤0.4 artifacts.
    return config.run_config_from_artifact(base, version.MODEL_VERSION)


def test_key_changes_with_embedder_shapes(tmp_path: pathlib.Path):
    base = _cfg(tmp_path)
    assert (
        _cfg(tmp_path, card_embed_dim=16).setup_architecture_key
        != base.setup_architecture_key
    )
    assert (
        _cfg(tmp_path, card_encoder_layers=(64,)).setup_architecture_key
        != base.setup_architecture_key
    )
    # Default is now pooled (use_distinct_hand_model=False); flipping to distinct
    # must change the setup key since the setup net's embedder path differs.
    distinct = _cfg(tmp_path, use_distinct_hand_model=True, tray_set_embedding=False)
    assert distinct.setup_architecture_key != base.setup_architecture_key
    assert (
        _cfg(tmp_path, hand_embed_dim=32).setup_architecture_key
        != base.setup_architecture_key
    )
    # A main-net knob that touches neither embedder leaves the setup key alone.
    assert (
        _cfg(tmp_path, trunk_layers=(64, 64)).setup_architecture_key
        == base.setup_architecture_key
    )
    # But a setup state-trunk change IS a setup-architecture change.
    assert (
        _cfg(tmp_path, setup_trunk_layers=(32,)).setup_architecture_key
        != base.setup_architecture_key
    )
    # As is a setup choice-trunk change (the second tower).
    assert (
        _cfg(tmp_path, setup_choice_layers=(48,)).setup_architecture_key
        != base.setup_architecture_key
    )


def _write_main_checkpoint(tmp_path: pathlib.Path, cfg: config.TrainConfig) -> None:
    """A loadable ``last.pt`` so the main net genuinely resumes (the setup gate
    must act independently of the main resume)."""
    source = loop.TrainingLoop(cfg)
    progress = runstate.RunProgress(iteration=3, total_games=12)
    payload: dict[str, object] = {
        "config": cfg.model_dump(),
        "model": source.net.state_dict(),
        "optimizer": source.optimizer.state_dict(),
        "progress": progress.model_dump(),
        "git_sha": None,
    }
    torch.save(payload, tmp_path / artifacts.LAST_CKPT)


def test_stale_feature_dim_starts_setup_fresh(
    tmp_path: pathlib.Path,
):
    """A setup.pt recorded under an old feature-dim layout triggers a fresh start."""
    cfg = _cfg(tmp_path, resume=True)
    _write_main_checkpoint(tmp_path, cfg)
    # A setup checkpoint with the old 477-wide feature dim.
    stale_payload: dict[str, object] = {
        "setup_config": cfg.model_dump(),
        "setup_feature_dim": _OLD_FEATURE_DIM,
        "setup_model": {},
        "setup_optimizer": {},
    }
    torch.save(stale_payload, tmp_path / artifacts.SETUP_CKPT)

    training = loop.TrainingLoop(cfg)

    # Main net resumed; setup net started fresh.
    assert training.state.total_games == 12
    assert training._setup_net is not None
    feature_dim = training.config.setup_encoding.total_dim
    with torch.no_grad():
        out = training._setup_net(torch.zeros(1, feature_dim))
    assert out.shape == (1,)


def test_incompatible_weights_fall_back_fresh(tmp_path: pathlib.Path):
    """Belt-and-suspenders: a payload that passes the key check but whose
    weights cannot load (foreign state dict) rebuilds the setup net fresh."""
    cfg = _cfg(tmp_path, resume=True)
    _write_main_checkpoint(tmp_path, cfg)
    bad_payload: dict[str, object] = {
        "setup_config": cfg.model_dump(),
        "setup_feature_dim": cfg.setup_encoding.total_dim,
        "setup_model": {"nonexistent.weight": torch.zeros(1, 1)},
        "setup_optimizer": {},
    }
    torch.save(bad_payload, tmp_path / artifacts.SETUP_CKPT)

    training = loop.TrainingLoop(cfg)

    assert training._setup_net is not None
    feature_dim = training.config.setup_encoding.total_dim
    with torch.no_grad():
        out = training._setup_net(torch.zeros(1, feature_dim))
    assert out.shape == (1,)


def test_matching_setup_checkpoint_still_resumes(tmp_path: pathlib.Path):
    """The happy path keeps working: a setup.pt saved by the current layout
    resumes (readout weights) instead of being discarded. The frozen embedder
    copies are re-synced from the resumed main net right after, so the resume
    assertion pins the readout MLP — the part that actually persists."""
    cfg = _cfg(tmp_path, resume=True)
    source = loop.TrainingLoop(cfg)
    loop_setup.save_setup_checkpoint(source)
    _write_main_checkpoint(tmp_path, cfg)

    resumed = loop.TrainingLoop(cfg)
    assert resumed._setup_net is not None and source._setup_net is not None
    source_value = source._setup_net.value_head.state_dict()
    for name, tensor in resumed._setup_net.value_head.state_dict().items():
        assert torch.equal(tensor, source_value[name])
    # And the post-resume sync re-pinned the embedder copies to the main net.
    assert torch.equal(resumed._setup_net.card_table(), resumed.net.card_table())
