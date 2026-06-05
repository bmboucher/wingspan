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

import json
import os
import pathlib
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

torch = pytest.importorskip("torch")

from wingspan import setup_model  # noqa: E402
from wingspan.training import artifacts, config, loop, runstate  # noqa: E402

_OLD_FEATURE_DIM = 477  # the pre-shared-embedder layout's width


def _cfg(tmp_path: pathlib.Path, **overrides: object) -> config.TrainConfig:
    base: dict[str, object] = {
        "device": "cpu",
        "checkpoint_dir": str(tmp_path),
        "use_setup_model": True,
        "trunk_layers": (32, 32),
        "choice_layers": (32, 32),
        "card_embed_dim": 8,
        "setup_hidden_layers": (16,),
    }
    base.update(overrides)
    return config.TrainConfig.model_validate(base)


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
    meanpool = _cfg(tmp_path, use_distinct_hand_model=False, tray_set_embedding=False)
    assert meanpool.setup_architecture_key != base.setup_architecture_key
    assert (
        _cfg(tmp_path, hand_embed_dim=32).setup_architecture_key
        != base.setup_architecture_key
    )
    # A main-net knob that touches neither embedder leaves the setup key alone.
    assert (
        _cfg(tmp_path, trunk_layers=(64, 64)).setup_architecture_key
        == base.setup_architecture_key
    )


def _write_store_row(path: pathlib.Path, width: int) -> None:
    row = {"features": [0.0] * width, "margin": 1.0, "iteration": 1}
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


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


def test_stale_feature_dim_starts_setup_fresh_and_clears_store(
    tmp_path: pathlib.Path,
):
    cfg = _cfg(tmp_path, resume=True)
    _write_main_checkpoint(tmp_path, cfg)
    # A setup checkpoint recorded under the old 477-wide layout: the saved
    # config validates to the *current* key (both sides recompute from current
    # code), so only the persisted feature dim can expose the stale layout.
    stale_payload: dict[str, object] = {
        "setup_config": cfg.model_dump(),
        "setup_feature_dim": _OLD_FEATURE_DIM,
        "setup_model": {},
        "setup_optimizer": {},
        "setup_fit_done": True,
    }
    torch.save(stale_payload, tmp_path / artifacts.SETUP_CKPT)
    store_path = tmp_path / artifacts.SETUP_DATA_LOG
    _write_store_row(store_path, _OLD_FEATURE_DIM)

    training = loop.TrainingLoop(cfg)

    # Main net resumed; setup net started fresh with the stale store cleared.
    assert training.state.total_games == 12
    assert training._setup_fit_done is False
    assert training._setup_store is not None
    assert training._setup_store.count() == 0
    assert training._setup_net is not None
    with torch.no_grad():
        out = training._setup_net(torch.zeros(1, setup_model.SETUP_FEATURE_DIM))
    assert out.shape == (1,)


def test_incompatible_weights_fall_back_fresh(tmp_path: pathlib.Path):
    """Belt-and-suspenders: a payload that passes the key check but whose
    weights cannot load (foreign state dict) rebuilds the setup net fresh and
    clears the store rather than crashing."""
    cfg = _cfg(tmp_path, resume=True)
    _write_main_checkpoint(tmp_path, cfg)
    bad_payload: dict[str, object] = {
        "setup_config": cfg.model_dump(),
        "setup_feature_dim": setup_model.SETUP_FEATURE_DIM,
        "setup_model": {"mlp.0.weight": torch.zeros(1, 1)},
        "setup_optimizer": {},
        "setup_fit_done": True,
    }
    torch.save(bad_payload, tmp_path / artifacts.SETUP_CKPT)
    store_path = tmp_path / artifacts.SETUP_DATA_LOG
    _write_store_row(store_path, setup_model.SETUP_FEATURE_DIM)

    training = loop.TrainingLoop(cfg)

    assert training._setup_fit_done is False
    assert training._setup_store is not None
    assert training._setup_store.count() == 0
    assert training._setup_net is not None
    with torch.no_grad():
        out = training._setup_net(torch.zeros(1, setup_model.SETUP_FEATURE_DIM))
    assert out.shape == (1,)


def test_matching_setup_checkpoint_still_resumes(tmp_path: pathlib.Path):
    """The happy path keeps working: a setup.pt saved by the current layout
    resumes (readout weights, fit flag) instead of being discarded. The frozen
    embedder copies are re-synced from the resumed main net right after, so the
    resume assertion pins the readout MLP — the part that actually persists."""
    cfg = _cfg(tmp_path, resume=True)
    source = loop.TrainingLoop(cfg)
    source._setup_fit_done = True
    source._save_setup_checkpoint()
    _write_main_checkpoint(tmp_path, cfg)

    resumed = loop.TrainingLoop(cfg)
    assert resumed._setup_fit_done is True
    assert resumed._setup_net is not None and source._setup_net is not None
    source_mlp = source._setup_net.mlp.state_dict()
    for name, tensor in resumed._setup_net.mlp.state_dict().items():
        assert torch.equal(tensor, source_mlp[name])
    # And the post-resume sync re-pinned the embedder copies to the main net.
    assert torch.equal(resumed._setup_net.card_table(), resumed.net.card_table())
