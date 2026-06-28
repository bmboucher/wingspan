"""The run-directory load paths in ``wingspan.players.loaders``.

``load_policy_net`` (a self-describing ``.pt`` payload) is exercised through the
bootstrap/dagger/resume tests; this file covers the *run-directory* loaders the
tournament and ``play`` CLIs use, which rebuild a net from the unified
``run_config_<stamp>.json`` descriptor rather than the payload:

* :func:`loaders.load_policy_net_from_run_dir` (+ the private ``_reconstruct_net``)
  — rebuild the main net in its saved shape and seat its weights.
* :func:`loaders.load_setup_net` — rebuild the optional ``SetupNet`` when both the
  setup checkpoint and a run-config sidecar are present, degrading to ``None``
  (random setup picks) when either is absent.

At the 1.0 baseline every loadable run is at the live era, so these resolve to
the live net classes at the live dims.
"""

from __future__ import annotations

import pathlib

import pytest

torch = pytest.importorskip("torch")

from wingspan import model, version  # noqa: E402
from wingspan.players import loaders  # noqa: E402
from wingspan.training import (  # noqa: E402
    artifacts,
    config,
    loop_checkpoint,
    runmeta,
    runstate,
)
from wingspan.training import setup_net as setup_net_module  # noqa: E402
from wingspan.training import (  # noqa: E402
    setup_runmeta,
)


def _setup_cfg(checkpoint_dir: pathlib.Path) -> config.RunConfig:
    """A tiny setup-enabled live-era run config (fast to build/seat)."""
    return config.RunConfig(
        misc=config.MiscConfig(device="cpu"),
        run=config.RunSettings(
            run_name="loader-test", checkpoint_dir=str(checkpoint_dir)
        ),
        architecture=config.ArchitectureConfig(
            main=config.MainNetArchitecture(
                trunk_layers=(8, 8),
                choice_layers=(8, 8),
                head_layers=(),
                value_layers=(),
                card_embed_dim=4,
                card_encoder_layers=(),
                hand_encoder_layers=(8,),
            ),
            setup=config.SetupNetArchitecture(hidden_layers=(8,)),
        ),
    )


def _build_net(cfg: config.RunConfig) -> model.PolicyValueNet:
    return model.PolicyValueNet(
        state_dim=cfg.state_dim,
        choice_dim=cfg.choice_dim,
        num_families=len(cfg.family_order),
        arch=cfg.arch,
        spec=cfg.encoding_spec,
    )


def _write_run_dir(
    checkpoint_dir: pathlib.Path, *, with_setup: bool
) -> config.RunConfig:
    """Lay down a complete run directory the run-dir loaders accept: the unified
    ``run_config_<stamp>.json`` descriptor, a ``last.pt`` carrying a real
    state_dict, and (optionally) a ``setup.pt``. No training run required."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg = _setup_cfg(checkpoint_dir)
    runmeta.write_run_config(
        str(checkpoint_dir),
        cfg,
        stamp="t0",
        started_at="t0",
        git_sha=None,
        resumed_from_iteration=0,
    )
    main_payload: dict[str, object] = {
        "config": cfg.model_dump(),
        "model": _build_net(cfg).state_dict(),
        "progress": runstate.RunProgress(iteration=1, total_games=2).model_dump(),
        "git_sha": None,
        "version": version.MODEL_VERSION,
    }
    loop_checkpoint.atomic_save(main_payload, checkpoint_dir / artifacts.LAST_CKPT)
    if with_setup:
        descriptor = setup_runmeta.read_setup_config(str(checkpoint_dir))
        setup_payload: dict[str, object] = {
            "setup_model": setup_net_module.SetupNet.from_setup_config(
                descriptor
            ).state_dict(),
            "version": version.MODEL_VERSION,
        }
        loop_checkpoint.atomic_save(
            setup_payload, checkpoint_dir / artifacts.SETUP_CKPT
        )
    return cfg


def test_load_policy_net_from_run_dir_rebuilds_live_net(tmp_path: pathlib.Path):
    """The tournament path: rebuild the main net from the run's descriptor and
    seat its weights, at the live dims."""
    cfg = _write_run_dir(tmp_path, with_setup=False)
    net = loaders.load_policy_net_from_run_dir(str(tmp_path), torch.device("cpu"))
    assert isinstance(net, model.PolicyValueNet)
    assert net.state_dim == cfg.state_dim
    assert not net.training  # seated in eval mode


def test_load_setup_net_present(tmp_path: pathlib.Path):
    """A run with both a setup checkpoint and a config sidecar yields a SetupNet."""
    _write_run_dir(tmp_path, with_setup=True)
    setup_net = loaders.load_setup_net(tmp_path, torch.device("cpu"))
    assert isinstance(setup_net, setup_net_module.SetupNet)


def test_load_setup_net_absent_returns_none(tmp_path: pathlib.Path):
    """No setup checkpoint (a run trained without a setup model) → ``None`` so
    play degrades to random setup picks rather than erroring."""
    _write_run_dir(tmp_path, with_setup=False)
    assert loaders.load_setup_net(tmp_path, torch.device("cpu")) is None
    # An empty directory (no config sidecar at all) likewise degrades to None.
    empty = tmp_path / "empty"
    empty.mkdir()
    assert loaders.load_setup_net(empty, torch.device("cpu")) is None


def test_load_policy_net_round_trips_a_self_describing_payload(tmp_path: pathlib.Path):
    """The ``play``-CLI path: rebuild from the ``.pt``'s own embedded config and
    return the parsed config alongside the seated net."""
    cfg = _setup_cfg(tmp_path)
    payload: dict[str, object] = {
        "config": cfg.model_dump(),
        "model": _build_net(cfg).state_dict(),
        "version": version.MODEL_VERSION,
    }
    path = tmp_path / artifacts.LAST_CKPT
    loop_checkpoint.atomic_save(payload, path)
    net, parsed = loaders.load_policy_net(path, torch.device("cpu"))
    assert isinstance(net, model.PolicyValueNet)
    assert parsed.encoding_version == version.MODEL_VERSION
    assert net.state_dim == cfg.state_dim


def test_load_policy_net_missing_file_raises(tmp_path: pathlib.Path):
    with pytest.raises(FileNotFoundError):
        loaders.load_policy_net(tmp_path / "nope.pt", torch.device("cpu"))


def test_load_policy_net_refuses_pre_1_0_payload(tmp_path: pathlib.Path):
    """A payload stamped with a pre-1.0 era is a different MAJOR and is refused."""
    cfg = _setup_cfg(tmp_path)
    payload: dict[str, object] = {
        "config": cfg.model_dump(),
        "model": _build_net(cfg).state_dict(),
        "version": "0.2",
    }
    path = tmp_path / artifacts.LAST_CKPT
    loop_checkpoint.atomic_save(payload, path)
    with pytest.raises(version.IncompatibleArtifactError):
        loaders.load_policy_net(path, torch.device("cpu"))
