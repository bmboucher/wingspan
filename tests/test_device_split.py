# pyright: reportPrivateUsage=false
# (calls configure.screen._detail_hint — deliberate intra-package coupling,
# same pattern as loop_*.py's own private-access header)
"""Tests for the ``misc.device`` → ``collect_device`` / ``train_device`` split.

Covers the seams introduced by the split: the legacy-key migration validator
(``MiscConfig._migrate_legacy_device``, exercised through a bare
``RunConfig``/``MiscConfig`` validation, the flat-artifact reshape, and a cloud
run-file), ``device_label``, the ``validate_launchable`` pairing + bootstrap
rules, the pure ``config.resolve_devices`` fallback helper, the dashboard's
``--collect-device`` / ``--train-device`` flags, the configurator's two
``ChoiceField``s and detail-panel hints, the user-defaults round trip, and
``TrainingLoop`` holding both device attributes with no process-wide thread
cap. The last test exercises an actual cpu-collect/cuda-train iteration and is
skipped on a CPU-only torch build.
"""

from __future__ import annotations

import json
import pathlib

import pytest

pytest.importorskip("torch")
pytest.importorskip("rich")
pytest.importorskip("boto3")

import torch

from wingspan import version
from wingspan.cloud import runfile
from wingspan.training import app, config, cpu_threads, loop, runstate
from wingspan.training.configure import controller, fields, screen, user_defaults

# ---------------------------------------------------------------------------
# Legacy-key migration
# ---------------------------------------------------------------------------


def test_legacy_device_key_seeds_both_roles():
    """A bare ``device`` key seeds both roles; an explicit new-key value wins
    over the legacy one; the legacy key never survives into a dump."""
    cfg = config.RunConfig.model_validate({"misc": {"device": "cuda"}})
    assert cfg.misc.collect_device == "cuda"
    assert cfg.misc.train_device == "cuda"

    explicit = config.MiscConfig.model_validate(
        {"device": "cpu", "train_device": "cuda"}
    )
    assert explicit.collect_device == "cpu"  # from the legacy key
    assert explicit.train_device == "cuda"  # explicit value wins

    assert "device" not in cfg.misc.model_dump()


def test_flat_legacy_device_migrates():
    """A ≤0.4 flat artifact dict's ``device`` key migrates through the
    flat→nested reshape into both new roles."""
    cfg = config.run_config_from_artifact(
        {"device": "cuda", "lr": 1e-3}, version.MODEL_VERSION
    )
    assert cfg.misc.collect_device == "cuda"
    assert cfg.misc.train_device == "cuda"
    assert cfg.training.lr == 1e-3


def test_cloud_run_file_legacy_device_key():
    """A cloud run-file's embedded ``train.misc.device`` migrates the same way
    (pydantic validates the nested ``misc`` dict through the same
    ``mode="before"`` validator, independent of the top-level entry point)."""
    yaml_text = """
run_name: testrun
train:
  misc:
    device: cuda
"""
    run = runfile.parse_run_file(yaml_text)
    assert run.train.misc.collect_device == "cuda"
    assert run.train.misc.train_device == "cuda"


# ---------------------------------------------------------------------------
# device_label
# ---------------------------------------------------------------------------


def test_device_label():
    """``device_label`` collapses to one word when both roles agree, and
    spells the split as ``collect→train`` otherwise."""
    same = config.MiscConfig(collect_device="cpu", train_device="cpu")
    assert same.device_label == "cpu"

    split = config.MiscConfig(collect_device="cpu", train_device="cuda")
    assert split.device_label == "cpu→cuda"


# ---------------------------------------------------------------------------
# validate_launchable
# ---------------------------------------------------------------------------


def test_validate_launchable_device_pairs():
    """The three supported pairs launch clean; a mismatched pair (in-process
    collection naming a different device than the learner) is flagged; a
    checkpoint bootstrap still requires cpu collection specifically, and no
    longer blocks a cpu-collect/cuda-train pairing."""
    for collect, train in (("cpu", "cpu"), ("cpu", "cuda"), ("cuda", "cuda")):
        cfg = config.RunConfig(
            misc=config.MiscConfig(collect_device=collect, train_device=train)
        )
        assert config.validate_launchable(cfg) == []

    mismatched = config.RunConfig(
        misc=config.MiscConfig(collect_device="cuda", train_device="cpu")
    )
    problems = config.validate_launchable(mismatched)
    assert any("train_device" in problem for problem in problems)

    indexed = config.RunConfig(
        misc=config.MiscConfig(collect_device="cuda:1", train_device="cuda:0")
    )
    problems = config.validate_launchable(indexed)
    assert any("train_device" in problem for problem in problems)

    bootstrap_on_cuda = config.RunConfig(
        misc=config.MiscConfig(collect_device="cuda", train_device="cuda"),
        opponent=config.OpponentConfig(bootstrap_opponent="some/path.pt"),
    )
    problems = config.validate_launchable(bootstrap_on_cuda)
    assert any("collect_device='cpu'" in problem for problem in problems)

    # The new capability: a checkpoint bootstrap pairs fine with a cpu-collect
    # / cuda-train run, since the bootstrap opponent only ever runs through
    # the cpu worker pool.
    bootstrap_split = config.RunConfig(
        misc=config.MiscConfig(collect_device="cpu", train_device="cuda"),
        opponent=config.OpponentConfig(bootstrap_opponent="some/path.pt"),
    )
    problems = config.validate_launchable(bootstrap_split)
    assert not any("device" in problem for problem in problems)


# ---------------------------------------------------------------------------
# resolve_devices
# ---------------------------------------------------------------------------


def test_resolve_devices():
    """Every ``cuda`` role downgrades to ``cpu`` when CUDA is unavailable;
    nothing changes (and the same object is returned) when it is, or when a
    config has no ``cuda`` role to begin with."""
    cpu_cpu = config.RunConfig(
        misc=config.MiscConfig(collect_device="cpu", train_device="cpu")
    )
    assert config.resolve_devices(cpu_cpu, cuda_available=True) is cpu_cpu
    assert config.resolve_devices(cpu_cpu, cuda_available=False) is cpu_cpu

    cpu_cuda = config.RunConfig(
        misc=config.MiscConfig(collect_device="cpu", train_device="cuda")
    )
    resolved = config.resolve_devices(cpu_cuda, cuda_available=False)
    assert resolved.misc.collect_device == "cpu"
    assert resolved.misc.train_device == "cpu"

    cuda_cuda = config.RunConfig(
        misc=config.MiscConfig(collect_device="cuda", train_device="cuda")
    )
    resolved = config.resolve_devices(cuda_cuda, cuda_available=False)
    assert resolved.misc.collect_device == "cpu"
    assert resolved.misc.train_device == "cpu"


# ---------------------------------------------------------------------------
# Dashboard flags
# ---------------------------------------------------------------------------


def test_dashboard_flags_build_both_roles():
    """``--collect-device`` / ``--train-device`` build both roles; the retired
    ``--device`` flag is no longer recognized."""
    args = app._parse_args(["--collect-device", "cpu", "--train-device", "cuda"])
    cfg = app._config_from_namespace(args)
    assert cfg.misc.collect_device == "cpu"
    assert cfg.misc.train_device == "cuda"

    with pytest.raises(SystemExit):
        app._parse_args(["--device", "cpu"])


# ---------------------------------------------------------------------------
# Configurator
# ---------------------------------------------------------------------------


def test_configurator_device_fields_replace_single_device():
    """``collect_device`` / ``train_device`` are RUN SETTINGS choice fields;
    the retired ``device`` attr is no longer a valid field spec."""
    collect_spec = fields.spec_for("collect_device")
    train_spec = fields.spec_for("train_device")
    assert isinstance(collect_spec, fields.ChoiceField)
    assert isinstance(train_spec, fields.ChoiceField)
    assert collect_spec.group_path == ("RUN SETTINGS",)
    assert train_spec.group_path == ("RUN SETTINGS",)

    with pytest.raises(KeyError):
        fields.spec_for("device")


def test_detail_hint_device_roles(tmp_path: pathlib.Path):
    """The detail-panel hint falls back to a cuda-unavailable note per role,
    and separately flags a collect/train pairing mismatch when cuda IS
    available."""
    collect_spec = fields.spec_for("collect_device")
    train_spec = fields.spec_for("train_device")

    fallback_cfg = config.RunConfig(
        misc=config.MiscConfig(collect_device="cuda", train_device="cuda"),
        run=config.RunSettings(checkpoint_dir=str(tmp_path)),
    )
    fallback_view = controller.build_initial_state(fallback_cfg, cuda_available=False)
    assert (
        screen._detail_hint(fallback_view, collect_spec)
        == "→ cuda unavailable — will fall back to cpu"
    )
    assert (
        screen._detail_hint(fallback_view, train_spec)
        == "→ cuda unavailable — will fall back to cpu"
    )

    pairing_cfg = config.RunConfig(
        misc=config.MiscConfig(collect_device="cuda", train_device="cpu"),
        run=config.RunSettings(checkpoint_dir=str(tmp_path)),
    )
    pairing_view = controller.build_initial_state(pairing_cfg, cuda_available=True)
    assert (
        screen._detail_hint(pairing_view, collect_spec)
        == "→ cuda collection requires train device = cuda"
    )


# ---------------------------------------------------------------------------
# User defaults
# ---------------------------------------------------------------------------


def test_user_defaults_round_trip_keeps_both_roles(tmp_path: pathlib.Path):
    """Neither device role (nor the legacy key) is ever persisted to the
    defaults file; loading always takes both roles from the caller's current
    config, whether or not a saved file carries the pre-split key."""
    saved_cfg = config.RunConfig(
        misc=config.MiscConfig(collect_device="cuda", train_device="cuda"),
        training=config.TrainingConfig(lr=5e-4),
    )
    path = user_defaults.save_defaults(saved_cfg, directory=tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    misc_settings = raw["settings"].get("misc", {})
    assert "collect_device" not in misc_settings
    assert "train_device" not in misc_settings
    assert "device" not in misc_settings

    current = config.RunConfig(
        misc=config.MiscConfig(collect_device="cpu", train_device="cuda")
    )
    loaded = user_defaults.load_defaults(current, directory=tmp_path)
    assert loaded.warning is None and loaded.train_config is not None
    assert loaded.train_config.misc.collect_device == "cpu"
    assert loaded.train_config.misc.train_device == "cuda"

    # A hand-written defaults file still carrying the pre-split legacy key
    # loads fine too — the current run's roles win either way, since device
    # roles are per-run identity and never sourced from the defaults file.
    legacy_envelope = {
        "saved_with_version": version.MODEL_VERSION,
        "saved_at": "2026-01-01T00:00:00",
        "settings": {"misc": {"device": "cuda"}},
    }
    (tmp_path / user_defaults.DEFAULTS_FILENAME).write_text(
        json.dumps(legacy_envelope), encoding="utf-8"
    )
    loaded_legacy = user_defaults.load_defaults(current, directory=tmp_path)
    assert loaded_legacy.warning is None and loaded_legacy.train_config is not None
    assert loaded_legacy.train_config.misc.collect_device == "cpu"
    assert loaded_legacy.train_config.misc.train_device == "cuda"


# ---------------------------------------------------------------------------
# TrainingLoop
# ---------------------------------------------------------------------------


def _tiny_config(
    tmp_path: pathlib.Path,
    *,
    collect_device: str = "cpu",
    train_device: str = "cpu",
    games_per_iter: int = 256,
    max_iterations: int = 0,
    eval_every: int = 5,
) -> config.TrainConfig:
    """A fast-building config mirroring ``test_loop_setup_update.py``'s tiny
    helper, with the loop-shape knobs the CUDA iteration test needs exposed."""
    return config.RunConfig(
        misc=config.MiscConfig(
            collect_device=collect_device, train_device=train_device
        ),
        run=config.RunSettings(
            checkpoint_dir=str(tmp_path),
            resume=False,
            games_per_iter=games_per_iter,
            max_iterations=max_iterations,
            eval_every=eval_every,
        ),
        architecture=config.ArchitectureConfig(
            use_setup_model=True,
            main=config.MainNetArchitecture(
                trunk_layers=(32, 32),
                choice_layers=(32, 32),
                card_embed_dim=8,
            ),
            setup=config.SetupNetArchitecture(head_layers=(16,)),
        ),
    )


def test_loop_holds_both_roles_and_keeps_thread_count(tmp_path: pathlib.Path):
    """``TrainingLoop`` exposes ``collect_device`` / ``train_device`` (no
    ``device`` attribute), and no longer pins the process-wide torch thread
    count at construction time (the cap moved to ``cpu_threads``)."""
    torch.set_num_threads(4)
    training = loop.TrainingLoop(_tiny_config(tmp_path))
    assert training.collect_device == torch.device("cpu")
    assert training.train_device == torch.device("cpu")
    assert next(training.net.parameters()).device.type == "cpu"
    assert torch.get_num_threads() == 4  # untouched — conftest restores 2 after


def test_inference_thread_cap():
    """The context manager caps a CPU device to the measured sweet spot,
    restores the prior count after, never raises the count, and is a no-op
    for a non-cpu device (constructible without an actual CUDA build)."""
    torch.set_num_threads(6)
    with cpu_threads.inference_thread_cap(torch.device("cpu")):
        assert torch.get_num_threads() == cpu_threads.INFERENCE_INTRAOP_THREADS
    assert torch.get_num_threads() == 6

    torch.set_num_threads(1)
    with cpu_threads.inference_thread_cap(torch.device("cpu")):
        assert torch.get_num_threads() == 1  # already at/below the cap
    assert torch.get_num_threads() == 1

    torch.set_num_threads(6)
    with cpu_threads.inference_thread_cap(torch.device("cuda")):
        assert torch.get_num_threads() == 6  # non-cpu device — no-op


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA torch build")
def test_cpu_collect_cuda_train_iteration(tmp_path: pathlib.Path):
    """An actual cpu-collect / cuda-train iteration runs end to end: the pool
    collects on cpu, the learner's net lives and updates on cuda."""
    cfg = _tiny_config(
        tmp_path,
        collect_device="cpu",
        train_device="cuda",
        games_per_iter=2,
        max_iterations=1,
        eval_every=0,
    )
    training = loop.TrainingLoop(cfg)
    training.run()
    assert training.state.error is None
    assert training.state.phase is runstate.Phase.DONE
    assert training.state.last_iter is not None
    assert next(training.net.parameters()).device.type == "cuda"
