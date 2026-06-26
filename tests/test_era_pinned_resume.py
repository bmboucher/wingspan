# pyright: reportPrivateUsage=false
# (white-box tests of the loop's resume seam, the configurator's seeding, and
#  the mp/batched collectors' private net builders — the mechanisms under test)
"""Era-pinned training resume (docs/VERSIONING.md "Training resume: era pinning").

A run carries its artifact era in ``TrainConfig.encoding_version``: dims are
era-routed from it, the net is constructed as the era's compat class, and every
artifact the run writes is stamped with it. These tests pin the whole chain on
a synthetic **pre-field** v0.2 run directory (the production payload shape of a
run trained before ``encoding_version`` existed): the era is adopted from the
payload's ``version`` stamp, the loop resumes as ``PolicyValueNetV02`` at the
frozen 771-dim geometry, re-saved artifacts stay era-0.2, and the configurator
seeds a RESUMABLE working config from the saved run.
"""

from __future__ import annotations

import os
import pathlib
import sys
import typing

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

torch = pytest.importorskip("torch")

from wingspan import compat, encode, model, version  # noqa: E402
from wingspan.compat import v0_0, v0_1, v0_2, v0_4  # noqa: E402
from wingspan.training import (  # noqa: E402
    artifacts,
    batched_collect,
    config,
    loop,
    loop_checkpoint,
    loop_resume,
    metrics,
    mp_collect,
    runmeta,
    runstate,
)
from wingspan.training.configure import controller  # noqa: E402
from wingspan.training.configure import runs as config_runs  # noqa: E402

# The frozen pre-0.3 state width and pre-0.1 choice width, pinned as literals —
# these must never track the live layout (that is the point of the freeze).
_V02_STATE_DIM = 771
_V00_CHOICE_DIM = 397

# A deliberately tiny topology so era nets construct and play fast.
_SMALL_ARCH: dict[str, object] = {
    "trunk_layers": (8, 8),
    "choice_layers": (8, 8),
    "head_layers": (),
    "value_layers": (),
    "card_embed_dim": 4,
    "card_encoder_layers": (),
    "hand_encoder_layers": (8,),
}

_RESUME_ITERATION = 5  # the iteration the synthetic checkpoint was saved at


def _cfg(tmp_path: pathlib.Path, **overrides: object) -> config.TrainConfig:
    base: dict[str, object] = {
        "run_name": "era-test",
        "device": "cpu",
        "checkpoint_dir": str(tmp_path),
        "games_per_iter": 2,
        "eval_games": 2,
        **_SMALL_ARCH,
    }
    base.update(overrides)
    # The flat keys above are routed to their nested sections by the same
    # migration the loaders use for ≤0.4 artifacts.
    return config.run_config_from_artifact(base, version.MODEL_VERSION)


def _iter_metrics(iteration: int) -> metrics.IterationMetrics:
    family = metrics.FamilyCounts()
    family.bump(0)
    return metrics.IterationMetrics(
        iteration=iteration,
        total_games=12,
        games_this_iter=2,
        loss=1.0,
        policy_loss=0.5,
        value_loss=0.3,
        entropy=0.6,
        grad_norm=1.5,
        advantage_mean=0.0,
        advantage_std=1.0,
        avg_self_score=50.0,
        avg_margin=0.0,
        avg_breakdown=metrics.ScoreBreakdown(),
        avg_decisions=140.0,
        avg_winner_breakdown=metrics.ScoreBreakdown(),
        avg_abs_margin=0.0,
        margin_std=0.0,
        abs_margin_std=0.0,
        decisions_std=0.0,
        family_counts=family,
        collect_seconds=1.0,
        update_seconds=0.5,
        eval_seconds=0.0,
        games_per_sec=2.0,
    )


def _build_era_net(era_cfg: config.TrainConfig) -> model.PolicyValueNet:
    net_cls = model.PolicyValueNet.class_for_version(era_cfg.encoding_version)
    return net_cls(
        state_dim=era_cfg.state_dim,
        choice_dim=era_cfg.choice_dim,
        num_families=len(era_cfg.family_order),
        arch=era_cfg.arch,
        spec=era_cfg.encoding_spec,
    )


def _write_prefield_v02_checkpoint(
    tmp_path: pathlib.Path, *, model_state: dict[str, typing.Any] | None = None
) -> config.TrainConfig:
    """A ``last.pt`` exactly as a v0.2-era trainer left it: the embedded config
    has no ``encoding_version`` key (the field postdates the run) and the
    payload's ``version`` stamp — ``"0.2"`` — is the only era marker."""
    era_cfg = _cfg(tmp_path, encoding_version="0.2")
    net = _build_era_net(era_cfg)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    raw_config = era_cfg.model_dump()
    # Pre-field shape: the era marker postdates the run, so the embedded config
    # carries no encoding_version (now nested under ``architecture``).
    raw_config["architecture"].pop("encoding_version")
    payload: dict[str, typing.Any] = {
        "config": raw_config,
        "model": net.state_dict() if model_state is None else model_state,
        "optimizer": optimizer.state_dict(),
        "metrics": {},
        "progress": runstate.RunProgress(
            iteration=_RESUME_ITERATION, total_games=10
        ).model_dump(),
        "git_sha": None,
        "version": "0.2",
    }
    loop_checkpoint.atomic_save(payload, tmp_path / artifacts.LAST_CKPT)
    return era_cfg


#### The config carries its era ####


def test_default_config_is_live_era():
    cfg = config.RunConfig(misc=config.MiscConfig(device="cpu"))
    assert cfg.encoding_version == version.MODEL_VERSION
    assert cfg.state_dim == encode.state_size(cfg.encoding_spec)
    assert cfg.architecture_key[0] == version.MODEL_VERSION


def test_era_pinned_config_derives_frozen_dims(tmp_path: pathlib.Path):
    live = _cfg(tmp_path)
    pinned = _cfg(tmp_path, encoding_version="0.2")
    assert pinned.state_dim == _V02_STATE_DIM
    # v0.2 choice geometry is the pre-0.6 frozen dim (becomes_playable was added in 0.6).
    assert pinned.choice_dim == v0_4.choice_feature_dim_v04(pinned.encoding_spec)
    assert pinned.architecture_key[0] == "0.2"
    assert pinned.architecture_key != live.architecture_key


def test_unknown_or_future_eras_are_rejected(tmp_path: pathlib.Path):
    for bad in ("1.0", "garbage"):
        with pytest.raises(ValueError):
            _cfg(tmp_path, encoding_version=bad)


def test_encoding_dims_for_era_routes_each_era():
    spec = encode.spec_for(True)
    live_dims = (encode.state_size(spec), encode.choice_feature_dim(spec))
    assert compat.encoding_dims_for_era("0.0", spec) == (
        _V02_STATE_DIM,
        _V00_CHOICE_DIM,
    )
    # v0.1 and v0.2 predate the v0.6 becomes_playable stripe; their choice dim is frozen.
    pre_v06_choice_dim = v0_4.choice_feature_dim_v04(spec)
    assert compat.encoding_dims_for_era("0.1", spec) == (
        _V02_STATE_DIM,
        pre_v06_choice_dim,
    )
    assert compat.encoding_dims_for_era("0.2", spec) == (
        _V02_STATE_DIM,
        pre_v06_choice_dim,
    )
    assert compat.encoding_dims_for_era(version.MODEL_VERSION, spec) == live_dims
    # The include-setup axis stays orthogonal to the era axis.
    setup_spec = encode.spec_for(False)
    assert compat.encoding_dims_for_era(version.MODEL_VERSION, setup_spec) == (
        encode.state_size(setup_spec),
        encode.choice_feature_dim(setup_spec),
    )


def test_class_for_version_routes_each_era():
    assert model.PolicyValueNet.class_for_version("0.0") is v0_0.PolicyValueNetV00
    assert model.PolicyValueNet.class_for_version("0.1") is v0_1.PolicyValueNetV01
    assert model.PolicyValueNet.class_for_version("0.2") is v0_2.PolicyValueNetV02
    live_cls = model.PolicyValueNet.class_for_version(version.MODEL_VERSION)
    assert live_cls is model.PolicyValueNet


def test_train_config_from_artifact_adopts_the_payload_stamp(tmp_path: pathlib.Path):
    raw = _cfg(tmp_path).model_dump()
    raw["architecture"].pop("encoding_version")
    adopted = config.train_config_from_artifact(raw, "0.2")
    assert adopted.encoding_version == "0.2"
    assert adopted.state_dim == _V02_STATE_DIM
    # A config that already carries the field keeps it — the field is the
    # config's own record; the stamp only fills the pre-field gap.
    explicit = _cfg(tmp_path).model_dump()  # architecture.encoding_version == live
    kept = config.train_config_from_artifact(explicit, "0.2")
    assert kept.encoding_version == version.MODEL_VERSION


def test_with_encoding_version_resyncs_dims(tmp_path: pathlib.Path):
    live = _cfg(tmp_path)
    pinned = config.with_encoding_version(live, "0.2")
    assert (pinned.encoding_version, pinned.state_dim) == ("0.2", _V02_STATE_DIM)
    unpinned = config.with_encoding_version(pinned, version.MODEL_VERSION)
    assert unpinned.state_dim == live.state_dim


#### Era adoption from the run directory ####


def test_adopt_checkpoint_era_pins_only_when_it_reconciles(tmp_path: pathlib.Path):
    _write_prefield_v02_checkpoint(tmp_path)

    # A live config whose only difference is the era: pinned.
    pinned = loop_resume.adopt_checkpoint_era(_cfg(tmp_path))
    assert pinned.encoding_version == "0.2"
    assert pinned.state_dim == _V02_STATE_DIM

    # A genuinely different architecture: left alone (the gate will refuse).
    other = loop_resume.adopt_checkpoint_era(_cfg(tmp_path, trunk_layers=(16, 16)))
    assert other.encoding_version == version.MODEL_VERSION

    # Resume disabled, or no checkpoint at all: left alone.
    no_resume = loop_resume.adopt_checkpoint_era(_cfg(tmp_path, resume=False))
    assert no_resume.encoding_version == version.MODEL_VERSION
    empty_dir = tmp_path / "empty"
    untouched = loop_resume.adopt_checkpoint_era(_cfg(empty_dir))
    assert untouched.encoding_version == version.MODEL_VERSION


def test_adopt_checkpoint_era_unpins_fresh_launches(tmp_path: pathlib.Path):
    """The other direction of the seam: any launch that will not resume must
    not inherit a stale era — the config is re-keyed at the live version."""
    _write_prefield_v02_checkpoint(tmp_path)

    # Resume disabled: a pinned config (e.g. seeded by the configurator from
    # the saved run, then launched as a new run) un-pins to the live era.
    fresh = loop_resume.adopt_checkpoint_era(
        _cfg(tmp_path, encoding_version="0.2", resume=False)
    )
    assert fresh.encoding_version == version.MODEL_VERSION
    assert fresh.state_dim != _V02_STATE_DIM

    # Resume enabled but nothing to resume: same un-pin.
    empty_dir = tmp_path / "empty"
    no_checkpoint = loop_resume.adopt_checkpoint_era(
        _cfg(empty_dir, encoding_version="0.2")
    )
    assert no_checkpoint.encoding_version == version.MODEL_VERSION

    # Resume enabled but the architecture genuinely differs (the gate will
    # refuse and start fresh): the pinned era is dropped too.
    mismatched = loop_resume.adopt_checkpoint_era(
        _cfg(tmp_path, encoding_version="0.2", trunk_layers=(16, 16))
    )
    assert mismatched.encoding_version == version.MODEL_VERSION


def test_fresh_launch_over_old_era_dir_trains_and_stamps_live(
    tmp_path: pathlib.Path,
):
    """The headline fix: a new run started over a 0.2-era directory with a
    working config still seeded at that era trains at the live version — the
    live net class, live dims, and a ``model_config.json`` stamped with the
    current MODEL_VERSION instead of 0.2."""
    _write_prefield_v02_checkpoint(tmp_path)
    training = loop.TrainingLoop(_cfg(tmp_path, encoding_version="0.2", resume=False))

    assert training.config.encoding_version == version.MODEL_VERSION
    assert type(training.net) is model.PolicyValueNet
    assert training.net.state_dim != _V02_STATE_DIM
    assert training._start_iteration == 0

    descriptor = runmeta.read_model_config(str(tmp_path))
    assert descriptor.version == version.MODEL_VERSION


def test_training_loop_resumes_a_prefield_v02_run(tmp_path: pathlib.Path):
    """The headline guarantee: a pre-field 0.2 run dir handed a *live* config
    resumes era-pinned — the V02 net at 771 dims, progress restored — and the
    resume event names the pin."""
    _write_prefield_v02_checkpoint(tmp_path)
    training = loop.TrainingLoop(_cfg(tmp_path))

    assert training.config.encoding_version == "0.2"
    assert isinstance(training.net, v0_2.PolicyValueNetV02)
    assert training.net.state_dim == _V02_STATE_DIM
    assert training._start_iteration == _RESUME_ITERATION + 1
    assert training.state.total_games == 10
    resumed_events = [e.text for e in training.state.events if "resumed" in e.text]
    assert resumed_events and "era 0.2 (pinned)" in resumed_events[-1]


def test_resumed_run_keeps_stamping_its_own_era(tmp_path: pathlib.Path):
    """Everything an era-pinned run writes reads as its era: the checkpoint
    payload, its embedded config, the run descriptor — and the play loaders
    route the re-saved artifact back to the era class."""
    import wingspan.players.loaders as loaders

    _write_prefield_v02_checkpoint(tmp_path)
    training = loop.TrainingLoop(_cfg(tmp_path))
    loop_checkpoint.checkpoint(training, _iter_metrics(_RESUME_ITERATION + 1), None, [])

    payload = typing.cast(
        "dict[str, typing.Any]",
        torch.load(
            tmp_path / artifacts.LAST_CKPT, map_location="cpu", weights_only=False
        ),
    )
    assert payload["version"] == "0.2"
    assert payload["config"]["architecture"]["encoding_version"] == "0.2"
    assert payload["config"]["architecture"]["state_dim"] == _V02_STATE_DIM

    runmeta.write_model_config(str(tmp_path), training.config)
    descriptor = runmeta.read_model_config(str(tmp_path))
    assert (descriptor.version, descriptor.state_dim) == ("0.2", _V02_STATE_DIM)

    loaded, loaded_cfg = loaders.load_policy_net(
        tmp_path / artifacts.LAST_CKPT, torch.device("cpu")
    )
    assert isinstance(loaded, v0_2.PolicyValueNetV02)
    assert loaded_cfg.encoding_version == "0.2"


def test_resume_gate_weight_mismatch_alarms_and_starts_fresh(
    tmp_path: pathlib.Path,
):
    """The non-fatal contract holds even past the key check: a payload whose
    config matches but whose tensors cannot load (here: an empty model dict)
    alarms and starts fresh instead of crashing ``__init__``."""
    _write_prefield_v02_checkpoint(tmp_path, model_state={})
    training = loop.TrainingLoop(_cfg(tmp_path))
    assert training._start_iteration == 0
    assert training.state.total_games == 0
    alarms = [e.text for e in training.state.events if "do not fit" in e.text]
    assert alarms


#### The configurator flow ####


def test_configurator_seeds_the_era_and_reads_resumable(tmp_path: pathlib.Path):
    """The dashboard path: the saved summary carries the payload's era, the
    seeded working config adopts it (RESUMABLE), and a raw live config —
    which would rebuild a different net — correctly reads INCOMPATIBLE."""
    _write_prefield_v02_checkpoint(tmp_path)
    live = _cfg(tmp_path)
    summary = config_runs.inspect_run(str(tmp_path))
    assert summary.train_config is not None
    assert summary.train_config.encoding_version == "0.2"

    working, seeded = controller._seed_from_summary(live, summary)
    assert seeded and working.encoding_version == "0.2"
    assert (
        config_runs.resolve_status(summary, working) is config_runs.RunStatus.RESUMABLE
    )
    assert (
        config_runs.resolve_status(summary, live) is config_runs.RunStatus.INCOMPATIBLE
    )


#### The collectors honor the era ####


def test_worker_arch_and_builder_are_era_routed(tmp_path: pathlib.Path):
    era_cfg = _cfg(tmp_path, encoding_version="0.2")
    collector = mp_collect.ProcessCollector(era_cfg, num_workers=1)
    assert collector._arch.encoding_version == "0.2"
    worker_net = mp_collect._build_worker_net(collector._arch)
    assert isinstance(worker_net, v0_2.PolicyValueNetV02)
    assert worker_net.state_dim == _V02_STATE_DIM


def test_batched_collect_plays_through_an_era_net(tmp_path: pathlib.Path):
    """The batched (CUDA-path) collector encodes through the served net, so an
    era net collects era-width vectors — a live-encoder pairing would feed 790
    dims into a 771-dim trunk and could not even complete the game."""
    era_cfg = _cfg(tmp_path, encoding_version="0.2")
    net = _build_era_net(era_cfg)
    net.eval()
    records = batched_collect.collect_games(net, torch.device("cpu"), seeds=[11])
    assert len(records) == 1
    assert records[0].steps
    for step in records[0].steps:
        assert step.state.shape == (_V02_STATE_DIM,)
