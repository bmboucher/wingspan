# pyright: reportPrivateUsage=false
# (white-box tests of the loop's resume seam, the configurator's seeding, and
#  the mp/batched collectors' private net builders — the mechanisms under test)
"""Era-pinned training resume (docs/VERSIONING.md "Training resume: era pinning").

A run carries its artifact era in ``RunConfig.encoding_version``: dims are
era-routed from it, the net is constructed as the era's net class, and every
artifact the run writes is stamped with it. Since the v1.4 FRESH bump (the two
food-unlock state stripes) there are superseded same-MAJOR eras again: 1.0-1.3
route to compat net classes and derive a narrower state_dim than the live 1.4,
while ``check_artifact_compatible`` still refuses any different-MAJOR or
future-MINOR era. These tests exercise the whole chain — the dims router, the
version-routed net class, the configurator's seeding, and the loop's
resume/collector seams — and pin the era-divergence for the pre-1.4 shims.
"""

from __future__ import annotations

import pathlib
import typing

import pytest

torch = pytest.importorskip("torch")

from wingspan import compat, encode, model, version  # noqa: E402
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

_RESUME_ITERATION = 5  # the iteration the synthetic checkpoint was saved at


def _cfg(
    tmp_path: pathlib.Path,
    *,
    resume: bool = True,
    trunk_layers: tuple[int, ...] = (8, 8),
) -> config.RunConfig:
    """A tiny live-era run config rooted at ``tmp_path`` (fast to construct/play)."""
    return config.RunConfig(
        misc=config.MiscConfig(device="cpu"),
        run=config.RunSettings(
            run_name="era-test",
            checkpoint_dir=str(tmp_path),
            games_per_iter=2,
            eval_games=2,
            resume=resume,
        ),
        architecture=config.ArchitectureConfig(
            main=config.MainNetArchitecture(
                trunk_layers=trunk_layers,
                choice_layers=(8, 8),
                head_layers=(),
                value_layers=(),
                card_embed_dim=4,
                card_encoder_layers=(),
                hand_encoder_layers=(8,),
            ),
        ),
    )


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


def _build_net(cfg: config.RunConfig) -> model.PolicyValueNet:
    net_cls = model.PolicyValueNet.class_for_version(cfg.encoding_version)
    return net_cls(
        state_dim=cfg.state_dim,
        choice_dim=cfg.choice_dim,
        num_families=len(cfg.family_order),
        arch=cfg.arch,
        spec=cfg.encoding_spec,
    )


def _write_live_checkpoint(
    tmp_path: pathlib.Path, *, model_state: dict[str, typing.Any] | None = None
) -> config.RunConfig:
    """A ``last.pt`` exactly as a live-era trainer left it: its embedded config
    and ``version`` stamp both carry the live MODEL_VERSION."""
    cfg = _cfg(tmp_path)
    net = _build_net(cfg)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    payload: dict[str, typing.Any] = {
        "config": cfg.model_dump(),
        "model": net.state_dict() if model_state is None else model_state,
        "optimizer": optimizer.state_dict(),
        "metrics": {},
        "progress": runstate.RunProgress(
            iteration=_RESUME_ITERATION, total_games=10
        ).model_dump(),
        "git_sha": None,
        "version": version.MODEL_VERSION,
    }
    loop_checkpoint.atomic_save(payload, tmp_path / artifacts.LAST_CKPT)
    return cfg


#### The config carries its era ####


def test_default_config_is_live_era():
    cfg = config.RunConfig(misc=config.MiscConfig(device="cpu"))
    assert cfg.encoding_version == version.MODEL_VERSION
    assert cfg.state_dim == encode.state_size(cfg.encoding_spec)
    assert cfg.architecture_key[0] == version.MODEL_VERSION


def test_unknown_or_future_eras_are_rejected():
    """The era validator refuses any era this code cannot load: a different
    MAJOR, a future MINOR, or a malformed string. The live era is accepted."""
    live = config.RunConfig(misc=config.MiscConfig(device="cpu"))
    current = version.parse_version(version.MODEL_VERSION)
    future_minor = f"{current.major}.{current.minor + 1}"
    for bad in ("0.9", "2.0", future_minor, "garbage"):
        with pytest.raises(ValueError):
            config.with_encoding_version(live, bad)
    assert (
        config.with_encoding_version(live, version.MODEL_VERSION).encoding_version
        == version.MODEL_VERSION
    )


def test_encoding_dims_for_era_state_narrows_pre_1_4():
    """The v1.4 bump narrows ``state_dim`` by the two food-unlock stripes (10) and
    ``choice_dim`` by the ``resets_feeder`` stripe (1) for every pre-1.4 same-MAJOR
    era; 0.x eras (untouched by the major-1 router branch) and the live era keep the
    live widths. A malformed string is rejected."""
    spec = encode.spec_for(True)
    live_state = encode.state_size(spec)
    live_choice = encode.choice_feature_dim(spec)
    stripe_width = 2 * encode.STATE_FOOD_UNLOCK_DIM
    for era in ("1.1", "1.2", "1.3"):
        state_dim, choice_dim = compat.encoding_dims_for_era(era, spec)
        assert live_state - state_dim == stripe_width
        assert live_choice - choice_dim == encode.CHOICE_RESETS_FEEDER_DIM
    for era in ("0.0", "0.2", version.MODEL_VERSION):
        assert compat.encoding_dims_for_era(era, spec) == (live_state, live_choice)
    with pytest.raises(ValueError):
        compat.encoding_dims_for_era("garbage", spec)
    # The include-setup axis stays orthogonal to the era axis.
    setup_spec = encode.spec_for(False)
    assert compat.encoding_dims_for_era(version.MODEL_VERSION, setup_spec) == (
        encode.state_size(setup_spec),
        encode.choice_feature_dim(setup_spec),
    )


def test_class_for_version_routes_pre_1_4_to_shims():
    """Pre-1.4 same-MAJOR eras route to their compat net class (1.0 -> v1_0,
    1.1-1.3 -> v1_3); 0.x and the live era route to the live net. A malformed
    string is rejected."""
    from wingspan.compat import v1_0, v1_3

    assert model.PolicyValueNet.class_for_version("1.0") is v1_0.PolicyValueNetV1_0
    for era in ("1.1", "1.2", "1.3"):
        assert model.PolicyValueNet.class_for_version(era) is v1_3.PolicyValueNetV1_3
    for era in ("0.0", "0.2", version.MODEL_VERSION):
        assert model.PolicyValueNet.class_for_version(era) is model.PolicyValueNet
    with pytest.raises(ValueError):
        model.PolicyValueNet.class_for_version("garbage")


def test_run_config_from_artifact_defaults_the_stamp_when_absent(
    tmp_path: pathlib.Path,
):
    """A pre-field config (no ``encoding_version``) adopts the passed stamp; a
    config that already carries the field validates and keeps it."""
    raw = _cfg(tmp_path).model_dump()
    raw["architecture"].pop("encoding_version")
    adopted = config.run_config_from_artifact(raw, version.MODEL_VERSION)
    assert adopted.encoding_version == version.MODEL_VERSION
    assert adopted.state_dim == encode.state_size(adopted.encoding_spec)

    explicit = _cfg(tmp_path).model_dump()  # architecture.encoding_version == live
    kept = config.run_config_from_artifact(explicit, version.MODEL_VERSION)
    assert kept.encoding_version == version.MODEL_VERSION


def test_with_encoding_version_at_live_keeps_dims(tmp_path: pathlib.Path):
    live = _cfg(tmp_path)
    same = config.with_encoding_version(live, version.MODEL_VERSION)
    assert (same.encoding_version, same.state_dim) == (
        version.MODEL_VERSION,
        live.state_dim,
    )
    # A different-MAJOR era is refused — no pre-1.0 artifact loads under 1.0.
    with pytest.raises(ValueError):
        config.with_encoding_version(live, "0.2")


#### Era adoption from the run directory ####


def test_adopt_checkpoint_era_keeps_live_runs_live(tmp_path: pathlib.Path):
    """The era seam is a no-op at the single live era: every launch — resuming,
    architecture-mismatched, resume-disabled, or checkpoint-less — stays keyed at
    the live MODEL_VERSION."""
    _write_live_checkpoint(tmp_path)

    # A live config that reconciles with the saved run: returned at the live era.
    pinned = loop_resume.adopt_checkpoint_era(_cfg(tmp_path))
    assert pinned.encoding_version == version.MODEL_VERSION
    assert pinned.state_dim == _cfg(tmp_path).state_dim

    # A genuinely different architecture: left alone (the gate will refuse).
    other = loop_resume.adopt_checkpoint_era(_cfg(tmp_path, trunk_layers=(16, 16)))
    assert other.encoding_version == version.MODEL_VERSION

    # Resume disabled, or no checkpoint at all: left alone.
    no_resume = loop_resume.adopt_checkpoint_era(_cfg(tmp_path, resume=False))
    assert no_resume.encoding_version == version.MODEL_VERSION
    empty_dir = tmp_path / "empty"
    untouched = loop_resume.adopt_checkpoint_era(_cfg(empty_dir))
    assert untouched.encoding_version == version.MODEL_VERSION


def test_training_loop_resumes_a_live_run(tmp_path: pathlib.Path):
    """The headline guarantee: a run dir handed a live config resumes — the live
    net at the live dims, progress restored — and a resume event is recorded."""
    _write_live_checkpoint(tmp_path)
    training = loop.TrainingLoop(_cfg(tmp_path))

    assert training.config.encoding_version == version.MODEL_VERSION
    assert type(training.net) is model.PolicyValueNet
    assert training.net.state_dim == training.config.state_dim
    assert training._start_iteration == _RESUME_ITERATION + 1
    assert training.state.total_games == 10
    assert [event.text for event in training.state.events if "resumed" in event.text]


def test_resume_gate_weight_mismatch_alarms_and_starts_fresh(
    tmp_path: pathlib.Path,
):
    """The non-fatal contract holds even past the key check: a payload whose
    config matches but whose tensors cannot load (here: an empty model dict)
    alarms and starts fresh instead of crashing ``__init__``."""
    _write_live_checkpoint(tmp_path, model_state={})
    training = loop.TrainingLoop(_cfg(tmp_path))
    assert training._start_iteration == 0
    assert training.state.total_games == 0
    alarms = [
        event.text for event in training.state.events if "do not fit" in event.text
    ]
    assert alarms


def test_resumed_run_stamps_live_era(tmp_path: pathlib.Path):
    """Everything a run writes reads as its era: the checkpoint payload, its
    embedded config, the run descriptor — and the play loaders route the
    re-saved artifact back to the live net class."""
    import wingspan.players.loaders as loaders

    _write_live_checkpoint(tmp_path)
    training = loop.TrainingLoop(_cfg(tmp_path))
    loop_checkpoint.checkpoint(training, _iter_metrics(_RESUME_ITERATION + 1), None, [])

    payload = typing.cast(
        "dict[str, typing.Any]",
        torch.load(
            tmp_path / artifacts.LAST_CKPT, map_location="cpu", weights_only=False
        ),
    )
    assert payload["version"] == version.MODEL_VERSION
    assert (
        payload["config"]["architecture"]["encoding_version"] == version.MODEL_VERSION
    )
    assert payload["config"]["architecture"]["state_dim"] == training.config.state_dim

    descriptor = runmeta.read_model_config(str(tmp_path))
    assert (descriptor.version, descriptor.state_dim) == (
        version.MODEL_VERSION,
        training.config.state_dim,
    )

    loaded, loaded_cfg = loaders.load_policy_net(
        tmp_path / artifacts.LAST_CKPT, torch.device("cpu")
    )
    assert isinstance(loaded, model.PolicyValueNet)
    assert loaded_cfg.encoding_version == version.MODEL_VERSION


#### The configurator flow ####


def test_configurator_seeds_and_reads_resumable(tmp_path: pathlib.Path):
    """The dashboard path: the saved summary carries the run's era, the seeded
    working config adopts it (RESUMABLE), and a config that would rebuild a
    different net reads INCOMPATIBLE."""
    _write_live_checkpoint(tmp_path)
    summary = config_runs.inspect_run(str(tmp_path))
    assert summary.train_config is not None
    assert summary.train_config.encoding_version == version.MODEL_VERSION

    working, seeded = controller._seed_from_summary(_cfg(tmp_path), summary)
    assert seeded and working.encoding_version == version.MODEL_VERSION
    assert (
        config_runs.resolve_status(summary, working) is config_runs.RunStatus.RESUMABLE
    )
    incompatible = _cfg(tmp_path, trunk_layers=(16, 16))
    assert (
        config_runs.resolve_status(summary, incompatible)
        is config_runs.RunStatus.INCOMPATIBLE
    )


#### The collectors honor the era ####


def test_worker_arch_and_builder_are_live(tmp_path: pathlib.Path):
    cfg = _cfg(tmp_path)
    collector = mp_collect.ProcessCollector(cfg, num_workers=1)
    assert collector._arch.encoding_version == version.MODEL_VERSION
    worker_net = mp_collect._build_worker_net(collector._arch)
    assert isinstance(worker_net, model.PolicyValueNet)
    assert worker_net.state_dim == cfg.state_dim


def test_batched_collect_plays_through_the_net(tmp_path: pathlib.Path):
    """The batched (CUDA-path) collector encodes through the served net, so the
    steps it records carry the net's own state width."""
    cfg = _cfg(tmp_path)
    net = _build_net(cfg)
    net.eval()
    records = batched_collect.collect_games(net, torch.device("cpu"), seeds=[11])
    assert len(records) == 1
    assert records[0].steps
    for step in records[0].steps:
        assert step.state.shape == (cfg.state_dim,)
