# pyright: reportPrivateUsage=false
# (accesses TrainingLoop's private fields — deliberate intra-package coupling)
"""Resume, initialization, and run-metadata helpers for ``TrainingLoop``.

Free functions whose first argument is a ``TrainingLoop`` instance handle the
``__init__``-time sequence: restoring from a checkpoint, initializing the
training phase and target milestone, clearing stale history logs on a fresh
run, and writing the session JSON sidecars.

:func:`adopt_checkpoint_era` is the era seam (``docs/VERSIONING.md``): called
before the loop builds its net, it adopts the resumable checkpoint's artifact
era into the config whenever that adoption is what makes the resume possible —
so a run started before a FRESH encoding change keeps training at its own
frozen geometry under newer code — and, symmetrically, re-keys any *fresh*
launch at the live ``MODEL_VERSION`` so a new run never inherits a stale era.
Both directions hold for every entry point (dashboard, cloud runner, tests)
by construction.
"""

from __future__ import annotations

import datetime
import logging
import pathlib
import typing

import pydantic
import torch

from wingspan import version
from wingspan.training import (
    artifacts,
)
from wingspan.training import config as training_config
from wingspan.training import (
    loop_checkpoint,
    loop_eval,
    runmeta,
    runstate,
)

if typing.TYPE_CHECKING:
    from wingspan.training import loop


def maybe_resume(training_loop: "loop.TrainingLoop") -> None:
    """Restore the network, optimizer, and run progress from ``last.pt``.

    A restarted run continues where it left off instead of from scratch.
    No-ops when resuming is disabled or there is no checkpoint.  A checkpoint
    that can't be read, or whose architecture differs from this run's, is
    skipped with a dashboard alarm rather than crashing — the run then starts
    fresh (and the next checkpoint will overwrite the mismatched one).
    """
    if not training_loop.config.run.resume:
        return
    last = training_loop._ckpt_dir / artifacts.LAST_CKPT
    if not last.exists():
        return
    try:
        # Our own trusted checkpoint carries a config dict + metrics, not just
        # tensors, so the full (non weights-only) unpickler is required.
        payload = typing.cast(
            "dict[str, typing.Any]",
            torch.load(last, map_location=training_loop.device, weights_only=False),
        )
    except Exception:  # noqa: BLE001 — a corrupt/unreadable checkpoint starts fresh
        training_loop.state.push_event(
            runstate.EventKind.ALARM,
            f"could not read {artifacts.LAST_CKPT} — starting fresh",
        )
        return
    if not architecture_matches(training_loop, payload):
        training_loop.state.push_event(
            runstate.EventKind.ALARM,
            f"{artifacts.LAST_CKPT} architecture differs — starting fresh",
        )
        return

    try:
        training_loop.net.load_state_dict(payload["model"])
        training_loop.optimizer.load_state_dict(payload["optimizer"])
    except (RuntimeError, ValueError, KeyError):
        # The key matched but the tensors did not fit (a hand-edited or
        # corrupted payload) — keep the non-fatal contract: alarm and start
        # fresh rather than crashing ``__init__``.
        training_loop.state.push_event(
            runstate.EventKind.ALARM,
            f"{artifacts.LAST_CKPT} weights do not fit this architecture — "
            "starting fresh",
        )
        return
    reset_optimizer_lr(training_loop)  # honor this run's --lr over the saved one
    progress = runstate.RunProgress.model_validate(payload["progress"])
    training_loop.state.restore_progress(progress)
    training_loop._start_iteration = progress.iteration + 1
    if training_loop.state.opponent_generation > 0:
        loop_eval.load_opponent(training_loop)  # may reset generation to 0 if gone
    era = training_loop.config.encoding_version
    era_note = "" if era == version.MODEL_VERSION else f" · era {era} (pinned)"
    training_loop.state.push_event(
        runstate.EventKind.INFO,
        f"resumed {artifacts.LAST_CKPT} · iter {progress.iteration:04d} · "
        f"{progress.total_games:,} games · "
        f"opponent {loop_eval.opponent_label(training_loop)}{era_note}",
    )


def init_training_phase(training_loop: "loop.TrainingLoop") -> None:
    """Open a fresh run in the random-opponent bootstrap phase when
    ``config.initial_vs_random`` asks for it (collect vs random, eval paused).

    A resumed run keeps the phase restored from its checkpoint —
    ``_start_iteration`` is 0 only on a fresh start — so this never overrides
    a run that already graduated to self-play.
    """
    if training_loop._start_iteration > 0 or not training_loop.config.initial_vs_random:
        return
    with training_loop.lock:
        training_loop.state.training_phase = runstate.TrainingPhase.RANDOM_OPPONENT
        training_loop.state.push_event(
            runstate.EventKind.INFO,
            "bootstrap: collecting vs random opponent · eval paused "
            f"until {training_loop.config.opponent.random_phase_win_rate * 100:.0f}% win-rate",
        )


def init_target_if_fresh(training_loop: "loop.TrainingLoop") -> None:
    """Seed the live target from the config on fresh runs.

    Resumed runs restore ``state.target_iterations`` from ``RunProgress``
    (via ``restore_progress``), so we never overwrite a live target the user
    may have updated in a prior continuation.
    """
    if training_loop._start_iteration > 0:
        return
    with training_loop.lock:
        training_loop.state.target_iterations = (
            training_loop.config.run.target_iterations
        )


def reset_history_logs_if_fresh(training_loop: "loop.TrainingLoop") -> None:
    """Clear a previous run's history when this run did not resume.

    Truncates both append-only logs (``metrics.jsonl`` / ``games.jsonl``) and
    removes the prior run's dated session records (``run_config_*.json`` for
    ≥0.5 runs, ``process_*.json`` for legacy), so a fresh run never appends its
    rows onto stale history.  A resumed run keeps and continues its logs.
    """
    if training_loop._start_iteration > 0:
        return
    for name in (artifacts.METRICS_LOG, artifacts.GAMES_LOG):
        log_path = training_loop._ckpt_dir / name
        if log_path.exists():
            log_path.write_text("", encoding="utf-8")
    for stale_session in training_loop._ckpt_dir.glob(artifacts.RUN_CONFIG_GLOB):
        stale_session.unlink(missing_ok=True)
    for stale_session in training_loop._ckpt_dir.glob(artifacts.PROCESS_GLOB):
        stale_session.unlink(missing_ok=True)


def write_run_metadata(training_loop: "loop.TrainingLoop") -> None:
    """Drop this startup's JSON sidecars.

    For ≥0.5 runs: writes a single dated ``run_config_<stamp>.json`` (the
    unified config file) plus the inspect report and model summary HTML.
    The three legacy files (model_config.json, setup_config.json,
    process_<stamp>.json) are no longer written. Called once per session after
    the resume decision so the unified file can note where it resumed from.
    """
    now = datetime.datetime.now()
    checkpoint_dir = training_loop.config.run.checkpoint_dir
    session_path = runmeta.write_run_config(
        checkpoint_dir,
        training_loop.config,
        stamp=now.strftime("%Y%m%d-%H%M%S"),
        started_at=now.isoformat(timespec="seconds"),
        git_sha=loop_checkpoint.git_sha(),
        resumed_from_iteration=training_loop._start_iteration,
    )
    runmeta.write_inspect_report(checkpoint_dir, training_loop.config)
    runmeta.write_model_summary_html(checkpoint_dir, training_loop.config)
    training_loop.state.push_event(
        runstate.EventKind.INFO, f"session log → {session_path.name}"
    )


def adopt_checkpoint_era(
    cfg: training_config.RunConfig,
) -> training_config.RunConfig:
    """The era seam: pin ``cfg`` to a resumable checkpoint's artifact era, or
    un-pin a fresh launch back to the live :data:`version.MODEL_VERSION`.

    When resume is enabled and ``last.pt`` holds a readable saved config,
    ``cfg`` re-keyed at the saved era is compared against the saved run's
    ``architecture_key``: a match means the configs agree on everything
    *except* the era — the exact situation of a run started before a FRESH
    encoding change — so the era-adopted config is returned and the caller's
    net build / resume gate proceed at the run's own frozen geometry.

    Every other situation starts fresh (``maybe_resume`` either no-ops or
    alarms), and a fresh run must never inherit a stale era — e.g. a working
    config the configurator seeded from an old run and then launched with
    ``resume=False`` — so the config is re-keyed at the live MODEL_VERSION.
    A deliberately era-pinned config therefore cannot train a *fresh* run
    through ``TrainingLoop``; regenerating old-era artifacts must build the
    era net directly (the ``tests/test_era_pinned_resume.py`` pattern).

    Called by ``TrainingLoop.__init__`` before the net is constructed, so the
    era is right for every entry point by construction rather than relying on
    each launcher (dashboard, cloud runner, tests) to remember it.
    """
    saved = _resumable_saved_config(cfg) if cfg.run.resume else None
    if saved is not None:
        candidate = _config_at_era(cfg, saved.encoding_version)
        if candidate.architecture_key == saved.architecture_key:
            if candidate is not cfg:
                logging.info(
                    "Resumable checkpoint in %s is era %s — pinning this run "
                    "to it (state_dim %d, choice_dim %d)",
                    cfg.run.checkpoint_dir,
                    saved.encoding_version,
                    candidate.state_dim,
                    candidate.choice_dim,
                )
            return candidate
    if cfg.encoding_version != version.MODEL_VERSION:
        logging.info(
            "Fresh run — un-pinning stale era %s back to the live %s",
            cfg.encoding_version,
            version.MODEL_VERSION,
        )
        return training_config.with_encoding_version(cfg, version.MODEL_VERSION)
    return cfg


def _resumable_saved_config(
    cfg: training_config.RunConfig,
) -> training_config.RunConfig | None:
    """The saved config embedded in ``cfg``'s ``last.pt``, rehydrated at the
    payload's own artifact era; ``None`` when there is no checkpoint, the
    payload is unreadable, or it carries no valid config (all of which defer
    the fresh-vs-alarm decision to ``maybe_resume``)."""
    last = pathlib.Path(cfg.run.checkpoint_dir) / artifacts.LAST_CKPT
    if not last.exists():
        return None
    try:
        payload = typing.cast(
            "dict[str, typing.Any]",
            torch.load(last, map_location="cpu", weights_only=False),
        )
        raw_config = payload.get("config")
        if raw_config is None:
            return None
        artifact_version = str(payload.get("version", version.PRE_VERSIONING_VERSION))
        return training_config.run_config_from_artifact(raw_config, artifact_version)
    except Exception:  # noqa: BLE001 — an unreadable payload defers to maybe_resume
        return None


def _config_at_era(
    cfg: training_config.RunConfig, era: str
) -> training_config.RunConfig:
    """``cfg`` re-keyed at ``era`` — ``cfg`` itself when already there, so the
    common no-op path skips a full re-validation."""
    if cfg.encoding_version == era:
        return cfg
    return training_config.with_encoding_version(cfg, era)


def validate_bootstrap_opponent(training_loop: "loop.TrainingLoop") -> None:
    """Load the bootstrap checkpoint once at startup to fail fast on bad paths.

    A missing file, a corrupt payload, or an incompatible encoding layout all
    raise immediately so the run never starts a multi-hour training session
    against an opponent it cannot load.  Resumes re-validate on every session
    startup because ``_WorkerArch`` is rebuilt from config each time (the path
    is not persisted in the run checkpoint).
    """
    path = training_loop.config.bootstrap_opponent_checkpoint
    if path is None:
        return
    # Function-level import: loaders imports from wingspan.training (artifacts,
    # config, runmeta, …). Importing at module level would create a cycle since
    # loop_resume is itself part of wingspan.training.
    import wingspan.players.loaders as loaders  # noqa: PLC0415

    net, saved = loaders.load_policy_net(pathlib.Path(path), torch.device("cpu"))
    logging.info(
        "Bootstrap opponent loaded: path=%s state_dim=%d choice_dim=%d",
        path,
        saved.state_dim,
        saved.choice_dim,
    )
    del net  # free immediately; workers reload from the path on demand


def validate_dagger_expert(training_loop: "loop.TrainingLoop") -> None:
    """Load the DAgger expert checkpoint once at startup to fail fast on bad paths.

    Mirrors :func:`validate_bootstrap_opponent`: a missing file, corrupt payload,
    or incompatible encoding layout raises immediately.  The expert may be a
    different architecture/era than the student — ``load_policy_net`` handles
    era-routing — so only basic load-ability is verified here.
    """
    path = training_loop.config.dagger_expert_checkpoint
    if path is None:
        return
    import wingspan.players.loaders as loaders  # noqa: PLC0415

    net, saved = loaders.load_policy_net(pathlib.Path(path), torch.device("cpu"))
    logging.info(
        "DAgger expert loaded: path=%s state_dim=%d choice_dim=%d clone_iters=%d",
        path,
        saved.state_dim,
        saved.choice_dim,
        training_loop.config.dagger.clone_iters,
    )
    del net  # free immediately; workers reload from the path on demand


###### PRIVATE #######


def architecture_matches(
    training_loop: "loop.TrainingLoop", payload: dict[str, typing.Any]
) -> bool:
    """Whether ``payload``'s saved network shape matches this run's.

    Checkpoints are self-describing: a payload with no embedded ``config``,
    or one that no longer validates (e.g. a value since constrained out of
    bounds), is treated as a mismatch so the run starts fresh with an alarm
    rather than crashing ``__init__`` — preserving the non-fatal contract.
    The saved config is rehydrated at the payload's own artifact era
    (``train_config_from_artifact``), so its key carries the era it actually
    trained at — never the live one a bare re-validation would substitute.
    """
    raw_config = payload.get("config")
    if raw_config is None:
        return False  # not a self-describing checkpoint — refuse
    artifact_version = str(payload.get("version", version.PRE_VERSIONING_VERSION))
    try:
        saved = training_config.run_config_from_artifact(raw_config, artifact_version)
    except pydantic.ValidationError:
        return False
    return saved.architecture_key == training_loop.config.architecture_key


def reset_optimizer_lr(training_loop: "loop.TrainingLoop") -> None:
    """Apply this run's learning rate after loading an optimizer that may have
    saved a different one (Adam's momentum is kept; only the step size moves)."""
    for group in training_loop.optimizer.param_groups:
        group["lr"] = training_loop.config.training.lr
