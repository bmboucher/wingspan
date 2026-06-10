# pyright: reportPrivateUsage=false
# (accesses TrainingLoop's private fields — deliberate intra-package coupling)
"""Resume, initialization, and run-metadata helpers for ``TrainingLoop``.

Free functions whose first argument is a ``TrainingLoop`` instance handle the
``__init__``-time sequence: restoring from a checkpoint, initializing the
training phase and target milestone, clearing stale history logs on a fresh
run, and writing the session JSON sidecars.
"""

from __future__ import annotations

import datetime
import logging
import pathlib
import typing

import pydantic
import torch

from wingspan.training import (
    artifacts,
)
from wingspan.training import config as training_config
from wingspan.training import (
    loop_checkpoint,
    loop_eval,
    runmeta,
    runstate,
    setup_runmeta,
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
    if not training_loop.config.resume:
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

    training_loop.net.load_state_dict(payload["model"])
    training_loop.optimizer.load_state_dict(payload["optimizer"])
    reset_optimizer_lr(training_loop)  # honor this run's --lr over the saved one
    progress = runstate.RunProgress.model_validate(payload["progress"])
    training_loop.state.restore_progress(progress)
    training_loop._start_iteration = progress.iteration + 1
    if training_loop.state.opponent_generation > 0:
        loop_eval.load_opponent(training_loop)  # may reset generation to 0 if gone
    training_loop.state.push_event(
        runstate.EventKind.INFO,
        f"resumed {artifacts.LAST_CKPT} · iter {progress.iteration:04d} · "
        f"{progress.total_games:,} games · "
        f"opponent {loop_eval.opponent_label(training_loop)}",
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
            f"until {training_loop.config.random_phase_win_rate * 100:.0f}% win-rate",
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
        training_loop.state.target_iterations = training_loop.config.target_iterations


def reset_history_logs_if_fresh(training_loop: "loop.TrainingLoop") -> None:
    """Clear a previous run's history when this run did not resume.

    Truncates both append-only logs (``metrics.jsonl`` / ``games.jsonl``) and
    removes the prior run's dated ``process_*.json`` session records, so a
    fresh run never appends its rows onto stale history.  A resumed run keeps
    and continues its logs.
    """
    if training_loop._start_iteration > 0:
        return
    for name in (artifacts.METRICS_LOG, artifacts.GAMES_LOG):
        log_path = training_loop._ckpt_dir / name
        if log_path.exists():
            log_path.write_text("", encoding="utf-8")
    for stale_session in training_loop._ckpt_dir.glob(artifacts.PROCESS_GLOB):
        stale_session.unlink(missing_ok=True)
    # The setup-sample log is append-only history too — clear it on a fresh
    # run so a new run's offline fit never reads a prior run's samples.
    if training_loop._setup_store is not None:
        training_loop._setup_store.clear()


def write_run_metadata(training_loop: "loop.TrainingLoop") -> None:
    """Drop this startup's JSON sidecars.

    Writes (overwrites) the model descriptor and a fresh dated process record.
    Called once per session after the resume decision, so the process record
    can note where it resumed from.
    """
    now = datetime.datetime.now()
    runmeta.write_model_config(
        training_loop.config.checkpoint_dir, training_loop.config
    )
    runmeta.write_inspect_report(
        training_loop.config.checkpoint_dir, training_loop.config
    )
    runmeta.write_model_summary_html(
        training_loop.config.checkpoint_dir, training_loop.config
    )
    if training_loop.config.use_setup_model:
        setup_runmeta.write_setup_config(
            training_loop.config.checkpoint_dir, training_loop.config
        )
    session_path = runmeta.write_session_record(
        training_loop.config.checkpoint_dir,
        training_loop.config,
        stamp=now.strftime("%Y%m%d-%H%M%S"),
        started_at=now.isoformat(timespec="seconds"),
        git_sha=loop_checkpoint.git_sha(),
        resumed_from_iteration=training_loop._start_iteration,
    )
    training_loop.state.push_event(
        runstate.EventKind.INFO, f"session log → {session_path.name}"
    )


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


###### PRIVATE #######


def architecture_matches(
    training_loop: "loop.TrainingLoop", payload: dict[str, typing.Any]
) -> bool:
    """Whether ``payload``'s saved network shape matches this run's.

    Checkpoints are self-describing: a payload with no embedded ``config``,
    or one that no longer validates (e.g. a value since constrained out of
    bounds), is treated as a mismatch so the run starts fresh with an alarm
    rather than crashing ``__init__`` — preserving the non-fatal contract.
    """
    raw_config = payload.get("config")
    if raw_config is None:
        return False  # not a self-describing checkpoint — refuse
    try:
        saved = training_config.TrainConfig.model_validate(raw_config)
    except pydantic.ValidationError:
        return False
    return saved.architecture_key == training_loop.config.architecture_key


def reset_optimizer_lr(training_loop: "loop.TrainingLoop") -> None:
    """Apply this run's learning rate after loading an optimizer that may have
    saved a different one (Adam's momentum is kept; only the step size moves)."""
    for group in training_loop.optimizer.param_groups:
        group["lr"] = training_loop.config.lr
