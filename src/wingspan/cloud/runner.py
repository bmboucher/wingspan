"""The headless supervisor: run the training loop unattended with S3 persistence.

``HeadlessRunner`` is the FLIGHT PLAN training worker with the dashboard replaced
by an S3 sync sidecar. It pulls any prior run state down from S3, runs the same
:class:`~wingspan.training.loop.TrainingLoop` on a worker thread, and on the main
thread periodically publishes a tiny status snapshot, uploads the checkpoint set,
and offloads the per-game log in chunks. ``SIGTERM`` / ``SIGINT`` (a Spot
reclaim, or Ctrl-C) requests a graceful stop and a final sync, so a relaunched
container picks the run up exactly where it left off; reaching the target
milestone writes + uploads the final-eval artifact and exits.
"""

from __future__ import annotations

import datetime
import logging
import os
import pathlib
import signal
import subprocess
import threading
import types
import typing

import torch

from wingspan.cloud import runfile, s3sync, status
from wingspan.training import artifacts, loop, runstate

_LOG = logging.getLogger(__name__)

# Spot reclaim warns ~2 minutes ahead, so allow that long for the in-flight game
# to finish, the final checkpoint to be written, and the closing sync to upload.
_STOP_GRACE_SECONDS = 120.0
# Once a stop is requested we poll the worker more tightly than the status
# cadence so the closing sync fires promptly rather than up to an interval later.
_STOPPING_POLL_SECONDS = 2.0


class HeadlessRunner:
    """Run one :class:`CloudRunFile` to completion, mirroring artifacts to S3."""

    def __init__(self, run: runfile.CloudRunFile, sync: s3sync.S3Sync):
        self._run = run
        self._config = run.train
        self._sync = sync
        self._ckpt_dir = pathlib.Path(self._config.run.checkpoint_dir)
        self._git_sha = _git_sha()
        self._session_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self._started_at = datetime.datetime.now(datetime.UTC).isoformat(
            timespec="seconds"
        )
        # Built in ``run`` after the startup download, so the loop's own resume
        # reads the freshly-pulled checkpoints.
        self._training: loop.TrainingLoop | None = None
        self._worker: threading.Thread | None = None
        # Game-log offload bookkeeping for this session (byte offset + chunk seq).
        self._games_offset = 0
        self._games_seq = 0
        # Iteration of the last checkpoint / game-chunk upload (-1 = none yet).
        self._last_ckpt_iter = -1
        self._last_games_iter = -1
        # Set by the signal handler; ``_wake`` breaks the supervisor's sleep early.
        self._stop_requested = False
        self._wake = threading.Event()

    def run(self) -> int:
        """Run the supervisor to completion. Returns a process exit code (0 on a
        clean finish or graceful interrupt, 1 on a training crash)."""
        self._install_signal_handlers()
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._restore_from_s3()
        self._resolve_device()

        self._training = loop.TrainingLoop(self._config, pause_at_target=False)
        self._worker = threading.Thread(
            target=self._training.run, name="wingspan-trainer", daemon=True
        )
        self._worker.start()
        _LOG.info(
            "run '%s' started · %d games/iter · target %s · device %s",
            self._run.run_name,
            self._config.run.games_per_iter,
            self._config.run.target_iterations or "—",
            self._config.misc.device,
        )

        self._supervise()
        self._final_sync()
        return self._exit_code()

    ###### PRIVATE #######

    #### Startup ####

    def _restore_from_s3(self) -> None:
        """Pull prior run state from S3 so the loop's resume continues the run."""
        if not (self._config.run.resume and self._run.sync.download_on_start):
            return
        try:
            pulled = self._sync.download_run(self._ckpt_dir)
        except Exception as error:  # noqa: BLE001 — first launch / transient S3 issue
            _LOG.warning("startup download failed (%s) — starting fresh", error)
            return
        _LOG.info("pulled %d object(s) from S3 into %s", pulled, self._ckpt_dir)

    def _resolve_device(self) -> None:
        """Downgrade a ``cuda`` request to ``cpu`` when no GPU is present, matching
        the dashboard's fallback so a misconfigured device never crashes the loop
        at model construction (training is CPU-only anyway)."""
        if (
            self._config.misc.device.startswith("cuda")
            and not torch.cuda.is_available()
        ):
            _LOG.warning("cuda requested but unavailable — falling back to cpu")
            self._config = self._config.model_copy(
                update={"misc": self._config.misc.model_copy(update={"device": "cpu"})}
            )

    #### Supervision loop ####

    def _supervise(self) -> None:
        """Publish status + offload artifacts until the worker reaches a terminal
        phase (or finishes after a stop request)."""
        assert self._training is not None and self._worker is not None
        while True:
            with self._training.lock:
                snapshot = status.build_status(
                    self._training.state,
                    run_name=self._run.run_name,
                    started_at=self._started_at,
                    status_interval_seconds=self._run.sync.status_interval_seconds,
                    git_sha=self._git_sha,
                )
                completed = (
                    self._training.state.iteration
                    if self._training.state.last_iter is not None
                    else -1
                )
                terminal = self._training.state.phase.is_terminal

            self._publish_status(snapshot)
            self._log_progress(snapshot)
            if not terminal:
                self._maybe_offload(completed)
            if terminal and not self._worker.is_alive():
                return
            self._sleep_until_next_tick()

    def _maybe_offload(self, completed_iter: int) -> None:
        """Upload the checkpoint set and a game-log chunk on their iteration cadences.

        Uses ``completed - last_uploaded >= cadence`` (rather than a modulo) so an
        upload is never skipped when several iterations elapse between status
        ticks. The game log also flushes early once its un-offloaded tail exceeds
        ``games_chunk_mb``.
        """
        if completed_iter < 0:
            return
        sync = self._run.sync
        if completed_iter - self._last_ckpt_iter >= sync.checkpoint_upload_iters:
            self._guard(
                "checkpoint upload",
                lambda: self._sync.upload_checkpoint_set(self._ckpt_dir),
            )
            self._last_ckpt_iter = completed_iter

        games_path = self._ckpt_dir / artifacts.GAMES_LOG
        unsent = (
            games_path.stat().st_size - self._games_offset if games_path.exists() else 0
        )
        size_due = unsent >= sync.games_chunk_mb * 1024 * 1024
        iters_due = completed_iter - self._last_games_iter >= sync.games_chunk_iters
        if size_due or iters_due:
            self._offload_games(games_path)
            self._last_games_iter = completed_iter

    def _offload_games(self, games_path: pathlib.Path) -> None:
        def _do() -> None:
            new_offset = self._sync.offload_game_chunk(
                games_path, self._session_stamp, self._games_seq, self._games_offset
            )
            if new_offset > self._games_offset:
                self._games_offset = new_offset
                self._games_seq += 1

        self._guard("game-chunk offload", _do)

    #### Shutdown ####

    def _final_sync(self) -> None:
        """One last consistent upload after the worker stops: the checkpoint set
        (including any ``final_*`` milestone files), the trailing game chunk, and
        a closing status snapshot."""
        assert self._training is not None
        self._guard(
            "final checkpoint upload",
            lambda: self._sync.upload_checkpoint_set(self._ckpt_dir),
        )
        self._offload_games(self._ckpt_dir / artifacts.GAMES_LOG)
        with self._training.lock:
            snapshot = status.build_status(
                self._training.state,
                run_name=self._run.run_name,
                started_at=self._started_at,
                status_interval_seconds=self._run.sync.status_interval_seconds,
                git_sha=self._git_sha,
            )
        self._publish_status(snapshot)
        _LOG.info("final sync complete · phase=%s", snapshot.phase)

    def _exit_code(self) -> int:
        assert self._training is not None
        return 1 if self._training.state.phase is runstate.Phase.ERROR else 0

    #### Status I/O ####

    def _publish_status(self, snapshot: status.RunStatus) -> None:
        """Write ``status.json`` locally (atomically) and upload it to S3."""
        text = snapshot.model_dump_json(indent=2)
        path = self._ckpt_dir / artifacts.STATUS_JSON
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
        self._guard(
            "status upload",
            lambda: self._sync.upload_bytes(
                text.encode("utf-8"), artifacts.STATUS_JSON
            ),
        )

    def _log_progress(self, snapshot: status.RunStatus) -> None:
        """One-line progress to stdout (captured by the container log driver)."""
        win = (
            f"{snapshot.win_rate * 100:.0f}%" if snapshot.win_rate is not None else "—"
        )
        _LOG.info(
            "iter %d (%.0f%%) · %s · %d games · avg %.1f · win %s vs %s",
            snapshot.iteration,
            snapshot.pct_complete,
            snapshot.phase,
            snapshot.total_games,
            snapshot.avg_score,
            win,
            snapshot.opponent_label,
        )

    #### Signals + sleep ####

    def _install_signal_handlers(self) -> None:
        """Route SIGTERM / SIGINT to a graceful stop (runs on the main thread)."""

        def _handler(signum: int, _frame: types.FrameType | None) -> None:
            _LOG.info("signal %d received — requesting graceful stop", signum)
            self._stop_requested = True
            if self._training is not None:
                self._training.request_stop()
            self._wake.set()

        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, _handler)

    def _sleep_until_next_tick(self) -> None:
        """Sleep one status interval, but wake early once a stop is pending so the
        closing sync fires within a couple of seconds of the signal."""
        interval = self._run.sync.status_interval_seconds
        if self._stop_requested:
            interval = min(interval, _STOPPING_POLL_SECONDS)
        self._wake.wait(interval)
        self._wake.clear()

    def _guard(self, what: str, action: typing.Callable[[], None]) -> None:
        """Run an S3 action, logging (not raising) any failure so a transient S3
        error never kills a run that is otherwise making local progress."""
        try:
            action()
        except Exception as error:  # noqa: BLE001 — S3 hiccups must not stop training
            _LOG.warning("%s failed: %s", what, error)


def _git_sha() -> str | None:
    """Best-effort short git SHA (None in the container, where .git is absent)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        return result.stdout.strip() or None if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None
