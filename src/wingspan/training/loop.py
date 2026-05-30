"""The training orchestrator: the worker side of the dashboard.

``TrainingLoop`` owns the network, optimizer, RNG, and the shared
:class:`runstate.RunState`. Its :meth:`run` drives the TRAINING.md Phase-1
program — collect a batch of self-play games, run one length-bucketed REINFORCE
update, periodically evaluate against the random agent, checkpoint — looping
until ``max_iterations`` or an external stop request.

It runs on a background thread; every mutation of the shared state is made under
``self.lock`` so the main-thread renderer always reads a consistent frame. The
loop never touches the terminal — all presentation lives in the dashboard.
"""

from __future__ import annotations

import os
import pathlib
import random
import subprocess
import threading
import time
import traceback
import typing

import numpy as np
import torch
from torch import optim

from wingspan import model
from wingspan.training import (
    collect,
    config,
    evaluate,
    learner,
    metrics,
    runstate,
    sysmon,
)

# How often the side thread refreshes the SYSTEM band's host telemetry. One
# second keeps psutil's CPU sampling window meaningful while adding negligible
# overhead next to self-play collection.
_SYSMON_INTERVAL_SECONDS = 1.0

# Checkpoint filenames within ``checkpoint_dir``.
_LAST_CKPT = "last.pt"
_BEST_CKPT = "best.pt"


class TrainingLoop:
    """A resumable, stoppable self-play training run feeding a live RunState."""

    def __init__(self, cfg: config.TrainConfig):
        self.config = cfg
        self.device = torch.device(cfg.device)
        _seed_everything(cfg.seed)
        self.net = model.PolicyValueNet(hidden=cfg.hidden).to(self.device)
        self.optimizer: optim.Optimizer = optim.Adam(self.net.parameters(), lr=cfg.lr)
        self.rng = random.Random(cfg.seed)
        self.lock = threading.RLock()
        self.state = runstate.new_run_state(cfg)
        self._stop = threading.Event()
        self._ckpt_dir = pathlib.Path(cfg.checkpoint_dir)
        self._monitor = sysmon.SystemMonitor()
        self._monitor_stop = threading.Event()
        self._monitor_thread: threading.Thread | None = None
        # Iteration the loop starts numbering from (advanced past a resumed
        # checkpoint). Set last so resume can mutate net / optimizer / state.
        self._start_iteration = 0
        self._maybe_resume()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def request_stop(self) -> None:
        """Ask the loop to finish the current game and shut down gracefully."""
        self._stop.set()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def run(self) -> None:
        """Run iterations until ``max_iterations`` or a stop request. Intended
        as the target of a worker thread; never raises — failures land in
        ``state.phase = ERROR`` with the traceback in ``state.error``."""
        self._start_monitor()
        with self.lock:
            self.state.push_event(
                runstate.EventKind.INFO,
                f"run started · {self.config.games_per_iter} games/iter · {self.device}",
            )
        try:
            iteration = self._start_iteration
            while not self._stop.is_set() and not self._reached_limit(iteration):
                self._run_iteration(iteration)
                iteration += 1
            self._finish(
                runstate.Phase.STOPPED if self._stop.is_set() else runstate.Phase.DONE
            )
        except Exception:  # noqa: BLE001 — surface any failure on the dashboard
            with self.lock:
                self.state.phase = runstate.Phase.ERROR
                self.state.error = traceback.format_exc()
                self.state.push_event(
                    runstate.EventKind.ALARM, "training crashed — see console"
                )
        finally:
            self._monitor_stop.set()

    ###### PRIVATE #######

    def _reached_limit(self, iteration: int) -> bool:
        # ``max_iterations`` caps iterations run *this session*, so resuming a run
        # with ``--iterations N`` does N more rather than stopping immediately.
        done_this_session = iteration - self._start_iteration
        return (
            self.config.max_iterations > 0
            and done_this_session >= self.config.max_iterations
        )

    def _maybe_resume(self) -> None:
        """Restore the network, optimizer, and run progress from ``last.pt`` so a
        restarted run continues where it left off instead of from scratch.

        No-ops when resuming is disabled or there is no checkpoint. A checkpoint
        that can't be read, or whose architecture differs from this run's, is
        skipped with a dashboard alarm rather than crashing — the run then starts
        fresh (and the next checkpoint will overwrite the mismatched one)."""
        if not self.config.resume:
            return
        last = self._ckpt_dir / _LAST_CKPT
        if not last.exists():
            return
        try:
            # Our own trusted checkpoint carries a config dict + metrics, not just
            # tensors, so the full (non weights-only) unpickler is required.
            payload = typing.cast(
                "dict[str, typing.Any]",
                torch.load(last, map_location=self.device, weights_only=False),
            )
        except Exception:  # noqa: BLE001 — a corrupt/unreadable checkpoint starts fresh
            self.state.push_event(
                runstate.EventKind.ALARM,
                f"could not read {_LAST_CKPT} — starting fresh",
            )
            return
        if not self._architecture_matches(payload):
            self.state.push_event(
                runstate.EventKind.ALARM,
                f"{_LAST_CKPT} architecture differs — starting fresh",
            )
            return

        self.net.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self._reset_optimizer_lr()  # honor this run's --lr over the saved one
        progress = _progress_from_payload(payload)
        self.state.restore_progress(progress)
        self._start_iteration = progress.iteration + 1
        self.state.push_event(
            runstate.EventKind.INFO,
            f"resumed {_LAST_CKPT} · iter {progress.iteration:04d} · "
            f"{progress.total_games:,} games",
        )

    def _architecture_matches(self, payload: dict[str, typing.Any]) -> bool:
        """Whether ``payload``'s saved network shape matches this run's, so its
        weights can be loaded without misrouting heads (TRAINING.md §5.1)."""
        raw_config = payload.get("config")
        if raw_config is None:
            return True  # pre-descriptor checkpoint — assume compatible
        saved = config.TrainConfig.model_validate(raw_config)
        return (
            saved.state_dim == self.config.state_dim
            and saved.choice_dim == self.config.choice_dim
            and saved.family_order == self.config.family_order
            and saved.hidden == self.config.hidden
        )

    def _reset_optimizer_lr(self) -> None:
        """Apply this run's learning rate after loading an optimizer that may have
        saved a different one (Adam's momentum is kept; only the step size moves)."""
        for group in self.optimizer.param_groups:
            group["lr"] = self.config.lr

    def _start_monitor(self) -> None:
        """Take one telemetry sample now (so the SYSTEM band paints immediately),
        then keep sampling on a side thread so the gauges stay live even through
        the blocking update and eval phases."""
        self._sample_system()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, name="wingspan-sysmon", daemon=True
        )
        self._monitor_thread.start()

    def _monitor_loop(self) -> None:
        # ``wait`` returns the moment the run ends, so a finished run never lingers
        # a full interval before the sampler exits.
        while not self._monitor_stop.wait(_SYSMON_INTERVAL_SECONDS):
            self._sample_system()

    def _sample_system(self) -> None:
        stats = self._monitor.sample()
        with self.lock:
            self.state.system = stats

    def _run_iteration(self, iteration: int) -> None:
        with self.lock:
            self.state.phase = runstate.Phase.COLLECTING
            self.state.iteration = iteration
            self.state.game_in_iter = 0
            self.state.iter_start_monotonic = time.monotonic()

        collect_start = time.monotonic()
        records = self._collect(iteration)
        collect_seconds = time.monotonic() - collect_start
        if not records:
            return  # stopped before completing any game this iteration

        with self.lock:
            self.state.phase = runstate.Phase.UPDATING
        update_start = time.monotonic()
        stats = learner.update(
            self.net, self.optimizer, records, self.config, self.device
        )
        update_seconds = time.monotonic() - update_start

        eval_result, eval_seconds = self._maybe_evaluate(iteration)

        iter_metrics = _build_iteration_metrics(
            iteration,
            self.state.total_games,
            records,
            stats,
            eval_result,
            collect_seconds,
            update_seconds,
            eval_seconds,
        )
        self._commit_iteration(iter_metrics, stats, eval_result)

    def _collect(self, iteration: int) -> list[collect.GameRecord]:
        """Play ``games_per_iter`` self-play games, updating the live state per
        game so the dashboard advances mid-iteration."""
        records: list[collect.GameRecord] = []
        for game_idx in range(self.config.games_per_iter):
            if self._stop.is_set():
                break
            seed = self.config.seed * 1_000_000 + iteration * 10_000 + game_idx
            record = collect.play_game(self.net, self.device, self.rng, seed)
            records.append(record)

            decisions_seen = len(record.steps)
            family = _family_counts(record)
            with self.lock:
                self.state.record_game(record.breakdowns, decisions_seen, family)
                self.state.game_in_iter = game_idx + 1
                self.state.games_per_sec = (game_idx + 1) / max(
                    self.state.iter_elapsed(), 1e-6
                )
        return records

    def _maybe_evaluate(
        self, iteration: int
    ) -> tuple[metrics.EvalResult | None, float]:
        if self.config.eval_every <= 0 or iteration % self.config.eval_every != 0:
            return None, 0.0
        with self.lock:
            self.state.phase = runstate.Phase.EVALUATING
            self.state.eval_game_in_iter = 0
            self.state.eval_games_in_iter = 2 * self.config.eval_games
        start = time.monotonic()
        eval_seed = self.config.seed * 7919 + iteration * 101 + 1
        result = evaluate.evaluate_vs_random(
            self.net,
            self.device,
            self.config.eval_games,
            eval_seed,
            on_progress=self._record_eval_progress,
        )
        return result, time.monotonic() - start

    def _record_eval_progress(self, games_done: int, total_games: int) -> None:
        """Publish held-out eval progress so the header bar tracks eval games."""
        with self.lock:
            self.state.eval_game_in_iter = games_done
            self.state.eval_games_in_iter = total_games

    def _commit_iteration(
        self,
        iter_metrics: metrics.IterationMetrics,
        stats: learner.UpdateStats,
        eval_result: metrics.EvalResult | None,
    ) -> None:
        with self.lock:
            self.state.last_iter = iter_metrics
            self.state.history.append(iter_metrics)
            cap = self.config.history_len
            if len(self.state.history) > cap:
                del self.state.history[: len(self.state.history) - cap]
            if not np.isfinite(stats.loss):
                self.state.push_event(
                    runstate.EventKind.ALARM,
                    f"non-finite loss at iter {iter_metrics.iteration}",
                )
            if eval_result is not None:
                self.state.push_event(
                    runstate.EventKind.EVAL,
                    f"eval iter {iter_metrics.iteration:04d} · "
                    f"{eval_result.win_rate * 100:.1f}% ±{eval_result.ci95 * 100:.1f}% "
                    f"vs random",
                )
            self.state.phase = runstate.Phase.CHECKPOINTING

        self._checkpoint(iter_metrics, eval_result)

    def _checkpoint(
        self,
        iter_metrics: metrics.IterationMetrics,
        eval_result: metrics.EvalResult | None,
    ) -> None:
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)
        with self.lock:
            improved = eval_result is not None and (
                self.state.best_win_rate is None
                or eval_result.win_rate > self.state.best_win_rate
            )
            prev_best = self.state.best_win_rate
            if improved and eval_result is not None:
                self.state.best_win_rate = eval_result.win_rate
            # Snapshot the resumable progress (counters, aggregates, charts) so a
            # later run picks up exactly here rather than from zero.
            progress = self.state.to_progress()
        payload: dict[str, object] = {
            "config": self.config.model_dump(),
            "model": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "iteration": iter_metrics.iteration,
            "total_games": self.state.total_games,
            "metrics": iter_metrics.model_dump(),
            "progress": progress.model_dump(),
            "git_sha": _git_sha(),
        }
        _atomic_save(payload, self._ckpt_dir / _LAST_CKPT)
        if improved and eval_result is not None:
            _atomic_save(payload, self._ckpt_dir / _BEST_CKPT)

        with self.lock:
            if improved and eval_result is not None:
                prev_txt = (
                    f" > prev {prev_best * 100:.1f}%" if prev_best is not None else ""
                )
                self.state.push_event(
                    runstate.EventKind.BEST,
                    f"new {_BEST_CKPT} (eval {eval_result.win_rate * 100:.1f}%{prev_txt})",
                )
            else:
                self.state.push_event(
                    runstate.EventKind.CHECKPOINT, f"checkpoint saved  {_LAST_CKPT}"
                )

        with open(self._ckpt_dir / "metrics.jsonl", "a", encoding="utf-8") as handle:
            handle.write(iter_metrics.model_dump_json() + "\n")

    def _finish(self, phase: runstate.Phase) -> None:
        with self.lock:
            self.state.phase = phase
            self.state.stopped_monotonic = time.monotonic()
            self.state.push_event(
                runstate.EventKind.INFO,
                (
                    "run stopped by user"
                    if phase is runstate.Phase.STOPPED
                    else "run complete"
                ),
            )


# ---------------------------------------------------------------------------
# Pure helpers


def _family_counts(record: collect.GameRecord) -> metrics.FamilyCounts:
    counts = metrics.FamilyCounts()
    for step in record.steps:
        counts.bump(step.family_idx)
    return counts


def _build_iteration_metrics(
    iteration: int,
    total_games: int,
    records: list[collect.GameRecord],
    stats: learner.UpdateStats,
    eval_result: metrics.EvalResult | None,
    collect_seconds: float,
    update_seconds: float,
    eval_seconds: float,
) -> metrics.IterationMetrics:
    n_games = len(records)
    sum_breakdown = metrics.ScoreBreakdown()
    family = metrics.FamilyCounts()
    total_steps = 0
    margin_sum = 0.0
    self_score_sum = 0.0
    for record in records:
        sum_breakdown = sum_breakdown + record.breakdowns[0] + record.breakdowns[1]
        self_score_sum += record.breakdowns[0].total + record.breakdowns[1].total
        margin_sum += record.breakdowns[0].total - record.breakdowns[1].total
        total_steps += len(record.steps)
        family = family + _family_counts(record)

    player_games = max(2 * n_games, 1)
    return metrics.IterationMetrics(
        iteration=iteration,
        total_games=total_games,
        games_this_iter=n_games,
        loss=stats.loss,
        policy_loss=stats.policy_loss,
        value_loss=stats.value_loss,
        entropy=stats.entropy,
        grad_norm=stats.grad_norm,
        advantage_mean=stats.advantage_mean,
        advantage_std=stats.advantage_std,
        avg_self_score=self_score_sum / player_games,
        avg_margin=margin_sum / max(n_games, 1),
        avg_breakdown=sum_breakdown.scaled(1.0 / player_games),
        avg_decisions=total_steps / max(n_games, 1),
        family_counts=family,
        collect_seconds=collect_seconds,
        update_seconds=update_seconds,
        eval_seconds=eval_seconds,
        games_per_sec=n_games / collect_seconds if collect_seconds > 0 else 0.0,
        eval=eval_result,
    )


def _progress_from_payload(payload: dict[str, typing.Any]) -> runstate.RunProgress:
    """The resumable progress stored in a checkpoint. New checkpoints carry a
    full ``progress`` snapshot; older ones only carry the iteration / total-games
    counters, which still suffice to retain the run's place."""
    raw_progress = payload.get("progress")
    if raw_progress is not None:
        return runstate.RunProgress.model_validate(raw_progress)
    return runstate.RunProgress(
        iteration=int(payload.get("iteration", 0)),
        total_games=int(payload.get("total_games", 0)),
    )


def _seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and torch (TRAINING.md §5 reproducibility)."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    # torch's seeding stubs are typed with unknown parameters; suppress the
    # stub-gap report narrowly rather than leaving the seed unset (§5).
    torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # pyright: ignore[reportUnknownMemberType]


def _atomic_save(payload: dict[str, object], path: pathlib.Path) -> None:
    """Write a checkpoint to a temp file then ``os.replace`` it into place so a
    crash mid-write never corrupts the destination (TRAINING.md §5.2)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _git_sha() -> str | None:
    """Best-effort short git SHA of the working tree (None if unavailable)."""
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
