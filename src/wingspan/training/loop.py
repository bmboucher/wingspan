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

import numpy as np
import torch
from torch import optim

from wingspan import model
from wingspan.training import collect, config, evaluate, learner, metrics, runstate


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
        with self.lock:
            self.state.push_event(
                runstate.EventKind.INFO,
                f"run started · {self.config.games_per_iter} games/iter · {self.device}",
            )
        try:
            iteration = 0
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

    ###### PRIVATE #######

    def _reached_limit(self, iteration: int) -> bool:
        return (
            self.config.max_iterations > 0 and iteration >= self.config.max_iterations
        )

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
        start = time.monotonic()
        eval_seed = self.config.seed * 7919 + iteration * 101 + 1
        result = evaluate.evaluate_vs_random(
            self.net, self.device, self.config.eval_games, eval_seed
        )
        return result, time.monotonic() - start

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
        payload: dict[str, object] = {
            "config": self.config.model_dump(),
            "model": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "iteration": iter_metrics.iteration,
            "total_games": self.state.total_games,
            "metrics": iter_metrics.model_dump(),
            "git_sha": _git_sha(),
        }
        _atomic_save(payload, self._ckpt_dir / "last.pt")

        improved = eval_result is not None and (
            self.state.best_win_rate is None
            or eval_result.win_rate > self.state.best_win_rate
        )
        with self.lock:
            if improved and eval_result is not None:
                prev = self.state.best_win_rate
                _atomic_save(payload, self._ckpt_dir / "best.pt")
                self.state.best_win_rate = eval_result.win_rate
                prev_txt = f" > prev {prev * 100:.1f}%" if prev is not None else ""
                self.state.push_event(
                    runstate.EventKind.BEST,
                    f"new best.pt (eval {eval_result.win_rate * 100:.1f}%{prev_txt})",
                )
            else:
                self.state.push_event(
                    runstate.EventKind.CHECKPOINT, "checkpoint saved  last.pt"
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
