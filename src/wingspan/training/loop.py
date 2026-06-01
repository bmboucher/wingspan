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

import datetime
import os
import pathlib
import random
import subprocess
import threading
import time
import traceback
import typing

import numpy as np
import pydantic
import torch
from torch import optim

from wingspan import model
from wingspan.training import (
    artifacts,
    batched_collect,
    collect,
    config,
    evaluate,
    learner,
    metrics,
    mp_collect,
    runmeta,
    runstate,
    sysmon,
)

# How often the side thread refreshes the SYSTEM band's host telemetry. One
# second keeps psutil's CPU sampling window meaningful while adding negligible
# overhead next to self-play collection.
_SYSMON_INTERVAL_SECONDS = 1.0

# torch CPU intra-op thread count for the run. Self-play collection — the
# throughput bottleneck — runs one small forward pass per decision (~130 per
# game), and on CPU those tiny ops run *slower* when many torch threads
# contend over them: measured ~5.7 games/sec at torch's default 12 threads vs
# ~7.5 games/sec at 1-2 threads (+33%). The batched backprop in the update
# phase would prefer more threads, but it costs <0.2s/iter against ~8s of
# collection, so a low global count wins overall. Eval (also per-decision
# inference) benefits identically to collection.
_CPU_INTRAOP_THREADS = 2


class TrainingLoop:
    """A resumable, stoppable self-play training run feeding a live RunState."""

    def __init__(self, cfg: config.TrainConfig):
        self.config = cfg
        self.device = torch.device(cfg.device)
        if self.device.type == "cpu":
            torch.set_num_threads(_CPU_INTRAOP_THREADS)
        _seed_everything(cfg.seed)
        self.net = model.PolicyValueNet(arch=cfg.arch).to(self.device)
        self.optimizer: optim.Optimizer = optim.Adam(self.net.parameters(), lr=cfg.lr)
        self.lock = threading.RLock()
        self.state = runstate.new_run_state(cfg)
        self._stop = threading.Event()
        self._ckpt_dir = pathlib.Path(cfg.checkpoint_dir)
        # The frozen reference opponent the eval plays against; None = the random
        # agent (generation 0). Loaded from ``opponent.pt`` on resume when the
        # restored run had already advanced past the random agent.
        self._opponent_net: model.PolicyValueNet | None = None
        self._monitor = sysmon.SystemMonitor()
        self._monitor_stop = threading.Event()
        self._monitor_thread: threading.Thread | None = None
        # Process-parallel CPU collector, created on first collect and reused
        # across iterations (None until then, and unused on non-CPU devices).
        self._collector: mp_collect.ProcessCollector | None = None
        # Iteration the loop starts numbering from (advanced past a resumed
        # checkpoint). Set last so resume can mutate net / optimizer / state.
        self._start_iteration = 0
        self._maybe_resume()
        self._init_training_phase()
        self._reset_history_logs_if_fresh()
        self._write_run_metadata()

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
            if self._collector is not None:
                self._collector.close()

    ###### PRIVATE #######

    #### Resume & init ####

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
        last = self._ckpt_dir / artifacts.LAST_CKPT
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
                f"could not read {artifacts.LAST_CKPT} — starting fresh",
            )
            return
        if not self._architecture_matches(payload):
            self.state.push_event(
                runstate.EventKind.ALARM,
                f"{artifacts.LAST_CKPT} architecture differs — starting fresh",
            )
            return

        self.net.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self._reset_optimizer_lr()  # honor this run's --lr over the saved one
        progress = _progress_from_payload(payload)
        self.state.restore_progress(progress)
        self._start_iteration = progress.iteration + 1
        if self.state.opponent_generation > 0:
            self._load_opponent()  # may reset generation to 0 if opponent.pt is gone
        self.state.push_event(
            runstate.EventKind.INFO,
            f"resumed {artifacts.LAST_CKPT} · iter {progress.iteration:04d} · "
            f"{progress.total_games:,} games · opponent {self._opponent_label()}",
        )

    def _init_training_phase(self) -> None:
        """Open a fresh run in the random-opponent bootstrap phase when
        ``config.initial_vs_random`` asks for it (collect vs random, eval
        paused). A resumed run keeps the phase restored from its checkpoint —
        ``_start_iteration`` is 0 only on a fresh start — so this never overrides
        a run that already graduated to self-play."""
        if self._start_iteration > 0 or not self.config.initial_vs_random:
            return
        with self.lock:
            self.state.training_phase = runstate.TrainingPhase.RANDOM_OPPONENT
            self.state.push_event(
                runstate.EventKind.INFO,
                "bootstrap: collecting vs random opponent · eval paused "
                f"until {self.config.random_phase_win_rate * 100:.0f}% win-rate",
            )

    def _reset_history_logs_if_fresh(self) -> None:
        """Clear a previous run's history when this run did not resume, so a fresh
        run (``--no-resume``, or one started over an overwritten directory) never
        appends its rows onto stale history. Truncates both append-only logs
        (``metrics.jsonl`` / ``games.jsonl``) and removes the prior run's dated
        ``process_*.json`` session records, leaving the directory reflecting only
        the run starting here. A resumed run (``_start_iteration > 0``) keeps and
        continues its logs (and just adds this session's record)."""
        if self._start_iteration > 0:
            return
        for name in (artifacts.METRICS_LOG, artifacts.GAMES_LOG):
            log_path = self._ckpt_dir / name
            if log_path.exists():
                log_path.write_text("", encoding="utf-8")
        for stale_session in self._ckpt_dir.glob(artifacts.PROCESS_GLOB):
            stale_session.unlink(missing_ok=True)

    def _write_run_metadata(self) -> None:
        """Drop this startup's JSON sidecars: the (overwritten) model descriptor
        and a fresh dated process record. Called once per session after the
        resume decision, so the process record can note where it resumed from."""
        now = datetime.datetime.now()
        runmeta.write_model_config(self.config.checkpoint_dir, self.config)
        session_path = runmeta.write_session_record(
            self.config.checkpoint_dir,
            self.config,
            stamp=now.strftime("%Y%m%d-%H%M%S"),
            started_at=now.isoformat(timespec="seconds"),
            git_sha=_git_sha(),
            resumed_from_iteration=self._start_iteration,
        )
        self.state.push_event(
            runstate.EventKind.INFO, f"session log → {session_path.name}"
        )

    def _architecture_matches(self, payload: dict[str, typing.Any]) -> bool:
        """Whether ``payload``'s saved network shape matches this run's, so its
        weights can be loaded without misrouting heads (TRAINING.md §5.1).

        A saved config that no longer validates (e.g. a value since constrained
        out of bounds) is treated as a mismatch so the run starts fresh with an
        alarm rather than crashing ``__init__`` — preserving the non-fatal
        contract a corrupt/incompatible checkpoint has everywhere else."""
        raw_config = payload.get("config")
        if raw_config is None:
            return True  # pre-descriptor checkpoint — assume compatible
        try:
            saved = config.TrainConfig.model_validate(raw_config)
        except pydantic.ValidationError:
            return False
        return saved.architecture_key == self.config.architecture_key

    def _reset_optimizer_lr(self) -> None:
        """Apply this run's learning rate after loading an optimizer that may have
        saved a different one (Adam's momentum is kept; only the step size moves)."""
        for group in self.optimizer.param_groups:
            group["lr"] = self.config.lr

    #### System monitor ####

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

    #### Iteration orchestration ####

    # One training iteration -- the heart of the loop. Five phases run in order:
    #   1. collect  -- self-play games into recorded forked decisions (_collect)
    #   2. update   -- one length-bucketed REINFORCE step (learner.update)
    #   3. evaluate -- periodic paired games vs the reference opponent (_maybe_evaluate)
    #   4. measure  -- fold the above into one IterationMetrics row
    #   5. commit   -- graduate/advance the opponent, checkpoint, log (_commit_iteration)
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

        games_per_sec = len(records) / collect_seconds if collect_seconds > 0 else 0.0
        with self.lock:
            self.state.phase = runstate.Phase.UPDATING
            self.state.push_event(
                runstate.EventKind.INFO,
                f"COLLECT {len(records)} games in {collect_seconds:.1f}s · "
                f"{games_per_sec:.1f} g/s · avg {_avg_points(records):.1f} pts/game",
            )
        update_start = time.monotonic()
        stats = learner.update(
            self.net, self.optimizer, records, self.config, self.device
        )
        update_seconds = time.monotonic() - update_start
        with self.lock:
            self.state.push_event(
                runstate.EventKind.INFO,
                f"UPDATE in {update_seconds:.2f}s · loss {stats.loss:.3f} · "
                f"entropy {stats.entropy:.3f} · |grad| {stats.grad_norm:.2f}",
            )

        eval_result, eval_seconds = self._maybe_evaluate(iteration)

        # Only the bootstrap phase has a meaningful collection win-rate: the net
        # is seat 0 against the random agent, so winner == 0 is a net win.
        collection_win_rate = (
            _collection_win_rate(records)
            if self.state.training_phase == runstate.TrainingPhase.RANDOM_OPPONENT
            else None
        )
        iter_metrics = _build_iteration_metrics(
            iteration,
            self.state.total_games,
            records,
            stats,
            eval_result,
            collect_seconds,
            update_seconds,
            eval_seconds,
            collection_win_rate,
        )
        self._commit_iteration(iter_metrics, stats, eval_result, records)

    #### Collection ####

    def _collect(self, iteration: int) -> list[collect.GameRecord]:
        """Play ``games_per_iter`` games with batched inference, updating the
        live state as each game finishes so the dashboard advances mid-iteration.
        Games run concurrently and complete out of order; the per-game callback
        runs under ``self.lock`` so the shared state stays consistent. In the
        bootstrap phase the games are net-vs-random rather than self-play."""
        seeds = [
            self.config.seed * 1_000_000 + iteration * 10_000 + game_idx
            for game_idx in range(self.config.games_per_iter)
        ]
        vs_random = self.state.training_phase == runstate.TrainingPhase.RANDOM_OPPONENT
        # CPU collection is GIL-bound under threads, so it fans across worker
        # processes; CUDA collection keeps the in-process batched-inference path
        # (one shared GPU forward beats one model copy per process).
        if self.device.type == "cpu":
            return self._collect_multiprocess(seeds, vs_random)
        return batched_collect.collect_games(
            self.net,
            self.device,
            seeds,
            on_game_done=self._record_collected_game,
            should_stop=self._stop.is_set,
            vs_random=vs_random,
        )

    def _collect_multiprocess(
        self, seeds: list[int], vs_random: bool
    ) -> list[collect.GameRecord]:
        """Collect across worker processes; the pool is built on first use and
        reused across iterations (closed in ``run``'s teardown)."""
        return self._ensure_collector().collect_games(
            self.net,
            self.device,
            seeds,
            on_game_done=self._record_collected_game,
            should_stop=self._stop.is_set,
            vs_random=vs_random,
        )

    def _ensure_collector(self) -> mp_collect.ProcessCollector:
        """The shared worker pool, built on first use and reused for both
        collection and evaluation across iterations."""
        if self._collector is None:
            self._collector = mp_collect.ProcessCollector(self.config)
        return self._collector

    def _record_collected_game(self, record: collect.GameRecord) -> None:
        """Fold one finished self-play game into the live dashboard state."""
        with self.lock:
            self.state.record_game(
                record.breakdowns,
                len(record.steps),
                _family_counts(record),
                record.winner,
            )
            self.state.game_in_iter += 1

    #### Evaluation ####

    def _maybe_evaluate(
        self, iteration: int
    ) -> tuple[metrics.EvalResult | None, float]:
        # The bootstrap phase reads strength from the collection win-rate vs
        # random, so the separate eval block is paused until it graduates.
        if self.state.training_phase == runstate.TrainingPhase.RANDOM_OPPONENT:
            return None, 0.0
        if self.config.eval_every <= 0 or iteration % self.config.eval_every != 0:
            return None, 0.0
        with self.lock:
            self.state.phase = runstate.Phase.EVALUATING
            self.state.eval_game_in_iter = 0
            self.state.eval_games_in_iter = 2 * self.config.eval_pairs
        start = time.monotonic()
        eval_seed = self.config.seed * 7919 + iteration * 101 + 1
        # CPU eval fans across the same worker pool collection uses; CUDA keeps
        # the in-process sequential path (one shared GPU beats a model per
        # process). Both paths run identical per-game logic, so results match.
        if self.device.type == "cpu":
            result = self._ensure_collector().evaluate_games(
                self.net,
                self._opponent_net,
                self.device,
                self.config.eval_pairs,
                eval_seed,
                opponent_generation=self.state.opponent_generation,
                on_progress=self._record_eval_progress,
            )
        else:
            result = evaluate.evaluate_vs_opponent(
                self.net,
                self._opponent_net,
                self.device,
                self.config.eval_pairs,
                eval_seed,
                opponent_generation=self.state.opponent_generation,
                on_progress=self._record_eval_progress,
            )
        return result, time.monotonic() - start

    def _record_eval_progress(self, games_done: int, total_games: int) -> None:
        """Publish held-out eval progress so the header bar tracks eval games."""
        with self.lock:
            self.state.eval_game_in_iter = games_done
            self.state.eval_games_in_iter = total_games

    # ------------------------------------------------------------------
    # Reference-opponent advancement (TRAINING.md §7)
    #### Opponent advancement ####

    def _opponent_label(self) -> str:
        """A short name for the current reference opponent (for events)."""
        gen = self.state.opponent_generation
        return "random" if gen == 0 else f"self·gen{gen}"

    def _maybe_graduate_from_random_phase(self) -> None:
        """Leave the random-opponent bootstrap phase once the smoothed collection
        win-rate clears ``config.random_phase_win_rate``: freeze the current
        policy as the first self-play opponent (self·gen1), switch collection to
        self-play, and resume evaluation against it. A no-op outside the
        bootstrap phase, so it is safe to call every iteration."""
        if self.state.training_phase != runstate.TrainingPhase.RANDOM_OPPONENT:
            return
        threshold = self.config.random_phase_win_rate
        with self.lock:
            ewma = self.state.collection_win_rate_ewma()
            if ewma is None or ewma < threshold:
                return

        frozen = self._clone_net()
        self._save_opponent(frozen, generation=1)
        with self.lock:
            self._opponent_net = frozen
            self.state.training_phase = runstate.TrainingPhase.SELF_PLAY
            self.state.opponent_generation = 1
            self.state.opponent_since_iteration = self.state.iteration
            self.state.best_win_rate = None  # best is per-opponent-generation
            self.state.push_event(
                runstate.EventKind.BEST,
                f"graduated random phase → self·gen1 "
                f"(collection win-rate {ewma * 100:.0f}%) · self-play + eval resume",
            )

    def _maybe_advance_opponent(self, eval_result: metrics.EvalResult | None) -> None:
        """When the smoothed win-rate against the current opponent clears the
        configured threshold, freeze the current policy as the new "player to
        beat": save it to ``opponent.pt``, bump the generation, and reset the
        win-rate trend (best + EWMA) so progress against the stronger opponent
        starts a fresh climb from ~50%."""
        threshold = self.config.opponent_reset_win_rate
        if eval_result is None or threshold <= 0.0:
            return
        with self.lock:
            ewma = self.state.eval_ewma()
            if ewma is None or ewma.win_rate < threshold:
                return
            new_generation = self.state.opponent_generation + 1

        frozen = self._clone_net()
        self._save_opponent(frozen, new_generation)
        with self.lock:
            self._opponent_net = frozen
            self.state.opponent_generation = new_generation
            self.state.opponent_since_iteration = self.state.iteration
            self.state.best_win_rate = None  # best is per-opponent-generation
            self.state.push_event(
                runstate.EventKind.BEST,
                f"opponent advanced → self·gen{new_generation} "
                f"(beat {self._prev_opponent_label(new_generation)} "
                f"{ewma.win_rate * 100:.0f}%) · win-rate reset",
            )

    def _prev_opponent_label(self, new_generation: int) -> str:
        return "random" if new_generation == 1 else f"self·gen{new_generation - 1}"

    def _clone_net(self) -> model.PolicyValueNet:
        """An independent, eval-mode copy of the current policy network."""
        clone = model.PolicyValueNet(
            state_dim=self.config.state_dim,
            choice_dim=self.config.choice_dim,
            arch=self.config.arch,
        ).to(self.device)
        clone.load_state_dict(self.net.state_dict())
        clone.eval()
        return clone

    def _save_opponent(self, opponent: model.PolicyValueNet, generation: int) -> None:
        """Persist the frozen opponent so a resumed run keeps the same reference."""
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)
        payload: dict[str, object] = {
            "config": self.config.model_dump(),
            "model": opponent.state_dict(),
            "opponent_generation": generation,
            "git_sha": _git_sha(),
        }
        _atomic_save(payload, self._ckpt_dir / artifacts.OPPONENT_CKPT)

    def _load_opponent(self) -> None:
        """Restore the frozen opponent from ``opponent.pt`` on resume. If it is
        missing or unreadable, fall back to the random agent (generation 0) so
        the run stays consistent rather than evaluating against nothing."""
        path = self._ckpt_dir / artifacts.OPPONENT_CKPT
        try:
            payload = typing.cast(
                "dict[str, typing.Any]",
                torch.load(path, map_location=self.device, weights_only=False),
            )
            opponent = self._clone_net()
            opponent.load_state_dict(payload["model"])
            opponent.eval()
        except Exception:  # noqa: BLE001 — a missing/corrupt opponent resets to random
            self.state.opponent_generation = 0
            self.state.push_event(
                runstate.EventKind.ALARM,
                f"could not read {artifacts.OPPONENT_CKPT} — opponent reset to random",
            )
            return
        self._opponent_net = opponent

    #### Checkpointing ####

    def _commit_iteration(
        self,
        iter_metrics: metrics.IterationMetrics,
        stats: learner.UpdateStats,
        eval_result: metrics.EvalResult | None,
        records: list[collect.GameRecord],
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
                    f"EVAL {eval_result.n_games} games in "
                    f"{iter_metrics.eval_seconds:.1f}s · "
                    f"{eval_result.win_rate * 100:.1f}% ±{eval_result.ci95 * 100:.1f}% "
                    f"vs {self._opponent_label()} · "
                    f"margin {eval_result.mean_margin:+.1f}",
                )

        # Graduate out of the bootstrap phase (freezes self·gen1) or advance the
        # frozen opponent before checkpointing, so ``last.pt`` records the new
        # generation / phase alongside the matching ``opponent.pt`` snapshot. The
        # two are mutually exclusive within an iteration: graduation runs only in
        # the random phase (where ``eval_result`` is always None), and opponent
        # advancement only acts on a non-None eval result.
        self._maybe_graduate_from_random_phase()
        self._maybe_advance_opponent(eval_result)

        with self.lock:
            self.state.phase = runstate.Phase.CHECKPOINTING
        self._checkpoint(iter_metrics, eval_result, records)

    def _checkpoint(
        self,
        iter_metrics: metrics.IterationMetrics,
        eval_result: metrics.EvalResult | None,
        records: list[collect.GameRecord],
    ) -> None:
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)
        with self.lock:
            # "best" is per opponent-generation: the eval that triggers an
            # advancement belongs to the old opponent (its generation no longer
            # matches), so it is not credited as the new generation's best.
            improved = (
                eval_result is not None
                and eval_result.opponent_generation == self.state.opponent_generation
                and (
                    self.state.best_win_rate is None
                    or eval_result.win_rate > self.state.best_win_rate
                )
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
        _atomic_save(payload, self._ckpt_dir / artifacts.LAST_CKPT)
        if improved and eval_result is not None:
            _atomic_save(payload, self._ckpt_dir / artifacts.BEST_CKPT)

        with self.lock:
            # A new best is worth surfacing (it carries an eval number); a routine
            # last.pt write is not — the RECENT EVENTS log tracks phase
            # transitions (collect / update / eval), not file saves.
            if improved and eval_result is not None:
                prev_txt = (
                    f" > prev {prev_best * 100:.1f}%" if prev_best is not None else ""
                )
                self.state.push_event(
                    runstate.EventKind.BEST,
                    f"new {artifacts.BEST_CKPT} (eval {eval_result.win_rate * 100:.1f}%{prev_txt})",
                )

        with open(
            self._ckpt_dir / artifacts.METRICS_LOG, "a", encoding="utf-8"
        ) as handle:
            handle.write(iter_metrics.model_dump_json() + "\n")

        # Per-game history is appended after ``last.pt`` (written above) so a
        # crash between the two only ever loses this iteration's rows rather than
        # duplicating them on the resume that re-plays the un-checkpointed cycle.
        self._append_game_history(_build_game_outcomes(records, iter_metrics.iteration))

    def _append_game_history(self, outcomes: list[metrics.GameOutcome]) -> None:
        """Append one ``games.jsonl`` line per finished game (a single buffered
        write per iteration — ~256 lines every few seconds — so the per-game log
        never becomes a throughput drag)."""
        if not outcomes:
            return
        rows = "".join(outcome.model_dump_json() + "\n" for outcome in outcomes)
        with open(
            self._ckpt_dir / artifacts.GAMES_LOG, "a", encoding="utf-8"
        ) as handle:
            handle.write(rows)

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

#### Metrics aggregation ####


def _family_counts(record: collect.GameRecord) -> metrics.FamilyCounts:
    counts = metrics.FamilyCounts()
    for step in record.steps:
        counts.bump(step.family_idx)
    return counts


def _build_game_outcomes(
    records: list[collect.GameRecord], iteration: int
) -> list[metrics.GameOutcome]:
    """One persisted :class:`metrics.GameOutcome` per finished game, tagged with
    the ``iteration`` that produced it — the rows appended to ``games.jsonl``."""
    return [
        metrics.GameOutcome(
            iteration=iteration,
            seed=record.seed,
            winner=record.winner,
            decisions=len(record.steps),
            breakdowns=record.breakdowns,
            family_counts=_family_counts(record),
        )
        for record in records
    ]


def _pop_std(sum_sq: float, mean: float, n: int) -> float:
    """Population σ from a Σx², a mean, and a sample count (clamped at 0)."""
    var = sum_sq / max(n, 1) - mean * mean
    return var**0.5 if var > 0.0 else 0.0


def _avg_points(records: list[collect.GameRecord]) -> float:
    """Mean final score across both seats of every game in a collected batch."""
    if not records:
        return 0.0
    total = sum(rec.breakdowns[0].total + rec.breakdowns[1].total for rec in records)
    return total / (2 * len(records))


def _collection_win_rate(records: list[collect.GameRecord]) -> float:
    """Win fraction for the net over a bootstrap-phase batch, ties as half. The
    net always plays seat 0 against the random agent, so ``winner == 0`` is a net
    win and ``winner == -1`` is a tie."""
    if not records:
        return 0.0
    wins = sum(1 for record in records if record.winner == 0)
    ties = sum(1 for record in records if record.winner < 0)
    return (wins + 0.5 * ties) / len(records)


#### Iteration metrics ####


def _build_iteration_metrics(
    iteration: int,
    total_games: int,
    records: list[collect.GameRecord],
    stats: learner.UpdateStats,
    eval_result: metrics.EvalResult | None,
    collect_seconds: float,
    update_seconds: float,
    eval_seconds: float,
    collection_win_rate: float | None,
) -> metrics.IterationMetrics:
    n_games = len(records)
    sum_breakdown = metrics.ScoreBreakdown()
    winner_breakdown = metrics.ScoreBreakdown()
    decided_games = 0
    family = metrics.FamilyCounts()
    total_steps = 0
    total_steps_sq = 0
    margin_sum = 0.0
    margin_sq_sum = 0.0
    abs_margin_sum = 0.0
    self_score_sum = 0.0
    for record in records:
        sum_breakdown = sum_breakdown + record.breakdowns[0] + record.breakdowns[1]
        self_score_sum += record.breakdowns[0].total + record.breakdowns[1].total
        margin = record.breakdowns[0].total - record.breakdowns[1].total
        margin_sum += margin
        margin_sq_sum += margin * margin
        abs_margin_sum += abs(margin)
        if record.winner >= 0:
            winner_breakdown = winner_breakdown + record.breakdowns[record.winner]
            decided_games += 1
        steps = len(record.steps)
        total_steps += steps
        total_steps_sq += steps * steps
        family = family + _family_counts(record)

    player_games = max(2 * n_games, 1)
    games = max(n_games, 1)
    margin_mean = margin_sum / games
    abs_margin_mean = abs_margin_sum / games
    # Per-cycle population σ over this iteration's games. ``|margin|² == margin²``,
    # so the signed-margin second moment also yields the winning-margin σ.
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
        avg_margin=margin_mean,
        avg_breakdown=sum_breakdown.scaled(1.0 / player_games),
        avg_decisions=total_steps / games,
        avg_winner_breakdown=winner_breakdown.scaled(1.0 / max(decided_games, 1)),
        avg_abs_margin=abs_margin_mean,
        margin_std=_pop_std(margin_sq_sum, margin_mean, games),
        abs_margin_std=_pop_std(margin_sq_sum, abs_margin_mean, games),
        decisions_std=_pop_std(total_steps_sq, total_steps / games, games),
        family_counts=family,
        collect_seconds=collect_seconds,
        update_seconds=update_seconds,
        eval_seconds=eval_seconds,
        games_per_sec=n_games / collect_seconds if collect_seconds > 0 else 0.0,
        eval=eval_result,
        collection_win_rate=collection_win_rate,
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


#### Seeding ####


def _seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and torch (TRAINING.md §5 reproducibility)."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    # torch's seeding stubs are typed with unknown parameters; suppress the
    # stub-gap report narrowly rather than leaving the seed unset (§5).
    torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # pyright: ignore[reportUnknownMemberType]


#### Checkpoint I/O ####


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
