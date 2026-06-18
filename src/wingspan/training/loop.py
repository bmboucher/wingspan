"""The training orchestrator: the worker side of the dashboard.

``TrainingLoop`` owns the network, optimizer, RNG, and the shared
:class:`runstate.RunState`. Its :meth:`run` drives the TRAINING.md Phase-1
program — collect a batch of self-play games, run one length-bucketed REINFORCE
update, periodically evaluate against the random agent, checkpoint — looping
until ``max_iterations`` or an external stop request.

It runs on a background thread; every mutation of the shared state is made under
``self.lock`` so the main-thread renderer always reads a consistent frame. The
loop never touches the terminal — all presentation lives in the dashboard.

The per-concern logic is split across sibling modules:

- ``loop_resume``     — checkpoint resume, init phase/target, metadata writes
- ``loop_collect``    — batched / multiprocess self-play collection
- ``loop_setup``      — setup-model lifecycle (fit, update, sync, resume)
- ``loop_eval``       — paired-game eval, opponent graduation / advancement
- ``loop_target``     — target-milestone sequence (checkpoint → eval → pause)
- ``loop_checkpoint`` — commit, checkpoint write, finish; I/O + seed helpers
- ``loop_metrics``    — pure metrics aggregation (no loop state)
"""

from __future__ import annotations

import pathlib
import threading
import time
import traceback
import typing

import torch
from torch import optim

from wingspan import model, setup_model
from wingspan.training import (
    artifacts,
    collect,
    config,
    learner,
    loop_checkpoint,
    loop_collect,
    loop_eval,
    loop_metrics,
    loop_resume,
    loop_setup,
    loop_target,
    mp_collect,
    runstate,
    setup_net,
    sysmon,
)

# torch CPU intra-op thread count for the run. Self-play collection — the
# throughput bottleneck — runs one small forward pass per decision (~130 per
# game), and on CPU those tiny ops run *slower* when many torch threads
# contend over them: measured ~5.7 games/sec at torch's default 12 threads vs
# ~7.5 games/sec at 1-2 threads (+33%). The batched backprop in the update
# phase would prefer more threads, but it costs <0.2s/iter against ~8s of
# collection, so a low global count wins overall. Eval (also per-decision
# inference) benefits identically to collection.
_CPU_INTRAOP_THREADS = 2

# How often the side thread refreshes the SYSTEM band's host telemetry.
_SYSMON_INTERVAL_SECONDS = 1.0


class TrainingLoop:
    """A resumable, stoppable self-play training run feeding a live RunState."""

    def __init__(self, cfg: config.RunConfig, *, pause_at_target: bool = True):
        # Pin the config to a resumable checkpoint's artifact era before
        # anything derives from it (net class and dims, encoders, stamps), so a
        # run started before a FRESH encoding change resumes at its own frozen
        # geometry from any entry point (docs/VERSIONING.md).
        cfg = loop_resume.adopt_checkpoint_era(cfg)
        self.config = cfg
        # Whether reaching ``target_iterations`` pauses for interactive
        # [C]ontinue / [E]nd input (the dashboard) or finalizes the milestone and
        # ends the run (the headless cloud runner passes ``False``).
        self._pause_at_target = pause_at_target
        self.device = torch.device(cfg.misc.device)
        if self.device.type == "cpu":
            torch.set_num_threads(_CPU_INTRAOP_THREADS)
        loop_checkpoint.seed_everything(cfg.misc.seed)
        # The net class and dims are era-routed: an era-pinned run constructs
        # the matching compat subclass at its frozen widths.
        net_cls = model.PolicyValueNet.class_for_version(cfg.encoding_version)
        self.net = net_cls(
            state_dim=cfg.state_dim,
            choice_dim=cfg.choice_dim,
            num_families=len(cfg.family_order),
            arch=cfg.arch,
            spec=cfg.encoding_spec,
        ).to(self.device)
        self.optimizer: optim.Optimizer = optim.Adam(
            self.net.parameters(), lr=cfg.training.lr
        )
        self.lock = threading.RLock()
        self.state = runstate.new_run_state(cfg)
        self._stop = threading.Event()
        self._ckpt_dir = pathlib.Path(cfg.run.checkpoint_dir)
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
        # Setup model: a separate value-regression net trained on a different
        # schedule (built only when enabled). Its optimizer, on-disk sample store,
        # and one-time-offline-fit flag live alongside the main net's.
        self._setup_net: setup_net.SetupNet | None = None
        self._setup_optimizer: optim.Optimizer | None = None
        self._setup_store: setup_model.SetupDataStore | None = None
        # Pre-mark the offline fit done when there is no warmup schedule
        # (setup_train_iter == 0 means MODEL_DRIVEN from iteration 0 — no
        # offline fit window was ever recorded, so the one-time fit is skipped).
        self._setup_fit_done = cfg.training.setup.train_iter == 0
        if cfg.architecture.use_setup_model:
            self._setup_net, self._setup_optimizer = loop_setup.build_setup_net(self)
            self._setup_store = setup_model.SetupDataStore(
                self._ckpt_dir / artifacts.SETUP_DATA_LOG
            )
        # Iteration the loop starts numbering from (advanced past a resumed
        # checkpoint). Set last so resume can mutate net / optimizer / state.
        self._start_iteration = 0
        # Signals the loop to wake from PAUSED_AT_TARGET when the dashboard
        # receives a user [C]ontinue or [E]nd keypress.
        self._target_reached_event = threading.Event()
        loop_resume.maybe_resume(self)
        loop_setup.maybe_resume_setup(self)
        # The setup net's frozen embedder copies must reflect the (possibly just
        # resumed) main net before any collection or checkpointing happens.
        loop_setup.sync_setup_embedders(self)
        loop_resume.init_training_phase(self)
        loop_resume.validate_bootstrap_opponent(self)
        loop_resume.validate_dagger_expert(self)
        loop_resume.init_target_if_fresh(self)
        loop_resume.reset_history_logs_if_fresh(self)
        loop_resume.write_run_metadata(self)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def request_stop(self) -> None:
        """Ask the loop to finish the current game and shut down gracefully."""
        self._stop.set()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def signal_target_response(
        self,
        choice: typing.Literal["continue", "end"],
        new_target: int = 0,
    ) -> None:
        """Unblock the loop from PAUSED_AT_TARGET.

        Called from the dashboard thread. ``choice`` is ``"continue"`` or
        ``"end"``; ``new_target`` sets the next milestone when > 0, or clears
        the target entirely when 0 (the user chose to continue without a new
        target and training runs until ``max_iterations`` or Ctrl+C).
        """
        with self.lock:
            self.state.user_target_choice = choice
            if choice == "continue":
                self.state.target_iterations = new_target
        self._target_reached_event.set()

    def run(self) -> None:
        """Run iterations until ``max_iterations`` or a stop request.

        Intended as the target of a worker thread; never raises — failures land
        in ``state.phase = ERROR`` with the traceback in ``state.error``.
        """
        self._start_monitor()
        with self.lock:
            self.state.push_event(
                runstate.EventKind.INFO,
                f"run started · {self.config.run.games_per_iter} games/iter · {self.device}",
            )
        try:
            iteration = self._start_iteration
            while not self._stop.is_set() and not self._reached_limit(iteration):
                self._run_iteration(iteration)
                if loop_target.handle_target_if_reached(self, iteration):
                    break  # user chose "end" → exit the iteration loop
                iteration += 1
            loop_checkpoint.finish(
                self,
                runstate.Phase.STOPPED if self._stop.is_set() else runstate.Phase.DONE,
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

    #### Run control ####

    def _reached_limit(self, iteration: int) -> bool:
        # ``max_iterations`` caps iterations run *this session*, so resuming a run
        # with ``--iterations N`` does N more rather than stopping immediately.
        done_this_session = iteration - self._start_iteration
        return (
            self.config.run.max_iterations > 0
            and done_this_session >= self.config.run.max_iterations
        )

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
    #   1. collect  -- self-play games into recorded forked decisions (loop_collect)
    #   2. update   -- one length-bucketed REINFORCE step (learner.update)
    #   3. evaluate -- periodic paired games vs the reference opponent (loop_eval)
    #   4. measure  -- fold the above into one IterationMetrics row (loop_metrics)
    #   5. commit   -- graduate/advance the opponent, checkpoint, log (loop_checkpoint)
    def _run_iteration(self, iteration: int) -> None:
        with self.lock:
            self.state.phase = runstate.Phase.COLLECTING
            self.state.iteration = iteration
            self.state.game_in_iter = 0
            self.state.iter_start_monotonic = time.monotonic()

        # Setup-model schedule: which regime this iteration runs under (None when
        # the feature is off). The one-time offline fit happens before this
        # iteration's collection so the net is trained before it drives selection.
        setup_phase = (
            loop_setup.setup_phase_for(self, iteration)
            if self.config.architecture.use_setup_model
            else None
        )
        if setup_phase is not None:
            with self.lock:
                self.state.setup_phase = setup_phase.name
            if (
                setup_phase is collect.SetupPhase.MODEL_DRIVEN
                and not self._setup_fit_done
            ):
                loop_setup.run_offline_setup_fit(self)

        # DAgger clone phase: pure imitation for the first clone_iters iterations.
        # vs_random is driven independently by training_phase; the DAgger validator
        # ensures bootstrap_opponent == "none" when clone_iters > 0, so there is
        # no conflict between the two modes.
        imitation_phase = self.config.dagger_active_at(iteration)

        collect_start = time.monotonic()
        records = loop_collect.collect_games(
            self, iteration, setup_phase, dagger_active=imitation_phase
        )
        collect_seconds = time.monotonic() - collect_start
        if not records:
            return  # stopped before completing any game this iteration

        games_per_sec = len(records) / collect_seconds if collect_seconds > 0 else 0.0
        with self.lock:
            self.state.phase = runstate.Phase.UPDATING
            self.state.push_event(
                runstate.EventKind.INFO,
                f"COLLECT {len(records)} games in {collect_seconds:.1f}s · "
                f"{games_per_sec:.1f} g/s · "
                f"avg {loop_metrics.avg_points(records):.1f} pts/game"
                + (" · DAgger clone" if imitation_phase else ""),
            )
        update_start = time.monotonic()
        stats = learner.update(
            self.net,
            self.optimizer,
            records,
            self.config,
            self.device,
            imitation_phase=imitation_phase,
        )
        update_seconds = time.monotonic() - update_start
        with self.lock:
            self.state.push_event(
                runstate.EventKind.INFO,
                f"UPDATE in {update_seconds:.2f}s · loss {stats.loss:.3f} · "
                f"entropy {stats.entropy:.3f} · |grad| {stats.grad_norm:.2f}",
            )

        # Re-sync the setup net's frozen embedder copies to the just-updated main
        # net before the setup update / checkpoint / next broadcast, so setup.pt
        # and the worker weights always carry this iteration's representations.
        loop_setup.sync_setup_embedders(self)

        # Setup-model update: record this iteration's samples (random-record
        # phase) or run one on-policy MSE step (model-driven phase).
        setup_stats = (
            loop_setup.update_setup(self, setup_phase, records)
            if setup_phase is not None
            else None
        )

        eval_result, eval_seconds = loop_eval.maybe_evaluate(self, iteration)

        # Only the bootstrap phase has a meaningful collection win-rate: the net
        # is seat 0 against the random agent, so winner == 0 is a net win.
        win_rate = (
            loop_metrics.collection_win_rate(records)
            if self.state.training_phase == runstate.TrainingPhase.RANDOM_OPPONENT
            else None
        )
        iter_metrics = loop_metrics.build_iteration_metrics(
            iteration,
            self.state.total_games,
            records,
            stats,
            eval_result,
            collect_seconds,
            update_seconds,
            eval_seconds,
            win_rate,
            setup_phase,
            setup_stats,
            imitation_phase=imitation_phase,
        )
        loop_checkpoint.commit_iteration(
            self, iter_metrics, stats, eval_result, records
        )
