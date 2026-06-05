"""The live, shared state the training loop writes and the dashboard reads.

``RunState`` is the single mutable object the worker thread updates as games
complete and iterations finish; the render thread reads it (under the loop's
lock) to repaint. It holds the cumulative running aggregates (so the dashboard
can show "average since start"), the per-iteration history (the raw material for
the convergence charts), the monotonic wall-clock anchors (so the two
chronometers tick every refresh regardless of training activity), and a bounded
ring of recent events.

All read-side derivations (averages, elapsed times, the chart series) are pure
methods so the dashboard never reaches into the raw counters.
"""

from __future__ import annotations

import enum
import time

import pydantic

from wingspan.training import config, metrics


class Phase(enum.StrEnum):
    """What the training loop is doing right now (drives the LED + accents)."""

    STARTING = "starting"
    COLLECTING = "collecting"
    UPDATING = "updating"
    EVALUATING = "evaluating"
    CHECKPOINTING = "checkpointing"
    FINAL_EVALUATING = "final_evaluating"  # large fixed-model eval at target milestone
    PAUSED_AT_TARGET = "paused_at_target"  # waiting for user [C]ontinue / [E]nd input
    DONE = "done"
    STOPPED = "stopped"
    ERROR = "error"

    @property
    def is_terminal(self) -> bool:
        return self in (Phase.DONE, Phase.STOPPED, Phase.ERROR)


class TrainingPhase(enum.StrEnum):
    """Which opponent regime the run is in — distinct from :class:`Phase` (the
    live collect/update/eval activity).

    ``RANDOM_OPPONENT`` is the bootstrap phase: collection games pit the net
    (seat 0) against the random agent and evaluation is paused, so strength is
    read from the collection win-rate. ``SELF_PLAY`` is the ordinary regime:
    both seats are the net and evaluation runs against the frozen reference
    opponent. A fresh run starts in whichever phase ``config.initial_vs_random``
    selects and graduates to ``SELF_PLAY`` once the smoothed collection win-rate
    clears ``config.random_phase_win_rate``. ``SELF_PLAY`` is the default — the
    steady-state regime every run graduates into.
    """

    RANDOM_OPPONENT = "random_opponent"
    SELF_PLAY = "self_play"


class EventKind(enum.StrEnum):
    """Category of a recent-events log line (drives its glyph + color)."""

    INFO = "info"
    EVAL = "eval"
    CHECKPOINT = "checkpoint"
    BEST = "best"
    ALARM = "alarm"


class EventLine(pydantic.BaseModel):
    """One recent-events entry: a wall-clock stamp, a kind, and the text."""

    clock: str  # HH:MM:SS since start
    kind: EventKind
    text: str


_MAX_EVENTS = 40


def _new_history() -> list[metrics.IterationMetrics]:
    return []


def _new_events() -> list[EventLine]:
    return []


def _new_int_list() -> list[int]:
    return []


class RunProgress(pydantic.BaseModel):
    """The resumable slice of a run's live state, persisted in every checkpoint
    so a restarted run continues its counters, cumulative aggregates, and
    convergence charts instead of starting from zero.

    Every field defaults to its fresh-run value, so ``RunProgress()`` is the
    empty snapshot. Fields added to this model later must also default (with a
    comment naming why), so checkpoints already written by current-era runs keep
    resuming — see CLAUDE.md "Checkpoint compatibility policy".
    """

    iteration: int = 0
    total_games: int = 0
    total_decisions: int = 0
    # Wall-clock seconds the run had already accumulated at checkpoint time, so
    # the dashboard's ``T+`` chronometer resumes where it left off instead of
    # restarting at zero.
    elapsed_seconds: float = 0.0
    # The recent-events ring, so the RECENT EVENTS panel reopens with its log
    # rather than empty.
    events: list[EventLine] = pydantic.Field(default_factory=_new_events)
    cum_breakdown: metrics.ScoreBreakdown = pydantic.Field(
        default_factory=metrics.ScoreBreakdown
    )
    cum_player_games: int = 0
    cum_games: int = 0
    cum_decisions: int = 0
    cum_decisions_sq: float = 0.0
    cum_family: metrics.FamilyCounts = pydantic.Field(
        default_factory=metrics.FamilyCounts
    )
    cum_margin_sum: float = 0.0
    cum_margin_sq: float = 0.0
    cum_abs_margin_sum: float = 0.0
    cum_winner_breakdown: metrics.ScoreBreakdown = pydantic.Field(
        default_factory=metrics.ScoreBreakdown
    )
    cum_decided_games: int = 0
    game_len_min: int | None = None
    game_len_max: int | None = None
    best_win_rate: float | None = None
    opponent_generation: int = 0
    # The iteration at which the current reference opponent was frozen, so the
    # dashboard can show how many iterations have passed since the frozen self
    # model was last advanced.
    opponent_since_iteration: int = 0
    # The iterations at which the reference opponent was advanced, so the WIN RATE
    # convergence chart can mark each challenger upgrade with a vertical line.
    opponent_change_iterations: list[int] = pydantic.Field(
        default_factory=_new_int_list
    )
    # Which opponent regime the run is in (SELF_PLAY is the steady state).
    training_phase: TrainingPhase = TrainingPhase.SELF_PLAY
    last_iter: metrics.IterationMetrics | None = None
    history: list[metrics.IterationMetrics] = pydantic.Field(
        default_factory=_new_history
    )

    # Target milestone (initialized from config on fresh run; updated when the user
    # continues with a new target). Stored here so it persists across checkpoints.
    target_iterations: int = 0

    # Per-phase timing counters for time-to-target estimation.  Each iteration
    # records its wall-clock duration in the appropriate bucket so the rate can be
    # computed without any transient clock state.
    random_phase_iter_count: int = 0
    random_phase_seconds: float = 0.0
    self_play_iter_count: int = 0
    self_play_seconds: float = 0.0


class RunState(pydantic.BaseModel):
    """Everything the dashboard needs to repaint a single frame."""

    config: config.TrainConfig
    phase: Phase = Phase.STARTING

    # Monotonic wall-clock anchors (seconds). ``stopped_monotonic`` freezes the
    # clocks once the run ends so a finished dashboard stops ticking.
    start_monotonic: float
    iter_start_monotonic: float
    stopped_monotonic: float | None = None
    # Wall-clock seconds carried over from previous sessions (restored from a
    # checkpoint on resume), added to the live ``now - start`` so ``T+`` counts
    # total run time across restarts rather than just this session's.
    elapsed_offset: float = 0.0

    # Live counters.
    iteration: int = 0
    game_in_iter: int = 0
    games_in_iter: int = 0
    # During an EVALUATING phase these track held-out eval games (out of the
    # block's 2 * eval_pairs games) so the header progress bar reports eval
    # progress instead of the already-finished collection's counts.
    eval_game_in_iter: int = 0
    eval_games_in_iter: int = 0
    total_games: int = 0
    total_decisions: int = 0

    # Cumulative running aggregates (since the run started).
    cum_breakdown: metrics.ScoreBreakdown = pydantic.Field(
        default_factory=metrics.ScoreBreakdown
    )
    cum_player_games: int = 0  # = 2 * cum_games (one breakdown per seat)
    cum_games: int = 0
    cum_decisions: int = 0
    cum_decisions_sq: float = 0.0  # Σ (decisions per game)^2, for a length σ
    cum_family: metrics.FamilyCounts = pydantic.Field(
        default_factory=metrics.FamilyCounts
    )
    cum_margin_sum: float = 0.0  # Σ (player0 − player1)
    cum_margin_sq: float = 0.0  # Σ (player0 − player1)^2
    cum_abs_margin_sum: float = 0.0  # Σ |player0 − player1| (winning margin)
    cum_winner_breakdown: metrics.ScoreBreakdown = pydantic.Field(
        default_factory=metrics.ScoreBreakdown
    )  # Σ winning-seat score split over decided games
    cum_decided_games: int = 0  # games with a winner (excludes ties)
    game_len_min: int | None = None
    game_len_max: int | None = None

    # Latest readouts + history (capped to config.history_len).
    last_iter: metrics.IterationMetrics | None = None
    history: list[metrics.IterationMetrics] = pydantic.Field(
        default_factory=_new_history
    )
    best_win_rate: float | None = None
    # Which reference opponent evals are currently played against (0 = random
    # agent; advanced to a frozen past self each time the policy crushes it).
    opponent_generation: int = 0
    # The iteration the current reference opponent was frozen at (0 while still
    # evaluating against the random agent), so the EVAL inset can report how many
    # iterations have passed since the frozen self model was last advanced.
    opponent_since_iteration: int = 0
    # The iterations at which the reference opponent was advanced, so the WIN RATE
    # convergence chart can mark each challenger upgrade with a vertical line.
    opponent_change_iterations: list[int] = pydantic.Field(
        default_factory=_new_int_list
    )
    # The opponent regime collection plays under (random-opponent bootstrap vs
    # self-play). Drives whether evaluation runs and which win-rate the dashboard
    # plots.
    training_phase: TrainingPhase = TrainingPhase.SELF_PLAY

    events: list[EventLine] = pydantic.Field(default_factory=_new_events)
    error: str | None = None

    # Target-milestone counters (persisted via RunProgress → to_progress /
    # restore_progress). ``target_iterations`` is the live goal (initialised from
    # config on fresh runs; updated when the user continues with a new target).
    target_iterations: int = 0
    random_phase_iter_count: int = 0
    random_phase_seconds: float = 0.0
    self_play_iter_count: int = 0
    self_play_seconds: float = 0.0

    # Setup-model live readouts (None when the setup model is off). ``setup_phase``
    # is the current regime label (random / recording / model); ``last_setup`` is
    # the most recent setup-net update summary. Transient (dashboard-only) — not
    # persisted in ``RunProgress``; the durable record is ``metrics.jsonl``.
    setup_phase: str | None = None
    last_setup: metrics.SetupUpdateStats | None = None

    # Live host / accelerator telemetry, refreshed by the monitor thread
    # (None until the first sample lands).
    system: metrics.SystemStats | None = None

    # Target-milestone transient state (not persisted in RunProgress).
    # ``pinned_stats`` is set after the final self-play eval and cleared when the
    # first new training iteration completes after a [C]ontinue; it drives the
    # IN-GAME PERFORMANCE "[FINAL]" display.
    pinned_stats: metrics.FinalEvalStats | None = None
    # Progress of the FINAL_EVALUATING phase: (games_done, games_total).
    final_eval_progress: tuple[int, int] = (0, 0)
    # Written by the dashboard key-handler; read by the loop after it wakes from
    # the PAUSED_AT_TARGET wait.
    user_target_choice: str | None = None  # "continue" | "end" | None

    # ----- writer-side helpers (called by the loop, under its lock) -----

    def record_game(
        self,
        breakdowns: tuple[metrics.ScoreBreakdown, metrics.ScoreBreakdown],
        decisions_seen: int,
        family: metrics.FamilyCounts,
        winner: int,
    ) -> None:
        """Fold one finished game into the cumulative aggregates. ``winner`` is
        the winning seat (0 or 1, or -1 for a tie) so the winning-seat score
        split can be aggregated alongside the all-seats one."""
        self.cum_breakdown = self.cum_breakdown + breakdowns[0] + breakdowns[1]
        self.cum_player_games += 2
        self.cum_games += 1
        self.cum_decisions += decisions_seen
        self.cum_decisions_sq += decisions_seen * decisions_seen
        self.cum_family = self.cum_family + family
        margin = breakdowns[0].total - breakdowns[1].total
        self.cum_margin_sum += margin
        self.cum_margin_sq += margin * margin
        self.cum_abs_margin_sum += abs(margin)
        if winner >= 0:
            self.cum_winner_breakdown = self.cum_winner_breakdown + breakdowns[winner]
            self.cum_decided_games += 1
        self.game_len_min = (
            decisions_seen
            if self.game_len_min is None
            else min(self.game_len_min, decisions_seen)
        )
        self.game_len_max = (
            decisions_seen
            if self.game_len_max is None
            else max(self.game_len_max, decisions_seen)
        )
        self.total_games += 1
        self.total_decisions += decisions_seen

    def record_setup_trained(self, n_samples: int) -> None:
        """Fold trained setup samples into the SETUP slot of cum_family."""
        self.cum_family.counts[metrics.SETUP_FAMILY_IDX] += n_samples

    def push_event(self, kind: EventKind, text: str) -> None:
        """Append a recent-events line stamped with elapsed wall time."""
        self.events.append(
            EventLine(clock=_fmt_clock(self.elapsed()), kind=kind, text=text)
        )
        if len(self.events) > _MAX_EVENTS:
            del self.events[: len(self.events) - _MAX_EVENTS]

    # ----- reader-side derivations (called by the dashboard) -----

    def now(self) -> float:
        return (
            self.stopped_monotonic
            if self.stopped_monotonic is not None
            else time.monotonic()
        )

    def elapsed(self) -> float:
        """Total wall time since iteration 0, carrying the offset restored from a
        checkpoint so it accumulates across restarts."""
        return self.elapsed_offset + max(0.0, self.now() - self.start_monotonic)

    def session_elapsed(self) -> float:
        """Wall time of just this process's session (no resumed offset)."""
        return max(0.0, self.now() - self.start_monotonic)

    def iter_elapsed(self) -> float:
        return max(0.0, self.now() - self.iter_start_monotonic)

    def time_remaining_seconds(self) -> float | None:
        """Estimated seconds until ``target_iterations`` is reached.

        During the RANDOM_OPPONENT bootstrap phase the estimate uses half the
        current random-phase iteration rate (self-play is ~2× slower, so we
        assume the transition happens "now" and the rest runs at the slower
        rate). After graduation the estimate tracks the actual self-play rate.
        Returns ``None`` while the rate is still unknown (< 2 iterations
        recorded) and 0.0 once the target is reached.
        """
        if self.target_iterations <= 0:
            return None
        remaining = self.target_iterations - (self.iteration + 1)
        if remaining <= 0:
            return 0.0

        if self.training_phase == TrainingPhase.RANDOM_OPPONENT:
            if self.random_phase_iter_count < 2 or self.random_phase_seconds <= 0:
                return None
            # Assume the rest runs at the (slower) self-play rate.
            rate = (self.random_phase_iter_count / self.random_phase_seconds) / 2.0
        else:
            if self.self_play_iter_count < 2 or self.self_play_seconds <= 0:
                return None
            rate = self.self_play_iter_count / self.self_play_seconds

        return remaining / rate if rate > 0 else None

    def avg_breakdown(self) -> metrics.ScoreBreakdown:
        return self.cum_breakdown.scaled(1.0 / max(self.cum_player_games, 1))

    def avg_total_score(self) -> float:
        return self.avg_breakdown().total

    def avg_decisions(self) -> float:
        return self.cum_decisions / max(self.cum_games, 1)

    def avg_margin(self) -> float:
        return self.cum_margin_sum / max(self.cum_games, 1)

    def margin_std(self) -> float:
        if self.cum_games <= 1:
            return 0.0
        mean = self.avg_margin()
        var = self.cum_margin_sq / self.cum_games - mean * mean
        return var**0.5 if var > 0 else 0.0

    def avg_abs_margin(self) -> float:
        """Mean winning margin |player0 − player1| — by how much the winner of a
        self-play game typically wins (the signed mean is ~0 by symmetry)."""
        return self.cum_abs_margin_sum / max(self.cum_games, 1)

    def abs_margin_std(self) -> float:
        if self.cum_games <= 1:
            return 0.0
        mean = self.avg_abs_margin()
        # |margin|² == margin², so the running sum-of-squares already holds
        # Σ|margin|² and serves the winning margin's variance too.
        var = self.cum_margin_sq / self.cum_games - mean * mean
        return var**0.5 if var > 0 else 0.0

    def eval_ewma(self) -> metrics.EvalEwma | None:
        """EWMA-smoothed eval win-rate and margin against the *current* reference
        opponent.

        Folds the eval blocks in ``history`` with ``config.eval_ewma_alpha`` so
        the dashboard can damp the per-eval sampling noise. Only evals played
        against the current ``opponent_generation`` are folded, so advancing the
        opponent resets the trend to a fresh climb rather than carrying the old
        opponent's saturated win-rate forward. None until the first eval against
        the current opponent lands.
        """
        alpha = self.config.eval_ewma_alpha
        win: float | None = None
        margin: float | None = None
        for item in self.history:
            if (
                item.eval is None
                or item.eval.opponent_generation != self.opponent_generation
            ):
                continue
            if win is None or margin is None:
                win, margin = item.eval.win_rate, item.eval.mean_margin
            else:
                win = alpha * item.eval.win_rate + (1.0 - alpha) * win
                margin = alpha * item.eval.mean_margin + (1.0 - alpha) * margin
        if win is None or margin is None:
            return None
        return metrics.EvalEwma(win_rate=win, mean_margin=margin)

    def collection_win_rate_ewma(self) -> float | None:
        """EWMA-smoothed collection win-rate (vs random) across the bootstrap
        phase's iterations.

        Folds every iteration that recorded a ``collection_win_rate`` (set only
        while in the random-opponent phase) with ``config.eval_ewma_alpha``, so
        the graduation gate and the dashboard read a steadier trend than any one
        noisy 256-game iteration. None until the first such iteration lands.
        """
        alpha = self.config.eval_ewma_alpha
        win: float | None = None
        for item in self.history:
            if item.collection_win_rate is None:
                continue
            if win is None:
                win = item.collection_win_rate
            else:
                win = alpha * item.collection_win_rate + (1.0 - alpha) * win
        return win

    def collection_margin_ewma(self) -> float | None:
        """EWMA-smoothed collection margin (net − random) across the bootstrap
        phase's iterations — the margin twin of :meth:`collection_win_rate_ewma`.

        Folds the ``avg_margin`` of every iteration that recorded a
        ``collection_win_rate`` (so only bootstrap rows count) with
        ``config.eval_ewma_alpha``, so the COLLECT inset can show an EWMA margin
        beside its EWMA win-rate. None until the first such iteration lands.
        """
        alpha = self.config.eval_ewma_alpha
        margin: float | None = None
        for item in self.history:
            if item.collection_win_rate is None:
                continue
            if margin is None:
                margin = item.avg_margin
            else:
                margin = alpha * item.avg_margin + (1.0 - alpha) * margin
        return margin

    def produce_stats(self) -> metrics.ProduceStats | None:
        """The PRODUCING band's readouts: a per-iteration EWMA once at least one
        iteration has finished, otherwise the cumulative average folded over the
        games of the in-progress first iteration (so the panel is live from the
        very first game). None only before any game has been recorded.

        When ``pinned_stats`` is set (at a target milestone) the fixed measurement
        is returned instead of the EWMA so the panel shows the "landed" model's
        clean values rather than the smoothed training history.
        """
        if self.pinned_stats is not None:
            pinned = self.pinned_stats
            return metrics.ProduceStats(
                breakdown=pinned.avg_breakdown,
                winner_breakdown=pinned.avg_winner_breakdown,
                decisions=pinned.decisions_per_game,
                decisions_std=0.0,
                margin=pinned.mean_margin,
                margin_std=0.0,
                abs_margin=pinned.mean_margin,
                abs_margin_std=0.0,
            )
        return self._produce_ewma() or self._produce_cumulative()

    def _produce_ewma(self) -> metrics.ProduceStats | None:
        """EWMA of each iteration's outcome aggregates (None until iteration 1).

        Only iterations from the *current* training phase are folded (a bootstrap
        row is one with a ``collection_win_rate``), so the EWMA restarts fresh at
        the bootstrap → self-play graduation rather than dragging the vs-random
        score / margin character slowly into the self-play trend.

        The dispersion σ values are the EWMA of each iteration's own per-cycle σ
        (computed over that iteration's games), not a σ re-derived from EWMA'd
        moments — the dashboard then divides by √games_per_iter for its 95% CI.
        """
        alpha = self.config.produce_ewma_alpha
        in_random_phase = self.training_phase == TrainingPhase.RANDOM_OPPONENT
        breakdown: metrics.ScoreBreakdown | None = None
        winner: metrics.ScoreBreakdown | None = None
        decisions = decisions_std = margin = margin_std = abs_margin = abs_std = 0.0
        for item in self.history:
            if (item.collection_win_rate is not None) != in_random_phase:
                continue
            if breakdown is None or winner is None:
                breakdown, winner = item.avg_breakdown, item.avg_winner_breakdown
                decisions, decisions_std = item.avg_decisions, item.decisions_std
                margin, margin_std = item.avg_margin, item.margin_std
                abs_margin, abs_std = item.avg_abs_margin, item.abs_margin_std
            else:
                breakdown = _ewma_breakdown(item.avg_breakdown, breakdown, alpha)
                winner = _ewma_breakdown(item.avg_winner_breakdown, winner, alpha)
                decisions = _ewma(item.avg_decisions, decisions, alpha)
                decisions_std = _ewma(item.decisions_std, decisions_std, alpha)
                margin = _ewma(item.avg_margin, margin, alpha)
                margin_std = _ewma(item.margin_std, margin_std, alpha)
                abs_margin = _ewma(item.avg_abs_margin, abs_margin, alpha)
                abs_std = _ewma(item.abs_margin_std, abs_std, alpha)
        if breakdown is None or winner is None:
            return None
        return metrics.ProduceStats(
            breakdown=breakdown,
            winner_breakdown=winner,
            decisions=decisions,
            decisions_std=decisions_std,
            margin=margin,
            margin_std=margin_std,
            abs_margin=abs_margin,
            abs_margin_std=abs_std,
        )

    def _produce_cumulative(self) -> metrics.ProduceStats | None:
        """Since-start average — the fallback shown mid-first-iteration."""
        if self.cum_games == 0:
            return None
        games = max(self.cum_games, 1)
        breakdown = self.cum_breakdown.scaled(1.0 / max(self.cum_player_games, 1))
        winner = self.cum_winner_breakdown.scaled(1.0 / max(self.cum_decided_games, 1))
        decisions = self.cum_decisions / games
        margin = self.cum_margin_sum / games
        abs_margin = self.cum_abs_margin_sum / games
        margin_sq = self.cum_margin_sq / games
        return metrics.ProduceStats(
            breakdown=breakdown,
            winner_breakdown=winner,
            decisions=decisions,
            decisions_std=_std(self.cum_decisions_sq / games, decisions),
            margin=margin,
            margin_std=_std(margin_sq, margin),
            abs_margin=abs_margin,
            # |margin|² == margin², so the same second moment serves both.
            abs_margin_std=_std(margin_sq, abs_margin),
        )

    # ----- checkpoint resume -----

    def to_progress(self) -> RunProgress:
        """Snapshot the resumable counters, aggregates, and charts for a
        checkpoint (the transient clocks, phase, and telemetry are not saved)."""
        return RunProgress(
            iteration=self.iteration,
            total_games=self.total_games,
            total_decisions=self.total_decisions,
            elapsed_seconds=self.elapsed(),
            events=list(self.events),
            cum_breakdown=self.cum_breakdown,
            cum_player_games=self.cum_player_games,
            cum_games=self.cum_games,
            cum_decisions=self.cum_decisions,
            cum_decisions_sq=self.cum_decisions_sq,
            cum_family=self.cum_family,
            cum_margin_sum=self.cum_margin_sum,
            cum_margin_sq=self.cum_margin_sq,
            cum_abs_margin_sum=self.cum_abs_margin_sum,
            cum_winner_breakdown=self.cum_winner_breakdown,
            cum_decided_games=self.cum_decided_games,
            game_len_min=self.game_len_min,
            game_len_max=self.game_len_max,
            best_win_rate=self.best_win_rate,
            opponent_generation=self.opponent_generation,
            opponent_since_iteration=self.opponent_since_iteration,
            opponent_change_iterations=list(self.opponent_change_iterations),
            training_phase=self.training_phase,
            last_iter=self.last_iter,
            history=self.history,
            target_iterations=self.target_iterations,
            random_phase_iter_count=self.random_phase_iter_count,
            random_phase_seconds=self.random_phase_seconds,
            self_play_iter_count=self.self_play_iter_count,
            self_play_seconds=self.self_play_seconds,
        )

    def restore_progress(self, progress: RunProgress) -> None:
        """Load a checkpoint's :class:`RunProgress` back into the live state, so
        the dashboard reopens with the prior counts, averages, and charts."""
        self.iteration = progress.iteration
        self.total_games = progress.total_games
        self.total_decisions = progress.total_decisions
        self.elapsed_offset = progress.elapsed_seconds
        self.events = list(progress.events)
        self.cum_breakdown = progress.cum_breakdown
        self.cum_player_games = progress.cum_player_games
        self.cum_games = progress.cum_games
        self.cum_decisions = progress.cum_decisions
        self.cum_decisions_sq = progress.cum_decisions_sq
        self.cum_family = progress.cum_family
        self.cum_margin_sum = progress.cum_margin_sum
        self.cum_margin_sq = progress.cum_margin_sq
        self.cum_abs_margin_sum = progress.cum_abs_margin_sum
        self.cum_winner_breakdown = progress.cum_winner_breakdown
        self.cum_decided_games = progress.cum_decided_games
        self.game_len_min = progress.game_len_min
        self.game_len_max = progress.game_len_max
        self.best_win_rate = progress.best_win_rate
        self.opponent_generation = progress.opponent_generation
        self.opponent_since_iteration = progress.opponent_since_iteration
        self.opponent_change_iterations = list(progress.opponent_change_iterations)
        self.training_phase = progress.training_phase
        self.last_iter = progress.last_iter
        self.history = list(progress.history)
        self.target_iterations = progress.target_iterations
        self.random_phase_iter_count = progress.random_phase_iter_count
        self.random_phase_seconds = progress.random_phase_seconds
        self.self_play_iter_count = progress.self_play_iter_count
        self.self_play_seconds = progress.self_play_seconds


def new_run_state(cfg: config.TrainConfig) -> RunState:
    """Build a fresh ``RunState`` with both clocks anchored at 'now'."""
    anchor = time.monotonic()
    return RunState(
        config=cfg,
        start_monotonic=anchor,
        iter_start_monotonic=anchor,
        games_in_iter=cfg.games_per_iter,
    )


def _fmt_clock(seconds: float) -> str:
    """``H:MM:SS`` elapsed-time string."""
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}"


def _ewma(new: float, prev: float, alpha: float) -> float:
    """One exponentially-weighted moving-average step."""
    return alpha * new + (1.0 - alpha) * prev


def _ewma_breakdown(
    new: metrics.ScoreBreakdown, prev: metrics.ScoreBreakdown, alpha: float
) -> metrics.ScoreBreakdown:
    """Per-source EWMA step over a whole score breakdown."""
    return new.scaled(alpha) + prev.scaled(1.0 - alpha)


def _std(mean_sq: float, mean: float) -> float:
    """Standard deviation from a mean-of-squares and a mean (clamped at 0)."""
    var = mean_sq - mean * mean
    return var**0.5 if var > 0 else 0.0
