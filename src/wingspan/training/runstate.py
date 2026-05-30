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
    DONE = "done"
    STOPPED = "stopped"
    ERROR = "error"

    @property
    def is_terminal(self) -> bool:
        return self in (Phase.DONE, Phase.STOPPED, Phase.ERROR)


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


class RunState(pydantic.BaseModel):
    """Everything the dashboard needs to repaint a single frame."""

    config: config.TrainConfig
    phase: Phase = Phase.STARTING

    # Monotonic wall-clock anchors (seconds). ``stopped_monotonic`` freezes the
    # clocks once the run ends so a finished dashboard stops ticking.
    start_monotonic: float
    iter_start_monotonic: float
    stopped_monotonic: float | None = None

    # Live counters.
    iteration: int = 0
    game_in_iter: int = 0
    games_in_iter: int = 0
    total_games: int = 0
    total_decisions: int = 0
    games_per_sec: float = 0.0

    # Cumulative running aggregates (since the run started).
    cum_breakdown: metrics.ScoreBreakdown = pydantic.Field(
        default_factory=metrics.ScoreBreakdown
    )
    cum_player_games: int = 0  # = 2 * cum_games (one breakdown per seat)
    cum_games: int = 0
    cum_decisions: int = 0
    cum_family: metrics.FamilyCounts = pydantic.Field(
        default_factory=metrics.FamilyCounts
    )
    cum_margin_sum: float = 0.0  # Σ (player0 − player1)
    cum_margin_sq: float = 0.0  # Σ (player0 − player1)^2
    game_len_min: int | None = None
    game_len_max: int | None = None

    # Latest readouts + history (capped to config.history_len).
    last_iter: metrics.IterationMetrics | None = None
    history: list[metrics.IterationMetrics] = pydantic.Field(
        default_factory=_new_history
    )
    best_win_rate: float | None = None

    events: list[EventLine] = pydantic.Field(default_factory=_new_events)
    error: str | None = None

    # ----- writer-side helpers (called by the loop, under its lock) -----

    def record_game(
        self,
        breakdowns: tuple[metrics.ScoreBreakdown, metrics.ScoreBreakdown],
        decisions_seen: int,
        family: metrics.FamilyCounts,
    ) -> None:
        """Fold one finished game into the cumulative aggregates."""
        self.cum_breakdown = self.cum_breakdown + breakdowns[0] + breakdowns[1]
        self.cum_player_games += 2
        self.cum_games += 1
        self.cum_decisions += decisions_seen
        self.cum_family = self.cum_family + family
        margin = breakdowns[0].total - breakdowns[1].total
        self.cum_margin_sum += margin
        self.cum_margin_sq += margin * margin
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
        return max(0.0, self.now() - self.start_monotonic)

    def iter_elapsed(self) -> float:
        return max(0.0, self.now() - self.iter_start_monotonic)

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
