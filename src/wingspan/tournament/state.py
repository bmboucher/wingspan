"""The live, shared state the runner writes and the dashboard reads.

``TournamentState`` is the single mutable object the worker thread updates as
games complete (live Elo, per-competitor records, a recent-events ring) and the
render thread reads — under the app's lock — to repaint. The live Elo here moves
in game-completion order for the "as it progresses" feel; the report's final Elo
is recomputed deterministically by :func:`results.aggregate`.
"""

from __future__ import annotations

import time

import pydantic

from wingspan.tournament import models
from wingspan.training import runstate

# Recent-events ring size, and how many trailing live-Elo points to keep per
# competitor for the standings sparkline.
_MAX_EVENTS = 40
_ELO_HISTORY_CAP = 240


class TournamentState(pydantic.BaseModel):
    """Everything the dashboard needs to repaint one frame."""

    config: models.TournamentConfig
    phase: models.TournamentPhase = models.TournamentPhase.RUNNING
    start_monotonic: float
    stopped_monotonic: float | None = None

    total_games: int
    games_done: int = 0

    live_table: models.EloTable
    records: dict[str, models.LiveRecord]
    elo_history: dict[str, list[float]]
    events: list[runstate.EventLine] = pydantic.Field(
        default_factory=list[runstate.EventLine]
    )
    error: str | None = None

    # ----- writer-side helpers (called by the runner callback, under the lock) -----

    def record_game(self, result: models.GameResult) -> None:
        """Fold one finished game into the live Elo, records, and sparkline."""
        self.games_done += 1
        self.live_table.update(result.player_a_id, result.player_b_id, result.score_a)
        margin_a = float(result.a_score - result.b_score)
        self._record_side(result.player_a_id, result.score_a, margin_a)
        self._record_side(result.player_b_id, 1.0 - result.score_a, -margin_a)
        for competitor_id, rating in self.live_table.ratings.items():
            history = self.elo_history[competitor_id]
            history.append(rating)
            if len(history) > _ELO_HISTORY_CAP:
                del history[: len(history) - _ELO_HISTORY_CAP]

    def push_event(self, kind: runstate.EventKind, text: str) -> None:
        """Append a recent-events line stamped with elapsed wall time."""
        self.events.append(
            runstate.EventLine(clock=_fmt_clock(self.elapsed()), kind=kind, text=text)
        )
        if len(self.events) > _MAX_EVENTS:
            del self.events[: len(self.events) - _MAX_EVENTS]

    def finish(self, phase: models.TournamentPhase) -> None:
        """Freeze the clock and mark the terminal phase."""
        self.phase = phase
        self.stopped_monotonic = time.monotonic()

    # ----- reader-side derivations (called by the dashboard) -----

    def now(self) -> float:
        return (
            self.stopped_monotonic
            if self.stopped_monotonic is not None
            else time.monotonic()
        )

    def elapsed(self) -> float:
        return max(0.0, self.now() - self.start_monotonic)

    def throughput(self) -> float:
        """Finished games per second since the tournament started."""
        elapsed = self.elapsed()
        return self.games_done / elapsed if elapsed > 0 else 0.0

    def progress(self) -> float:
        """Fraction of the scheduled games that have finished, in ``[0, 1]``."""
        return self.games_done / self.total_games if self.total_games else 0.0

    def standings(self) -> list[models.StandingRow]:
        """The competitor rows, sorted by live Elo (descending)."""
        display = {spec.id: spec.display_name for spec in self.config.participants}
        rows = [
            models.StandingRow(
                id=competitor_id,
                display_name=display[competitor_id],
                elo=self.live_table.ratings[competitor_id],
                games=record.games,
                wins=record.wins,
                losses=record.losses,
                ties=record.ties,
                win_rate=record.win_rate,
                avg_margin=record.avg_margin,
                elo_spark=list(self.elo_history[competitor_id]),
            )
            for competitor_id, record in self.records.items()
        ]
        rows.sort(key=lambda row: row.elo, reverse=True)
        return rows

    ###### PRIVATE #######

    def _record_side(self, competitor_id: str, score: float, margin: float) -> None:
        record = self.records[competitor_id]
        record.games += 1
        record.margin_sum += margin
        if score == 1.0:
            record.wins += 1
        elif score == 0.0:
            record.losses += 1
        else:
            record.ties += 1


def new_tournament_state(cfg: models.TournamentConfig) -> TournamentState:
    """A fresh live state with every competitor seeded at the initial Elo."""
    ids = cfg.participant_ids
    return TournamentState(
        config=cfg,
        start_monotonic=time.monotonic(),
        total_games=cfg.total_games,
        live_table=models.EloTable.initial(ids, cfg.elo_init, cfg.elo_k),
        records={competitor_id: models.LiveRecord() for competitor_id in ids},
        elo_history={competitor_id: [cfg.elo_init] for competitor_id in ids},
    )


def _fmt_clock(seconds: float) -> str:
    """``H:MM:SS`` elapsed-time string."""
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}"
