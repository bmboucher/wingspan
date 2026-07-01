"""Pure data models for the tournament package.

All Pydantic models and enums that describe tournament data shapes live here so
every other tournament module can import from a single, dependency-free source.
This module depends only on ``pydantic`` and the standard library — no torch,
no engine, no training code — so it can be imported anywhere without pulling in
heavy dependencies or creating circular imports.

Public names (grouped by concern):

*Competitors*
  :class:`ParticipantKind`, :class:`ParticipantSpec`, :class:`RunOption`

*Schedule*
  :class:`Orientation`, :class:`GameTask`

*Configuration*
  :data:`DEFAULT_ELO_INIT`, :data:`DEFAULT_ELO_K`, :data:`DEFAULT_GAMES_PER_PAIR`,
  :class:`TournamentConfig`, :class:`RegimeFlags`

*Results*
  :class:`GameResult`, :class:`SplitStats`, :class:`MatchupResult`,
  :class:`ParticipantResult`, :class:`TournamentReport`

*Ratings*
  :class:`EloTable`

*Live state*
  :class:`TournamentPhase`, :class:`LiveRecord`, :class:`StandingRow`
"""

from __future__ import annotations

import enum
import math
import typing

import pydantic

# ---------------------------------------------------------------------------
# Competitors


class ParticipantKind(enum.StrEnum):
    """Whether a competitor is a trained model or the random agent."""

    MODEL = "model"
    RANDOM = "random"


class ParticipantSpec(pydantic.BaseModel):
    """One tournament competitor, identified by a stable ``id``.

    ``checkpoint_dir`` is the run directory for a ``MODEL`` and ``None`` for the
    ``RANDOM`` agent. Frozen so a spec can be a dict key and shipped to worker
    processes as immutable pool ``initargs``.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    id: str
    display_name: str
    kind: ParticipantKind
    checkpoint_dir: str | None = None


class RunOption(pydantic.BaseModel):
    """One discoverable trained run, as shown to the user in the picker."""

    checkpoint_dir: str
    display_name: str
    iteration: int | None = None
    best_win_rate: float | None = None
    modified: float | None = None

    def to_spec(self) -> ParticipantSpec:
        """The competitor spec for selecting this run as a model player."""
        return ParticipantSpec(
            id=self.display_name,
            display_name=self.display_name,
            kind=ParticipantKind.MODEL,
            checkpoint_dir=self.checkpoint_dir,
        )


# ---------------------------------------------------------------------------
# Schedule


class Orientation(enum.IntEnum):
    """Which board seat competitor A occupies for a game (the mirror swaps it).

    Across a deal's two orientations each competitor is the start player exactly
    once, regardless of the deal's randomly chosen first player.
    """

    A_SEAT_0 = 0
    A_SEAT_1 = 1


class GameTask(pydantic.BaseModel):
    """One scheduled game: a deal seed, the two competitors, and A's seat.

    Frozen so it ships to worker processes as immutable work. ``a_seat`` is
    derived from ``orientation`` (they are the same value) so the seat that
    competitor A takes is always consistent with the orientation label.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    round_index: int
    pair_index: int
    deal_seed: int
    orientation: Orientation
    player_a_id: str
    player_b_id: str

    @property
    def a_seat(self) -> int:
        """The board seat (0 or 1) competitor A occupies in this game."""
        return int(self.orientation)


# ---------------------------------------------------------------------------
# Configuration

# Classic Elo defaults: every competitor starts at 1500 and a game moves the
# pair by at most ``elo_k`` rating points.
DEFAULT_ELO_INIT = 1500.0
DEFAULT_ELO_K = 24.0
DEFAULT_GAMES_PER_PAIR = 32


class TournamentConfig(pydantic.BaseModel):
    """A round-robin tournament's competitors and parameters.

    ``games_per_pair`` must be even: every unordered pair plays it as
    ``games_per_pair / 2`` mirrored deals (each deal played from both seat
    orderings), which gives each competitor the start-player seat an equal number
    of times.
    """

    participants: typing.Annotated[list[ParticipantSpec], pydantic.Field(min_length=2)]
    games_per_pair: typing.Annotated[int, pydantic.Field(ge=2)] = DEFAULT_GAMES_PER_PAIR
    elo_k: float = DEFAULT_ELO_K
    elo_init: float = DEFAULT_ELO_INIT
    base_seed: int = 0
    out_path: str = "tournament_report.json"
    device: str = "cpu"

    @pydantic.model_validator(mode="after")
    def _games_per_pair_even(self) -> "TournamentConfig":
        if self.games_per_pair % 2 != 0:
            raise ValueError("games_per_pair must be even (games are mirrored deals)")
        return self

    @property
    def participant_ids(self) -> list[str]:
        """Every competitor's id, in configured order."""
        return [spec.id for spec in self.participants]

    @property
    def n_pairs(self) -> int:
        """The number of unordered competitor pairs in the round-robin."""
        count = len(self.participants)
        return math.comb(count, 2)

    @property
    def total_games(self) -> int:
        """Total games the tournament will play across every pair."""
        return self.n_pairs * self.games_per_pair


class RegimeFlags(pydantic.BaseModel):
    """The setup/food engine regimes every game in a tournament runs under.

    Resolved once from the competitors' training configs (see
    ``participants.resolve_regime_flags``) so each game mirrors how the nets
    were trained — the tournament-wide analogue of the per-matchup resolution
    ``wingspan play`` performs. Frozen so it ships to worker processes as
    immutable pool ``initargs``. All-``False`` (the default) is the engine's own
    default and the resolution for a config-free (random-only) field.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    split_setup_bonus: bool = False
    split_setup_food: bool = False
    combine_gain_food: bool = False


# ---------------------------------------------------------------------------
# Results


class GameResult(pydantic.BaseModel):
    """One finished tournament game, from competitor A's point of view.

    ``a_was_start_player`` records who actually went first (the deal's random
    first player mapped through A's seat), which is what the first/second splits
    bucket on — not the orientation label.
    """

    round_index: int
    pair_index: int
    orientation: Orientation
    player_a_id: str
    player_b_id: str
    a_score: int
    b_score: int
    a_was_start_player: bool

    @property
    def score_a(self) -> float:
        """A's game outcome as an Elo/​win-rate score: 1.0 win, 0.5 tie, 0.0 loss."""
        if self.a_score > self.b_score:
            return 1.0
        if self.a_score < self.b_score:
            return 0.0
        return 0.5


class SplitStats(pydantic.BaseModel):
    """Win rate and average point margin over one subset of a pair's games, from
    a fixed competitor's perspective (ties count as half a win)."""

    games: int = 0
    wins: float = 0.0
    win_rate: float = 0.0
    avg_margin: float = 0.0


class MatchupResult(pydantic.BaseModel):
    """One unordered pair's full result, broken down from each side as first
    player, as second player, and overall."""

    player_a_id: str
    player_b_id: str
    a_first: SplitStats
    a_second: SplitStats
    a_overall: SplitStats
    b_first: SplitStats
    b_second: SplitStats
    b_overall: SplitStats


class ParticipantResult(pydantic.BaseModel):
    """One competitor's tournament-wide standing."""

    id: str
    display_name: str
    final_elo: float
    wins: int
    losses: int
    ties: int
    win_rate: float
    avg_margin: float


class TournamentReport(pydantic.BaseModel):
    """The complete tournament result — the JSON written to ``out_path``."""

    config: TournamentConfig
    participants: list[ParticipantResult]
    matchups: list[MatchupResult]
    games: list[GameResult]


# ---------------------------------------------------------------------------
# Ratings

# The Elo logistic scale: a 400-point gap is a 10:1 expected-score ratio.
_ELO_SCALE = 400.0


class EloTable(pydantic.BaseModel):
    """Mutable Elo ratings keyed by competitor id, with a shared K-factor."""

    ratings: dict[str, float]
    k: float

    @classmethod
    def initial(cls, ids: typing.Sequence[str], init: float, k: float) -> "EloTable":
        """A table with every id seeded at the ``init`` rating."""
        return cls(ratings={competitor_id: init for competitor_id in ids}, k=k)

    def expected(self, a: str, b: str) -> float:
        """``a``'s expected score against ``b`` — the logistic of the gap, in
        ``(0, 1)``. ``expected(a, b) + expected(b, a) == 1``."""
        gap = self.ratings[b] - self.ratings[a]
        return 1.0 / (1.0 + 10.0 ** (gap / _ELO_SCALE))

    def update(self, a: str, b: str, score_a: float) -> None:
        """Apply one game's result (``score_a`` is 1.0 / 0.5 / 0.0 for an a-win /
        draw / a-loss). The pair moves by equal and opposite amounts."""
        expected_a = self.expected(a, b)
        delta = self.k * (score_a - expected_a)
        self.ratings[a] += delta
        self.ratings[b] -= delta


# ---------------------------------------------------------------------------
# Live state


class TournamentPhase(enum.StrEnum):
    """What the tournament is doing (drives the header LED + the exit check)."""

    RUNNING = "running"
    DONE = "done"
    STOPPED = "stopped"
    ERROR = "error"

    @property
    def is_terminal(self) -> bool:
        return self in (
            TournamentPhase.DONE,
            TournamentPhase.STOPPED,
            TournamentPhase.ERROR,
        )


class LiveRecord(pydantic.BaseModel):
    """One competitor's running win/loss/tie record and margin total."""

    games: int = 0
    wins: int = 0
    losses: int = 0
    ties: int = 0
    margin_sum: float = 0.0

    @property
    def win_rate(self) -> float:
        return (self.wins + 0.5 * self.ties) / self.games if self.games else 0.0

    @property
    def avg_margin(self) -> float:
        return self.margin_sum / self.games if self.games else 0.0


class StandingRow(pydantic.BaseModel):
    """One row of the live standings table (already sorted-by-Elo by the caller)."""

    id: str
    display_name: str
    elo: float
    games: int
    wins: int
    losses: int
    ties: int
    win_rate: float
    avg_margin: float
    elo_spark: list[float]
