"""The tournament's run configuration — the competitors and the knobs.

``TournamentConfig`` is the single self-describing record the runner, the live
dashboard, and the JSON report all read. It is embedded verbatim in the report
so a result file records exactly the settings that produced it.
"""

from __future__ import annotations

import math
import typing

import pydantic

from wingspan.tournament import participants

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

    participants: typing.Annotated[
        list[participants.ParticipantSpec], pydantic.Field(min_length=2)
    ]
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
