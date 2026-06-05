"""Sequential Elo ratings for tournament competitors.

:class:`EloTable` is the classic Elo update: each competitor holds a rating, a
game shifts the pair toward the loser by ``k * (actual - expected)``, and the
expected score is the logistic of the rating gap. The dashboard updates a table
live as games complete (so the standings move "as it progresses"); the report's
final ratings come from :func:`replay`, which applies the same updates over a
*fixed* game order, so the saved numbers are reproducible regardless of the
order games happened to finish in across the worker pool.
"""

from __future__ import annotations

import typing

import pydantic

if typing.TYPE_CHECKING:
    from wingspan.tournament import results

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


def replay(
    ids: typing.Sequence[str],
    init: float,
    k: float,
    game_results: "typing.Sequence[results.GameResult]",
) -> EloTable:
    """Final Elo from a fresh table over ``game_results`` applied in a fixed
    order (by round, then pair, then orientation), so the result is deterministic
    regardless of the order the games completed in."""
    table = EloTable.initial(ids, init, k)
    ordered = sorted(
        game_results,
        key=lambda result: (
            result.round_index,
            result.pair_index,
            int(result.orientation),
        ),
    )
    for result in ordered:
        table.update(result.player_a_id, result.player_b_id, result.score_a)
    return table
