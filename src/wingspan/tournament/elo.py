"""Sequential Elo ratings for tournament competitors.

:func:`replay` applies the classic Elo update over a *fixed* game order so the
saved ratings are reproducible regardless of the order games happened to finish
in across the worker pool. The :class:`~models.EloTable` data shape lives in
:mod:`models`; this module provides the :func:`replay` function.
"""

from __future__ import annotations

import typing

from wingspan.tournament import models


def replay(
    ids: typing.Sequence[str],
    init: float,
    k: float,
    game_results: typing.Sequence[models.GameResult],
) -> models.EloTable:
    """Final Elo from a fresh table over ``game_results`` applied in a fixed
    order (by round, then pair, then orientation), so the result is deterministic
    regardless of the order the games completed in."""
    table = models.EloTable.initial(ids, init, k)
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
