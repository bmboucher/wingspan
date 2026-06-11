"""The round-robin game schedule.

Every unordered pair of competitors plays ``games_per_pair`` games as
``games_per_pair / 2`` *mirrored deals*: each deal seed is played twice, once
with competitor A in board seat 0 and once in seat 1. Because a deal's
randomly-chosen first player is fixed by its seed, swapping the seats makes each
competitor the start player in exactly one of the two games — so over the whole
schedule each competitor goes first an equal number of times, and the
first-player / deal advantage cancels within every pair (the variance-reduction
trick :func:`evaluate.play_eval_game` uses).

Games are emitted round-interleaved (every pair plays deal 0, then every pair
plays deal 1, …) so the live standings fill in evenly rather than one matchup
finishing before the next begins.

Data shapes (:class:`~models.Orientation`, :class:`~models.GameTask`) live in
:mod:`models`; this module provides :func:`build_schedule`.
"""

from __future__ import annotations

import itertools
import typing

from wingspan.tournament import models

# Deal-seed strides: large, coprime-ish multipliers that keep the per-pair,
# per-deal seeds from colliding across the schedule while staying a pure function
# of (base_seed, pair, deal) so the whole schedule is reproducible.
_BASE_SEED_STRIDE = 1_000_000_007
_PAIR_SEED_STRIDE = 1_000_003
_DEAL_SEED_STRIDE = 101


def build_schedule(
    competitors: typing.Sequence[models.ParticipantSpec],
    games_per_pair: int,
    base_seed: int,
) -> list[models.GameTask]:
    """Every game of the round-robin, round-interleaved. ``games_per_pair`` must
    be even; each pair plays ``games_per_pair // 2`` mirrored deals."""
    ids = sorted(spec.id for spec in competitors)
    pairs = list(itertools.combinations(ids, 2))
    n_deals = games_per_pair // 2

    tasks: list[models.GameTask] = []
    for round_index in range(n_deals):
        for pair_index, (id_a, id_b) in enumerate(pairs):
            deal_seed = _deal_seed(base_seed, pair_index, round_index)
            for orientation in (
                models.Orientation.A_SEAT_0,
                models.Orientation.A_SEAT_1,
            ):
                tasks.append(
                    models.GameTask(
                        round_index=round_index,
                        pair_index=pair_index,
                        deal_seed=deal_seed,
                        orientation=orientation,
                        player_a_id=id_a,
                        player_b_id=id_b,
                    )
                )
    return tasks


###### PRIVATE #######


def _deal_seed(base_seed: int, pair_index: int, round_index: int) -> int:
    """A deterministic, well-separated deal seed for one mirrored deal."""
    return (
        base_seed * _BASE_SEED_STRIDE
        + pair_index * _PAIR_SEED_STRIDE
        + round_index * _DEAL_SEED_STRIDE
    )
