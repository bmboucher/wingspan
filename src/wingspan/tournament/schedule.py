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
"""

from __future__ import annotations

import enum
import itertools
import typing

import pydantic

from wingspan.tournament import participants

# Deal-seed strides: large, coprime-ish multipliers that keep the per-pair,
# per-deal seeds from colliding across the schedule while staying a pure function
# of (base_seed, pair, deal) so the whole schedule is reproducible.
_BASE_SEED_STRIDE = 1_000_000_007
_PAIR_SEED_STRIDE = 1_000_003
_DEAL_SEED_STRIDE = 101


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


def build_schedule(
    competitors: typing.Sequence[participants.ParticipantSpec],
    games_per_pair: int,
    base_seed: int,
) -> list[GameTask]:
    """Every game of the round-robin, round-interleaved. ``games_per_pair`` must
    be even; each pair plays ``games_per_pair // 2`` mirrored deals."""
    ids = sorted(spec.id for spec in competitors)
    pairs = list(itertools.combinations(ids, 2))
    n_deals = games_per_pair // 2

    tasks: list[GameTask] = []
    for round_index in range(n_deals):
        for pair_index, (id_a, id_b) in enumerate(pairs):
            deal_seed = _deal_seed(base_seed, pair_index, round_index)
            for orientation in (Orientation.A_SEAT_0, Orientation.A_SEAT_1):
                tasks.append(
                    GameTask(
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
