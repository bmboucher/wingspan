"""Tests for the round-robin schedule.

Checks the game count, the equal-seat invariant that gives each competitor the
start-player seat an equal number of times, and that a deal's two mirrored
orientations share a seed (so model-vs-model games are true mirrors).
"""

from __future__ import annotations

import collections

from wingspan.tournament import models, schedule


def _random_specs(count: int) -> list[models.ParticipantSpec]:
    return [
        models.ParticipantSpec(
            id=f"p{index}",
            display_name=f"p{index}",
            kind=models.ParticipantKind.RANDOM,
        )
        for index in range(count)
    ]


def test_schedule_total_game_count() -> None:
    tasks = schedule.build_schedule(_random_specs(4), games_per_pair=6, base_seed=0)
    n_pairs = 6  # C(4, 2)
    assert len(tasks) == n_pairs * 6


def test_each_pair_balanced_across_seats() -> None:
    tasks = schedule.build_schedule(_random_specs(3), games_per_pair=8, base_seed=1)
    per_pair: dict[int, collections.Counter[models.Orientation]] = {}
    for task in tasks:
        per_pair.setdefault(task.pair_index, collections.Counter())[
            task.orientation
        ] += 1
    assert len(per_pair) == 3  # C(3, 2)
    for counter in per_pair.values():
        assert counter[models.Orientation.A_SEAT_0] == 4  # games_per_pair // 2
        assert counter[models.Orientation.A_SEAT_1] == 4


def test_mirrored_orientations_share_deal_seed() -> None:
    tasks = schedule.build_schedule(_random_specs(2), games_per_pair=4, base_seed=2)
    by_deal: dict[tuple[int, int], list[models.GameTask]] = {}
    for task in tasks:
        by_deal.setdefault((task.pair_index, task.round_index), []).append(task)
    for group in by_deal.values():
        assert len(group) == 2
        assert group[0].deal_seed == group[1].deal_seed
        assert {task.orientation for task in group} == {
            models.Orientation.A_SEAT_0,
            models.Orientation.A_SEAT_1,
        }


def test_a_seat_tracks_orientation() -> None:
    for task in schedule.build_schedule(
        _random_specs(3), games_per_pair=2, base_seed=0
    ):
        assert task.a_seat == int(task.orientation)
