"""Tests for result aggregation.

Checks tie handling (half a win), the first/second/overall splits, the
per-competitor record + Elo ordering, and that the report round-trips through
JSON unchanged.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from wingspan.tournament import models, results


def _cfg(ids: list[str]) -> models.TournamentConfig:
    specs = [
        models.ParticipantSpec(
            id=competitor_id,
            display_name=competitor_id,
            kind=models.ParticipantKind.RANDOM,
        )
        for competitor_id in ids
    ]
    return models.TournamentConfig(participants=specs, games_per_pair=2, base_seed=0)


def _two_game_pair() -> list[models.GameResult]:
    """Pair (a, b): a goes first and wins by 5; b goes first and the game ties."""
    return [
        models.GameResult(
            round_index=0,
            pair_index=0,
            orientation=models.Orientation.A_SEAT_0,
            player_a_id="a",
            player_b_id="b",
            a_score=20,
            b_score=15,
            a_was_start_player=True,
        ),
        models.GameResult(
            round_index=0,
            pair_index=0,
            orientation=models.Orientation.A_SEAT_1,
            player_a_id="a",
            player_b_id="b",
            a_score=10,
            b_score=10,
            a_was_start_player=False,
        ),
    ]


def test_matchup_splits_bucket_by_who_went_first() -> None:
    report = results.aggregate(_cfg(["a", "b"]), _two_game_pair())
    assert len(report.matchups) == 1
    matchup = report.matchups[0]

    assert matchup.a_first.games == 1
    assert matchup.a_first.wins == 1.0
    assert matchup.a_first.win_rate == 1.0
    assert matchup.a_first.avg_margin == 5.0

    assert matchup.a_second.games == 1
    assert matchup.a_second.wins == 0.5  # the tie counts as half a win
    assert matchup.a_second.avg_margin == 0.0

    assert matchup.a_overall.games == 2
    assert matchup.a_overall.wins == 1.5

    # b went first in the tie game and second in the game it lost.
    assert matchup.b_first.games == 1
    assert matchup.b_first.wins == 0.5
    assert matchup.b_second.wins == 0.0
    assert matchup.b_second.avg_margin == -5.0


def test_participant_records_and_elo_ordering() -> None:
    report = results.aggregate(_cfg(["a", "b"]), _two_game_pair())
    by_id = {participant.id: participant for participant in report.participants}

    assert (by_id["a"].wins, by_id["a"].ties, by_id["a"].losses) == (1, 1, 0)
    assert by_id["a"].win_rate == 0.75
    assert (by_id["b"].wins, by_id["b"].ties, by_id["b"].losses) == (0, 1, 1)
    # Sorted by final Elo descending — the net winner leads.
    assert report.participants[0].id == "a"
    assert by_id["a"].final_elo > by_id["b"].final_elo


def test_report_round_trips_through_json() -> None:
    report = results.aggregate(_cfg(["a", "b"]), _two_game_pair())
    restored = models.TournamentReport.model_validate_json(report.model_dump_json())
    assert restored.model_dump_json() == report.model_dump_json()
