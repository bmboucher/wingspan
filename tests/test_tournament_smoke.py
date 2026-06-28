"""End-to-end tournament smoke test.

Runs a tiny round-robin between random-only competitors via the in-process
driver (no worker pool, no model checkpoints), confirming the schedule plays
out, the report shape is right, and the mirror gives every competitor the
start-player seat an equal number of times.
"""

from __future__ import annotations

import pathlib

from wingspan.tournament import models, runner


def _random_cfg(ids: list[str], out_path: str) -> models.TournamentConfig:
    specs = [
        models.ParticipantSpec(
            id=competitor_id,
            display_name=competitor_id,
            kind=models.ParticipantKind.RANDOM,
        )
        for competitor_id in ids
    ]
    return models.TournamentConfig(
        participants=specs, games_per_pair=4, base_seed=0, out_path=out_path
    )


def test_in_process_tournament_plays_and_reports(tmp_path: pathlib.Path) -> None:
    ids = ["r0", "r1", "r2"]
    cfg = _random_cfg(ids, str(tmp_path / "report.json"))
    streamed: list[models.GameResult] = []

    report = runner.run_tournament(cfg, on_result=streamed.append, in_process=True)

    total = cfg.total_games  # C(3, 2) * 4 = 12
    assert len(streamed) == total
    assert len(report.games) == total
    assert {participant.id for participant in report.participants} == set(ids)
    assert len(report.matchups) == 3

    # Each competitor plays both opponents over 4 games each = 8 games.
    for participant in report.participants:
        assert participant.wins + participant.losses + participant.ties == 8


def test_mirror_gives_equal_first_player_counts(tmp_path: pathlib.Path) -> None:
    ids = ["r0", "r1", "r2"]
    cfg = _random_cfg(ids, str(tmp_path / "report.json"))
    report = runner.run_tournament(cfg, in_process=True)

    first_counts = {competitor_id: 0 for competitor_id in ids}
    for game in report.games:
        starter = game.player_a_id if game.a_was_start_player else game.player_b_id
        first_counts[starter] += 1

    # 8 games each, first in exactly half of them (one per mirrored deal).
    assert all(count == 4 for count in first_counts.values())


def test_report_serializes_to_json(tmp_path: pathlib.Path) -> None:
    out = tmp_path / "report.json"
    cfg = _random_cfg(["r0", "r1"], str(out))
    report = runner.run_tournament(cfg, in_process=True)
    out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    restored = models.TournamentReport.model_validate_json(
        out.read_text(encoding="utf-8")
    )
    assert len(restored.games) == cfg.total_games
