"""End-to-end tournament smoke test.

Runs a tiny round-robin between random-only competitors via the in-process
driver (no worker pool, no model checkpoints), confirming the schedule plays
out, the report shape is right, and the mirror gives every competitor the
start-player seat an equal number of times.
"""

from __future__ import annotations

import json
import pathlib

from wingspan.tournament import models, runner

# The flat log's shard helpers are private to the runner, but they are the part
# of that wiring with something to get wrong and a real pool costs seconds to
# spawn — so the tests below drive them directly.
_clear_shards = runner._clear_shards  # pyright: ignore[reportPrivateUsage]
_merge_shards = runner._merge_shards  # pyright: ignore[reportPrivateUsage]
_shard_path = runner._shard_path  # pyright: ignore[reportPrivateUsage]


def _random_cfg(
    ids: list[str], out_path: str, jsonl_path: str | None = None
) -> models.TournamentConfig:
    specs = [
        models.ParticipantSpec(
            id=competitor_id,
            display_name=competitor_id,
            kind=models.ParticipantKind.RANDOM,
        )
        for competitor_id in ids
    ]
    return models.TournamentConfig(
        participants=specs,
        games_per_pair=4,
        base_seed=0,
        out_path=out_path,
        jsonl_path=jsonl_path,
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


def test_jsonl_path_logs_every_game_once(tmp_path: pathlib.Path) -> None:
    """``jsonl_path`` collects the whole round-robin into one flat log.

    Each game contributes exactly one header row, and ``game_id`` is built from
    the schedule coordinates so a mirrored deal's two orientations — same seed,
    different seats — stay distinct."""
    jsonl_path = tmp_path / "games.jsonl"
    cfg = _random_cfg(["r0", "r1"], str(tmp_path / "report.json"), str(jsonl_path))
    runner.run_tournament(cfg, in_process=True)

    rows = [
        json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()
    ]
    headers = [row for row in rows if row["row"] == "game"]
    assert len(headers) == cfg.total_games
    assert len({row["game_id"] for row in headers}) == cfg.total_games
    assert all(row["source"] == "tournament" for row in headers)
    assert all(sorted(row["seats"]) == ["r0", "r1"] for row in headers)
    # Node rows carry the game they belong to, so the file is groupable.
    assert {row["game_id"] for row in rows} == {row["game_id"] for row in headers}


def test_tournament_without_jsonl_writes_nothing(tmp_path: pathlib.Path) -> None:
    """The flat log stays off by default — it costs a recorder per game."""
    cfg = _random_cfg(["r0", "r1"], str(tmp_path / "report.json"))
    runner.run_tournament(cfg, in_process=True)
    assert list(tmp_path.glob("*.jsonl")) == []


def test_worker_shards_merge_back_into_one_file(tmp_path: pathlib.Path) -> None:
    """Per-worker shards concatenate into the configured path and are removed."""
    base = str(tmp_path / "games.jsonl")
    _clear_shards(base)
    for worker_id, text in ((11, '{"row":"game","w":11}\n'), (22, '{"w":22}\n')):
        pathlib.Path(_shard_path(base, worker_id)).write_text(text, encoding="utf-8")

    _merge_shards(base)

    merged = pathlib.Path(base).read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["w"] for line in merged] == [11, 22]
    assert list(tmp_path.glob("games.w*.jsonl")) == []


def test_clear_shards_removes_a_previous_runs_leftovers(
    tmp_path: pathlib.Path,
) -> None:
    """Shard names are keyed on pid, so a rerun must not inherit stale rows."""
    base = str(tmp_path / "games.jsonl")
    stale = pathlib.Path(_shard_path(base, 99))
    stale.write_text('{"row":"stale"}\n', encoding="utf-8")
    pathlib.Path(base).write_text('{"row":"stale"}\n', encoding="utf-8")

    _clear_shards(base)

    assert not stale.exists()
    assert pathlib.Path(base).read_text(encoding="utf-8") == ""


def test_report_serializes_to_json(tmp_path: pathlib.Path) -> None:
    out = tmp_path / "report.json"
    cfg = _random_cfg(["r0", "r1"], str(out))
    report = runner.run_tournament(cfg, in_process=True)
    out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    restored = models.TournamentReport.model_validate_json(
        out.read_text(encoding="utf-8")
    )
    assert len(restored.games) == cfg.total_games
