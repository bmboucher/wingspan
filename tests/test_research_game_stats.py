"""The model-performance game-stats study (``wingspan.research.game_stats``).

Black-box over the study's seams: the per-seat capture of a real greedy
self-play game (score series, actions, outcome), shard-plan / worker-count
invariance of the raw CSV, the CSV column contract, the aggregate math on a
hand-built fixture (means, standard errors, action fractions, the won cohort,
the markdown table), and the ``wingspan research game-stats`` CLI end to end.
"""

from __future__ import annotations

import csv
import io
import math
import pathlib

import pytest

torch = pytest.importorskip("torch")

from wingspan import decisions, model, version  # noqa: E402
from wingspan.engine import core, scoring  # noqa: E402
from wingspan.instrumentation.handlers import game_stats  # noqa: E402
from wingspan.research import app, constants  # noqa: E402
from wingspan.research import game_stats as study  # noqa: E402
from wingspan.research import game_stats_agg, models
from wingspan.training import (  # noqa: E402
    artifacts,
    collect,
    config,
    loop_checkpoint,
    policy,
    runmeta,
    runstate,
)
from wingspan.training import setup_net as setup_net_module  # noqa: E402
from wingspan.training import setup_runmeta

_DEVICE = torch.device("cpu")
_TURNS = 26
_SCALAR_COLUMNS = [
    "game_index",
    "seed",
    "seat",
    "won",
    "tie",
    "final_score",
    "birds",
    "bonus",
    "eggs",
    "tucked",
    "cached",
    "goals",
]


def _write_run_dir(checkpoint_dir: pathlib.Path, *, with_setup: bool = True) -> None:
    """A minimal run directory the study loads: the unified run config, a
    ``last.pt`` with a fresh tiny policy net, and (optionally) ``setup.pt``."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg = config.RunConfig(
        misc=config.MiscConfig(collect_device="cpu", train_device="cpu"),
        run=config.RunSettings(
            run_name="game-stats-test", checkpoint_dir=str(checkpoint_dir)
        ),
        architecture=config.ArchitectureConfig(
            main=config.MainNetArchitecture(
                trunk_layers=(8, 8),
                choice_layers=(8, 8),
                head_layers=(),
                value_layers=(),
                card_embed_dim=4,
                card_encoder_layers=(),
                hand_encoder_layers=(8,),
            ),
            setup=config.SetupNetArchitecture(head_layers=(8,)),
        ),
    )
    runmeta.write_run_config(
        str(checkpoint_dir),
        cfg,
        stamp="t0",
        started_at="t0",
        git_sha=None,
        resumed_from_iteration=0,
    )
    net = model.PolicyValueNet(
        state_dim=cfg.state_dim,
        choice_dim=cfg.choice_dim,
        num_families=len(cfg.family_order),
        arch=cfg.arch,
        spec=cfg.encoding_spec,
    )
    main_payload: dict[str, object] = {
        "config": cfg.model_dump(),
        "model": net.state_dict(),
        "progress": runstate.RunProgress(iteration=1, total_games=2).model_dump(),
        "git_sha": None,
        "version": version.MODEL_VERSION,
    }
    loop_checkpoint.atomic_save(main_payload, checkpoint_dir / artifacts.LAST_CKPT)
    if with_setup:
        descriptor = setup_runmeta.read_setup_config(str(checkpoint_dir))
        setup_payload: dict[str, object] = {
            "setup_model": setup_net_module.SetupNet.from_setup_config(
                descriptor
            ).state_dict(),
            "version": version.MODEL_VERSION,
        }
        loop_checkpoint.atomic_save(
            setup_payload, checkpoint_dir / artifacts.SETUP_CKPT
        )


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    directory = tmp_path_factory.mktemp("game_stats_run")
    _write_run_dir(directory)
    return directory


###### Capture ######


def _play_with_capture(
    nets: study.GameNets, seed: int
) -> tuple[core.Engine, game_stats.GameStatsCapture]:
    """One greedy self-play game with the capture handler attached."""
    eng = collect.new_engine(seed, nets.num_players)
    agent = policy.greedy_agent(nets.policy_net, nets.device, setup_net=nets.setup_net)
    capture = game_stats.GameStatsCapture()
    played = core.Engine.play_one_game(
        eng.state,
        [agent] * nets.num_players,
        instrumentation=capture.instrumentation(),
        verbose_turn_log=False,
        split_setup_bonus=nets.split_setup_bonus,
        split_setup_food=nets.split_setup_food,
        combine_gain_food=nets.combine_gain_food,
    )
    return played, capture


def test_capture_records_a_full_real_game(run_dir: pathlib.Path):
    """Per seat: 27 score entries per convention, 26 valid actions, both
    conventions converging on the recorded final score, and outcome flags
    consistent with ``scoring.winners``."""
    nets = study.load_game_nets(run_dir, _DEVICE)
    valid_actions = {action.value for action in decisions.MainAction}
    for seed in (3, 4):
        played, capture = _play_with_capture(nets, seed)
        assert len(capture.seats) == nets.num_players
        for seat in capture.seats:
            player = played.state.players[seat.seat]
            assert len(seat.score_proj) == _TURNS + 1
            assert len(seat.score_raw) == _TURNS + 1
            assert len(seat.actions) == _TURNS
            assert set(seat.actions) <= valid_actions
            assert seat.final_score == player.final_score
            assert seat.score_raw[_TURNS] == seat.final_score
            assert seat.score_proj[_TURNS] == seat.final_score
            assert seat.final_score == (
                seat.bird_pts
                + seat.bonus_pts
                + seat.eggs
                + seat.tucked
                + seat.cached
                + seat.round_goal
            )
            # The baseline is the pre-game score; the series never decreases
            # below it (no bird, egg, or card is ever removed from the score).
            assert seat.score_proj[0] == 0
        winner_ids = scoring.winners(played.state.players)
        assert capture.tie == (len(winner_ids) > 1)
        for seat in capture.seats:
            assert seat.won == (not capture.tie and seat.seat in winner_ids)


def test_play_game_returns_one_record_per_seat(run_dir: pathlib.Path):
    nets = study.load_game_nets(run_dir, _DEVICE)
    spec = models.GameStatsSpec(games=2, seed=11)
    records = study.play_game(spec, 1, nets)
    assert [record.seat for record in records] == list(range(nets.num_players))
    assert all(record.game_index == 1 for record in records)
    assert all(record.seed == spec.game_seed(1) for record in records)
    assert records[0].tie == records[1].tie


def test_load_game_nets_requires_a_setup_model(tmp_path: pathlib.Path):
    _write_run_dir(tmp_path, with_setup=False)
    with pytest.raises(FileNotFoundError, match="no setup model"):
        study.load_game_nets(tmp_path, _DEVICE)


###### Column contract ######


def test_csv_columns_contract():
    expected = (
        _SCALAR_COLUMNS
        + [f"score_proj_{turn}" for turn in range(27)]
        + [f"score_raw_{turn}" for turn in range(27)]
        + [f"action_{turn}" for turn in range(1, 27)]
    )
    assert models.game_stats_csv_columns() == expected
    assert len(expected) == 12 + 27 + 27 + 26


def test_seat_record_validates_series_lengths():
    with pytest.raises(ValueError):
        models.GameSeatRecord(
            game_index=0,
            seed=0,
            seat=0,
            won=False,
            tie=False,
            final_score=0,
            birds=0,
            bonus=0,
            eggs=0,
            tucked=0,
            cached=0,
            goals=0,
            score_proj=[0] * 5,
            score_raw=[0] * 27,
            actions=["gain_food"] * 26,
        )


###### Determinism ######


def test_workers_match_in_process_byte_for_byte(
    run_dir: pathlib.Path, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    """Seed-pure shards: the process pool writes exactly the in-process CSV."""
    monkeypatch.setattr(constants, "GAME_STATS_GAMES_PER_SHARD", 2)
    in_process = tmp_path / "in_process.csv"
    pooled = tmp_path / "pooled.csv"
    study.run_game_stats_study(
        models.GameStatsSpec(games=8, seed=5, workers=1), run_dir, _DEVICE, in_process
    )
    summary = study.run_game_stats_study(
        models.GameStatsSpec(games=8, seed=5, workers=2), run_dir, _DEVICE, pooled
    )
    assert summary.workers == 2
    assert summary.rows == 16
    assert in_process.read_bytes() == pooled.read_bytes()
    with in_process.open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert header == models.game_stats_csv_columns()


def test_progress_stream_leaves_csv_unchanged(
    run_dir: pathlib.Path, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    """Progress goes to its own stream: the CSV is byte-identical with or
    without it, and the last progress line reports every game."""
    monkeypatch.setattr(constants, "GAME_STATS_GAMES_PER_SHARD", 1)
    spec = models.GameStatsSpec(games=2, seed=5, workers=1)
    quiet = tmp_path / "quiet.csv"
    reported = tmp_path / "reported.csv"
    stream = io.StringIO()
    study.run_game_stats_study(spec, run_dir, _DEVICE, quiet)
    study.run_game_stats_study(spec, run_dir, _DEVICE, reported, progress_stream=stream)
    assert quiet.read_bytes() == reported.read_bytes()
    last_line = stream.getvalue().splitlines()[-1]
    assert last_line.startswith("games 2/2 (100.0%)")


def test_plan_shards_covers_every_game_once(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(constants, "GAME_STATS_GAMES_PER_SHARD", 3)
    shards = study.plan_shards(models.GameStatsSpec(games=7))
    indices = [index for shard in shards for index in shard.game_indices]
    assert indices == list(range(7))


###### Aggregation (hand-built fixture) ######

_ACTIONS = [
    decisions.MainAction.GAIN_FOOD.value,
    decisions.MainAction.LAY_EGGS.value,
    decisions.MainAction.DRAW_CARDS.value,
    decisions.MainAction.PLAY_BIRD.value,
]
# Row i is game i // 2, seat i % 2. Games: seat 0 wins, seat 1 wins, tie.
_FINALS = [60, 50, 40, 70, 55, 55]
_WON = [True, False, False, True, False, False]
_TIE = [False, False, False, False, True, True]
_TUCKED = [1, 2, 3, 4, 5, 6]
# Row i's projected score at turn t is ``2 * i + t``; raw is one lower after
# turn 0.


def _fixture_record(row: int) -> models.GameSeatRecord:
    base = 2 * row
    return models.GameSeatRecord(
        game_index=row // 2,
        seed=100 + row // 2,
        seat=row % 2,
        won=_WON[row],
        tie=_TIE[row],
        final_score=_FINALS[row],
        birds=_FINALS[row] - 25,
        bonus=5,
        eggs=10,
        tucked=_TUCKED[row],
        cached=0,
        goals=2,
        score_proj=[base + turn for turn in range(_TURNS + 1)],
        score_raw=[base + turn - (1 if turn else 0) for turn in range(_TURNS + 1)],
        actions=[_ACTIONS[(turn + row) % 4] for turn in range(1, _TURNS + 1)],
    )


@pytest.fixture
def fixture_csv(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "games_raw.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=models.game_stats_csv_columns())
        writer.writeheader()
        for row in range(6):
            writer.writerow(_fixture_record(row).to_csv_row())
    return path


def test_breakdown_markdown_exact(fixture_csv: pathlib.Path):
    raw = game_stats_agg.read_raw_games(fixture_csv)
    markdown = game_stats_agg.breakdown_markdown(game_stats_agg.breakdown_rows(raw))
    assert markdown == (
        "|Category|Average (All Games)|Average (Games Won)|Difference|\n"
        "|---|---|---|---|\n"
        "|Bird Cards|30.0|40.0|+10.0|\n"
        "|Eggs|10.0|10.0|+0.0|\n"
        "|Tucked Cards|3.5|2.5|-1.0|\n"
        "|Cached Food|0.0|0.0|+0.0|\n"
        "|Bonus Cards|5.0|5.0|+0.0|\n"
        "|End-of-round Goals|2.0|2.0|+0.0|\n"
        "|Total|55.0|65.0|+10.0|\n"
    )


def test_turn_scores_match_hand_computation(fixture_csv: pathlib.Path):
    raw = game_stats_agg.read_raw_games(fixture_csv)
    rows = game_stats_agg.turn_score_rows(raw)
    # 2 conventions x 2 cohorts x 27 turns.
    assert len(rows) == 2 * 2 * (_TURNS + 1)

    def pick(
        turn: int, cohort: models.Cohort, convention: models.ScoreConvention
    ) -> models.TurnScoreRow:
        (match,) = [
            r
            for r in rows
            if (r.turn, r.cohort, r.convention) == (turn, cohort, convention)
        ]
        return match

    # All six rows at turn 3 (projected): 3, 5, 7, 9, 11, 13.
    everyone = pick(3, models.Cohort.ALL, models.ScoreConvention.PROJ)
    assert everyone.n == 6
    assert everyone.mean == pytest.approx(8.0)
    assert everyone.std == pytest.approx(math.sqrt(14.0))
    assert everyone.sem == pytest.approx(math.sqrt(14.0 / 6.0))
    assert everyone.ci95_lo == pytest.approx(8.0 - 1.96 * math.sqrt(14.0 / 6.0))
    assert everyone.ci95_hi == pytest.approx(8.0 + 1.96 * math.sqrt(14.0 / 6.0))
    assert everyone.p50 == pytest.approx(8.0)
    assert everyone.p5 == pytest.approx(3.0 + 0.25 * 2.0)
    assert everyone.p95 == pytest.approx(13.0 - 0.25 * 2.0)

    # Won rows are 0 and 3: scores 3 and 9.
    winners = pick(3, models.Cohort.WON, models.ScoreConvention.PROJ)
    assert winners.n == 2
    assert winners.mean == pytest.approx(6.0)
    assert winners.sem == pytest.approx(3.0)
    assert winners.ci95_hi == pytest.approx(6.0 + 1.96 * 3.0)

    # Raw is one lower after the baseline, equal at turn 0.
    assert pick(3, models.Cohort.ALL, models.ScoreConvention.RAW).mean == (
        pytest.approx(7.0)
    )
    assert pick(0, models.Cohort.ALL, models.ScoreConvention.RAW).mean == (
        pytest.approx(5.0)
    )


def test_action_fractions_sum_to_one_and_exclude_ties(fixture_csv: pathlib.Path):
    raw = game_stats_agg.read_raw_games(fixture_csv)
    rows = game_stats_agg.action_freq_rows(raw)
    assert len(rows) == 2 * _TURNS
    for row in rows:
        total = (
            row.frac_gain_food
            + row.frac_lay_eggs
            + row.frac_draw_cards
            + row.frac_play_bird
        )
        assert total == pytest.approx(1.0)
    (turn1_all,) = [r for r in rows if r.turn == 1 and r.cohort is models.Cohort.ALL]
    # Row i plays _ACTIONS[(1 + i) % 4] on turn 1: 1, 2, 3, 0, 1, 2.
    assert turn1_all.n == 6
    assert (
        turn1_all.count_gain_food,
        turn1_all.count_lay_eggs,
        turn1_all.count_draw_cards,
        turn1_all.count_play_bird,
    ) == (1, 2, 2, 1)
    (turn1_won,) = [r for r in rows if r.turn == 1 and r.cohort is models.Cohort.WON]
    # Won rows 0 and 3 (ties and losses excluded): actions 1 and 0.
    assert turn1_won.n == 2
    assert (turn1_won.count_gain_food, turn1_won.count_lay_eggs) == (1, 1)


def test_game_pairs(fixture_csv: pathlib.Path):
    raw = game_stats_agg.read_raw_games(fixture_csv)
    pairs = game_stats_agg.game_pair_rows(raw)
    assert [(p.game_index, p.p0_final, p.p1_final, p.winner, p.tie) for p in pairs] == [
        (0, 60, 50, 0, False),
        (1, 40, 70, 1, False),
        (2, 55, 55, -1, True),
    ]


def test_preview_contains_scattergram_and_percentile_band(
    fixture_csv: pathlib.Path, tmp_path: pathlib.Path
):
    out_dir = tmp_path / "out"
    game_stats_agg.aggregate(fixture_csv, out_dir)
    html = (out_dir / game_stats_agg.PREVIEW_HTML).read_text(encoding="utf-8")
    assert '"id": "game_pairs"' in html
    assert "scattergl" in html
    assert "p5-p95" in html
    assert "p0 wins (n=1, 33.3%)" in html
    assert "p1 wins (n=1, 33.3%)" in html
    assert "tie (n=1, 33.3%)" in html
    assert "ci95" not in html


def test_aggregate_writes_every_output(
    fixture_csv: pathlib.Path, tmp_path: pathlib.Path
):
    out_dir = tmp_path / "out"
    written = game_stats_agg.aggregate(fixture_csv, out_dir)
    names = {path.name for path in written}
    assert names == {
        game_stats_agg.BREAKDOWN_CSV,
        game_stats_agg.BREAKDOWN_MD,
        game_stats_agg.TURN_SCORES_CSV,
        game_stats_agg.ACTION_FREQ_CSV,
        game_stats_agg.GAME_PAIRS_CSV,
        game_stats_agg.PREVIEW_HTML,
    }
    assert all(path.stat().st_size > 0 for path in written)
    html = (out_dir / game_stats_agg.PREVIEW_HTML).read_text(encoding="utf-8")
    assert "cdn.plot.ly/plotly-2.35.2.min.js" in html
    assert "<table>" in html


###### CLI ######


def test_cli_plays_and_aggregates(
    run_dir: pathlib.Path, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
):
    out_dir = tmp_path / "study"
    code = app.main(
        [
            "game-stats",
            "--games",
            "2",
            "--seed",
            "9",
            "--out",
            str(out_dir),
            "--checkpoint-dir",
            str(run_dir),
        ]
    )
    assert code == 0
    assert "games/s" in capsys.readouterr().out
    assert (out_dir / constants.GAMES_RAW_FILENAME).stat().st_size > 0
    assert (out_dir / game_stats_agg.PREVIEW_HTML).stat().st_size > 0

    # Aggregating the same raw table elsewhere reproduces the aggregates.
    again = tmp_path / "again"
    code = app.main(
        [
            "game-stats",
            "--out",
            str(again),
            "--aggregate-only",
            str(out_dir / constants.GAMES_RAW_FILENAME),
        ]
    )
    assert code == 0
    assert (again / game_stats_agg.BREAKDOWN_MD).read_text(encoding="utf-8") == (
        out_dir / game_stats_agg.BREAKDOWN_MD
    ).read_text(encoding="utf-8")


def test_cli_missing_run_exits_one(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
):
    code = app.main(
        [
            "game-stats",
            "--games",
            "1",
            "--out",
            str(tmp_path / "out"),
            "--checkpoint-dir",
            str(tmp_path / "no_such_run"),
        ]
    )
    assert code == 1
    assert "wingspan research" in capsys.readouterr().err
