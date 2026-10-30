"""The general card-stats study (``wingspan.research.card_stats``).

Black-box over the study's seams: the capture handler's round-by-round holding
rules (driven event by event against a real engine state), the CSV column
contract, the aggregate math on a hand-built fixture (keep rates, games-in-hand
tallies, Wilson intervals, a never-seen card, a tied game), worker-count
invariance of the raw CSV, and the ``wingspan research card-stats`` CLI.
"""

from __future__ import annotations

import csv
import pathlib

import pytest

torch = pytest.importorskip("torch")

from wingspan import cards, model, release, state, version  # noqa: E402
from wingspan.instrumentation.handlers import card_stats as card_capture  # noqa: E402
from wingspan.research import app  # noqa: E402
from wingspan.research import card_stats as study  # noqa: E402
from wingspan.research import card_stats_agg, constants, game_stats, models
from wingspan.training import setup_runmeta  # noqa: E402
from wingspan.training import (  # noqa: E402
    artifacts,
    collect,
    config,
    loop_checkpoint,
    runmeta,
    runstate,
)
from wingspan.training import setup_net as setup_net_module  # noqa: E402

_DEVICE = torch.device("cpu")
_ROUNDS = 4
_SEAT = 0


def _write_run_dir(checkpoint_dir: pathlib.Path) -> None:
    """A minimal run directory the study loads: the unified run config, a
    ``last.pt`` with a fresh tiny policy net, and ``setup.pt``."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg = config.RunConfig(
        misc=config.MiscConfig(collect_device="cpu", train_device="cpu"),
        run=config.RunSettings(
            run_name="card-stats-test", checkpoint_dir=str(checkpoint_dir)
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
    descriptor = setup_runmeta.read_setup_config(str(checkpoint_dir))
    setup_payload: dict[str, object] = {
        "setup_model": setup_net_module.SetupNet.from_setup_config(
            descriptor
        ).state_dict(),
        "version": version.MODEL_VERSION,
    }
    loop_checkpoint.atomic_save(setup_payload, checkpoint_dir / artifacts.SETUP_CKPT)


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    directory = tmp_path_factory.mktemp("card_stats_run")
    _write_run_dir(directory)
    return directory


###### Capture handler ######


def _bird_name(index: int) -> str:
    return cards.birds_ordered()[index].name


def test_capture_applies_the_final_turn_exclusion():
    """Cards that arrive on a seat's final turn of a round miss that round and
    appear in the next; birds drawn and played within a non-final turn count;
    the original bonus never enters the acquired sets."""
    eng = collect.new_engine(1, 2)
    player = eng.state.players[_SEAT]
    birds = cards.birds_ordered()
    bonuses = cards.bonus_cards_ordered()
    capture = card_capture.CardStatsCapture()

    # Setup: five birds dealt, three kept, bonus 0 kept out of {0, 1}.
    player.hand = list(birds[0:5])
    capture.setup_start(engine=eng, player=player, dealt_bonus=list(bonuses[0:2]))
    player.hand = list(birds[0:3])
    player.bonus_cards = [bonuses[0]]
    other = eng.state.players[1]
    other.hand = list(birds[10:15])
    capture.setup_start(engine=eng, player=other, dealt_bonus=list(bonuses[2:4]))
    capture.round_start(engine=eng, round_num=0)

    # Round 1, turn 1 (a cube remains afterwards): draw bird 5, place bird 0,
    # acquire bonus 4.
    eng.state.round_idx = 0
    player.action_cubes_left = 7
    capture.turn_start(engine=eng, player=player)
    player.hand.append(birds[5])
    placed = player.hand.pop(0)
    capture.bird_placed(
        engine=eng,
        player=player,
        bird=placed,
        habitat=cards.Habitat.FOREST,
        played_bird=state.PlayedBird(bird=placed),
    )
    player.bonus_cards.append(bonuses[4])
    capture.turn_end(engine=eng, player=player)

    # Round 1, final turn: draw bird 6 and acquire bonus 5 (both excluded).
    player.action_cubes_left = 1
    capture.turn_start(engine=eng, player=player)
    player.hand.append(birds[6])
    player.bonus_cards.append(bonuses[5])
    player.action_cubes_left = 0
    capture.turn_end(engine=eng, player=player)

    # Round 2: the late arrivals show up at the first turn_start.
    eng.state.round_idx = 1
    player.action_cubes_left = 7
    capture.turn_start(engine=eng, player=player)
    player.action_cubes_left = 0
    capture.turn_end(engine=eng, player=player)

    seat = capture.seats[_SEAT]
    assert seat.dealt_birds == [birds[index].name for index in range(5)]
    assert seat.kept_birds == [birds[index].name for index in range(3)]
    assert set(seat.kept_birds) <= set(seat.dealt_birds)
    assert seat.dealt_bonus == [bonuses[0].name, bonuses[1].name]
    assert seat.original_bonus == bonuses[0].name
    assert seat.round_birds[0] == {birds[index].name for index in (0, 1, 2, 5)}
    assert birds[6].name not in seat.round_birds[0]
    assert seat.round_birds[1] == {birds[index].name for index in (1, 2, 5, 6)}
    assert seat.round_acquired_bonus[0] == {bonuses[4].name}
    assert seat.round_acquired_bonus[1] == {bonuses[4].name, bonuses[5].name}
    assert bonuses[0].name not in seat.round_acquired_bonus[0]
    assert seat.round_birds[2] == set()
    assert seat.played_birds == {birds[0].name}


def test_capture_on_a_real_game(run_dir: pathlib.Path):
    """A real greedy game: the opener is a subset of the deal, every round has a
    hand, and the outcome flags match the engine's."""
    nets = game_stats.load_game_nets(run_dir, _DEVICE)
    records = study.play_game(models.GameStatsSpec(games=1, seed=3), 0, nets)
    assert [record.seat for record in records] == list(range(nets.num_players))
    for record in records:
        assert len(record.dealt_birds) == 5
        assert set(record.kept_birds) <= set(record.dealt_birds)
        assert record.original_bonus in record.dealt_bonus
        assert len(record.birds_by_round) == _ROUNDS
        assert len(set(record.played_birds)) == len(record.played_birds)
        assert record.original_bonus not in {
            name for names in record.acquired_bonus_by_round for name in names
        }
    assert sum(record.won for record in records) <= 1


###### Column contract ######


def test_csv_columns_contract():
    expected = (
        ["game_index", "seed", "seat", "won", "tie", "final_score"]
        + ["dealt_birds", "kept_birds", "dealt_bonus", "original_bonus"]
        + ["played_birds"]
        + [f"birds_r{k}" for k in range(1, 5)]
        + [f"acquired_bonus_r{k}" for k in range(1, 5)]
    )
    assert models.card_stats_csv_columns() == expected


def test_seat_record_round_trips_through_csv_cells():
    record = _record(
        0,
        True,
        False,
        ["A", "B"],
        ["A"],
        "X",
        [["A"], [], ["A", "B"], []],
        played=["A"],
    )
    assert (
        models.CardSeatRecord.from_csv_row(
            {key: str(value) for key, value in record.to_csv_row().items()}
        )
        == record
    )


def test_bird_and_bonus_stat_columns_contract():
    gih = ["n", "wins", "win_rate", "ci95_lo", "ci95_hi"]
    bird = ["bird_id", "bird_name", "dealt_n", "kept_n", "keep_rate", "dealt_win_rate"]
    assert models.BirdStatsRow.csv_columns() == (
        bird
        + [f"gih_{stat}" for stat in gih]
        + [f"gih_r{k}_{stat}" for k in range(1, 5) for stat in gih]
        + [f"played_{stat}" for stat in gih]
        + ["played_in_hand_n", "play_rate"]
    )
    bonus = ["bonus_name", "offered_n", "kept_n", "keep_rate", "offered_win_rate"]
    assert models.BonusStatsRow.csv_columns() == (
        bonus
        + [f"original_{stat}" for stat in gih]
        + [f"gih_{stat}" for stat in gih]
        + [f"acquired_gih_{stat}" for stat in gih]
        + [f"acquired_r{k}_{stat}" for k in range(1, 5) for stat in gih]
    )


###### Aggregation (hand-built fixture) ######


def _record(
    row: int,
    won: bool,
    tie: bool,
    dealt: list[str],
    kept: list[str],
    original: str,
    birds_by_round: list[list[str]],
    acquired_by_round: list[list[str]] | None = None,
    played: list[str] | None = None,
) -> models.CardSeatRecord:
    return models.CardSeatRecord(
        game_index=row // 2,
        seed=100 + row // 2,
        seat=row % 2,
        won=won,
        tie=tie,
        final_score=50,
        dealt_birds=dealt,
        kept_birds=kept,
        dealt_bonus=[_BONUS_X, _BONUS_Y],
        original_bonus=original,
        played_birds=played or [],
        birds_by_round=birds_by_round,
        acquired_bonus_by_round=acquired_by_round or [[], [], [], []],
    )


_BIRD_A = _bird_name(0)
_BIRD_B = _bird_name(1)
_BIRD_C = _bird_name(2)
_BIRD_UNSEEN = _bird_name(3)
_BONUS_X = cards.bonus_cards_ordered()[0].name
_BONUS_Y = cards.bonus_cards_ordered()[1].name
_BONUS_UNSEEN = cards.bonus_cards_ordered()[2].name


def _fixture_records() -> list[models.CardSeatRecord]:
    a, b, c = _BIRD_A, _BIRD_B, _BIRD_C
    return [
        # Win; acquires bonus Y in rounds 2-3.
        _record(
            0,
            True,
            False,
            [a, b],
            [a],
            _BONUS_X,
            [[a], [a], [a, c], []],
            [[], [_BONUS_Y], [_BONUS_Y], []],
            [a, c],
        ),
        # Loss; holds a but never plays it.
        _record(1, False, False, [a], [a], _BONUS_X, [[a], [], [], []]),
        # Tie: never wins, stays in every denominator; keeps nothing.
        _record(2, False, True, [b], [], _BONUS_Y, [[], [], [], []]),
        # Win; plays a only.
        _record(
            3,
            True,
            False,
            [a, c],
            [a, c],
            _BONUS_X,
            [[a, c], [a, c], [a], [a]],
            played=[a],
        ),
    ]


@pytest.fixture
def fixture_csv(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / constants.CARD_GAMES_RAW_FILENAME
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=models.card_stats_csv_columns())
        writer.writeheader()
        for record in _fixture_records():
            writer.writerow(record.to_csv_row())
    return path


def _bird(rows: list[models.BirdStatsRow], name: str) -> models.BirdStatsRow:
    (match,) = [row for row in rows if row.bird_name == name]
    return match


def _bonus(rows: list[models.BonusStatsRow], name: str) -> models.BonusStatsRow:
    (match,) = [row for row in rows if row.bonus_name == name]
    return match


def test_wilson_interval_matches_hand_values():
    low, high = card_stats_agg.wilson_interval(2, 3)
    assert low == pytest.approx(0.20765, abs=1e-4)
    assert high == pytest.approx(0.93851, abs=1e-4)
    low, high = card_stats_agg.wilson_interval(0, 1)
    assert low == 0.0
    assert high == pytest.approx(0.79346, abs=1e-4)
    assert card_stats_agg.wilson_interval(2, 2)[1] == pytest.approx(1.0)


def test_baseline_counts_ties_as_non_wins(fixture_csv: pathlib.Path):
    base = card_stats_agg.baseline(card_stats_agg.read_raw_cards(fixture_csv))
    assert (base.seat_games, base.wins, base.ties) == (4, 2, 1)
    assert base.win_rate == pytest.approx(0.5)


def test_bird_rows_match_hand_computation(fixture_csv: pathlib.Path):
    rows = card_stats_agg.bird_rows(card_stats_agg.read_raw_cards(fixture_csv))
    assert len(rows) == len(cards.birds_ordered())

    bird_a = _bird(rows, _BIRD_A)
    assert (bird_a.dealt_n, bird_a.kept_n) == (3, 3)
    assert bird_a.keep_rate == pytest.approx(1.0)
    assert bird_a.dealt_win_rate == pytest.approx(2 / 3)
    assert (bird_a.gih.n, bird_a.gih.wins) == (3, 2)
    assert bird_a.gih.ci95_lo == pytest.approx(0.20765, abs=1e-4)
    assert bird_a.gih.ci95_hi == pytest.approx(0.93851, abs=1e-4)
    assert [(stat.n, stat.wins) for stat in bird_a.gih_rounds] == [
        (3, 2),
        (2, 2),
        (2, 2),
        (1, 1),
    ]

    # Held in 3 seat-games, played in 2 (both wins).
    assert (bird_a.played.n, bird_a.played.wins) == (2, 2)
    assert bird_a.played_in_hand_n == 2
    assert bird_a.play_rate == pytest.approx(2 / 3)

    # Dealt twice (a win and a tie), never kept, never held after setup.
    bird_b = _bird(rows, _BIRD_B)
    assert (bird_b.dealt_n, bird_b.kept_n) == (2, 0)
    assert bird_b.keep_rate == pytest.approx(0.0)
    assert bird_b.dealt_win_rate == pytest.approx(0.5)
    assert bird_b.gih.n == 0
    assert bird_b.gih.win_rate is None
    assert bird_b.played.n == 0
    assert bird_b.play_rate is None

    # Held in a won game in round 3 and a won game in rounds 1, 2.
    bird_c = _bird(rows, _BIRD_C)
    assert (bird_c.gih.n, bird_c.gih.wins) == (2, 2)
    assert [stat.n for stat in bird_c.gih_rounds] == [1, 1, 1, 0]
    assert (bird_c.played.n, bird_c.played.wins) == (1, 1)
    assert bird_c.play_rate == pytest.approx(0.5)

    unseen = _bird(rows, _BIRD_UNSEEN)
    assert unseen.dealt_n == 0
    assert unseen.keep_rate is None
    assert unseen.dealt_win_rate is None
    assert unseen.gih.n == 0
    assert unseen.gih.ci95_lo is None


def test_bonus_rows_match_hand_computation(fixture_csv: pathlib.Path):
    rows = card_stats_agg.bonus_rows(card_stats_agg.read_raw_cards(fixture_csv))
    assert len(rows) == len(cards.bonus_cards_ordered())

    bonus_x = _bonus(rows, _BONUS_X)
    assert (bonus_x.offered_n, bonus_x.kept_n) == (4, 3)
    assert bonus_x.keep_rate == pytest.approx(0.75)
    assert bonus_x.offered_win_rate == pytest.approx(0.5)
    assert (bonus_x.original.n, bonus_x.original.wins) == (3, 2)
    assert (bonus_x.combined.n, bonus_x.combined.wins) == (3, 2)
    assert bonus_x.acquired.n == 0
    assert bonus_x.acquired.win_rate is None

    bonus_y = _bonus(rows, _BONUS_Y)
    assert (bonus_y.offered_n, bonus_y.kept_n) == (4, 1)
    # The tied seat kept it: a non-win.
    assert (bonus_y.original.n, bonus_y.original.wins) == (1, 0)
    assert (bonus_y.acquired.n, bonus_y.acquired.wins) == (1, 1)
    assert [stat.n for stat in bonus_y.acquired_rounds] == [0, 1, 1, 0]
    # Original (tied game) OR acquired (won game): two seat-games, one win.
    assert (bonus_y.combined.n, bonus_y.combined.wins) == (2, 1)

    unseen = _bonus(rows, _BONUS_UNSEEN)
    assert unseen.offered_n == 0
    assert unseen.offered_win_rate is None
    assert unseen.combined.n == 0


def test_aggregate_writes_csvs_with_blank_rates(
    fixture_csv: pathlib.Path, tmp_path: pathlib.Path
):
    out_dir = tmp_path / "out"
    written = card_stats_agg.aggregate(fixture_csv, out_dir)
    assert {path.name for path in written} == {
        constants.BIRD_STATS_FILENAME,
        constants.BONUS_STATS_FILENAME,
        constants.BIRD_STATS_MARKDOWN_FILENAME,
        constants.BONUS_STATS_MARKDOWN_FILENAME,
    }
    with (out_dir / constants.BIRD_STATS_FILENAME).open(
        newline="", encoding="utf-8"
    ) as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == models.BirdStatsRow.csv_columns()
        rows = {row["bird_name"]: row for row in reader}
    assert len(rows) == len(cards.birds_ordered())
    assert rows[_BIRD_UNSEEN]["gih_n"] == "0"
    assert rows[_BIRD_UNSEEN]["gih_win_rate"] == ""
    assert rows[_BIRD_A]["gih_wins"] == "2"
    markdown = (out_dir / constants.BIRD_STATS_MARKDOWN_FILENAME).read_text(
        encoding="utf-8"
    )
    assert markdown.startswith("Baseline win rate: 50.0% over 4 seat-games")


###### Determinism ######


def test_workers_match_in_process_byte_for_byte(
    run_dir: pathlib.Path, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    """Seed-pure shards: the process pool writes exactly the in-process CSV."""
    monkeypatch.setattr(constants, "CARD_STATS_GAMES_PER_SHARD", 1)
    in_process = tmp_path / "in_process.csv"
    pooled = tmp_path / "pooled.csv"
    study.run_card_stats_study(
        models.GameStatsSpec(games=2, seed=5, workers=1), run_dir, _DEVICE, in_process
    )
    summary = study.run_card_stats_study(
        models.GameStatsSpec(games=2, seed=5, workers=2), run_dir, _DEVICE, pooled
    )
    assert summary.rows == 4
    assert in_process.read_bytes() == pooled.read_bytes()
    with in_process.open(newline="", encoding="utf-8") as handle:
        assert next(csv.reader(handle)) == models.card_stats_csv_columns()


def test_games_match_game_stats_outcomes(run_dir: pathlib.Path):
    """Same seeds and nets as game-stats: the winners and scores agree."""
    nets = game_stats.load_game_nets(run_dir, _DEVICE)
    spec = models.GameStatsSpec(games=1, seed=7)
    card_records = study.play_game(spec, 0, nets)
    stats_records = game_stats.play_game(spec, 0, nets)
    assert [(r.won, r.tie, r.final_score) for r in card_records] == [
        (r.won, r.tie, r.final_score) for r in stats_records
    ]


###### CLI ######


def test_cli_release_and_checkpoint_dir_are_mutually_exclusive(tmp_path: pathlib.Path):
    with pytest.raises(SystemExit) as raised:
        app.main(
            [
                "card-stats",
                "--out",
                str(tmp_path / "out"),
                "--release",
                "--checkpoint-dir",
                str(tmp_path),
            ]
        )
    assert raised.value.code == 2


def test_cli_release_without_bundle_exits_one(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    monkeypatch.setattr(release, "has_release", lambda: False)
    code = app.main(
        ["card-stats", "--games", "1", "--out", str(tmp_path / "out"), "--release"]
    )
    assert code == 1
    assert "no bundled release model" in capsys.readouterr().err


def test_cli_plays_and_aggregates(
    run_dir: pathlib.Path, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
):
    out_dir = tmp_path / "study"
    code = app.main(
        [
            "card-stats",
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
    assert "baseline win rate" in capsys.readouterr().out
    raw = out_dir / constants.CARD_GAMES_RAW_FILENAME
    assert raw.stat().st_size > 0
    assert (out_dir / constants.BIRD_STATS_FILENAME).stat().st_size > 0
    assert (out_dir / constants.BONUS_STATS_FILENAME).stat().st_size > 0

    again = tmp_path / "again"
    assert (
        app.main(["card-stats", "--out", str(again), "--aggregate-only", str(raw)]) == 0
    )
    assert (again / constants.BIRD_STATS_FILENAME).read_bytes() == (
        out_dir / constants.BIRD_STATS_FILENAME
    ).read_bytes()


def test_cli_missing_run_exits_one(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
):
    code = app.main(
        [
            "card-stats",
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
