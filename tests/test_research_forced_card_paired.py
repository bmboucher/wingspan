"""Paired forced-card comparison (``wingspan.research.forced_card_paired``) and the
``wingspan research forced-card-summary`` CLI, on small hand-built raw CSVs."""

from __future__ import annotations

import csv
import math
import pathlib

import pytest

from wingspan.research import (
    app,
    constants,
    forced_card_agg,
    forced_card_paired,
    models,
)

_NUM_PLAYERS = 2
_BASE_SEED = 100
_OPPONENT_SCORE = 50


def _game(
    game_index: int,
    *,
    margin: int,
    won: bool,
    in_hand: bool = False,
    played: bool = False,
    seed: int | None = None,
    biased_seat: int | None = None,
) -> list[models.ForcedCardSeatRecord]:
    """The two seat rows of one game: the biased seat's ``margin`` over a fixed
    opponent score."""
    seat = game_index % _NUM_PLAYERS if biased_seat is None else biased_seat
    game_seed = _BASE_SEED + game_index if seed is None else seed
    biased = models.ForcedCardSeatRecord(
        game_index=game_index,
        seed=game_seed,
        seat=seat,
        biased=True,
        won=won,
        tie=False,
        final_score=_OPPONENT_SCORE + margin,
        forced_card_in_hand=in_hand,
        forced_card_played=played,
    )
    opponent = models.ForcedCardSeatRecord(
        game_index=game_index,
        seed=game_seed,
        seat=1 - seat,
        biased=False,
        won=not won,
        tie=False,
        final_score=_OPPONENT_SCORE,
        forced_card_in_hand=False,
        forced_card_played=False,
    )
    return [biased, opponent]


def _baseline_records() -> list[models.ForcedCardSeatRecord]:
    return [
        *_game(0, margin=10, won=True, in_hand=True, played=True),
        *_game(1, margin=-15, won=False, in_hand=True, played=False),
        *_game(2, margin=-5, won=False),
        *_game(3, margin=10, won=True),
    ]


def _factor_records() -> list[models.ForcedCardSeatRecord]:
    return [
        *_game(0, margin=-3, won=False, in_hand=True, played=True),
        *_game(1, margin=12, won=True, in_hand=True, played=True),
        *_game(2, margin=-12, won=False),
        *_game(3, margin=12, won=True),
    ]


def _runs(
    baseline: list[models.ForcedCardSeatRecord],
    factor: list[models.ForcedCardSeatRecord],
) -> tuple[forced_card_agg.FactorRecords, forced_card_agg.FactorRecords]:
    return (
        forced_card_agg.FactorRecords(0.5, 1, factor),
        forced_card_agg.FactorRecords(0.0, 1, baseline),
    )


def _write_factor_dir(
    out_dir: pathlib.Path, factor: float, records: list[models.ForcedCardSeatRecord]
) -> None:
    factor_dir = out_dir / forced_card_agg.factor_dir_name(factor)
    factor_dir.mkdir(parents=True)
    with (factor_dir / constants.FORCED_CARD_RAW_FILENAME).open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=models.ForcedCardSeatRecord.csv_columns()
        )
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_csv_row())


def _by_subset(
    rows: list[models.ForcedCardPairedRow],
) -> dict[models.PairedSubset, models.ForcedCardPairedRow]:
    return {row.subset: row for row in rows}


###### Paired statistics ######


def test_paired_means_and_discordant_counts_match_hand_computation():
    rows = _by_subset(forced_card_paired.paired_rows(*_runs(*_pair_args())))
    everything = rows[models.PairedSubset.ALL]
    assert everything.n_pairs == 4
    # Win differences -1, +1, 0, 0.
    assert everything.win.mean == pytest.approx(0.0)
    assert (everything.x_win_baseline_loss, everything.x_loss_baseline_win) == (1, 1)
    # Margin differences -13, 27, -7, 2.
    assert everything.margin.mean == pytest.approx(2.25)
    margin_sd = math.sqrt(
        sum((diff - 2.25) ** 2 for diff in (-13, 27, -7, 2)) / (4 - 1)
    )
    half_width = constants.NORMAL_95_Z * margin_sd / math.sqrt(4)
    assert everything.margin.ci95_lo == pytest.approx(2.25 - half_width)
    assert everything.margin.ci95_hi == pytest.approx(2.25 + half_width)
    # Played differences 0, 1, 0, 0.
    assert everything.play.mean == pytest.approx(0.25)


def _pair_args() -> (
    tuple[list[models.ForcedCardSeatRecord], list[models.ForcedCardSeatRecord]]
):
    return _baseline_records(), _factor_records()


def test_baseline_held_subset_filters_on_the_baseline_run():
    rows = _by_subset(forced_card_paired.paired_rows(*_runs(*_pair_args())))
    held = rows[models.PairedSubset.BASELINE_HELD]
    # Games 0 and 1 only: margin differences -13 and 27.
    assert held.n_pairs == 2
    assert held.margin.mean == pytest.approx(7.0)
    assert held.margin.ci95_hi == pytest.approx(7.0 + constants.NORMAL_95_Z * 20.0)
    assert held.win.mean == pytest.approx(0.0)
    assert held.play.mean == pytest.approx(0.5)


def test_pairs_use_the_intersection_of_complete_games():
    baseline = _baseline_records()[:-2]  # game 3 absent
    factor = _factor_records()[:-1]  # game 3's opponent row missing: incomplete
    factor_run, baseline_run = _runs(baseline, factor)
    rows = _by_subset(forced_card_paired.paired_rows(factor_run, baseline_run))
    assert rows[models.PairedSubset.ALL].n_pairs == 3
    assert set(forced_card_paired.game_outcomes(factor)) == {0, 1, 2}


def test_mismatched_seed_or_biased_seat_raises():
    factor_run, baseline_run = _runs(
        _baseline_records(),
        [*_factor_records()[:-2], *_game(3, margin=1, won=True, seed=7)],
    )
    with pytest.raises(ValueError, match="game 3 is not paired"):
        forced_card_paired.paired_rows(factor_run, baseline_run)
    factor_run, baseline_run = _runs(
        _baseline_records(),
        [*_factor_records()[:-2], *_game(3, margin=1, won=True, biased_seat=0)],
    )
    with pytest.raises(ValueError, match="game 3 is not paired"):
        forced_card_paired.paired_rows(factor_run, baseline_run)


def test_single_pair_has_no_interval_and_no_pairs_has_no_mean():
    one = _runs(_game(0, margin=5, won=True), _game(0, margin=9, won=True))
    row = _by_subset(forced_card_paired.paired_rows(*one))[models.PairedSubset.ALL]
    assert row.margin.mean == pytest.approx(4.0)
    assert row.margin.ci95_lo is None and row.margin.ci95_hi is None
    none = _by_subset(forced_card_paired.paired_rows(*one))[
        models.PairedSubset.BASELINE_HELD
    ]
    assert none.n_pairs == 0
    assert none.margin.mean is None and none.margin.ci95_lo is None


def test_zero_variance_gives_a_zero_width_interval():
    same = [*_game(0, margin=5, won=True), *_game(1, margin=5, won=True)]
    shifted = [*_game(0, margin=8, won=True), *_game(1, margin=8, won=True)]
    row = _by_subset(forced_card_paired.paired_rows(*_runs(same, shifted)))[
        models.PairedSubset.ALL
    ]
    assert row.margin.mean == pytest.approx(3.0)
    assert row.margin.ci95_lo == pytest.approx(3.0)
    assert row.margin.ci95_hi == pytest.approx(3.0)


###### Directory discovery and CLI ######


def test_factor_dir_name_round_trips_and_ignores_other_names():
    for factor in (-1.0, -0.5, 0.0, 0.25, 1.0):
        name = forced_card_agg.factor_dir_name(factor)
        assert forced_card_agg.parse_factor_dir(name) == factor
    assert forced_card_agg.factor_dir_name(-0.0) == "factor_0.000"
    assert forced_card_agg.parse_factor_dir("notes") is None
    assert forced_card_agg.parse_factor_dir("factor_abc") is None


def test_cli_rebuilds_summary_and_paired_files(tmp_path: pathlib.Path):
    _write_factor_dir(tmp_path, 0.5, _factor_records())
    _write_factor_dir(tmp_path, -0.5, _factor_records())
    _write_factor_dir(tmp_path, 0.0, _baseline_records())
    (tmp_path / "factor_junk").mkdir()
    assert app.main(["forced-card-summary", "--out", str(tmp_path)]) == 0

    with (tmp_path / constants.FORCED_CARD_SUMMARY_FILENAME).open(
        newline="", encoding="utf-8"
    ) as handle:
        summary = list(csv.DictReader(handle))
    assert [row["force_card_factor"] for row in summary] == ["-0.5", "0.0", "0.5"]

    with (tmp_path / constants.FORCED_CARD_PAIRED_FILENAME).open(
        newline="", encoding="utf-8"
    ) as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == models.ForcedCardPairedRow.csv_columns()
        paired = list(reader)
    assert [(row["force_card_factor"], row["subset"]) for row in paired] == [
        ("-0.5", "all"),
        ("-0.5", "baseline_held"),
        ("0.5", "all"),
        ("0.5", "baseline_held"),
    ]
    assert float(paired[2]["margin_delta_mean"]) == pytest.approx(2.25)
    markdown = (tmp_path / constants.FORCED_CARD_PAIRED_MARKDOWN_FILENAME).read_text(
        encoding="utf-8"
    )
    assert markdown.startswith("Forced card: Cedar Waxwing")


def test_cli_honours_a_non_default_baseline(tmp_path: pathlib.Path):
    _write_factor_dir(tmp_path, 0.5, _factor_records())
    _write_factor_dir(tmp_path, 0.0, _baseline_records())
    code = app.main(
        ["forced-card-summary", "--out", str(tmp_path), "--baseline-factor", "0.5"]
    )
    assert code == 0
    with (tmp_path / constants.FORCED_CARD_PAIRED_FILENAME).open(
        newline="", encoding="utf-8"
    ) as handle:
        paired = list(csv.DictReader(handle))
    assert {row["force_card_factor"] for row in paired} == {"0.0"}
    assert {row["baseline_factor"] for row in paired} == {"0.5"}


def test_cli_missing_baseline_exits_nonzero_with_message(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
):
    _write_factor_dir(tmp_path, 0.5, _factor_records())
    assert app.main(["forced-card-summary", "--out", str(tmp_path)]) == 1
    assert "baseline factor 0 has no factor_0.000" in capsys.readouterr().err
    assert not (tmp_path / constants.FORCED_CARD_PAIRED_FILENAME).exists()


def test_cli_without_any_raw_files_exits_nonzero(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
):
    assert app.main(["forced-card-summary", "--out", str(tmp_path)]) == 1
    assert "no factor_*/forced_card_raw.csv" in capsys.readouterr().err
