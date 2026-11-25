"""Per-factor summary of the forced-card study.

Reads each factor's raw per-seat table (``forced_card_raw.csv``) and writes the
combined ``forced_card_summary.csv`` / ``.md``: one row per factor with the biased
seat's overall win rate, the unbiased seats' win rate, the forced bird's play rate
(plays over games in hand), and the biased seat's win rate in games that played
the bird versus held it without playing. Every rate carries a Wilson 95%
interval; ties are non-wins kept in the denominator, as in card-stats.
"""

from __future__ import annotations

import csv
import pathlib
import typing

from wingspan.research import card_stats_agg, constants, forced_card, models

# Percent formatting for the markdown rendering.
_PERCENT = 100.0
_BLANK_CELL = ""


class FactorRun(typing.NamedTuple):
    """One factor's results: the factor, the forced bird id, and its raw CSV."""

    force_card_factor: float
    forced_card_id: int
    raw_path: pathlib.Path


class FactorRecords(typing.NamedTuple):
    """One factor's parsed seat-game records."""

    force_card_factor: float
    forced_card_id: int
    records: list[models.ForcedCardSeatRecord]


def read_raw(path: pathlib.Path) -> list[models.ForcedCardSeatRecord]:
    """Parse ``forced_card_raw.csv`` into one record per seat-game."""
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            models.ForcedCardSeatRecord.from_csv_row(row)
            for row in csv.DictReader(handle)
        ]


def summary_row(
    force_card_factor: float,
    forced_card_id: int,
    records: typing.Sequence[models.ForcedCardSeatRecord],
) -> models.ForcedCardSummaryRow:
    """Summarise one factor's seat-game ``records``."""
    biased = _tally(record for record in records if record.biased)
    opponent = _tally(record for record in records if not record.biased)
    in_hand = [
        record for record in records if record.biased and record.forced_card_in_hand
    ]
    played = _tally(record for record in in_hand if record.forced_card_played)
    held_unplayed = _tally(
        record for record in in_hand if not record.forced_card_played
    )
    play_rate = card_stats_agg.win_stat(models.WinTally(n=len(in_hand), wins=played.n))
    return models.ForcedCardSummaryRow(
        force_card_factor=force_card_factor,
        forced_card_id=forced_card_id,
        biased=card_stats_agg.win_stat(biased),
        opponent=card_stats_agg.win_stat(opponent),
        in_hand_games=len(in_hand),
        play_rate=play_rate.win_rate,
        play_rate_ci95_lo=play_rate.ci95_lo,
        play_rate_ci95_hi=play_rate.ci95_hi,
        played=card_stats_agg.win_stat(played),
        held_unplayed=card_stats_agg.win_stat(held_unplayed),
    )


def summary_markdown(
    rows: typing.Sequence[models.ForcedCardSummaryRow], forced_card_name: str
) -> str:
    """A compact markdown table of the headline columns, headed by the forced bird."""
    lines = [
        f"Forced card: {forced_card_name}. Win rates are the biased seat's "
        "(ties are non-wins, kept in the denominator).",
        "",
        "|Factor|Games|Win rate|Win 95% CI|Opponent win rate|In hand|Play rate"
        "|Play 95% CI|Played win rate|Held-unplayed win rate|",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"|{row.force_card_factor:g}|{row.biased.n}"
            f"|{_percent(row.biased.win_rate)}"
            f"|{_interval(row.biased.ci95_lo, row.biased.ci95_hi)}"
            f"|{_percent(row.opponent.win_rate)}|{row.in_hand_games}"
            f"|{_percent(row.play_rate)}"
            f"|{_interval(row.play_rate_ci95_lo, row.play_rate_ci95_hi)}"
            f"|{_percent(row.played.win_rate)}"
            f"|{_percent(row.held_unplayed.win_rate)}|"
        )
    return "\n".join(lines) + "\n"


def aggregate(
    runs: typing.Sequence[FactorRun], out_dir: pathlib.Path
) -> list[pathlib.Path]:
    """Write the combined summary CSV and markdown for ``runs`` into ``out_dir``
    (created if needed), returning the written files. Rows are ordered by factor
    ascending, negatives first."""
    return aggregate_records(
        [
            FactorRecords(
                run.force_card_factor, run.forced_card_id, read_raw(run.raw_path)
            )
            for run in runs
        ],
        out_dir,
    )


def aggregate_records(
    factors: typing.Sequence[FactorRecords], out_dir: pathlib.Path
) -> list[pathlib.Path]:
    """Like :func:`aggregate` over already-parsed records."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(factors, key=lambda factor: factor.force_card_factor)
    rows = [
        summary_row(factor.force_card_factor, factor.forced_card_id, factor.records)
        for factor in ordered
    ]
    csv_path = out_dir / constants.FORCED_CARD_SUMMARY_FILENAME
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=models.ForcedCardSummaryRow.csv_columns()
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_row())
    markdown_path = out_dir / constants.FORCED_CARD_SUMMARY_MARKDOWN_FILENAME
    forced_name = forced_card.bird_name(ordered[0].forced_card_id) if ordered else ""
    markdown_path.write_text(
        summary_markdown(rows, forced_name), encoding="utf-8", newline="\n"
    )
    return [csv_path, markdown_path]


def factor_dir_name(factor: float) -> str:
    """The per-factor directory name (``factor_-0.500``); ``-0.0`` is written as
    ``0.000``. The one place the format lives, shared with :func:`parse_factor_dir`."""
    return constants.FORCED_CARD_DIR_TEMPLATE.format(factor=factor + 0.0)


def parse_factor_dir(name: str) -> float | None:
    """The factor a directory name encodes, or ``None`` when it is not a factor
    directory (the inverse of :func:`factor_dir_name`)."""
    if not name.startswith(constants.FORCED_CARD_DIR_PREFIX):
        return None
    try:
        return float(name.removeprefix(constants.FORCED_CARD_DIR_PREFIX))
    except ValueError:
        return None


def discover_runs(out_dir: pathlib.Path, forced_card_id: int) -> list[FactorRecords]:
    """Parse every ``<out_dir>/factor_*/forced_card_raw.csv`` found, ascending by
    factor. Directories without a raw file or with an unparsable name are skipped."""
    found: list[FactorRecords] = []
    for factor_dir in out_dir.glob(f"{constants.FORCED_CARD_DIR_PREFIX}*"):
        factor = parse_factor_dir(factor_dir.name)
        raw_path = factor_dir / constants.FORCED_CARD_RAW_FILENAME
        if factor is not None and raw_path.is_file():
            found.append(FactorRecords(factor, forced_card_id, read_raw(raw_path)))
    return sorted(found, key=lambda factor: factor.force_card_factor)


###### PRIVATE #######


def _tally(records: typing.Iterable[models.ForcedCardSeatRecord]) -> models.WinTally:
    tally = models.WinTally()
    for record in records:
        tally.add(won=record.won)
    return tally


def _percent(rate: float | None) -> str:
    return _BLANK_CELL if rate is None else f"{_PERCENT * rate:.1f}%"


def _interval(low: float | None, high: float | None) -> str:
    if low is None or high is None:
        return _BLANK_CELL
    return f"{_percent(low)} - {_percent(high)}"
