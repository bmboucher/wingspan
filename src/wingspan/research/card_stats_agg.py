"""Per-card aggregates for the card-stats study.

Reads the study's raw per-seat table (``card_games_raw.csv``) — it is re-runnable
standalone — and writes, into an output directory:

* ``bird_stats.csv`` / ``bird_stats.md`` — one row per catalog bird: how often it
  is dealt and kept, and its "games in hand" (GIH) win rate overall and per
  round, each with a Wilson 95% interval; its "games played" win rate (bird
  placed on the board at any point); and its play rate (share of GIH
  seat-games in which it was also played).
* ``bonus_stats.csv`` / ``bonus_stats.md`` — one row per catalog bonus card: its
  offered / kept / original-card win rates, its acquired-mid-game win rates
  (overall and per round), and the combined original-or-acquired GIH rate.

Every seat-game is one observation; a tie counts as a non-win and stays in the
denominator. Cards never seen get ``n = 0`` rows with blank rates. Only the
standard library is used for the statistics.
"""

from __future__ import annotations

import collections
import csv
import math
import pathlib
import typing

from wingspan import cards
from wingspan.research import constants, models

# Percent formatting for the markdown renderings.
_PERCENT = 100.0
_BLANK_CELL = ""

type _Tallies = collections.defaultdict[str, models.WinTally]


def read_raw_cards(path: pathlib.Path) -> list[models.CardSeatRecord]:
    """Parse ``card_games_raw.csv`` into one record per seat-game."""
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            models.CardSeatRecord.from_csv_row(row) for row in csv.DictReader(handle)
        ]


def wilson_interval(wins: int, n: int) -> tuple[float, float]:
    """The Wilson score interval (95%) for ``wins`` successes in ``n`` trials,
    clamped to ``[0, 1]``. ``n`` must be positive."""
    z_squared = constants.Z_95**2
    rate = wins / n
    denominator = 1.0 + z_squared / n
    centre = (rate + z_squared / (2.0 * n)) / denominator
    spread = (
        constants.Z_95
        * math.sqrt(rate * (1.0 - rate) / n + z_squared / (4.0 * n * n))
        / denominator
    )
    return max(0.0, centre - spread), min(1.0, centre + spread)


def win_stat(tally: models.WinTally) -> models.WinStat:
    """The win rate and Wilson interval of ``tally`` (blank when it is empty)."""
    if tally.n == 0:
        return models.WinStat(n=0, wins=0, win_rate=None, ci95_lo=None, ci95_hi=None)
    low, high = wilson_interval(tally.wins, tally.n)
    return models.WinStat(
        n=tally.n,
        wins=tally.wins,
        win_rate=tally.wins / tally.n,
        ci95_lo=low,
        ci95_hi=high,
    )


def baseline(
    records: typing.Sequence[models.CardSeatRecord],
) -> models.CardStatsBaseline:
    """The overall seat-game win rate (ties are non-wins, kept in the count)."""
    wins = sum(record.won for record in records)
    return models.CardStatsBaseline(
        seat_games=len(records),
        wins=wins,
        ties=sum(record.tie for record in records),
        win_rate=wins / len(records) if records else None,
    )


def bird_rows(
    records: typing.Sequence[models.CardSeatRecord],
) -> list[models.BirdStatsRow]:
    """One row per catalog bird (in catalog order), including birds no seat ever
    held."""
    dealt = _new_tallies()
    gih = _new_tallies()
    rounds = [_new_tallies() for _ in range(constants.NUM_ROUNDS)]
    played = _new_tallies()
    kept: collections.Counter[str] = collections.Counter()
    played_in_hand: collections.Counter[str] = collections.Counter()
    for record in records:
        for name in set(record.dealt_birds):
            dealt[name].add(won=record.won)
        kept.update(set(record.kept_birds))
        in_hand = set[str]().union(*record.birds_by_round)
        for name in in_hand:
            gih[name].add(won=record.won)
        played_names = set(record.played_birds)
        for name in played_names:
            played[name].add(won=record.won)
        played_in_hand.update(in_hand & played_names)
        for round_tallies, names in zip(rounds, record.birds_by_round):
            for name in set(names):
                round_tallies[name].add(won=record.won)

    return [
        models.BirdStatsRow(
            bird_id=bird.id,
            bird_name=bird.name,
            dealt_n=dealt[bird.name].n,
            kept_n=kept[bird.name],
            keep_rate=_rate(kept[bird.name], dealt[bird.name].n),
            dealt_win_rate=_rate(dealt[bird.name].wins, dealt[bird.name].n),
            gih=win_stat(gih[bird.name]),
            gih_rounds=[win_stat(tallies[bird.name]) for tallies in rounds],
            played=win_stat(played[bird.name]),
            played_in_hand_n=played_in_hand[bird.name],
            play_rate=_rate(played_in_hand[bird.name], gih[bird.name].n),
        )
        for bird in cards.birds_ordered()
    ]


def bonus_rows(
    records: typing.Sequence[models.CardSeatRecord],
) -> list[models.BonusStatsRow]:
    """One row per catalog bonus card (in catalog order), including cards no seat
    was ever dealt. The original card is excluded from the acquired columns."""
    offered = _new_tallies()
    original = _new_tallies()
    acquired = _new_tallies()
    combined = _new_tallies()
    acquired_rounds = [_new_tallies() for _ in range(constants.NUM_ROUNDS)]
    for record in records:
        for name in set(record.dealt_bonus):
            offered[name].add(won=record.won)
        if record.original_bonus:
            original[record.original_bonus].add(won=record.won)
        acquired_names = set[str]().union(*record.acquired_bonus_by_round)
        for name in acquired_names:
            acquired[name].add(won=record.won)
        for name in acquired_names | ({record.original_bonus} - {_BLANK_CELL}):
            combined[name].add(won=record.won)
        for round_tallies, names in zip(
            acquired_rounds, record.acquired_bonus_by_round
        ):
            for name in set(names):
                round_tallies[name].add(won=record.won)

    return [
        models.BonusStatsRow(
            bonus_name=bonus.name,
            offered_n=offered[bonus.name].n,
            kept_n=original[bonus.name].n,
            keep_rate=_rate(original[bonus.name].n, offered[bonus.name].n),
            offered_win_rate=_rate(offered[bonus.name].wins, offered[bonus.name].n),
            original=win_stat(original[bonus.name]),
            combined=win_stat(combined[bonus.name]),
            acquired=win_stat(acquired[bonus.name]),
            acquired_rounds=[
                win_stat(tallies[bonus.name]) for tallies in acquired_rounds
            ],
        )
        for bonus in cards.bonus_cards_ordered()
    ]


def bird_markdown(
    rows: typing.Sequence[models.BirdStatsRow], base: models.CardStatsBaseline
) -> str:
    """A compact markdown table of the headline bird columns, headed by the
    baseline win rate."""
    lines = [
        _baseline_line(base),
        "",
        "|Bird|Dealt|Keep rate|Dealt win rate|GIH n|GIH win rate|GIH 95% CI"
        "|Play rate|Played n|Played win rate|Played 95% CI|",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"|{row.bird_name}|{row.dealt_n}|{_percent(row.keep_rate)}"
            f"|{_percent(row.dealt_win_rate)}|{row.gih.n}"
            f"|{_percent(row.gih.win_rate)}|{_interval(row.gih)}"
            f"|{_percent(row.play_rate)}|{row.played.n}"
            f"|{_percent(row.played.win_rate)}|{_interval(row.played)}|"
        )
    return "\n".join(lines) + "\n"


def bonus_markdown(
    rows: typing.Sequence[models.BonusStatsRow], base: models.CardStatsBaseline
) -> str:
    """A compact markdown table of the headline bonus columns, headed by the
    baseline win rate."""
    lines = [
        _baseline_line(base),
        "",
        "|Bonus card|Offered|Keep rate|Offered win rate|Original win rate"
        "|Acquired n|Acquired win rate|GIH n|GIH win rate|GIH 95% CI|",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"|{row.bonus_name}|{row.offered_n}|{_percent(row.keep_rate)}"
            f"|{_percent(row.offered_win_rate)}|{_percent(row.original.win_rate)}"
            f"|{row.acquired.n}|{_percent(row.acquired.win_rate)}|{row.combined.n}"
            f"|{_percent(row.combined.win_rate)}|{_interval(row.combined)}|"
        )
    return "\n".join(lines) + "\n"


def aggregate(raw_csv: pathlib.Path, out_dir: pathlib.Path) -> list[pathlib.Path]:
    """Aggregate ``raw_csv`` into ``out_dir`` (created if needed), returning the
    written files."""
    records = read_raw_cards(raw_csv)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = baseline(records)
    birds = bird_rows(records)
    bonuses = bonus_rows(records)
    return [
        _write_rows(
            out_dir / constants.BIRD_STATS_FILENAME,
            birds,
            models.BirdStatsRow.csv_columns(),
        ),
        _write_text(
            out_dir / constants.BIRD_STATS_MARKDOWN_FILENAME,
            bird_markdown(birds, base),
        ),
        _write_rows(
            out_dir / constants.BONUS_STATS_FILENAME,
            bonuses,
            models.BonusStatsRow.csv_columns(),
        ),
        _write_text(
            out_dir / constants.BONUS_STATS_MARKDOWN_FILENAME,
            bonus_markdown(bonuses, base),
        ),
    ]


###### PRIVATE #######


def _new_tallies() -> _Tallies:
    return collections.defaultdict(models.WinTally)


def _rate(numerator: int, denominator: int) -> float | None:
    """``numerator / denominator``, or ``None`` when nothing was observed."""
    return numerator / denominator if denominator else None


#### Output writers ####


class _RowLike(typing.Protocol):
    def to_csv_row(self) -> dict[str, object]: ...


def _write_rows(
    path: pathlib.Path, rows: typing.Sequence[_RowLike], columns: list[str]
) -> pathlib.Path:
    """Write ``rows`` as a CSV with the given header (``None`` cells are blank)."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_row())
    return path


def _write_text(path: pathlib.Path, text: str) -> pathlib.Path:
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


#### Markdown cells ####


def _baseline_line(base: models.CardStatsBaseline) -> str:
    return (
        f"Baseline win rate: {_percent(base.win_rate)} over {base.seat_games} "
        f"seat-games ({base.ties} in tied games, counted as non-wins)."
    )


def _percent(rate: float | None) -> str:
    return _BLANK_CELL if rate is None else f"{_PERCENT * rate:.1f}%"


def _interval(stat: models.WinStat) -> str:
    if stat.ci95_lo is None or stat.ci95_hi is None:
        return _BLANK_CELL
    return f"{_percent(stat.ci95_lo)} - {_percent(stat.ci95_hi)}"
