"""Paired comparison of forced-card factors against a baseline factor.

Every factor of the study plays the same per-game seeds and the same biased
seat(s) (one seat, ``game_index % num_players``, or every seat), so a factor's game
``g`` and the baseline's game ``g`` start from identical deals. Each biased seat of
a game is one unit, keyed by ``(game_index, seat)``; differencing its outcomes
removes the deal variance that dominates the marginal win rates. For each
non-baseline factor and each of three subsets ("all" pairs, pairs where the
baseline's seat held the forced bird, and pairs where the seat held it in either
run) this reports the paired mean difference (factor minus baseline) with a 95%
normal-approximation interval for: the win indicator (ties are non-wins), the
score margin (the seat's final score minus the best other seat's; the most
powerful of the three), and the forced bird's played indicator. When every seat
is biased the pairs within one game are not independent, so the intervals are
optimistic.

Only games complete in both runs are paired, so runs still in flight (or partial
files) compare on their intersection. A game is complete when it has a row for
every seat the file's games use. The held subsets condition on a post-setup
trajectory event, so they are descriptive slices, not randomised ones.
"""

from __future__ import annotations

import collections
import csv
import math
import pathlib
import statistics
import typing

from wingspan.research import constants, forced_card, forced_card_agg, models

_PERCENT = 100.0
_BLANK_CELL = ""
_MIN_PAIRS_FOR_INTERVAL = 2
# Markdown caveat appended to the heading of the subsets that condition on a hand.
_SUBSET_NOTES = {
    models.PairedSubset.BASELINE_HELD: (
        " (baseline's seat held the forced bird; a post-setup trajectory slice)"
    ),
    models.PairedSubset.ANY_HELD: (
        " (the seat held the forced bird in either run; a post-setup trajectory "
        "slice)"
    ),
}

type _Pair = tuple[models.ForcedCardGameOutcome, models.ForcedCardGameOutcome]
type _Outcomes = dict[tuple[int, int], models.ForcedCardGameOutcome]


def compare_to_baseline(
    factors: typing.Sequence[forced_card_agg.FactorRecords],
    baseline_factor: float,
    out_dir: pathlib.Path,
) -> list[pathlib.Path]:
    """Write ``forced_card_paired.csv`` / ``.md`` into ``out_dir`` comparing every
    factor in ``factors`` but the baseline against it. Raises ``FileNotFoundError``
    when no baseline factor is among ``factors``."""
    baseline_name = forced_card_agg.factor_dir_name(baseline_factor)
    baselines = [
        factor
        for factor in factors
        if forced_card_agg.factor_dir_name(factor.force_card_factor) == baseline_name
    ]
    if not baselines:
        raise FileNotFoundError(
            f"baseline factor {baseline_factor:g} has no "
            f"{baseline_name}/{constants.FORCED_CARD_RAW_FILENAME} under {out_dir}"
        )
    baseline = baselines[0]
    rows: list[models.ForcedCardPairedRow] = []
    for factor in sorted(factors, key=lambda run: run.force_card_factor):
        if factor is not baseline:
            rows.extend(paired_rows(factor, baseline))
    return _write(rows, baseline, out_dir)


def paired_rows(
    factor: forced_card_agg.FactorRecords, baseline: forced_card_agg.FactorRecords
) -> list[models.ForcedCardPairedRow]:
    """The "all", "baseline held", and "any held" rows of ``factor`` against
    ``baseline``. Raises ``ValueError`` when a game's biased seats or a seat's seed
    differ between runs."""
    pairs = _pair_games(game_outcomes(factor.records), game_outcomes(baseline.records))
    baseline_held = [pair for pair in pairs if pair[1].in_hand]
    any_held = [pair for pair in pairs if pair[0].in_hand or pair[1].in_hand]
    return [
        _row(factor.force_card_factor, baseline.force_card_factor, subset, subset_pairs)
        for subset, subset_pairs in (
            (models.PairedSubset.ALL, pairs),
            (models.PairedSubset.BASELINE_HELD, baseline_held),
            (models.PairedSubset.ANY_HELD, any_held),
        )
    ]


def game_outcomes(
    records: typing.Sequence[models.ForcedCardSeatRecord],
) -> _Outcomes:
    """Fold seat rows into one outcome per biased seat of each complete game,
    keyed by ``(game_index, seat)``. Incomplete games (fewer rows than the file's
    seat count) are dropped; a complete game without a biased row raises
    ``ValueError``."""
    by_game: dict[int, list[models.ForcedCardSeatRecord]] = collections.defaultdict(
        list
    )
    for record in records:
        by_game[record.game_index].append(record)
    seats_per_game = max((len(rows) for rows in by_game.values()), default=0)
    outcomes: _Outcomes = {}
    for game_index, rows in by_game.items():
        if len(rows) == seats_per_game:
            for outcome in _outcomes(game_index, rows):
                outcomes[(game_index, outcome.biased_seat)] = outcome
    return outcomes


def paired_markdown(
    rows: typing.Sequence[models.ForcedCardPairedRow], forced_card_name: str
) -> str:
    """Markdown tables, one per subset, headed by the forced bird. The margin
    column leads because it is the most sensitive comparison."""
    lines = [
        f"Forced card: {forced_card_name}. Each row is factor minus baseline on "
        "identical deals (each biased seat, paired by game and seat; ties are "
        "non-wins). Margin is the seat's score minus the best other seat's; "
        "intervals are 95% normal approximations of the per-pair differences.",
    ]
    for subset in models.PairedSubset:
        lines += [
            "",
            f"Subset: {subset.value}" + _SUBSET_NOTES.get(subset, ""),
            "",
            "|Factor|Baseline|Pairs|Margin delta|Margin 95% CI|Win delta|Win 95% CI"
            "|x wins / base loses|x loses / base wins|Play delta|Play 95% CI|",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        lines.extend(_markdown_row(row) for row in rows if row.subset is subset)
    return "\n".join(lines) + "\n"


###### PRIVATE #######

#### Pairing ####


def _outcomes(
    game_index: int, rows: typing.Sequence[models.ForcedCardSeatRecord]
) -> list[models.ForcedCardGameOutcome]:
    """One outcome per biased row of the game; the margin is that seat's score
    minus the best other seat's."""
    biased = [row for row in rows if row.biased]
    if not biased:
        raise ValueError(f"game {game_index} has no biased seat rows")
    return [
        models.ForcedCardGameOutcome(
            game_index=game_index,
            seed=row.seed,
            biased_seat=row.seat,
            won=row.won,
            margin=row.final_score - _best_other_score(row, rows),
            in_hand=row.forced_card_in_hand,
            played=row.forced_card_played,
        )
        for row in biased
    ]


def _best_other_score(
    seat_row: models.ForcedCardSeatRecord,
    rows: typing.Sequence[models.ForcedCardSeatRecord],
) -> int:
    return max((row.final_score for row in rows if row is not seat_row), default=0)


def _pair_games(factor: _Outcomes, baseline: _Outcomes) -> list[_Pair]:
    """``(factor, baseline)`` outcomes of the games complete in both, one pair per
    biased ``(game_index, seat)``. A game whose biased seats or a seat's seed
    differ between the runs raises ``ValueError``."""
    factor_seats = _seats_by_game(factor)
    baseline_seats = _seats_by_game(baseline)
    pairs: list[_Pair] = []
    for game_index in sorted(factor_seats.keys() & baseline_seats.keys()):
        if factor_seats[game_index] != baseline_seats[game_index]:
            raise ValueError(
                f"game {game_index} is not paired: biased seats "
                f"{sorted(factor_seats[game_index])} vs baseline "
                f"{sorted(baseline_seats[game_index])}"
            )
        for seat in sorted(factor_seats[game_index]):
            mine, theirs = factor[(game_index, seat)], baseline[(game_index, seat)]
            if mine.seed != theirs.seed:
                raise ValueError(
                    f"game {game_index} is not paired: seed {mine.seed} vs "
                    f"baseline {theirs.seed}"
                )
            pairs.append((mine, theirs))
    return pairs


def _seats_by_game(outcomes: _Outcomes) -> dict[int, set[int]]:
    """The biased seats of each game in ``outcomes``."""
    seats: dict[int, set[int]] = collections.defaultdict(set)
    for game_index, seat in outcomes:
        seats[game_index].add(seat)
    return seats


#### Statistics ####


def _row(
    factor: float,
    baseline_factor: float,
    subset: models.PairedSubset,
    pairs: typing.Sequence[_Pair],
) -> models.ForcedCardPairedRow:
    return models.ForcedCardPairedRow(
        force_card_factor=factor,
        baseline_factor=baseline_factor,
        subset=subset,
        n_pairs=len(pairs),
        win=_delta([int(mine.won) - int(theirs.won) for mine, theirs in pairs]),
        x_win_baseline_loss=sum(1 for mine, theirs in pairs if mine.won > theirs.won),
        x_loss_baseline_win=sum(1 for mine, theirs in pairs if mine.won < theirs.won),
        margin=_delta([mine.margin - theirs.margin for mine, theirs in pairs]),
        play=_delta([int(mine.played) - int(theirs.played) for mine, theirs in pairs]),
    )


def _delta(differences: typing.Sequence[int]) -> models.PairedDelta:
    """Mean of ``differences`` with a 95% interval from their sample standard
    error; the interval needs two pairs, and zero variance gives a zero-width one."""
    n = len(differences)
    if n == 0:
        return models.PairedDelta(n=0, mean=None, ci95_lo=None, ci95_hi=None)
    mean = statistics.fmean(differences)
    if n < _MIN_PAIRS_FOR_INTERVAL:
        return models.PairedDelta(n=n, mean=mean, ci95_lo=None, ci95_hi=None)
    half_width = constants.NORMAL_95_Z * statistics.stdev(differences) / math.sqrt(n)
    return models.PairedDelta(
        n=n, mean=mean, ci95_lo=mean - half_width, ci95_hi=mean + half_width
    )


#### Output ####


def _write(
    rows: typing.Sequence[models.ForcedCardPairedRow],
    baseline: forced_card_agg.FactorRecords,
    out_dir: pathlib.Path,
) -> list[pathlib.Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / constants.FORCED_CARD_PAIRED_FILENAME
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=models.ForcedCardPairedRow.csv_columns()
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_row())
    markdown_path = out_dir / constants.FORCED_CARD_PAIRED_MARKDOWN_FILENAME
    forced_name = forced_card.bird_name(baseline.forced_card_id)
    markdown_path.write_text(
        paired_markdown(rows, forced_name), encoding="utf-8", newline="\n"
    )
    return [csv_path, markdown_path]


def _markdown_row(row: models.ForcedCardPairedRow) -> str:
    return (
        f"|{row.force_card_factor:g}|{row.baseline_factor:g}|{row.n_pairs}"
        f"|{_number(row.margin.mean)}|{_interval(row.margin, _number)}"
        f"|{_percent(row.win.mean)}|{_interval(row.win, _percent)}"
        f"|{row.x_win_baseline_loss}|{row.x_loss_baseline_win}"
        f"|{_percent(row.play.mean)}|{_interval(row.play, _percent)}|"
    )


def _number(value: float | None) -> str:
    return _BLANK_CELL if value is None else f"{value:+.2f}"


def _percent(value: float | None) -> str:
    return _BLANK_CELL if value is None else f"{_PERCENT * value:+.1f} pp"


def _interval(
    delta: models.PairedDelta, render: typing.Callable[[float | None], str]
) -> str:
    if delta.ci95_lo is None or delta.ci95_hi is None:
        return _BLANK_CELL
    return f"{render(delta.ci95_lo)} to {render(delta.ci95_hi)}"
