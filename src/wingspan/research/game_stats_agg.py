"""Aggregates and the HTML preview for the game-stats study.

Reads the study's raw per-seat table (``games_raw.csv``) — it is re-runnable
standalone — and writes chart-ready aggregates into an output directory:

* ``breakdown_table.csv`` / ``breakdown_table.md`` — mean score per category,
  all seat-games versus won seat-games.
* ``turn_scores.csv`` — per (turn, cohort, convention) score distribution.
* ``action_freq.csv`` — per (turn, cohort) main-action counts and fractions.
* ``game_pairs.csv`` — each game's paired final scores and winner.
* ``preview.html`` — a self-contained Plotly page with four sanity-check
  figures (not the blog figures themselves).

Two cohorts: ``all`` (every seat-game) and ``won`` (sole winners; ties are
excluded). Only numpy is used for the statistics.
"""

from __future__ import annotations

import csv
import json
import math
import pathlib
import typing

import numpy as np
import pydantic

from wingspan import decisions
from wingspan.reporting import training_summary
from wingspan.research import constants, models

# Two-sided 95% normal quantile for the mean's confidence band.
_Z_95 = 1.96
# Percentiles reported per turn, in column order (p5, p25, p50, p75, p95).
_PERCENTILES = (5.0, 25.0, 50.0, 75.0, 95.0)

# Score categories: (raw-table column, display name), in table order.
_CATEGORIES: tuple[tuple[str, str], ...] = (
    ("birds", "Bird Cards"),
    ("eggs", "Eggs"),
    ("tucked", "Tucked Cards"),
    ("cached", "Cached Food"),
    ("bonus", "Bonus Cards"),
    ("goals", "End-of-round Goals"),
)
_TOTAL_COLUMN = "final_score"
_TOTAL_LABEL = "Total"

# Output file names inside the ``--out`` directory.
BREAKDOWN_CSV = "breakdown_table.csv"
BREAKDOWN_MD = "breakdown_table.md"
TURN_SCORES_CSV = "turn_scores.csv"
ACTION_FREQ_CSV = "action_freq.csv"
GAME_PAIRS_CSV = "game_pairs.csv"
PREVIEW_HTML = "preview.html"

# Main actions in the stable order of the ``action_freq.csv`` columns.
_ACTIONS: tuple[decisions.MainAction, ...] = (
    decisions.MainAction.GAIN_FOOD,
    decisions.MainAction.LAY_EGGS,
    decisions.MainAction.DRAW_CARDS,
    decisions.MainAction.PLAY_BIRD,
)

# Preview styling: series colors that stay distinguishable on white.
_COLOR_ALL = "#1f77b4"
_COLOR_WON = "#d95f02"
_ACTION_COLORS: dict[decisions.MainAction, str] = {
    decisions.MainAction.GAIN_FOOD: "#e6a817",
    decisions.MainAction.LAY_EGGS: "#7570b3",
    decisions.MainAction.DRAW_CARDS: "#1b9e77",
    decisions.MainAction.PLAY_BIRD: "#d62728",
}
# Empty x-axis slots left between rounds in the action charts.
_ROUND_GAP = 1.0
_BAND_OPACITY_HEX = "33"
_BAR_OPACITY = 0.6


class RawGames(pydantic.BaseModel):
    """The raw table as numpy columns (one row per seat-game)."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    game_index: np.ndarray
    seat: np.ndarray
    won: np.ndarray
    tie: np.ndarray
    # Scalar score columns by raw-table name (final_score, birds, ...).
    scores: dict[str, np.ndarray]
    # (rows, TURNS_PER_GAME + 1) projected / raw score series.
    score_proj: np.ndarray
    score_raw: np.ndarray
    # (rows, TURNS_PER_GAME) ``MainAction`` values.
    actions: np.ndarray

    def mask(self, cohort: models.Cohort) -> np.ndarray:
        """Boolean row selector for ``cohort`` (won excludes ties by design:
        ``won`` is already False for both seats of a tied game)."""
        if cohort is models.Cohort.WON:
            return self.won
        return np.ones(len(self.won), dtype=bool)


def read_raw_games(path: pathlib.Path) -> RawGames:
    """Parse ``games_raw.csv`` into numpy columns."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    score_columns = [column for column, _ in _CATEGORIES] + [_TOTAL_COLUMN]
    proj_columns = [
        f"{constants.SCORE_PROJ_PREFIX}{turn}"
        for turn in range(constants.TURNS_PER_GAME + 1)
    ]
    raw_columns = [
        f"{constants.SCORE_RAW_PREFIX}{turn}"
        for turn in range(constants.TURNS_PER_GAME + 1)
    ]
    action_columns = [
        f"{constants.ACTION_PREFIX}{turn}"
        for turn in range(1, constants.TURNS_PER_GAME + 1)
    ]
    return RawGames(
        game_index=_int_column(rows, "game_index"),
        seat=_int_column(rows, "seat"),
        won=_bool_column(rows, "won"),
        tie=_bool_column(rows, "tie"),
        scores={column: _int_column(rows, column) for column in score_columns},
        score_proj=_int_matrix(rows, proj_columns),
        score_raw=_int_matrix(rows, raw_columns),
        actions=np.array([[row[c] for c in action_columns] for row in rows]),
    )


def breakdown_rows(raw: RawGames) -> list[models.BreakdownRow]:
    """The six categories plus the total: mean over all seat-games, mean over
    won seat-games, and the difference."""
    won = raw.mask(models.Cohort.WON)
    rows: list[models.BreakdownRow] = []
    for column, label in [*_CATEGORIES, (_TOTAL_COLUMN, _TOTAL_LABEL)]:
        values = raw.scores[column]
        mean_all = float(values.mean())
        mean_won = float(values[won].mean()) if won.any() else math.nan
        rows.append(
            models.BreakdownRow(
                category=label,
                mean_all=mean_all,
                mean_won=mean_won,
                difference=mean_won - mean_all,
            )
        )
    return rows


def breakdown_markdown(rows: list[models.BreakdownRow]) -> str:
    """The breakdown as a ready-to-paste GitHub markdown table (one decimal
    place; the difference carries an explicit sign)."""
    lines = [
        "|Category|Average (All Games)|Average (Games Won)|Difference|",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"|{row.category}|{row.mean_all:.1f}|{row.mean_won:.1f}|"
            f"{_signed(row.difference)}|"
        )
    return "\n".join(lines) + "\n"


def turn_score_rows(raw: RawGames) -> list[models.TurnScoreRow]:
    """The score distribution at every turn for each cohort and convention;
    a cohort with no games is omitted."""
    rows: list[models.TurnScoreRow] = []
    series = {
        models.ScoreConvention.PROJ: raw.score_proj,
        models.ScoreConvention.RAW: raw.score_raw,
    }
    for convention, matrix in series.items():
        for cohort in models.Cohort:
            selected = matrix[raw.mask(cohort)]
            if len(selected) == 0:
                continue
            for turn in range(constants.TURNS_PER_GAME + 1):
                rows.append(
                    _turn_score_row(turn, cohort, convention, selected[:, turn])
                )
    return rows


def action_freq_rows(raw: RawGames) -> list[models.ActionFreqRow]:
    """Main-action counts and fractions at every turn for each cohort; a cohort
    with no games is omitted."""
    rows: list[models.ActionFreqRow] = []
    for cohort in models.Cohort:
        selected = raw.actions[raw.mask(cohort)]
        if len(selected) == 0:
            continue
        for turn in range(1, constants.TURNS_PER_GAME + 1):
            column = selected[:, turn - 1]
            counts = {
                action: int((column == action.value).sum()) for action in _ACTIONS
            }
            total = len(column)
            rows.append(
                models.ActionFreqRow(
                    turn=turn,
                    cohort=cohort,
                    n=total,
                    count_gain_food=counts[decisions.MainAction.GAIN_FOOD],
                    count_lay_eggs=counts[decisions.MainAction.LAY_EGGS],
                    count_draw_cards=counts[decisions.MainAction.DRAW_CARDS],
                    count_play_bird=counts[decisions.MainAction.PLAY_BIRD],
                    frac_gain_food=counts[decisions.MainAction.GAIN_FOOD] / total,
                    frac_lay_eggs=counts[decisions.MainAction.LAY_EGGS] / total,
                    frac_draw_cards=counts[decisions.MainAction.DRAW_CARDS] / total,
                    frac_play_bird=counts[decisions.MainAction.PLAY_BIRD] / total,
                )
            )
    return rows


def game_pair_rows(raw: RawGames) -> list[models.GamePairRow]:
    """One row per two-seat game: both final scores and the winner (the sole
    winner's seat, or -1 for a tie)."""
    rows: list[models.GamePairRow] = []
    finals = raw.scores[_TOTAL_COLUMN]
    for game_index in sorted(set(raw.game_index.tolist())):
        in_game = raw.game_index == game_index
        seats = raw.seat[in_game]
        game_finals = {int(s): int(f) for s, f in zip(seats, finals[in_game])}
        tie = bool(raw.tie[in_game][0])
        winner_seats = seats[raw.won[in_game]]
        rows.append(
            models.GamePairRow(
                game_index=int(game_index),
                p0_final=game_finals[0],
                p1_final=game_finals[1],
                winner=int(winner_seats[0]) if len(winner_seats) == 1 else -1,
                tie=tie,
            )
        )
    return rows


def aggregate(raw_csv: pathlib.Path, out_dir: pathlib.Path) -> list[pathlib.Path]:
    """Aggregate ``raw_csv`` into ``out_dir`` (created if needed), returning the
    written files."""
    raw = read_raw_games(raw_csv)
    out_dir.mkdir(parents=True, exist_ok=True)
    breakdown = breakdown_rows(raw)
    turn_scores = turn_score_rows(raw)
    action_freq = action_freq_rows(raw)
    pairs = game_pair_rows(raw)

    written = [
        _write_rows(out_dir / BREAKDOWN_CSV, breakdown),
        _write_text(out_dir / BREAKDOWN_MD, breakdown_markdown(breakdown)),
        _write_rows(out_dir / TURN_SCORES_CSV, turn_scores),
        _write_rows(out_dir / ACTION_FREQ_CSV, action_freq),
        _write_rows(out_dir / GAME_PAIRS_CSV, pairs),
        _write_text(
            out_dir / PREVIEW_HTML,
            preview_html(raw, breakdown, turn_scores, action_freq),
        ),
    ]
    return written


def preview_html(
    raw: RawGames,
    breakdown: list[models.BreakdownRow],
    turn_scores: list[models.TurnScoreRow],
    action_freq: list[models.ActionFreqRow],
) -> str:
    """The single self-contained preview page: the breakdown table, the
    final-score histogram, the per-turn mean-score line with its confidence
    band, and the per-turn action-fraction bars (all, then won)."""
    figures: list[dict[str, object]] = [
        {
            "id": "histogram",
            "traces": _histogram_traces(raw),
            "layout": {
                "barmode": "overlay",
                "xaxis": {"title": "Final score (VP)"},
                "yaxis": {"title": "Fraction of cohort's seat-games"},
            },
        },
        {
            "id": "turn_scores",
            "traces": _turn_score_traces(turn_scores),
            "layout": {
                "xaxis": {"title": "Cumulative turn (0 = start of game)"},
                "yaxis": {"title": "Mean score (VP, projected convention)"},
            },
        },
    ]
    for cohort in models.Cohort:
        figures.append(
            {
                "id": f"actions_{cohort.value}",
                "traces": _action_traces(action_freq, cohort),
                "layout": _action_layout(cohort),
            }
        )
    return (
        _PAGE_TEMPLATE.replace("__PLOTLY_CDN__", training_summary.PLOTLY_CDN_URL)
        .replace("__TABLE__", _table_html(breakdown))
        .replace("__FIGURES__", json.dumps(figures))
        .replace("__GAMES__", str(len(set(raw.game_index.tolist()))))
    )


###### PRIVATE #######

#### CSV parsing ####


def _int_column(rows: list[dict[str, str]], column: str) -> np.ndarray:
    return np.array([int(row[column]) for row in rows], dtype=np.int64)


def _bool_column(rows: list[dict[str, str]], column: str) -> np.ndarray:
    return np.array([row[column] == "True" for row in rows], dtype=bool)


def _int_matrix(rows: list[dict[str, str]], columns: list[str]) -> np.ndarray:
    return np.array([[int(row[c]) for c in columns] for row in rows], dtype=np.int64)


#### Statistics ####


def _turn_score_row(
    turn: int,
    cohort: models.Cohort,
    convention: models.ScoreConvention,
    values: np.ndarray,
) -> models.TurnScoreRow:
    n = len(values)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if n > 1 else 0.0
    sem = std / math.sqrt(n)
    p5, p25, p50, p75, p95 = (float(q) for q in np.percentile(values, _PERCENTILES))
    return models.TurnScoreRow(
        turn=turn,
        cohort=cohort,
        convention=convention,
        n=n,
        mean=mean,
        std=std,
        sem=sem,
        ci95_lo=mean - _Z_95 * sem,
        ci95_hi=mean + _Z_95 * sem,
        p5=p5,
        p25=p25,
        p50=p50,
        p75=p75,
        p95=p95,
    )


def _signed(value: float) -> str:
    """``value`` to one decimal with an explicit sign (``+0.0`` for zero)."""
    rounded = round(value, 1)
    return f"{(rounded if rounded != 0 else 0.0):+.1f}"


#### Output writers ####


def _write_rows(
    path: pathlib.Path, rows: typing.Sequence[pydantic.BaseModel]
) -> pathlib.Path:
    """Write ``rows`` (one model type) as a CSV with the model's field order."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not rows:
            return path
        writer = csv.DictWriter(handle, fieldnames=list(type(rows[0]).model_fields))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.model_dump(mode="json"))
    return path


def _write_text(path: pathlib.Path, text: str) -> pathlib.Path:
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


#### Preview figures ####


def _histogram_traces(raw: RawGames) -> list[dict[str, object]]:
    """Final-score histogram as per-VP fractions of each cohort, overlaid."""
    finals = raw.scores[_TOTAL_COLUMN]
    low, high = int(finals.min()), int(finals.max())
    edges = np.arange(low, high + 2)
    traces: list[dict[str, object]] = []
    for cohort, color in (
        (models.Cohort.ALL, _COLOR_ALL),
        (models.Cohort.WON, _COLOR_WON),
    ):
        values = finals[raw.mask(cohort)]
        if len(values) == 0:
            continue
        counts, _ = np.histogram(values, bins=edges)
        traces.append(
            {
                "type": "bar",
                "name": f"{cohort.value} (n={len(values)})",
                "x": list(range(low, high + 1)),
                "y": (counts / len(values)).tolist(),
                "marker": {"color": color},
                "opacity": _BAR_OPACITY,
            }
        )
    return traces


def _turn_score_traces(rows: list[models.TurnScoreRow]) -> list[dict[str, object]]:
    """Mean line plus shaded 95% band per cohort (projected convention)."""
    traces: list[dict[str, object]] = []
    for cohort, color in (
        (models.Cohort.ALL, _COLOR_ALL),
        (models.Cohort.WON, _COLOR_WON),
    ):
        series = [
            row
            for row in rows
            if row.cohort is cohort and row.convention is models.ScoreConvention.PROJ
        ]
        if not series:
            continue
        turns = [row.turn for row in series]
        traces.append(
            {
                "type": "scatter",
                "mode": "lines",
                "name": f"{cohort.value} 95% CI",
                "x": turns + turns[::-1],
                "y": [row.ci95_hi for row in series]
                + [row.ci95_lo for row in series][::-1],
                "fill": "toself",
                "fillcolor": color + _BAND_OPACITY_HEX,
                "line": {"width": 0},
                "hoverinfo": "skip",
                "showlegend": False,
            }
        )
        traces.append(
            {
                "type": "scatter",
                "mode": "lines",
                "name": cohort.value,
                "x": turns,
                "y": [row.mean for row in series],
                "line": {"color": color},
            }
        )
    return traces


def _round_x(turn: int) -> float:
    """The x position of cumulative ``turn`` with a gap left between rounds."""
    position = float(turn)
    boundary = 0
    for round_turns in constants.ROUND_TURNS[:-1]:
        boundary += round_turns
        if turn > boundary:
            position += _ROUND_GAP
    return position


def _action_traces(
    rows: list[models.ActionFreqRow], cohort: models.Cohort
) -> list[dict[str, object]]:
    """One stacked-bar trace per main action over the cohort's turns."""
    series = [row for row in rows if row.cohort is cohort]
    xs = [_round_x(row.turn) for row in series]
    traces: list[dict[str, object]] = []
    for action in _ACTIONS:
        traces.append(
            {
                "type": "bar",
                "name": action.value,
                "x": xs,
                "y": [getattr(row, f"frac_{action.value}") for row in series],
                "marker": {"color": _ACTION_COLORS[action]},
            }
        )
    return traces


def _action_layout(cohort: models.Cohort) -> dict[str, object]:
    turns = list(range(1, constants.TURNS_PER_GAME + 1))
    return {
        "title": f"Main action by turn: {cohort.value} games",
        "barmode": "stack",
        "xaxis": {
            "title": "Cumulative turn (gaps separate rounds of 8/7/6/5 turns)",
            "tickmode": "array",
            "tickvals": [_round_x(turn) for turn in turns],
            "ticktext": [str(turn) for turn in turns],
        },
        "yaxis": {"title": "Fraction of turns", "range": [0, 1]},
    }


def _table_html(rows: list[models.BreakdownRow]) -> str:
    body = "".join(
        f"<tr><td>{row.category}</td><td>{row.mean_all:.1f}</td>"
        f"<td>{row.mean_won:.1f}</td><td>{_signed(row.difference)}</td></tr>"
        for row in rows
    )
    return (
        "<table><thead><tr><th>Category</th><th>Average (All Games)</th>"
        "<th>Average (Games Won)</th><th>Difference</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


_PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Wingspan game stats preview</title>
<script src="__PLOTLY_CDN__"></script>
<style>
  body { font-family: system-ui, sans-serif; margin: 1.5rem auto; max-width: 70rem;
         padding: 0 1rem; background: #ffffff; color: #1a1a1a; }
  table { border-collapse: collapse; margin-bottom: 2rem; }
  th, td { border: 1px solid #bbb; padding: 0.35rem 0.8rem; text-align: right; }
  th:first-child, td:first-child { text-align: left; }
  .chart { height: 28rem; margin-bottom: 2rem; }
</style>
</head>
<body>
<h1>Wingspan game stats preview</h1>
<p>__GAMES__ games of greedy self-play. Sanity-check figures, not the final
artwork.</p>
<h2>Score breakdown by category</h2>
__TABLE__
<h2>Final score distribution</h2>
<div id="histogram" class="chart"></div>
<h2>Mean score by turn</h2>
<div id="turn_scores" class="chart"></div>
<h2>Main action by turn</h2>
<div id="actions_all" class="chart"></div>
<div id="actions_won" class="chart"></div>
<noscript><p>Charts need internet access to load Plotly from its CDN.</p></noscript>
<script>
  const figures = __FIGURES__;
  if (typeof Plotly !== "undefined") {
    for (const fig of figures) {
      Plotly.newPlot(fig.id, fig.traces, Object.assign(
        {margin: {t: 40}, legend: {orientation: "h", y: -0.2}}, fig.layout));
    }
  }
</script>
</body>
</html>
"""
