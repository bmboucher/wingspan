"""Standalone end-of-run training-progression HTML report.

Charts a run's ``metrics.jsonl`` history end to end (loss curves, strength vs.
the reference opponent, score composition, throughput, and — when present —
the architecture-probe and setup-model readouts) with `Plotly.js
<https://plotly.com/javascript/>`_ loaded from its CDN
(``https://cdn.plot.ly/plotly-2.35.2.min.js``). Unlike the live dashboard
(``training.charts``, a ``rich`` terminal renderer) this is a single
self-contained ``.html`` file a reviewer opens in a browser after the run has
stopped producing new data — every chart is interactive (pan/zoom/hover) via
Plotly rather than a fixed-size braille canvas.

:func:`write_training_summary` is the one disk-writing entry point, called
automatically at the target-milestone (``training.loop_target
.handle_target_reached``) and on demand via ``wingspan summary``
(``reporting.summary_cli``). This module is torch-free — it reads only the
Pydantic history/config/report shapes (``training.metrics``,
``training.config``, ``analysis.models``) — so the report can be regenerated
without loading any checkpoint weights.

Two-stage design, mirroring ``training.runmeta``'s descriptor seam:
:func:`build_summary_payload` turns a run's history into a plain,
JSON-serializable :class:`SummaryPayload` (pure, unit-testable without any
HTML); :func:`build_training_summary_html` renders that payload into the page
(also pure). The payload's ``model_dump_json()`` is embedded verbatim in a
``<script type="application/json">`` block; a small inline script reads it
back and drives one ``Plotly.newPlot`` call per chart — so the Python side
never generates a data-table's worth of hand-built SVG/JS, only the JSON and
the chart wiring.
"""

from __future__ import annotations

import html as html_lib
import pathlib
import typing

import pydantic

from wingspan import decisions, version
from wingspan.analysis import models as analysis_models
from wingspan.training import (
    artifacts,
    config,
    convergence,
    metrics,
    metrics_log,
    runmeta,
    theme,
)

_PLOTLY_CDN_URL = "https://cdn.plot.ly/plotly-2.35.2.min.js"

_SECONDS_PER_HOUR = 3600.0

# Mirrors ``config.OpponentConfig.eval_ewma_alpha``'s default — used only when
# a run has no ``run_config_*.json`` to read the run's own alpha from.
_DEFAULT_EVAL_EWMA_ALPHA = 0.3

# How far a "raw" series is blended toward the canvas to sit visually under
# its EWMA companion — the same relationship :data:`theme.WIN_RAW` has to
# :data:`theme.WIN_COLOR`, generalized to series with no dedicated dim
# constant of their own.
_DIM_BLEND = 0.4

# One color per slot of ``decisions.ALL_DECISION_FAMILIES`` (13 families,
# fixed order — the dataviz skill's categorical rule: assign hues in a fixed
# order, never cycled). Hues are spaced ~166 degrees apart in the family
# sequence (a non-adjacent sampling of 13 evenly-spaced hues, step 6 of 13)
# so that *consecutive* families — the only pairs a stacked-area chart's
# adjacency check cares about — stay far apart in hue even though 13 evenly
# spaced hues alone would put close neighbors next to each other. Lightness
# is hue-corrected (yellow/green/cyan darkened, blue/violet lightened) so
# every slot reads at a similar perceptual weight against the dark
# ``theme.CANVAS`` surface. Validated with the dataviz skill's
# ``validate_palette.js --mode dark --surface theme.CANVAS``: lightness band,
# chroma floor, normal-vision floor, and contrast all PASS; the worst
# adjacent CVD pair lands in the 6-8 floor band, which is legal because every
# chart that uses this palette also ships a legend and per-trace hover labels
# (the required secondary encoding).
_FAMILY_COLORS: tuple[str, ...] = (
    "#4695ce",
    "#c14e33",
    "#2ba19c",
    "#c1335b",
    "#2ba165",
    "#c1339c",
    "#2ba12f",
    "#b852d1",
    "#5ea12b",
    "#7d52d1",
    "#839112",
    "#5261d1",
    "#916412",
)


class ChartSeries(pydantic.BaseModel):
    """One plotted line/marker/area trace within a :class:`ChartSpec`."""

    name: str
    x: list[int]
    y: list[float]
    color: str
    mode: str = "lines"  # "lines" | "markers" | "lines+markers"
    dash: str = "solid"
    stack_group: str | None = None
    fill: bool = False
    y_axis: str = "y"  # "y" | "y2"
    hover_format: str = ".3f"  # d3-format spec for the hover tooltip's value


class ChartMarker(pydantic.BaseModel):
    """A vertical reference line at one iteration (e.g. an opponent advance)."""

    iteration: int
    label: str
    color: str


class ChartLine(pydantic.BaseModel):
    """A horizontal reference line at one y value (e.g. the 50% win-rate line)."""

    value: float
    label: str
    color: str


class ChartSpec(pydantic.BaseModel):
    """One Plotly chart: its series plus the vertical/horizontal reference
    lines drawn under them."""

    id: str  # DOM id / slug, unique across the whole page
    title: str
    subtitle: str
    y_label: str
    y2_label: str | None = None
    series: list[ChartSeries]
    markers: list[ChartMarker] = pydantic.Field(default_factory=list[ChartMarker])
    lines: list[ChartLine] = pydantic.Field(default_factory=list[ChartLine])
    y_range: tuple[float, float] | None = None


class ChartSection(pydantic.BaseModel):
    """A titled group of charts (Strength, Score, Losses, ...)."""

    title: str
    blurb: str
    charts: list[ChartSpec]


class RunHeader(pydantic.BaseModel):
    """The run-identity + headline-stats block at the top of the page."""

    run_name: str
    version: str
    git_sha: str | None
    started_at: str | None
    iterations: int
    total_games: int
    wall_clock_hours: float
    # Best held-out win rate within the latest reference-opponent generation
    # (see ``_build_header``), the iteration it landed on, and that generation.
    best_eval_win_rate: float | None
    best_eval_iteration: int | None
    best_eval_generation: int | None
    final_eval: metrics.FinalEvalStats | None


class SummaryPayload(pydantic.BaseModel):
    """The full report: :class:`RunHeader` plus every :class:`ChartSection`
    that had data to show."""

    header: RunHeader
    sections: list[ChartSection]


def build_summary_payload(
    rows: list[metrics.IterationMetrics],
    run_config: config.RunConfigFile | None,
    final_eval: metrics.FinalEvalStats | None,
    run_name: str,
) -> SummaryPayload:
    """Turn one run's history into a :class:`SummaryPayload` — pure, no I/O.

    ``rows`` is the full ``metrics.jsonl`` history in iteration order
    (:func:`wingspan.training.metrics_log.read_iteration_history`).
    ``run_config`` is the run's newest ``run_config_<stamp>.json``, or
    ``None`` when the run predates that artifact (the header and the
    config-dependent reference lines / markers degrade gracefully). A
    section is included only when at least one of its charts has data; a
    chart is included only when at least one of its series has a point.
    """
    target_marker = _target_marker(run_config)
    sections = [
        section
        for section in (
            _strength_section(rows, run_config, target_marker),
            _score_section(rows, run_config, target_marker),
            _losses_section(rows),
            _anneal_section(rows),
            _setup_section(rows),
            _decisions_section(rows),
            _throughput_section(rows),
            _representation_section(rows),
        )
        if section is not None
    ]
    return SummaryPayload(
        header=_build_header(rows, run_config, final_eval, run_name), sections=sections
    )


def build_training_summary_html(payload: SummaryPayload) -> str:
    """Render ``payload`` as a standalone, self-contained HTML page."""
    header_html = _render_header_html(payload.header)
    sections_html = "".join(
        _render_section_html(section) for section in payload.sections
    )
    notice_html = (
        '<p class="notice">No iterations logged yet — this page will fill in once '
        f"{artifacts.METRICS_LOG} has rows.</p>"
        if not payload.sections
        else ""
    )
    # ``model_dump_json`` embeds cleanly inside a ``<script type="application/
    # json">`` block except for a literal "</", which a naive HTML parser reads
    # as the start of a closing tag and would truncate the JSON mid-document.
    embedded_json = payload.model_dump_json().replace("</", "<\\/")
    return (
        _PAGE_TEMPLATE.replace(
            "__TITLE__",
            html_lib.escape(f"Training summary — {payload.header.run_name}"),
        )
        .replace("__CANVAS__", theme.CANVAS)
        .replace("__TEXT_PRIMARY__", theme.TEXT_PRIMARY)
        .replace("__TEXT_MUTED__", theme.TEXT_MUTED)
        .replace("__BORDER__", theme.BORDER_DEFAULT)
        .replace("__ACCENT__", theme.BORDER_HEADLINE)
        .replace("__PLOTLY_CDN__", _PLOTLY_CDN_URL)
        .replace("__HEADER_HTML__", header_html)
        .replace("__NOTICE_HTML__", notice_html)
        .replace("__SECTIONS_HTML__", sections_html)
        .replace("__PAYLOAD_JSON__", embedded_json)
    )


def write_training_summary(
    checkpoint_dir: str, out_path: pathlib.Path | None = None
) -> pathlib.Path:
    """Build and write ``training_summary.html`` for ``checkpoint_dir``.

    The one disk-writing entry point both the target-milestone loop hook
    (``training.loop_target.handle_target_reached``) and the ``wingspan
    summary`` CLI (``reporting.summary_cli``) call, so the two paths cannot
    drift. Reads the full metrics history, the newest run config (``None``
    when absent — an older run directory, or one with no config yet), and the
    newest ``final_eval_*.json`` (``None`` when the run hasn't reached its
    target milestone). Writes even a fresh, empty ``checkpoint_dir`` — the
    page then shows the header and a "no iterations logged yet" notice rather
    than raising. Returns the path written.
    """
    rows = metrics_log.read_iteration_history(checkpoint_dir)
    try:
        run_config = runmeta.read_run_config(checkpoint_dir)
    except (FileNotFoundError, version.IncompatibleArtifactError):
        # No unified config (a ≤0.4 run dir), or one from an era outside the
        # load guarantee: the metrics history is still worth charting, so the
        # config-derived header fields and reference lines degrade to absent
        # rather than refusing the whole report.
        run_config = None
    run_name = (
        run_config.config.run.run_name
        if run_config is not None
        else pathlib.Path(checkpoint_dir).name
    )
    payload = build_summary_payload(
        rows, run_config, _read_final_eval(checkpoint_dir), run_name
    )
    destination = (
        out_path or pathlib.Path(checkpoint_dir) / artifacts.TRAINING_SUMMARY_HTML
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(build_training_summary_html(payload), encoding="utf-8")
    return destination


###### PRIVATE #######

#### Header ####


def _build_header(
    rows: list[metrics.IterationMetrics],
    run_config: config.RunConfigFile | None,
    final_eval: metrics.FinalEvalStats | None,
    run_name: str,
) -> RunHeader:
    """Roll the run's identity, totals, and best-eval readout into one header.

    The best eval win rate is taken within the *latest* reference-opponent
    generation only: each opponent advance resets the eval to a stronger
    challenger, so a run-wide maximum would just report the easy early wins
    against the random agent."""
    best_win_rate: float | None = None
    best_iteration: int | None = None
    best_generation: int | None = None
    wall_clock_seconds = 0.0
    for row in rows:
        wall_clock_seconds += (
            row.collect_seconds
            + row.update_seconds
            + row.eval_seconds
            + row.probe_seconds
        )
        if row.eval is None:
            continue
        if best_generation is None or row.eval.opponent_generation > best_generation:
            best_generation = row.eval.opponent_generation
            best_win_rate = None
        if best_win_rate is None or row.eval.win_rate > best_win_rate:
            best_win_rate = row.eval.win_rate
            best_iteration = row.iteration
    return RunHeader(
        run_name=run_name,
        version=(
            run_config.version
            if run_config is not None
            else version.PRE_VERSIONING_VERSION
        ),
        git_sha=run_config.git_sha if run_config is not None else None,
        started_at=run_config.started_at if run_config is not None else None,
        iterations=rows[-1].iteration + 1 if rows else 0,
        total_games=rows[-1].total_games if rows else 0,
        wall_clock_hours=wall_clock_seconds / _SECONDS_PER_HOUR,
        best_eval_win_rate=best_win_rate,
        best_eval_iteration=best_iteration,
        best_eval_generation=best_generation,
        final_eval=final_eval,
    )


def _read_final_eval(checkpoint_dir: str) -> metrics.FinalEvalStats | None:
    """The latest-milestone ``final_eval_*.json`` in ``checkpoint_dir``, or
    ``None``. Chosen by each file's ``at_iteration`` rather than by filename,
    since the underscore-grouped names (``final_eval_500`` vs
    ``final_eval_1_000``) do not sort lexically by iteration."""
    candidates = [
        metrics.FinalEvalStats.model_validate_json(path.read_text(encoding="utf-8"))
        for path in pathlib.Path(checkpoint_dir).glob(artifacts.FINAL_EVAL_GLOB)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda stats: stats.at_iteration)


#### Shared chart helpers ####


def _dim(color: str) -> str:
    """A muted variant of ``color`` for a "raw" series plotted under its EWMA."""
    return theme.lerp_color(color, theme.CANVAS, _DIM_BLEND)


def _spec(
    chart_id: str,
    title: str,
    subtitle: str,
    y_label: str,
    series: list[ChartSeries],
    *,
    y2_label: str | None = None,
    markers: list[ChartMarker] | None = None,
    lines: list[ChartLine] | None = None,
    y_range: tuple[float, float] | None = None,
) -> ChartSpec:
    return ChartSpec(
        id=chart_id,
        title=title,
        subtitle=subtitle,
        y_label=y_label,
        y2_label=y2_label,
        series=series,
        markers=markers or [],
        lines=lines or [],
        y_range=y_range,
    )


def _xy(
    rows: list[metrics.IterationMetrics],
    value_of: typing.Callable[[metrics.IterationMetrics], float | None],
) -> tuple[list[int], list[float]]:
    """``(x, y)`` point lists over ``rows``, dropping iterations where
    ``value_of`` returns ``None`` (an optional field that isn't always set)."""
    xs: list[int] = []
    ys: list[float] = []
    for row in rows:
        value = value_of(row)
        if value is not None:
            xs.append(row.iteration)
            ys.append(value)
    return xs, ys


def _eval_ewma_alpha(run_config: config.RunConfigFile | None) -> float:
    """The EWMA decay the live dashboard's convergence charts use for the
    win-rate, score, and margin series alike (``convergence_chart.py`` reuses
    one alpha for all three)."""
    return (
        run_config.config.opponent.eval_ewma_alpha
        if run_config is not None
        else _DEFAULT_EVAL_EWMA_ALPHA
    )


def _target_marker(run_config: config.RunConfigFile | None) -> ChartMarker | None:
    """The target-milestone vertical marker, or ``None`` when no target is set
    (or the run has no config to read it from)."""
    if run_config is None or run_config.config.run.target_iterations <= 0:
        return None
    return ChartMarker(
        iteration=run_config.config.run.target_iterations,
        label="target",
        color=theme.GOOD,
    )


def _setup_markers(rows: list[metrics.IterationMetrics]) -> list[ChartMarker]:
    """Vertical markers at each setup-model phase transition
    (``convergence.setup_transition_iterations``), labeled with the phase the
    run entered at that iteration."""
    phase_at = {row.iteration: row.setup_phase for row in rows}
    return [
        ChartMarker(
            iteration=iteration,
            label=phase_at.get(iteration) or "setup phase change",
            color=theme.SETUP_MARK,
        )
        for iteration in convergence.setup_transition_iterations(rows)
    ]


#### Strength section ####


def _strength_section(
    rows: list[metrics.IterationMetrics],
    run_config: config.RunConfigFile | None,
    target_marker: ChartMarker | None,
) -> ChartSection | None:
    eval_rows = [row for row in rows if row.eval is not None]
    charts: list[ChartSpec] = []
    if eval_rows:
        charts.append(_eval_winrate_chart(rows, eval_rows, run_config, target_marker))
        charts.append(_eval_margin_chart(rows, eval_rows, run_config, target_marker))
    if any(row.collection_win_rate is not None for row in rows):
        charts.append(_bootstrap_winrate_chart(rows, target_marker))
    if not charts:
        return None
    return ChartSection(
        title="Strength",
        blurb=(
            "Win rate and margin against the reference opponent, and — during "
            "the random-opponent bootstrap phase — against the random agent."
        ),
        charts=charts,
    )


def _eval_winrate_chart(
    rows: list[metrics.IterationMetrics],
    eval_rows: list[metrics.IterationMetrics],
    run_config: config.RunConfigFile | None,
    target_marker: ChartMarker | None,
) -> ChartSpec:
    ewma = convergence.winrate_ewma_points(rows, _eval_ewma_alpha(run_config))
    lines = [ChartLine(value=50.0, label="50%", color=theme.FIFTY_PCT_LINE)]
    if run_config is not None:
        lines.append(
            ChartLine(
                value=run_config.config.opponent.opponent_reset_win_rate * 100.0,
                label="opponent-advance threshold",
                color=theme.WIN_THRESHOLD,
            )
        )
    markers = _opponent_gen_markers(eval_rows) + _setup_markers(rows)
    if target_marker is not None:
        markers.append(target_marker)
    return _spec(
        "strength-eval-win-rate",
        "Eval win rate",
        "Held-out win rate vs. the reference opponent — raw per-eval and EWMA-smoothed",
        "win rate (%)",
        [
            ChartSeries(
                name="raw",
                x=[row.iteration for row in eval_rows],
                y=[
                    row.eval.win_rate * 100.0
                    for row in eval_rows
                    if row.eval is not None
                ],
                color=theme.WIN_RAW,
                mode="lines+markers",
                hover_format=".1f",
            ),
            ChartSeries(
                name="EWMA",
                x=[point[0] for point in ewma],
                y=[point[1] for point in ewma],
                color=theme.WIN_COLOR,
                hover_format=".1f",
            ),
        ],
        markers=markers,
        lines=lines,
        y_range=(0.0, 100.0),
    )


def _opponent_gen_markers(
    eval_rows: list[metrics.IterationMetrics],
) -> list[ChartMarker]:
    """A marker each time the reference-opponent generation increases."""
    markers: list[ChartMarker] = []
    previous_generation: int | None = None
    for row in eval_rows:
        if row.eval is None:
            continue
        generation = row.eval.opponent_generation
        if previous_generation is not None and generation > previous_generation:
            markers.append(
                ChartMarker(
                    iteration=row.iteration,
                    label=f"opp gen {generation}",
                    color=theme.CHALLENGER_MARK,
                )
            )
        previous_generation = generation
    return markers


def _eval_margin_chart(
    rows: list[metrics.IterationMetrics],
    eval_rows: list[metrics.IterationMetrics],
    run_config: config.RunConfigFile | None,
    target_marker: ChartMarker | None,
) -> ChartSpec:
    ewma = convergence.margin_ewma_points(rows, _eval_ewma_alpha(run_config))
    # The margin EWMA resets at each opponent advance like the win rate does,
    # so the same challenger markers explain its sawtooth.
    markers = _opponent_gen_markers(eval_rows) + _setup_markers(rows)
    if target_marker is not None:
        markers.append(target_marker)
    return _spec(
        "strength-eval-margin",
        "Eval margin",
        "Held-out score margin (policy minus best other seat) vs. the reference opponent",
        "margin (pts)",
        [
            ChartSeries(
                name="raw",
                x=[row.iteration for row in eval_rows],
                y=[row.eval.mean_margin for row in eval_rows if row.eval is not None],
                color=_dim(theme.MARGIN_COLOR),
                mode="lines+markers",
            ),
            ChartSeries(
                name="EWMA",
                x=[point[0] for point in ewma],
                y=[point[1] for point in ewma],
                color=theme.MARGIN_COLOR,
            ),
        ],
        markers=markers,
        lines=[ChartLine(value=0.0, label="0", color=theme.TARGET_GRID)],
    )


def _bootstrap_winrate_chart(
    rows: list[metrics.IterationMetrics], target_marker: ChartMarker | None
) -> ChartSpec:
    x_values, y_values = _xy(
        rows,
        lambda row: (
            None if row.collection_win_rate is None else row.collection_win_rate * 100.0
        ),
    )
    return _spec(
        "strength-bootstrap-win-rate",
        "Bootstrap collection win rate",
        "Win rate vs. the random agent during the bootstrap phase's collection games",
        "win rate (%)",
        [
            ChartSeries(
                name="collection win rate",
                x=x_values,
                y=y_values,
                color=theme.WIN_RAW,
                mode="lines+markers",
                hover_format=".1f",
            )
        ],
        markers=[target_marker] if target_marker is not None else [],
        y_range=(0.0, 100.0),
    )


#### Score section ####


def _score_section(
    rows: list[metrics.IterationMetrics],
    run_config: config.RunConfigFile | None,
    target_marker: ChartMarker | None,
) -> ChartSection | None:
    if not rows:
        return None
    markers = [target_marker] if target_marker is not None else []
    return ChartSection(
        title="Score",
        blurb="Final-score composition per player-game, and the winning margin.",
        charts=[
            _score_chart(rows, run_config, markers),
            _score_breakdown_chart(rows, markers),
            _winner_breakdown_chart(rows, markers),
            _margin_chart(rows, markers),
        ],
    )


def _score_chart(
    rows: list[metrics.IterationMetrics],
    run_config: config.RunConfigFile | None,
    markers: list[ChartMarker],
) -> ChartSpec:
    ewma = convergence.score_ewma_points(rows, _eval_ewma_alpha(run_config))
    return _spec(
        "score-average-final-score",
        "Average final score",
        "Mean self-play final score per player-game — raw and EWMA-smoothed",
        "score (pts)",
        [
            ChartSeries(
                name="raw",
                x=[row.iteration for row in rows],
                y=[row.avg_self_score for row in rows],
                color=_dim(theme.POINTS_COLOR),
                mode="markers",
            ),
            ChartSeries(
                name="EWMA",
                x=[point[0] for point in ewma],
                y=[point[1] for point in ewma],
                color=theme.POINTS_COLOR,
            ),
        ],
        markers=markers,
    )


def _score_breakdown_chart(
    rows: list[metrics.IterationMetrics], markers: list[ChartMarker]
) -> ChartSpec:
    return _spec(
        "score-breakdown-all-seats",
        "Score breakdown (all seats)",
        "Mean final-score split across the six scoring sources, every player-game",
        "score (pts)",
        _breakdown_series(rows, "score", lambda row: row.avg_breakdown),
        markers=markers,
    )


def _winner_breakdown_chart(
    rows: list[metrics.IterationMetrics], markers: list[ChartMarker]
) -> ChartSpec:
    return _spec(
        "score-winner-breakdown",
        "Winner's score breakdown",
        "Mean final-score split of the winning seat only, over decided games",
        "score (pts)",
        _breakdown_series(rows, "winner-score", lambda row: row.avg_winner_breakdown),
        markers=markers,
    )


def _breakdown_series(
    rows: list[metrics.IterationMetrics],
    stack_group: str,
    breakdown_of: typing.Callable[[metrics.IterationMetrics], metrics.ScoreBreakdown],
) -> list[ChartSeries]:
    """The six stacked score-component series, in ``metrics.SCORE_COMPONENTS``
    order, colored via ``theme.SCORE_COLOR`` (the same six colors the live
    dashboard's stacked score bar uses)."""
    iterations = [row.iteration for row in rows]
    return [
        ChartSeries(
            name=name.capitalize(),
            x=iterations,
            y=[getattr(breakdown_of(row), name) for row in rows],
            color=theme.SCORE_COLOR[name],
            stack_group=stack_group,
            fill=True,
        )
        for name in metrics.SCORE_COMPONENTS
    ]


def _margin_chart(
    rows: list[metrics.IterationMetrics], markers: list[ChartMarker]
) -> ChartSpec:
    iterations = [row.iteration for row in rows]
    return _spec(
        "score-winning-margin",
        "Winning margin",
        "Mean winning margin, its per-cycle spread, and the signed margin (~0 by self-play symmetry)",
        "margin (pts)",
        [
            ChartSeries(
                name="|margin|",
                x=iterations,
                y=[row.avg_abs_margin for row in rows],
                color=theme.MARGIN_COLOR,
            ),
            ChartSeries(
                name="|margin| σ",
                x=iterations,
                y=[row.abs_margin_std for row in rows],
                color=_dim(theme.MARGIN_COLOR),
                dash="dash",
            ),
            ChartSeries(
                name="signed margin",
                x=iterations,
                y=[row.avg_margin for row in rows],
                color=theme.SPARK_COLOR,
            ),
        ],
        markers=markers,
        lines=[ChartLine(value=0.0, label="0", color=theme.TARGET_GRID)],
    )


#### Losses section ####

# Loss-component colors — no dedicated theme constants exist for the training
# losses, so these reuse convergence-chart accents that read distinctly
# together without touching the reserved GOOD/CAUTION/BAD verdict colors.
_LOSS_TOTAL_COLOR = theme.POINTS_COLOR
_LOSS_POLICY_COLOR = theme.WIN_COLOR
_LOSS_VALUE_COLOR = theme.MARGIN_COLOR
_LOSS_IMITATION_COLOR = theme.SETUP_MARK


def _losses_section(rows: list[metrics.IterationMetrics]) -> ChartSection | None:
    if not rows:
        return None
    return ChartSection(
        title="Losses",
        blurb="The optimizer's per-iteration loss components and diagnostics.",
        charts=[
            _loss_components_chart(rows),
            _entropy_chart(rows),
            _grad_norm_chart(rows),
            _advantages_chart(rows),
            _ppo_diagnostics_chart(rows),
        ],
    )


def _loss_components_chart(rows: list[metrics.IterationMetrics]) -> ChartSpec:
    iterations = [row.iteration for row in rows]
    series = [
        ChartSeries(
            name="loss",
            x=iterations,
            y=[row.loss for row in rows],
            color=_LOSS_TOTAL_COLOR,
        ),
        ChartSeries(
            name="policy loss",
            x=iterations,
            y=[row.policy_loss for row in rows],
            color=_LOSS_POLICY_COLOR,
        ),
        ChartSeries(
            name="value loss",
            x=iterations,
            y=[row.value_loss for row in rows],
            color=_LOSS_VALUE_COLOR,
        ),
    ]
    if any(row.imitation_loss is not None for row in rows):
        x_values, y_values = _xy(rows, lambda row: row.imitation_loss)
        series.append(
            ChartSeries(
                name="imitation loss",
                x=x_values,
                y=y_values,
                color=_LOSS_IMITATION_COLOR,
            )
        )
    return _spec(
        "losses-loss-components",
        "Loss components",
        "The combined loss and its policy/value (and, during DAgger, imitation) parts",
        "loss",
        series,
    )


def _entropy_chart(rows: list[metrics.IterationMetrics]) -> ChartSpec:
    iterations = [row.iteration for row in rows]
    return _spec(
        "losses-entropy",
        "Entropy",
        "Policy entropy — the exploration bonus's raw input, before its coefficient",
        "entropy (nats)",
        [
            ChartSeries(
                name="entropy",
                x=iterations,
                y=[row.entropy for row in rows],
                color=theme.SPARK_COLOR,
            )
        ],
    )


def _grad_norm_chart(rows: list[metrics.IterationMetrics]) -> ChartSpec:
    iterations = [row.iteration for row in rows]
    return _spec(
        "losses-grad-norm",
        "Gradient norm",
        "Global gradient norm before clipping",
        "grad norm",
        [
            ChartSeries(
                name="grad norm",
                x=iterations,
                y=[row.grad_norm for row in rows],
                color=theme.TEXT_DIM2,
            )
        ],
    )


def _advantages_chart(rows: list[metrics.IterationMetrics]) -> ChartSpec:
    iterations = [row.iteration for row in rows]
    return _spec(
        "losses-advantages",
        "Advantages",
        "Per-cycle advantage mean and spread, after whitening",
        "advantage",
        [
            ChartSeries(
                name="mean",
                x=iterations,
                y=[row.advantage_mean for row in rows],
                color=theme.POINTS_COLOR,
            ),
            ChartSeries(
                name="σ",
                x=iterations,
                y=[row.advantage_std for row in rows],
                color=_dim(theme.POINTS_COLOR),
                dash="dash",
            ),
        ],
    )


def _ppo_diagnostics_chart(rows: list[metrics.IterationMetrics]) -> ChartSpec:
    iterations = [row.iteration for row in rows]
    return _spec(
        "losses-ppo-diagnostics",
        "PPO diagnostics",
        "Clipped-fraction and approximate KL from the reuse-path update (0 on REINFORCE)",
        "clip fraction",
        [
            ChartSeries(
                name="clip fraction",
                x=iterations,
                y=[row.clip_fraction for row in rows],
                color=theme.WIN_COLOR,
            ),
            ChartSeries(
                name="approx KL",
                x=iterations,
                y=[row.approx_kl for row in rows],
                color=theme.MARGIN_COLOR,
                y_axis="y2",
            ),
        ],
        y2_label="approx KL (nats)",
    )


#### Anneal section ####


def _anneal_section(rows: list[metrics.IterationMetrics]) -> ChartSection | None:
    charts: list[ChartSpec] = []
    if any(row.entropy_coef is not None for row in rows):
        x_values, y_values = _xy(rows, lambda row: row.entropy_coef)
        charts.append(
            _spec(
                "anneal-entropy-coefficient",
                "Entropy coefficient",
                "The main net's effective entropy-bonus coefficient this iteration",
                "entropy coef",
                [
                    ChartSeries(
                        name="entropy coef",
                        x=x_values,
                        y=y_values,
                        color=theme.SPARK_COLOR,
                    )
                ],
            )
        )
    if any(row.dropout_p is not None for row in rows):
        x_values, y_values = _xy(rows, lambda row: row.dropout_p)
        charts.append(
            _spec(
                "anneal-dropout-p",
                "Dropout p",
                "The main net's effective global-dropout probability this iteration",
                "dropout p",
                [
                    ChartSeries(
                        name="dropout p",
                        x=x_values,
                        y=y_values,
                        color=theme.MARGIN_COLOR,
                    )
                ],
            )
        )
    if not charts:
        return None
    return ChartSection(
        title="Anneal",
        blurb="Scheduled knobs that taper from an initial value to a final one over the run.",
        charts=charts,
    )


#### Setup model section ####


def _setup_section(rows: list[metrics.IterationMetrics]) -> ChartSection | None:
    if not any(row.setup_loss is not None for row in rows):
        return None
    markers = _setup_markers(rows)
    return ChartSection(
        title="Setup model",
        blurb="The separate setup network's offline-fit / on-policy update readouts.",
        charts=[
            _setup_loss_chart(rows, markers),
            _setup_margins_chart(rows, markers),
            _setup_samples_chart(rows, markers),
        ],
    )


def _setup_loss_chart(
    rows: list[metrics.IterationMetrics], markers: list[ChartMarker]
) -> ChartSpec:
    x_values, y_values = _xy(rows, lambda row: row.setup_loss)
    return _spec(
        "setup-model-loss",
        "Setup loss",
        "Mean MSE over the setup net's update minibatches (normalized target)",
        "loss",
        [
            ChartSeries(
                name="setup loss", x=x_values, y=y_values, color=_LOSS_VALUE_COLOR
            )
        ],
        markers=markers,
    )


def _setup_margins_chart(
    rows: list[metrics.IterationMetrics], markers: list[ChartMarker]
) -> ChartSpec:
    pred_x, pred_y = _xy(rows, lambda row: row.setup_pred_margin_mean)
    target_x, target_y = _xy(rows, lambda row: row.setup_target_margin_mean)
    realized_x, realized_y = _xy(rows, lambda row: row.setup_realized_margin_mean)
    return _spec(
        "setup-model-margins",
        "Setup margins",
        "The setup critic's predicted margin vs. its regression target vs. the realized margin",
        "margin (pts)",
        [
            ChartSeries(name="predicted", x=pred_x, y=pred_y, color=theme.WIN_COLOR),
            ChartSeries(
                name="target", x=target_x, y=target_y, color=theme.MARGIN_COLOR
            ),
            ChartSeries(
                name="realized", x=realized_x, y=realized_y, color=theme.POINTS_COLOR
            ),
        ],
        markers=markers,
    )


def _setup_samples_chart(
    rows: list[metrics.IterationMetrics], markers: list[ChartMarker]
) -> ChartSpec:
    x_values, raw_counts = _xy(
        rows,
        lambda row: (
            None
            if row.setup_samples_recorded is None
            else float(row.setup_samples_recorded)
        ),
    )
    return _spec(
        "setup-model-samples",
        "Setup samples recorded",
        "Setup deals recorded into this iteration's update",
        "samples",
        [
            ChartSeries(
                name="samples",
                x=x_values,
                y=raw_counts,
                color=theme.SPARK_COLOR,
                hover_format=".0f",
            )
        ],
        markers=markers,
    )


#### Decisions section ####


def _decisions_section(rows: list[metrics.IterationMetrics]) -> ChartSection | None:
    if not rows:
        return None
    return ChartSection(
        title="Decisions",
        blurb="How many trainable decisions each game takes, and which judgment families they route to.",
        charts=[_decisions_per_game_chart(rows), _decision_family_share_chart(rows)],
    )


def _decisions_per_game_chart(rows: list[metrics.IterationMetrics]) -> ChartSpec:
    iterations = [row.iteration for row in rows]
    return _spec(
        "decisions-per-game",
        "Decisions per game",
        "Mean trainable decisions per game and its per-cycle spread",
        "decisions",
        [
            ChartSeries(
                name="mean",
                x=iterations,
                y=[row.avg_decisions for row in rows],
                color=theme.POINTS_COLOR,
            ),
            ChartSeries(
                name="σ",
                x=iterations,
                y=[row.decisions_std for row in rows],
                color=_dim(theme.POINTS_COLOR),
                dash="dash",
            ),
        ],
    )


def _decision_family_share_chart(rows: list[metrics.IterationMetrics]) -> ChartSpec:
    iterations = [row.iteration for row in rows]
    series = [
        ChartSeries(
            name=family.value,
            x=iterations,
            y=[_family_share(row, index) for row in rows],
            color=_FAMILY_COLORS[index],
            stack_group="family-share",
            fill=True,
            hover_format=".1%",
        )
        for index, family in enumerate(decisions.ALL_DECISION_FAMILIES)
    ]
    return _spec(
        "decisions-family-share",
        "Decision family share",
        "Each judgment family's share of this iteration's decisions",
        "share of decisions",
        series,
        y_range=(0.0, 1.0),
    )


def _family_share(row: metrics.IterationMetrics, family_index: int) -> float:
    total = row.family_counts.total()
    if total <= 0:
        return 0.0
    return row.family_counts.counts[family_index] / total


#### Throughput section ####


def _throughput_section(rows: list[metrics.IterationMetrics]) -> ChartSection | None:
    if not rows:
        return None
    return ChartSection(
        title="Throughput",
        blurb="Wall-clock cost per iteration and cumulative training progress.",
        charts=[
            _seconds_per_iteration_chart(rows),
            _games_per_second_chart(rows),
            _cumulative_games_chart(rows),
        ],
    )


def _seconds_per_iteration_chart(rows: list[metrics.IterationMetrics]) -> ChartSpec:
    iterations = [row.iteration for row in rows]
    phases: tuple[
        tuple[str, str, typing.Callable[[metrics.IterationMetrics], float]], ...
    ] = (
        ("collect", theme.WIN_COLOR, lambda row: row.collect_seconds),
        ("update", theme.POINTS_COLOR, lambda row: row.update_seconds),
        ("eval", theme.MARGIN_COLOR, lambda row: row.eval_seconds),
        ("probe", theme.SETUP_MARK, lambda row: row.probe_seconds),
    )
    series = [
        ChartSeries(
            name=name,
            x=iterations,
            y=[value_of(row) for row in rows],
            color=color,
            stack_group="phase-seconds",
            fill=True,
        )
        for name, color, value_of in phases
    ]
    return _spec(
        "throughput-seconds-per-iteration",
        "Seconds per iteration",
        "Wall-clock time spent collecting, updating, evaluating, and probing",
        "seconds",
        series,
    )


def _games_per_second_chart(rows: list[metrics.IterationMetrics]) -> ChartSpec:
    iterations = [row.iteration for row in rows]
    return _spec(
        "throughput-games-per-second",
        "Games per second",
        "Self-play collection throughput",
        "games / sec",
        [
            ChartSeries(
                name="games/sec",
                x=iterations,
                y=[row.games_per_sec for row in rows],
                color=theme.WIN_COLOR,
            )
        ],
    )


def _cumulative_games_chart(rows: list[metrics.IterationMetrics]) -> ChartSpec:
    iterations = [row.iteration for row in rows]
    cumulative_hours: list[float] = []
    running_seconds = 0.0
    for row in rows:
        running_seconds += (
            row.collect_seconds
            + row.update_seconds
            + row.eval_seconds
            + row.probe_seconds
        )
        cumulative_hours.append(running_seconds / _SECONDS_PER_HOUR)
    return _spec(
        "throughput-cumulative-games",
        "Cumulative games",
        "Total self-play games played, and cumulative wall-clock time spent",
        "total games",
        [
            ChartSeries(
                name="total games",
                x=iterations,
                y=[float(row.total_games) for row in rows],
                color=theme.POINTS_COLOR,
                hover_format=".0f",
            ),
            ChartSeries(
                name="wall-clock hours",
                x=iterations,
                y=cumulative_hours,
                color=theme.MARGIN_COLOR,
                y_axis="y2",
                hover_format=".1f",
            ),
        ],
        y2_label="wall-clock hours",
    )


#### Representation probe section ####


def _representation_section(
    rows: list[metrics.IterationMetrics],
) -> ChartSection | None:
    probed = [row for row in rows if row.representation is not None]
    if not probed:
        return None
    charts = [
        chart
        for chart in (
            _layer_metric_chart(
                probed,
                "representation-layer-rank95-width",
                "Layer rank95 / width",
                lambda layer: layer.rank95_over_width,
                "rank95 / width",
            ),
            _layer_metric_chart(
                probed,
                "representation-dead-unit-fraction",
                "Dead unit fraction",
                lambda layer: layer.dead_fraction,
                "dead fraction",
            ),
            _layer_metric_chart(
                probed,
                "representation-linear-r2",
                "Linear R²",
                lambda layer: layer.linear_r2,
                "linear R²",
            ),
            _attention_entropy_chart(probed),
            _ablation_kl_chart(probed),
        )
        if chart is not None
    ]
    if not charts:
        return None
    return ChartSection(
        title="Representation probe",
        blurb=(
            "Architecture-probe readouts (docs/TRAINING.md §6.5): per-layer capacity, "
            "board-attention behavior, and ablation sensitivity, at each probed iteration."
        ),
        charts=charts,
    )


def _layer_metric_chart(
    probed: list[metrics.IterationMetrics],
    chart_id: str,
    title: str,
    value_of: typing.Callable[[analysis_models.LayerSummary], float],
    y_label: str,
) -> ChartSpec | None:
    series: list[ChartSeries] = []
    for index, name in enumerate(_layer_names(probed)):
        x_values: list[int] = []
        y_values: list[float] = []
        for row in probed:
            layer = _layer_by_name(row, name)
            if layer is not None:
                x_values.append(row.iteration)
                y_values.append(value_of(layer))
        if x_values:
            series.append(
                ChartSeries(
                    name=name,
                    x=x_values,
                    y=y_values,
                    color=_FAMILY_COLORS[index % len(_FAMILY_COLORS)],
                )
            )
    if not series:
        return None
    return _spec(
        chart_id,
        title,
        "Per-layer capacity readout from the architecture probe",
        y_label,
        series,
    )


def _layer_names(probed: list[metrics.IterationMetrics]) -> list[str]:
    """Layer names in order of first appearance across the probed rows."""
    names: list[str] = []
    seen: set[str] = set()
    for row in probed:
        if row.representation is None:
            continue
        for layer in row.representation.layers:
            if layer.name not in seen:
                seen.add(layer.name)
                names.append(layer.name)
    return names


def _layer_by_name(
    row: metrics.IterationMetrics, name: str
) -> analysis_models.LayerSummary | None:
    if row.representation is None:
        return None
    for layer in row.representation.layers:
        if layer.name == name:
            return layer
    return None


def _attention_entropy_chart(
    probed: list[metrics.IterationMetrics],
) -> ChartSpec | None:
    max_heads = max(
        (
            len(row.representation.attention_entropy_median)
            for row in probed
            if row.representation is not None
        ),
        default=0,
    )
    series: list[ChartSeries] = []
    for head in range(max_heads):
        x_values: list[int] = []
        y_values: list[float] = []
        for row in probed:
            if row.representation is None:
                continue
            values = row.representation.attention_entropy_median
            if head < len(values):
                x_values.append(row.iteration)
                y_values.append(values[head])
        if x_values:
            series.append(
                ChartSeries(
                    name=f"head {head}",
                    x=x_values,
                    y=y_values,
                    color=_FAMILY_COLORS[head % len(_FAMILY_COLORS)],
                )
            )
    if not series:
        return None
    return _spec(
        "representation-attention-entropy",
        "Attention entropy median",
        "Median board-attention entropy per head, normalized by log(n_filled)",
        "entropy (normalized)",
        series,
    )


def _ablation_kl_chart(probed: list[metrics.IterationMetrics]) -> ChartSpec | None:
    uniform_kl = _xy(
        probed,
        lambda row: row.representation.uniform_kl if row.representation else None,
    )
    zero_kl = _xy(
        probed, lambda row: row.representation.zero_kl if row.representation else None
    )
    uniform_flip = _xy(
        probed,
        lambda row: (
            row.representation.uniform_flip_rate if row.representation else None
        ),
    )
    zero_flip = _xy(
        probed,
        lambda row: row.representation.zero_flip_rate if row.representation else None,
    )
    series: list[ChartSeries] = []
    if uniform_kl[0]:
        series.append(
            ChartSeries(
                name="uniform KL",
                x=uniform_kl[0],
                y=uniform_kl[1],
                color=theme.WIN_COLOR,
            )
        )
    if zero_kl[0]:
        series.append(
            ChartSeries(
                name="zero KL", x=zero_kl[0], y=zero_kl[1], color=theme.MARGIN_COLOR
            )
        )
    if uniform_flip[0]:
        series.append(
            ChartSeries(
                name="uniform flip rate",
                x=uniform_flip[0],
                y=uniform_flip[1],
                color=theme.POINTS_COLOR,
                dash="dash",
                y_axis="y2",
                hover_format=".1%",
            )
        )
    if zero_flip[0]:
        series.append(
            ChartSeries(
                name="zero flip rate",
                x=zero_flip[0],
                y=zero_flip[1],
                color=theme.SPARK_COLOR,
                dash="dash",
                y_axis="y2",
                hover_format=".1%",
            )
        )
    if not series:
        return None
    return _spec(
        "representation-ablation-kl",
        "Ablation KL",
        "Policy KL and greedy-flip rate from zeroing/uniformizing board attention",
        "KL (nats)",
        series,
        y2_label="flip rate",
    )


#### HTML rendering ####


def _render_header_html(header: RunHeader) -> str:
    best_eval = (
        f"{header.best_eval_win_rate * 100.0:.1f}% @ iter {header.best_eval_iteration:,}"
        if header.best_eval_win_rate is not None
        and header.best_eval_iteration is not None
        else "—"
    )
    tiles = [
        ("Run", header.run_name),
        ("Version", header.version),
        ("Git SHA", header.git_sha[:12] if header.git_sha else "—"),
        ("Started", header.started_at or "—"),
        ("Iterations", f"{header.iterations:,}"),
        ("Total games", f"{header.total_games:,}"),
        ("Wall clock", f"{header.wall_clock_hours:.1f} h"),
        (
            (
                f"Best eval win rate (opp gen {header.best_eval_generation})"
                if header.best_eval_generation is not None
                else "Best eval win rate"
            ),
            best_eval,
        ),
    ]
    stats_html = "".join(_render_stat_tile(label, value) for label, value in tiles)
    final_eval_html = (
        _render_final_eval_html(header.final_eval)
        if header.final_eval is not None
        else ""
    )
    return (
        f"<h1>{html_lib.escape(header.run_name)}</h1>"
        f'<div class="header-stats">{stats_html}</div>'
        f"{final_eval_html}"
    )


def _render_stat_tile(label: str, value: str) -> str:
    return (
        '<div class="stat-tile">'
        f'<div class="stat-label">{html_lib.escape(label)}</div>'
        f'<div class="stat-value">{html_lib.escape(value)}</div>'
        "</div>"
    )


def _render_final_eval_html(final_eval: metrics.FinalEvalStats) -> str:
    rows_html = "".join(
        f"<tr><td>{html_lib.escape(name.capitalize())}</td><td>{all_value:.2f}</td>"
        f"<td>{winner_value:.2f}</td></tr>"
        for (name, all_value), (_, winner_value) in zip(
            final_eval.avg_breakdown.components(),
            final_eval.avg_winner_breakdown.components(),
        )
    )
    stats_html = "".join(
        _render_stat_tile(label, value)
        for label, value in (
            ("Games", f"{final_eval.n_games:,}"),
            ("Decisions/game", f"{final_eval.decisions_per_game:.1f}"),
            ("Mean margin", f"{final_eval.mean_margin:.2f}"),
            ("Self-play win rate", f"{final_eval.self_play_win_rate * 100.0:.1f}%"),
            ("At iteration", f"{final_eval.at_iteration:,}"),
        )
    )
    return (
        "<h2>Final eval</h2>"
        '<table class="final-eval">'
        "<thead><tr><th>Component</th><th>All seats</th><th>Winner</th></tr></thead>"
        f"<tbody>{rows_html}"
        f"<tr><td>Total</td><td>{final_eval.avg_breakdown.total:.2f}</td>"
        f"<td>{final_eval.avg_winner_breakdown.total:.2f}</td></tr>"
        "</tbody></table>"
        f'<div class="header-stats">{stats_html}</div>'
    )


def _render_section_html(section: ChartSection) -> str:
    charts_html = "".join(_render_chart_card_html(chart) for chart in section.charts)
    return (
        f"<h2>{html_lib.escape(section.title)}</h2>"
        f'<p class="blurb">{html_lib.escape(section.blurb)}</p>'
        f'<div class="chart-grid">{charts_html}</div>'
    )


def _render_chart_card_html(chart: ChartSpec) -> str:
    return (
        '<div class="chart-card">'
        f"<h3>{html_lib.escape(chart.title)}</h3>"
        f'<p class="subtitle">{html_lib.escape(chart.subtitle)}</p>'
        f'<div class="chart-plot" id="{html_lib.escape(chart.id)}"></div>'
        "</div>"
    )


_PAGE_TEMPLATE = """<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>__TITLE__</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    padding: 24px;
    background: __CANVAS__;
    color: __TEXT_PRIMARY__;
    font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
    line-height: 1.4;
  }
  h1 { font-size: 1.5rem; margin: 0 0 4px; }
  h2 {
    font-size: 1.15rem;
    color: __ACCENT__;
    border-bottom: 1px solid __BORDER__;
    padding-bottom: 4px;
    margin: 32px 0 4px;
  }
  h3 { font-size: 0.95rem; margin: 0 0 2px; }
  .subtitle { color: __TEXT_MUTED__; font-size: 0.78rem; margin: 0 0 8px; }
  .blurb { color: __TEXT_MUTED__; font-size: 0.85rem; margin: 4px 0 12px; }
  .header-stats { display: flex; flex-wrap: wrap; gap: 12px; margin: 12px 0 20px; }
  .stat-tile {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid __BORDER__;
    border-radius: 6px;
    padding: 8px 12px;
    min-width: 120px;
  }
  .stat-label {
    color: __TEXT_MUTED__;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  .stat-value { font-size: 1.1rem; font-weight: 600; }
  table.final-eval {
    border-collapse: collapse;
    margin: 8px 0 20px;
    font-size: 0.85rem;
  }
  table.final-eval th,
  table.final-eval td {
    border: 1px solid __BORDER__;
    padding: 4px 10px;
    text-align: right;
  }
  table.final-eval th:first-child,
  table.final-eval td:first-child {
    text-align: left;
  }
  .chart-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(560px, 1fr));
    gap: 16px;
  }
  .chart-card {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid __BORDER__;
    border-radius: 6px;
    padding: 12px;
    min-width: 0;
  }
  .chart-plot { width: 100%; height: 360px; }
  .notice { color: __TEXT_MUTED__; font-style: italic; }
  .offline-warning {
    display: none;
    color: __TEXT_MUTED__;
    font-style: italic;
    border: 1px solid __BORDER__;
    border-radius: 6px;
    padding: 8px 12px;
    margin: 12px 0;
  }
  noscript p {
    color: __TEXT_MUTED__;
    font-style: italic;
  }
  @media (max-width: 640px) {
    body { padding: 12px 16px; }
    .chart-grid { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>
__HEADER_HTML__
__NOTICE_HTML__
<noscript><p>Charts need internet access to load Plotly from its CDN.</p></noscript>
<div id="offline-warning" class="offline-warning">
  Charts need internet access to load Plotly from its CDN.
</div>
__SECTIONS_HTML__
<script type="application/json" id="summary-data">__PAYLOAD_JSON__</script>
<script src="__PLOTLY_CDN__"></script>
<script>
(function () {
  if (typeof Plotly === "undefined") {
    var warning = document.getElementById("offline-warning");
    if (warning) {
      warning.style.display = "block";
    }
    return;
  }
  var dataEl = document.getElementById("summary-data");
  if (!dataEl) {
    return;
  }
  var payload = JSON.parse(dataEl.textContent);
  var canvas = "__CANVAS__";
  var textPrimary = "__TEXT_PRIMARY__";
  var gridColor = "__BORDER__";

  function traceFor(series) {
    return {
      type: "scatter",
      name: series.name,
      x: series.x,
      y: series.y,
      mode: series.mode,
      yaxis: series.y_axis,
      line: { color: series.color, dash: series.dash, width: 2 },
      marker: { color: series.color, size: 6 },
      stackgroup: series.stack_group || undefined,
      fill: series.fill ? (series.stack_group ? "tonexty" : "tozeroy") : undefined,
      hovertemplate:
        "<b>" + series.name + "</b>  %{y:" + series.hover_format + "}<extra></extra>",
    };
  }

  function markerShapes(chart) {
    var shapes = [];
    var annotations = [];
    var markers = (chart.markers || []).slice().sort(function (a, b) {
      return a.iteration - b.iteration;
    });
    markers.forEach(function (marker, index) {
      shapes.push({
        type: "line",
        xref: "x",
        yref: "paper",
        x0: marker.iteration,
        x1: marker.iteration,
        y0: 0,
        y1: 1,
        line: { color: marker.color, width: 1, dash: "dot" },
      });
      // Labels sit in the top margin, alternating between two rows so a
      // cluster of nearby markers (e.g. several early opponent advances)
      // stays legible instead of overprinting.
      annotations.push({
        x: marker.iteration,
        y: 1,
        yref: "paper",
        yshift: index % 2 === 0 ? 4 : 16,
        text: marker.label,
        showarrow: false,
        font: { color: marker.color, size: 10 },
        xanchor: "left",
        yanchor: "bottom",
      });
    });
    (chart.lines || []).forEach(function (line) {
      shapes.push({
        type: "line",
        xref: "paper",
        yref: "y",
        x0: 0,
        x1: 1,
        y0: line.value,
        y1: line.value,
        line: { color: line.color, width: 1, dash: "dash" },
      });
      // Reference-line labels sit just inside the right edge of the plot
      // (anchored right) so they are never clipped by the plot margin.
      annotations.push({
        x: 1,
        xref: "paper",
        y: line.value,
        yref: "y",
        xshift: -4,
        yshift: 2,
        text: line.label,
        showarrow: false,
        font: { color: line.color, size: 10 },
        xanchor: "right",
        yanchor: "bottom",
      });
    });
    return { shapes: shapes, annotations: annotations };
  }

  function layoutFor(chart) {
    var refs = markerShapes(chart);
    var layout = {
      paper_bgcolor: canvas,
      plot_bgcolor: canvas,
      font: { color: textPrimary, size: 12 },
      // Top margin holds the two rows of marker labels; bottom margin holds
      // the x-axis title with the legend parked below it (paper y < 0) so the
      // two never overprint.
      margin: { l: 56, r: chart.y2_label ? 56 : 20, t: 32, b: 88 },
      hovermode: "x unified",
      showlegend: true,
      legend: {
        orientation: "h",
        bgcolor: "rgba(0,0,0,0)",
        x: 0,
        xanchor: "left",
        y: -0.22,
        yanchor: "top",
      },
      xaxis: { title: "iteration", gridcolor: gridColor, zeroline: false },
      yaxis: { title: chart.y_label, gridcolor: gridColor, zeroline: false },
      shapes: refs.shapes,
      annotations: refs.annotations,
    };
    if (chart.y_range) {
      layout.yaxis.range = chart.y_range;
    }
    if (chart.y2_label) {
      layout.yaxis2 = {
        title: chart.y2_label,
        overlaying: "y",
        side: "right",
        gridcolor: gridColor,
        zeroline: false,
      };
    }
    return layout;
  }

  var plotConfig = { responsive: true, displaylogo: false };

  payload.sections.forEach(function (section) {
    section.charts.forEach(function (chart) {
      var el = document.getElementById(chart.id);
      if (!el) {
        return;
      }
      var traces = chart.series.map(traceFor);
      Plotly.newPlot(el, traces, layoutFor(chart), plotConfig);
    });
  });
})();
</script>
</body>
</html>
"""
