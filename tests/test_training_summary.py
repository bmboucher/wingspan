"""Tests for the training-progression HTML report (``reporting.training_summary``).

Covers the pure payload builder (:func:`training_summary.build_summary_payload`),
the HTML renderer, the ``write_training_summary`` disk-writing entry point, and
the ``wingspan summary`` CLI (``reporting.summary_cli``).
"""

from __future__ import annotations

import pathlib

import pytest

pytest.importorskip("torch")  # wingspan.training's package __init__ pulls torch

from wingspan import decisions
from wingspan.analysis import models as analysis_models
from wingspan.reporting import summary_cli, training_summary
from wingspan.training import artifacts, config, metrics, runmeta


def _row(
    iteration: int,
    score: float,
    *,
    eval_result: metrics.EvalResult | None = None,
    collection_win_rate: float | None = None,
    setup_phase: str | None = None,
    setup_loss: float | None = None,
    entropy_coef: float | None = None,
    representation: analysis_models.RepresentationMetrics | None = None,
) -> metrics.IterationMetrics:
    """A fabricated iteration row — extends ``test_metrics_log._row`` with the
    optional eval / bootstrap / setup / anneal / probe fields this report reads."""
    family = metrics.FamilyCounts()
    family.bump(iteration % len(decisions.ALL_DECISION_FAMILIES))
    return metrics.IterationMetrics(
        iteration=iteration,
        total_games=(iteration + 1) * 2,
        games_this_iter=2,
        loss=1.0,
        policy_loss=0.5,
        value_loss=0.3,
        entropy=0.6,
        grad_norm=1.5,
        advantage_mean=0.0,
        advantage_std=1.0,
        avg_self_score=score,
        avg_margin=0.0,
        avg_breakdown=metrics.ScoreBreakdown(birds=score),
        avg_decisions=140.0,
        avg_winner_breakdown=metrics.ScoreBreakdown(birds=score),
        avg_abs_margin=1.0,
        margin_std=0.0,
        abs_margin_std=0.0,
        decisions_std=0.0,
        family_counts=family,
        collect_seconds=1.0,
        update_seconds=0.5,
        eval_seconds=1.0,
        games_per_sec=2.0,
        eval=eval_result,
        collection_win_rate=collection_win_rate,
        setup_phase=setup_phase,
        setup_loss=setup_loss,
        setup_pred_margin_mean=1.0 if setup_loss is not None else None,
        setup_target_margin_mean=1.2 if setup_loss is not None else None,
        setup_realized_margin_mean=0.9 if setup_loss is not None else None,
        setup_samples_recorded=10 if setup_loss is not None else None,
        entropy_coef=entropy_coef,
        representation=representation,
    )


def _section_titles(payload: training_summary.SummaryPayload) -> set[str]:
    return {section.title for section in payload.sections}


def _chart(
    payload: training_summary.SummaryPayload, section_title: str, chart_title: str
) -> training_summary.ChartSpec:
    for section in payload.sections:
        if section.title == section_title:
            for chart in section.charts:
                if chart.title == chart_title:
                    return chart
    raise AssertionError(f"no chart {chart_title!r} in section {section_title!r}")


# ---------------------------------------------------------------------------
# build_summary_payload — section presence / absence


def test_minimal_rows_produce_core_sections_only():
    rows = [_row(0, 50.0), _row(1, 55.0)]
    payload = training_summary.build_summary_payload(rows, None, None, "test-run")
    titles = _section_titles(payload)
    assert {"Score", "Losses", "Decisions", "Throughput"} <= titles
    # No eval/collection rows, no anneal, no setup, no probe data.
    assert "Strength" not in titles
    assert "Anneal" not in titles
    assert "Setup model" not in titles
    assert "Representation probe" not in titles


def test_eval_rows_add_strength_section():
    rows = [
        _row(
            0,
            50.0,
            eval_result=metrics.EvalResult(
                n_games=4, win_rate=0.5, ci95=0.1, mean_margin=1.0
            ),
        ),
        _row(
            1,
            55.0,
            eval_result=metrics.EvalResult(
                n_games=4, win_rate=0.6, ci95=0.1, mean_margin=2.0
            ),
        ),
    ]
    payload = training_summary.build_summary_payload(rows, None, None, "test-run")
    assert "Strength" in _section_titles(payload)
    chart = _chart(payload, "Strength", "Eval win rate")
    raw = next(series for series in chart.series if series.name == "raw")
    assert raw.x == [0, 1]
    assert raw.y == [50.0, 60.0]


def test_bootstrap_rows_without_eval_add_bootstrap_chart_only():
    rows = [
        _row(0, 50.0, collection_win_rate=0.4),
        _row(1, 55.0, collection_win_rate=0.6),
    ]
    payload = training_summary.build_summary_payload(rows, None, None, "test-run")
    section = next(
        section for section in payload.sections if section.title == "Strength"
    )
    titles = {chart.title for chart in section.charts}
    assert titles == {"Bootstrap collection win rate"}


def test_anneal_section_present_only_with_entropy_coef():
    rows = [_row(0, 50.0, entropy_coef=0.05), _row(1, 55.0, entropy_coef=0.04)]
    payload = training_summary.build_summary_payload(rows, None, None, "test-run")
    assert "Anneal" in _section_titles(payload)
    section = next(section for section in payload.sections if section.title == "Anneal")
    titles = {chart.title for chart in section.charts}
    assert titles == {"Entropy coefficient"}


def test_setup_section_present_only_with_setup_loss():
    rows = [
        _row(0, 50.0, setup_phase="RANDOM", setup_loss=1.0),
        _row(1, 55.0, setup_phase="MODEL_DRIVEN", setup_loss=0.8),
    ]
    payload = training_summary.build_summary_payload(rows, None, None, "test-run")
    assert "Setup model" in _section_titles(payload)
    section = next(
        section for section in payload.sections if section.title == "Setup model"
    )
    titles = {chart.title for chart in section.charts}
    assert titles == {"Setup loss", "Setup margins", "Setup samples recorded"}
    # The setup-phase transition marks the win-rate chart too, once eval exists.
    markers = _chart(payload, "Setup model", "Setup loss").markers
    assert any(marker.iteration == 1 for marker in markers)


def test_representation_section_present_only_with_probe_data():
    layer = analysis_models.LayerSummary(
        name="trunk.L0",
        in_features=64,
        out_features=64,
        rank95=32,
        rank95_over_width=0.5,
        linear_r2=0.9,
        dead_fraction=0.1,
    )
    representation = analysis_models.RepresentationMetrics(
        probe_decisions=256,
        layers=[layer],
        attention_entropy_median=[0.8, 0.7],
        uniform_kl=0.05,
        zero_kl=0.1,
        uniform_flip_rate=0.01,
        zero_flip_rate=0.02,
    )
    rows = [_row(0, 50.0), _row(1, 55.0, representation=representation)]
    payload = training_summary.build_summary_payload(rows, None, None, "test-run")
    assert "Representation probe" in _section_titles(payload)
    section = next(
        section
        for section in payload.sections
        if section.title == "Representation probe"
    )
    titles = {chart.title for chart in section.charts}
    assert "Layer rank95 / width" in titles
    assert "Attention entropy median" in titles
    assert "Ablation KL" in titles
    layer_chart = _chart(payload, "Representation probe", "Layer rank95 / width")
    assert layer_chart.series[0].x == [1]  # only the probed iteration


def test_empty_rows_produce_no_sections():
    payload = training_summary.build_summary_payload([], None, None, "test-run")
    assert payload.sections == []
    assert payload.header.iterations == 0
    assert payload.header.total_games == 0


# ---------------------------------------------------------------------------
# Opponent-advance markers + family-share fractions


def test_opponent_generation_increase_adds_marker():
    rows = [
        _row(
            0,
            50.0,
            eval_result=metrics.EvalResult(
                n_games=4,
                win_rate=0.5,
                ci95=0.1,
                mean_margin=0.0,
                opponent_generation=0,
            ),
        ),
        _row(
            1,
            52.0,
            eval_result=metrics.EvalResult(
                n_games=4,
                win_rate=0.9,
                ci95=0.1,
                mean_margin=1.0,
                opponent_generation=0,
            ),
        ),
        _row(
            2,
            51.0,
            eval_result=metrics.EvalResult(
                n_games=4,
                win_rate=0.5,
                ci95=0.1,
                mean_margin=0.0,
                opponent_generation=1,
            ),
        ),
    ]
    payload = training_summary.build_summary_payload(rows, None, None, "test-run")
    chart = _chart(payload, "Strength", "Eval win rate")
    gen_markers = [marker for marker in chart.markers if marker.label == "opp gen 1"]
    assert len(gen_markers) == 1
    assert gen_markers[0].iteration == 2


def test_family_share_fractions_sum_to_one():
    rows = [_row(0, 50.0), _row(1, 55.0), _row(2, 60.0)]
    payload = training_summary.build_summary_payload(rows, None, None, "test-run")
    chart = _chart(payload, "Decisions", "Decision family share")
    assert len(chart.series) == len(decisions.ALL_DECISION_FAMILIES)
    for point_index in range(len(rows)):
        total = sum(series.y[point_index] for series in chart.series)
        assert total == pytest.approx(1.0)


def test_target_marker_appears_when_configured():
    run_config = _run_config_file(target_iterations=5)
    rows = [_row(0, 50.0), _row(1, 55.0)]
    payload = training_summary.build_summary_payload(rows, run_config, None, "test-run")
    score_chart = _chart(payload, "Score", "Average final score")
    assert any(
        marker.label == "target" and marker.iteration == 5
        for marker in score_chart.markers
    )


# ---------------------------------------------------------------------------
# HTML rendering


def test_build_training_summary_html_embeds_cdn_and_json():
    rows = [_row(0, 50.0)]
    payload = training_summary.build_summary_payload(rows, None, None, "test-run")
    html_content = training_summary.build_training_summary_html(payload)
    assert "https://cdn.plot.ly/plotly-2.35.2.min.js" in html_content
    assert 'id="summary-data"' in html_content
    # The embedded payload round-trips (validated before embedding, so this
    # exercises the same JSON the escape step operates on).
    assert (
        training_summary.SummaryPayload.model_validate_json(payload.model_dump_json())
        == payload
    )


def test_build_training_summary_html_notice_on_empty_payload():
    payload = training_summary.build_summary_payload([], None, None, "test-run")
    html_content = training_summary.build_training_summary_html(payload)
    assert "No iterations logged yet" in html_content


# ---------------------------------------------------------------------------
# write_training_summary


def test_write_training_summary_full(tmp_path: pathlib.Path):
    log_path = tmp_path / artifacts.METRICS_LOG
    with open(log_path, "a", encoding="utf-8") as handle:
        for row in (_row(0, 50.0), _row(1, 55.0)):
            handle.write(row.model_dump_json() + "\n")

    runmeta.write_run_config(
        str(tmp_path),
        config.RunConfig(),
        stamp="20260101-000000",
        started_at="2026-01-01T00:00:00",
        git_sha="abc123def456",
        resumed_from_iteration=0,
    )

    final_eval = metrics.FinalEvalStats(
        n_games=10,
        avg_breakdown=metrics.ScoreBreakdown(birds=20.0),
        avg_winner_breakdown=metrics.ScoreBreakdown(birds=22.0),
        decisions_per_game=140.0,
        mean_margin=3.0,
        self_play_win_rate=0.5,
        at_iteration=2,
    )
    (tmp_path / artifacts.final_eval_name(2)).write_text(
        final_eval.model_dump_json(), encoding="utf-8"
    )

    written = training_summary.write_training_summary(str(tmp_path))
    assert written == tmp_path / artifacts.TRAINING_SUMMARY_HTML
    assert written.exists()
    content = written.read_text(encoding="utf-8")
    assert "abc123def456" in content
    assert "Final eval" in content


def _final_eval(at_iteration: int, mean_margin: float) -> metrics.FinalEvalStats:
    return metrics.FinalEvalStats(
        n_games=10,
        avg_breakdown=metrics.ScoreBreakdown(birds=20.0),
        avg_winner_breakdown=metrics.ScoreBreakdown(birds=22.0),
        decisions_per_game=140.0,
        mean_margin=mean_margin,
        self_play_win_rate=0.5,
        at_iteration=at_iteration,
    )


def test_write_training_summary_picks_latest_final_eval_by_iteration(
    tmp_path: pathlib.Path,
):
    # ``final_eval_500.json`` sorts lexically *after* ``final_eval_1_000.json``,
    # so the newest milestone must be chosen by its ``at_iteration``, not its name.
    for at_iteration, margin in ((500, 5.0), (1000, 9.0)):
        (tmp_path / artifacts.final_eval_name(at_iteration)).write_text(
            _final_eval(at_iteration, margin).model_dump_json(), encoding="utf-8"
        )
    content = training_summary.write_training_summary(str(tmp_path)).read_text(
        encoding="utf-8"
    )
    assert '"at_iteration":1000' in content
    assert '"at_iteration":500' not in content


def test_header_best_eval_is_within_latest_opponent_generation():
    def _eval(win_rate: float, generation: int) -> metrics.EvalResult:
        return metrics.EvalResult(
            n_games=4,
            win_rate=win_rate,
            ci95=0.1,
            mean_margin=1.0,
            opponent_generation=generation,
        )

    # 95% vs the random agent (gen 0) must not outrank 60% vs the frozen self.
    rows = [
        _row(0, 50.0, eval_result=_eval(0.95, 0)),
        _row(1, 50.0, eval_result=_eval(0.55, 1)),
        _row(2, 50.0, eval_result=_eval(0.60, 1)),
        _row(3, 50.0, eval_result=_eval(0.58, 1)),
    ]
    header = training_summary.build_summary_payload(rows, None, None, "run").header
    assert header.best_eval_generation == 1
    assert header.best_eval_win_rate == 0.60
    assert header.best_eval_iteration == 2


def test_write_training_summary_empty_dir(tmp_path: pathlib.Path):
    written = training_summary.write_training_summary(str(tmp_path))
    assert written.exists()
    content = written.read_text(encoding="utf-8")
    assert "No iterations logged yet" in content


def test_write_training_summary_out_path(tmp_path: pathlib.Path):
    out_path = tmp_path / "reports" / "custom.html"
    written = training_summary.write_training_summary(str(tmp_path), out_path)
    assert written == out_path
    assert out_path.exists()


# ---------------------------------------------------------------------------
# wingspan summary CLI


def test_summary_cli_writes_default_path(tmp_path: pathlib.Path):
    exit_code = summary_cli.main_summary([str(tmp_path)])
    assert exit_code == 0
    assert (tmp_path / artifacts.TRAINING_SUMMARY_HTML).exists()


def test_summary_cli_missing_dir_returns_1(tmp_path: pathlib.Path):
    missing = tmp_path / "does-not-exist"
    exit_code = summary_cli.main_summary([str(missing)])
    assert exit_code == 1


def test_summary_cli_honors_out_flag(tmp_path: pathlib.Path):
    out_path = tmp_path / "out.html"
    exit_code = summary_cli.main_summary([str(tmp_path), "--out", str(out_path)])
    assert exit_code == 0
    assert out_path.exists()


###### PRIVATE test helpers #######


def _run_config_file(*, target_iterations: int) -> config.RunConfigFile:
    cfg = config.RunConfig(run=config.RunSettings(target_iterations=target_iterations))
    return config.RunConfigFile(
        version=cfg.encoding_version,
        saved_at="2026-01-01T00:00:00",
        started_at="2026-01-01T00:00:00",
        git_sha=None,
        resumed=False,
        resumed_from_iteration=0,
        config=cfg,
    )
