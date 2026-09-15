# pyright: reportPrivateUsage=false
# (exercises dashboard._probe_health_rows directly, the same pattern
# tests/test_loop_iteration_order.py uses for TrainingLoop._run_iteration)
"""Tests for ``wingspan.training.loop_probe`` and its dashboard/metrics wiring.

Covers:

1. ``maybe_probe`` on-cadence: populates a ``RepresentationMetrics`` result
   and pushes a PROBE event.
2. ``maybe_probe`` disabled (``probe_every=0``) / off-cadence: both return
   ``(None, 0.0)``.
3. A pre-existing serialized ``IterationMetrics`` row (missing the
   ``representation`` / ``probe_seconds`` fields) still parses.
4. The new TRAINING HEALTH dashboard rows: 3 rows once history carries a
   probe entry, 0 when it never has, and — with two differently-valued probed
   entries — that the rendered value is the *most recent* probe, that trunk-
   tail selection picks the highest ``L{index}`` (not list order or a
   non-trunk layer), and the mean-dead-fraction arithmetic.
5. ``probe_decisions=0`` is rejected by ``RunSettings`` (it would make
   ``subsample_steps`` return an empty sample and crash
   ``representation.measure``'s ``torch.cat([])`` mid-run).
6. A probe failure (any exception) is caught, alarmed, and reported as a skip
   rather than propagating out of ``maybe_probe`` and ending the run.

``tests/test_loop_iteration_order.py`` separately pins ``loop_probe.maybe_probe``
into ``TrainingLoop._run_iteration``'s expected phase order.
``tests/test_training_configurator.py`` separately pins that the configurator's
``nudge`` / ``commit`` paths for ``probe_decisions`` respect the same bound.
``tests/test_analysis_probe.py`` separately pins that ``representation.measure``
itself restores ``net``'s train/eval mode and uninstalls its attention wrappers
even when a probe pass raises.
"""

from __future__ import annotations

import pathlib
import random

import pydantic
import pytest

from wingspan.analysis import models as analysis_models
from wingspan.analysis import representation
from wingspan.training import (
    collect,
    config,
    dashboard,
    loop,
    loop_probe,
    metrics,
    runstate,
)

_SMALL_LAYERS = (32, 32)
_SMALL_CARD_EMBED_DIM = 8


###### HELPERS #######


def _tiny_loop_config(
    tmp_path: pathlib.Path, *, probe_every: int = 25, probe_decisions: int = 2048
) -> config.RunConfig:
    """A tiny, fast-collecting ``RunConfig``, mirroring
    ``tests/test_loop_iteration_order.py`` / ``tests/test_analysis_probe.py``'s
    tiny-net patterns. ``use_setup_model`` is off — irrelevant to probing, and
    keeps ``TrainingLoop`` construction fast."""
    return config.RunConfig(
        misc=config.MiscConfig(collect_device="cpu", train_device="cpu"),
        run=config.RunSettings(
            checkpoint_dir=str(tmp_path),
            resume=False,
            probe_every=probe_every,
            probe_decisions=probe_decisions,
        ),
        architecture=config.ArchitectureConfig(
            use_setup_model=False,
            main=config.MainNetArchitecture(
                trunk_layers=_SMALL_LAYERS,
                choice_layers=_SMALL_LAYERS,
                card_embed_dim=_SMALL_CARD_EMBED_DIM,
            ),
        ),
    )


def _play_games(
    training_loop: loop.TrainingLoop, n: int, seed: int
) -> list[collect.GameRecord]:
    rng = random.Random(seed)
    return [
        collect.play_game(
            training_loop.net,
            training_loop.train_device,
            rng,
            seed=seed + game_index,
            combine_gain_food=training_loop.config.engine.combine_gain_food,
            num_players=training_loop.config.num_players,
        )
        for game_index in range(n)
    ]


def _fake_representation(
    *, trunk_rank95s: tuple[float, ...] = (0.75, 0.25), uniform_kl: float = 0.01
) -> analysis_models.RepresentationMetrics:
    """A synthetic :class:`~analysis_models.RepresentationMetrics`, standing in
    for a real ``representation.measure`` pass in the dashboard-row tests
    below (those test pure rendering over a canned history, not the probe
    pipeline itself — already covered by the ``maybe_probe`` tests)."""
    layers = [
        analysis_models.LayerSummary(
            name=f"trunk.L{index}",
            in_features=32,
            out_features=32,
            rank95=max(1, round(rank95 * 32)),
            rank95_over_width=rank95,
            linear_r2=0.9,
            dead_fraction=0.1 * (index + 1),
        )
        for index, rank95 in enumerate(trunk_rank95s)
    ]
    return analysis_models.RepresentationMetrics(
        probe_decisions=64,
        layers=layers,
        attention_entropy_median=[0.9],
        uniform_kl=uniform_kl,
        uniform_flip_rate=0.05,
        zero_kl=0.02,
        zero_flip_rate=0.1,
    )


def _iter_metrics(
    iteration: int, representation: analysis_models.RepresentationMetrics | None
) -> metrics.IterationMetrics:
    """A minimal :class:`~metrics.IterationMetrics` row, only ``iteration`` and
    ``representation`` varying, for the dashboard-row tests' synthetic history."""
    return metrics.IterationMetrics(
        iteration=iteration,
        total_games=iteration,
        games_this_iter=1,
        loss=0.0,
        policy_loss=0.0,
        value_loss=0.0,
        entropy=0.0,
        grad_norm=0.0,
        advantage_mean=0.0,
        advantage_std=0.0,
        avg_self_score=0.0,
        avg_margin=0.0,
        avg_breakdown=metrics.ScoreBreakdown(),
        avg_decisions=0.0,
        avg_winner_breakdown=metrics.ScoreBreakdown(),
        avg_abs_margin=0.0,
        margin_std=0.0,
        abs_margin_std=0.0,
        decisions_std=0.0,
        family_counts=metrics.FamilyCounts(),
        collect_seconds=0.0,
        update_seconds=0.0,
        eval_seconds=0.0,
        games_per_sec=0.0,
        representation=representation,
    )


###### 1-2: maybe_probe #######


def test_maybe_probe_on_cadence_populates_representation_metrics(
    tmp_path: pathlib.Path,
) -> None:
    cfg = _tiny_loop_config(tmp_path, probe_every=1)
    training_loop = loop.TrainingLoop(cfg)
    records = _play_games(training_loop, 2, seed=100)

    result, elapsed = loop_probe.maybe_probe(training_loop, 1, records)

    assert isinstance(result, analysis_models.RepresentationMetrics)
    assert result.layers
    assert result.probe_decisions > 0
    assert elapsed >= 0.0
    assert any(
        event.kind == runstate.EventKind.INFO and event.text.startswith("PROBE ")
        for event in training_loop.state.events
    )


def test_maybe_probe_disabled_returns_none(tmp_path: pathlib.Path) -> None:
    cfg = _tiny_loop_config(tmp_path, probe_every=0)
    training_loop = loop.TrainingLoop(cfg)

    assert loop_probe.maybe_probe(training_loop, 1, []) == (None, 0.0)


def test_maybe_probe_off_cadence_returns_none(tmp_path: pathlib.Path) -> None:
    cfg = _tiny_loop_config(tmp_path, probe_every=5)
    training_loop = loop.TrainingLoop(cfg)

    assert loop_probe.maybe_probe(training_loop, 2, []) == (None, 0.0)


def test_maybe_probe_failure_is_non_fatal_and_pushes_alarm(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Any exception inside the probe pass (a CUDA OOM, a linalg error, a
    malformed probe set, ...) must be caught, alarmed, and reported as a skip
    — never propagate out of ``maybe_probe`` and kill the training run."""
    cfg = _tiny_loop_config(tmp_path, probe_every=1)
    training_loop = loop.TrainingLoop(cfg)
    records = _play_games(training_loop, 1, seed=250)

    def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("synthetic probe failure")

    monkeypatch.setattr(representation, "measure", _boom)

    result, elapsed = loop_probe.maybe_probe(training_loop, 1, records)

    assert result is None
    assert elapsed >= 0.0
    assert any(
        event.kind == runstate.EventKind.ALARM
        and "PROBE" in event.text
        and "synthetic probe failure" in event.text
        for event in training_loop.state.events
    )


###### 3: old metrics.jsonl rows still parse #######


def test_iteration_metrics_row_without_probe_fields_still_parses() -> None:
    """A ``metrics.jsonl`` row written before ``representation`` /
    ``probe_seconds`` existed lacks both keys; ``IterationMetrics`` must still
    parse it (both fields are optional/defaulted) so old history files stay
    readable."""
    full_row = _iter_metrics(1, _fake_representation()).model_dump()
    del full_row["representation"]
    del full_row["probe_seconds"]

    parsed = metrics.IterationMetrics.model_validate(full_row)

    assert parsed.representation is None
    assert parsed.probe_seconds == 0.0


###### 4: dashboard rows #######


def test_probe_health_rows_present_once_history_has_a_probe_entry(
    tmp_path: pathlib.Path,
) -> None:
    cfg = _tiny_loop_config(tmp_path)
    training_loop = loop.TrainingLoop(cfg)
    state = training_loop.state
    # The common shape: an earlier iteration probed, the latest one is
    # off-cadence (representation=None) — the rows must still show, reading
    # their "current value" off the most recent *probed* entry.
    state.history.append(_iter_metrics(1, _fake_representation()))
    state.history.append(_iter_metrics(2, None))
    state.last_iter = state.history[-1]

    rows = dashboard._probe_health_rows(state)

    assert len(rows) == 3
    assert [row[0].plain for row in rows] == [
        "trunk tail rank95/width",
        "attn uniform KL",
        "mean dead frac",
    ]


def test_probe_health_rows_use_most_recent_probed_values(
    tmp_path: pathlib.Path,
) -> None:
    """Two probed iterations with different values pin three things at once:
    the rendered "current value" comes from the *most recent* probed entry
    (not the first, and not an average); trunk-tail selection picks the
    highest ``L{index}`` regardless of list order or a non-trunk layer mixed
    in; and the mean-dead-fraction arithmetic covers every layer."""
    older = analysis_models.RepresentationMetrics(
        probe_decisions=32,
        layers=[
            analysis_models.LayerSummary(
                name="trunk.L0",
                in_features=32,
                out_features=32,
                rank95=3,
                rank95_over_width=0.10,
                linear_r2=0.9,
                dead_fraction=0.40,
            ),
            analysis_models.LayerSummary(
                name="trunk.L1",
                in_features=32,
                out_features=32,
                rank95=6,
                rank95_over_width=0.20,
                linear_r2=0.9,
                dead_fraction=0.60,
            ),
        ],
        attention_entropy_median=[0.9],
        uniform_kl=0.05,
    )
    newer = analysis_models.RepresentationMetrics(
        probe_decisions=32,
        # Deliberately out of index order, plus a non-trunk layer: a naive
        # "last in list" or "any layer" implementation would pick the wrong
        # value for the trunk-tail row.
        layers=[
            analysis_models.LayerSummary(
                name="trunk.L2",
                in_features=32,
                out_features=32,
                rank95=29,
                rank95_over_width=0.90,
                linear_r2=0.9,
                dead_fraction=0.10,
            ),
            analysis_models.LayerSummary(
                name="trunk.L0",
                in_features=32,
                out_features=32,
                rank95=10,
                rank95_over_width=0.30,
                linear_r2=0.9,
                dead_fraction=0.20,
            ),
            analysis_models.LayerSummary(
                name="choice.L0",
                in_features=32,
                out_features=32,
                rank95=32,
                rank95_over_width=0.99,
                linear_r2=0.9,
                dead_fraction=0.90,
            ),
        ],
        attention_entropy_median=[0.9],
        uniform_kl=0.15,
    )
    cfg = _tiny_loop_config(tmp_path)
    training_loop = loop.TrainingLoop(cfg)
    state = training_loop.state
    state.history.append(_iter_metrics(1, older))
    state.history.append(_iter_metrics(5, newer))
    state.last_iter = state.history[-1]

    rows = dashboard._probe_health_rows(state)

    assert len(rows) == 3
    values = {row[0].plain: row[1].plain for row in rows}
    # trunk.L2 (highest index) rank95_over_width from `newer` — not L0's 0.30,
    # not choice.L0's 0.99, and not `older`'s 0.20.
    assert values["trunk tail rank95/width"] == "0.9000"
    assert values["attn uniform KL"] == "0.1500"
    # Mean dead_fraction over ALL THREE of `newer`'s layers: (0.10+0.20+0.90)/3.
    assert values["mean dead frac"] == "0.4000"


def test_probe_health_rows_empty_without_a_probe_entry(tmp_path: pathlib.Path) -> None:
    cfg = _tiny_loop_config(tmp_path)
    training_loop = loop.TrainingLoop(cfg)
    state = training_loop.state
    state.history.append(_iter_metrics(1, None))
    state.last_iter = state.history[-1]

    assert dashboard._probe_health_rows(state) == []


###### 5: probe_decisions bound #######


def test_probe_decisions_zero_is_rejected_by_config() -> None:
    """0 would make ``subsample_steps`` return an empty sample -> an empty
    ``ProbeSet`` -> ``representation.measure``'s ``torch.cat([])`` crashes
    mid-run. The field's ``ge=1`` bound (not ``ge=0``) is what makes 0
    unreachable, whether set directly or via the configurator (pinned
    separately in ``tests/test_training_configurator.py``)."""
    with pytest.raises(pydantic.ValidationError, match="probe_decisions"):
        config.RunSettings(probe_decisions=0)
