# pyright: reportPrivateUsage=false
# (drives TrainingLoop._run_iteration directly to pin its phase ordering)
"""Regression pin for ``TrainingLoop._run_iteration``'s phase order.

The setup-model actor-critic update must run before the main net's update and
its following embedder re-sync, so its REINFORCE log-probs stay on-policy
(``setup_learner``'s module docstring). This test replaces every phase's
module-level function with a tag-recording stand-in and asserts the exact
call order, so a future reordering regresses loudly instead of silently
reintroducing the train/sample divergence that produced the observed −149
SETUP AC loss.
"""

from __future__ import annotations

import pathlib

import pytest

from wingspan.training import (
    collect,
    config,
    learner,
    loop,
    loop_checkpoint,
    loop_collect,
    loop_eval,
    loop_setup,
    metrics,
)


def _loop_config(tmp_path: pathlib.Path) -> config.TrainConfig:
    return config.RunConfig(
        misc=config.MiscConfig(device="cpu"),
        run=config.RunSettings(
            checkpoint_dir=str(tmp_path),
            resume=False,
        ),
        architecture=config.ArchitectureConfig(
            use_setup_model=True,
            main=config.MainNetArchitecture(
                trunk_layers=(32, 32),
                choice_layers=(32, 32),
                card_embed_dim=8,
            ),
            setup=config.SetupNetArchitecture(head_layers=(16,)),
        ),
    )


def test_run_iteration_setup_update_precedes_main_update_and_sync(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Constructed before any monkeypatching: __init__ itself calls
    # loop_setup.sync_setup_embedders once (the startup sync), which must run
    # against the real implementation, not the tag-recording stand-in below.
    training = loop.TrainingLoop(_loop_config(tmp_path))

    order: list[str] = []

    canned_record = collect.GameRecord(
        steps=[],
        breakdowns=(metrics.ScoreBreakdown(), metrics.ScoreBreakdown()),
        winner=-1,
        seed=0,
    )
    canned_update_stats = learner.UpdateStats(
        loss=0.0,
        policy_loss=0.0,
        value_loss=0.0,
        entropy=0.0,
        grad_norm=0.0,
        advantage_mean=0.0,
        advantage_std=0.0,
        n_steps=0,
    )
    canned_setup_stats = metrics.SetupUpdateStats(
        loss=0.0,
        pred_margin_mean=0.0,
        target_margin_mean=0.0,
        realized_margin_mean=0.0,
        n_samples=0,
        n_epochs=0,
    )

    def fake_collect_games(
        training_loop: loop.TrainingLoop,
        iteration: int,
        setup_enabled: bool,
        dagger_active: bool = False,
    ) -> list[collect.GameRecord]:
        order.append("collect")
        return [canned_record]

    def fake_learner_update(
        net: object,
        optimizer: object,
        records: list[collect.GameRecord],
        cfg: config.RunConfig,
        device: object,
        imitation_phase: bool = False,
        iteration: int = 0,
    ) -> learner.UpdateStats:
        order.append("learner.update")
        return canned_update_stats

    def fake_update_setup(
        training_loop: loop.TrainingLoop,
        records: list[collect.GameRecord],
        iteration: int = 0,
    ) -> metrics.SetupUpdateStats:
        order.append("update_setup")
        return canned_setup_stats

    def fake_sync_setup_embedders(training_loop: loop.TrainingLoop) -> None:
        order.append("sync_setup_embedders")

    def fake_maybe_evaluate(
        training_loop: loop.TrainingLoop, iteration: int
    ) -> tuple[metrics.EvalResult | None, float]:
        order.append("evaluate")
        return None, 0.0

    def fake_commit_iteration(
        training_loop: loop.TrainingLoop,
        iter_metrics: metrics.IterationMetrics,
        stats: learner.UpdateStats,
        eval_result: metrics.EvalResult | None,
        records: list[collect.GameRecord],
    ) -> None:
        order.append("commit")

    monkeypatch.setattr(loop_collect, "collect_games", fake_collect_games)
    monkeypatch.setattr(learner, "update", fake_learner_update)
    monkeypatch.setattr(loop_setup, "update_setup", fake_update_setup)
    monkeypatch.setattr(loop_setup, "sync_setup_embedders", fake_sync_setup_embedders)
    monkeypatch.setattr(loop_eval, "maybe_evaluate", fake_maybe_evaluate)
    monkeypatch.setattr(loop_checkpoint, "commit_iteration", fake_commit_iteration)

    training._run_iteration(0)

    assert order == [
        "collect",
        "update_setup",
        "learner.update",
        "sync_setup_embedders",
        "evaluate",
        "commit",
    ]
