# pyright: reportPrivateUsage=false
# (reads TrainingLoop._setup_net to confirm the post-update eval-mode invariant)
"""Coverage for ``loop_setup.update_setup`` — the ``TrainingLoop`` wrapper around
``setup_learner.actor_critic_update`` that pushes the SETUP AC dashboard event
and folds trained samples into the run's cumulative family counts.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from wingspan.setup_model import record
from wingspan.training import collect, config, loop, loop_setup, metrics


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


def _make_sample(feature_dim: int, seed: int, margin: float) -> record.SetupSample:
    rng = np.random.default_rng(seed)
    all_candidates = rng.standard_normal((6, feature_dim)).astype(np.float32)
    chosen_idx = seed % 6
    return record.SetupSample(
        features=all_candidates[chosen_idx],
        margin=margin,
        iteration=0,
        chosen_idx=chosen_idx,
        all_candidates=all_candidates,
        own_total=margin,
        opp_total=0.0,
        won=1 if margin > 0 else (-1 if margin < 0 else 0),
    )


def test_update_setup_wrapper(tmp_path: pathlib.Path) -> None:
    training = loop.TrainingLoop(_loop_config(tmp_path))
    feature_dim = training.config.setup_encoding.total_dim

    records = [
        collect.GameRecord(
            steps=[],
            breakdowns=(metrics.ScoreBreakdown(), metrics.ScoreBreakdown()),
            winner=-1,
            seed=seed,
            setup_samples=[_make_sample(feature_dim, seed, margin)],
        )
        for seed, margin in enumerate([1.0, 2.0, 3.0])
    ]

    stats = loop_setup.update_setup(training, records, iteration=0)

    assert stats.n_samples == 3
    assert np.isfinite(stats.loss)
    assert stats.target_margin_mean == pytest.approx(2.0)
    assert any(
        "SETUP AC" in event.text and "tgt" in event.text
        for event in training.state.events
    )
    assert training.state.last_setup is stats
    assert training.state.cum_family.counts[metrics.SETUP_FAMILY_IDX] == 3
    assert training._setup_net is not None
    assert training._setup_net.training is False
