"""One tiny end-to-end N=3 training iteration: collect -> update -> eval, all
in-process (no worker pool), proving the seat-count-generic training pipeline
actually trains at 3 players. Also covers the ``validate_launchable`` N=3
launch-blocker removal (Stage 3 of the N-player plan); the *other*,
permanent compat-era x num_players blocker is tested in
``test_run_config.py`` and is untouched by this stage.

CPU collection normally fans across a real worker pool
(``mp_collect.ProcessCollector``); this test instead calls
``collect.play_game`` / ``learner.update`` / ``evaluate.evaluate_vs_opponent``
directly (the same primitives ``TrainingLoop`` calls), so the "one iteration"
stays fast and process-spawn-free — mirroring the established 2-player
pattern in ``test_model_and_self_play.py``.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from wingspan import model, version
from wingspan.training import collect, config, evaluate, learner, loop_metrics

torch = pytest.importorskip("torch")

_SMALL_LAYERS = (32, 32)
_SMALL_CARD_EMBED_DIM = 16
_SMALL_CARD_ENCODER_LAYERS = (32,)


def _n3_config() -> config.RunConfig:
    return config.RunConfig(
        misc=config.MiscConfig(device="cpu"),
        run=config.RunSettings(games_per_iter=2, eval_games=2),
        architecture=config.ArchitectureConfig(
            num_players=3,
            main=config.MainNetArchitecture(
                trunk_layers=_SMALL_LAYERS,
                choice_layers=_SMALL_LAYERS,
                card_embed_dim=_SMALL_CARD_EMBED_DIM,
                card_encoder_layers=_SMALL_CARD_ENCODER_LAYERS,
            ),
        ),
    )


def test_validate_launchable_allows_n3_at_live_era():
    """The Stage-2 temporary blocker is gone: a live-era num_players=3 config
    launches cleanly."""
    cfg = _n3_config()
    assert cfg.architecture.encoding_version == version.MODEL_VERSION
    assert config.validate_launchable(cfg) == []


def test_n3_tiny_iteration_collect_update_eval():
    """One tiny in-process collect -> update -> eval cycle at num_players=3
    completes and records sane, seat-count-correct metrics."""
    cfg = _n3_config()
    net = model.PolicyValueNet(arch=cfg.arch, spec=cfg.encoding_spec)
    device = torch.device("cpu")
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    rng = random.Random(0)

    # 1. Collect -- in-process, no worker pool.
    records = [
        collect.play_game(net, device, rng, seed=seed, num_players=3)
        for seed in (101, 102)
    ]
    assert len(records) == cfg.run.games_per_iter
    for record in records:
        assert len(record.breakdowns) == 3
        assert {step.player_id for step in record.steps} <= {0, 1, 2}

    # 2. Update -- the real length-bucketed REINFORCE step.
    stats = learner.update(net, optimizer, records, cfg, device)
    assert stats.n_steps > 0
    assert np.isfinite(stats.loss)
    assert np.isfinite(stats.value_loss)

    # 3. Eval -- in-process, vs the random agent, rotated through all 3 seats.
    eval_result = evaluate.evaluate_vs_opponent(
        net, None, device, n_pairs=1, seed=555, num_players=3
    )
    assert eval_result.n_games == 3
    assert 0.0 <= eval_result.win_rate <= 1.0

    # 4. Measure -- fold into an IterationMetrics row exactly as loop.py does.
    iter_metrics = loop_metrics.build_iteration_metrics(
        iteration=0,
        total_games=len(records),
        records=records,
        stats=stats,
        eval_result=eval_result,
        collect_seconds=0.1,
        update_seconds=0.1,
        eval_seconds=0.1,
        win_rate=None,
        setup_enabled=False,
        setup_stats=None,
        entropy_coef=cfg.entropy_coef_at(0),
        dropout_p=cfg.dropout_p_at(0),
    )
    assert iter_metrics.games_this_iter == 2
    assert iter_metrics.avg_breakdown.total >= 0.0
    assert iter_metrics.family_counts.total() == sum(len(r.steps) for r in records)
