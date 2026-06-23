"""ON-path collection tests: ``play_game_with_setup`` records setup samples.

The setup model is always model-driven (actor-critic): every game records setup
samples for every net-controlled seat, with ``chosen_idx`` and ``all_candidates``
populated for REINFORCE training.
"""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from wingspan import agents, architecture, model, setup_model  # noqa: E402
from wingspan.training import collect, config, setup_net  # noqa: E402

_SMALL_ARCH = architecture.ModelArchitecture(
    trunk_layers=(32, 32),
    choice_layers=(32, 32),
    head_layers=(),
    value_layers=(),
    card_embed_dim=8,
)


def _net() -> model.PolicyValueNet:
    return model.PolicyValueNet(arch=_SMALL_ARCH)


def _spec() -> collect.SetupGameSpec:
    return collect.SetupGameSpec(
        deal_seed=4,
        continuation_seed=5,
        iteration=1500,
    )


def _generator() -> setup_model.RandomSetupGenerator:
    return setup_model.RandomSetupGenerator(hand_combos=3, food_sets=2)


def test_model_driven_self_play_records_both_seats():
    setup_policy_net = setup_net.SetupNet(
        arch=setup_model.SetupArchitecture(hidden_layers=(16,))
    )
    record = collect.play_game_with_setup(
        _net(),
        torch.device("cpu"),
        _spec(),
        _generator(),
        setup_policy_net=setup_policy_net,
        setup_temperature=0.5,
    )
    assert len(record.setup_samples) == 2
    for sample in record.setup_samples:
        assert sample.features.shape == (setup_model.SETUP_FEATURE_DIM,)
        assert sample.iteration == 1500
        # Actor-critic mode always populates chosen_idx and all_candidates.
        assert sample.chosen_idx is not None
        assert sample.all_candidates is not None
    assert record.steps  # in-game decisions are still recorded for the main net


def test_vs_random_records_only_net_seat():
    setup_policy_net = setup_net.SetupNet(
        arch=setup_model.SetupArchitecture(hidden_layers=(16,))
    )
    record = collect.play_game_with_setup(
        _net(),
        torch.device("cpu"),
        _spec(),
        _generator(),
        setup_policy_net=setup_policy_net,
        setup_temperature=0.5,
        opponent_agent=agents.random_agent(random.Random(0)),
    )
    assert len(record.setup_samples) == 1


def test_build_setup_specs_deal_per_game():
    """Each game in MODEL_DRIVEN mode gets its own independent deal."""
    cfg = config.RunConfig(
        architecture=config.ArchitectureConfig(use_setup_model=True),
        run=config.RunSettings(games_per_iter=4),
    )
    specs = collect.build_setup_specs(cfg, iteration=10)
    assert len(specs) == 4
    # Every game has a unique deal seed in MODEL_DRIVEN mode.
    assert len({spec.deal_seed for spec in specs}) == 4
