"""ON-path collection tests: ``play_game_with_setup`` records setup samples.

Exercise the three regimes a setup-enabled game runs under — random (recording),
model-driven, and net-vs-random — and confirm a sample is recorded for each
net-controlled seat with a finished game's realized margin.
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


def _spec(phase: collect.SetupPhase) -> collect.SetupGameSpec:
    return collect.SetupGameSpec(
        phase=phase,
        deal_seed=4,
        continuation_seed=5,
        tuple_index=0,
        iteration=1500,
    )


def _generator() -> setup_model.RandomSetupGenerator:
    return setup_model.RandomSetupGenerator(
        hand_combos=3, food_sets=2, tuples_per_batch=4
    )


def test_random_record_self_play_records_both_seats():
    record = collect.play_game_with_setup(
        _net(),
        torch.device("cpu"),
        _spec(collect.SetupPhase.RANDOM_RECORD),
        _generator(),
        setup_policy_net=None,
        setup_temperature=0.5,
    )
    assert len(record.setup_samples) == 2
    for sample in record.setup_samples:
        assert sample.features.shape == (setup_model.SETUP_FEATURE_DIM,)
        assert sample.iteration == 1500
    assert record.steps  # in-game decisions are still recorded for the main net


def test_no_record_phase_records_nothing():
    record = collect.play_game_with_setup(
        _net(),
        torch.device("cpu"),
        _spec(collect.SetupPhase.RANDOM_NO_RECORD),
        _generator(),
        setup_policy_net=None,
        setup_temperature=0.5,
    )
    assert record.setup_samples == []


def test_model_driven_uses_setup_net_and_records():
    setup_policy_net = setup_net.SetupNet(
        arch=setup_model.SetupArchitecture(hidden_layers=(16,))
    )
    record = collect.play_game_with_setup(
        _net(),
        torch.device("cpu"),
        _spec(collect.SetupPhase.MODEL_DRIVEN),
        _generator(),
        setup_policy_net=setup_policy_net,
        setup_temperature=0.5,
    )
    assert len(record.setup_samples) == 2


def test_vs_random_records_only_net_seat():
    record = collect.play_game_with_setup(
        _net(),
        torch.device("cpu"),
        _spec(collect.SetupPhase.RANDOM_RECORD),
        _generator(),
        setup_policy_net=None,
        setup_temperature=0.5,
        opponent_agent=agents.random_agent(random.Random(0)),
    )
    assert len(record.setup_samples) == 1


def test_build_setup_specs_batches_share_deal():
    cfg = config.TrainConfig(
        use_setup_model=True, games_per_iter=8, setup_tuples_per_batch=4
    )
    specs = collect.build_setup_specs(
        cfg, iteration=10, phase=collect.SetupPhase.RANDOM_RECORD
    )
    assert len(specs) == 8
    # Two batches of four: each batch's games share a deal seed, distinct
    # continuation seeds.
    assert specs[0].deal_seed == specs[3].deal_seed
    assert specs[4].deal_seed != specs[0].deal_seed
    assert len({spec.continuation_seed for spec in specs}) == 8
    assert [spec.tuple_index for spec in specs[:4]] == [0, 1, 2, 3]


def test_model_driven_specs_deal_per_game():
    cfg = config.TrainConfig(use_setup_model=True, games_per_iter=4)
    specs = collect.build_setup_specs(
        cfg, iteration=10, phase=collect.SetupPhase.MODEL_DRIVEN
    )
    for spec in specs:
        assert spec.deal_seed == spec.continuation_seed
