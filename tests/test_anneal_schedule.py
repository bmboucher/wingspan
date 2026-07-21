# pyright: reportPrivateUsage=false
# (calls loop_anneal._anneal_main_net_dropout directly — deliberate)
"""Tests for the dropout / entropy-coef annealing schedules.

Covers:
* ``config.anneal`` — the shared linear-taper-then-hold lerp.
* ``RunConfig.entropy_coef_at`` / ``dropout_p_at`` / ``setup_entropy_coef_at`` /
  ``setup_dropout_p_at`` — the per-net accessors built on ``anneal``.
* ``loop_anneal.set_dropout`` — the ``nn.Dropout`` sweep, for both the main net
  (incl. scorers + value head) and the setup net (frozen embedders stay eval()).
* ``loop_anneal._anneal_main_net_dropout`` — each main-net block anneals from
  its own resolved build-time dropout; a block with a 0.0 resolved dropout
  (globally or via a per-block override) has no Dropout module to sweep and is
  silently inert rather than rejected.
* The effective coefficient actually reaches the loss in ``learner.update`` /
  ``setup_learner.actor_critic_update`` (an annealed run at its target iteration
  matches an equivalent constant-coefficient run).
* ``validate_launchable`` — the sole anneal-specific launch rejection (a
  ``*_final`` set without a ``target_iterations`` horizon). Dropout anneal
  configs are never rejected for lacking Dropout modules to sweep, globally or
  per-block.
* Config round-trip: the accessors are a pure function of the (dumped and
  revalidated) config and the iteration argument.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wingspan import architecture, model, setup_model  # noqa: E402
from wingspan.setup_model import record  # noqa: E402
from wingspan.training import (  # noqa: E402
    collect,
    config,
    learner,
    loop_anneal,
    setup_learner,
    setup_net,
)

# ---------------------------------------------------------------------------
# config.anneal — the shared lerp
# ---------------------------------------------------------------------------


def test_anneal_at_iteration_zero_returns_initial():
    assert config.anneal(0.05, 0.01, 100, 0) == pytest.approx(0.05)


def test_anneal_midpoint_is_linear():
    assert config.anneal(0.05, 0.01, 100, 50) == pytest.approx(0.03)


def test_anneal_holds_final_at_and_beyond_target():
    assert config.anneal(0.05, 0.01, 100, 100) == pytest.approx(0.01)
    assert config.anneal(0.05, 0.01, 100, 500) == pytest.approx(0.01)


def test_anneal_passthrough_when_final_none():
    assert config.anneal(0.05, None, 100, 50) == pytest.approx(0.05)


def test_anneal_passthrough_when_no_target():
    assert config.anneal(0.05, 0.01, 0, 50) == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# RunConfig accessors
# ---------------------------------------------------------------------------


def _annealing_config() -> config.RunConfig:
    """0.05 -> 0.01 over 100 iterations, on every one of the four schedules."""
    return config.RunConfig(
        misc=config.MiscConfig(device="cpu"),
        run=config.RunSettings(target_iterations=100),
        architecture=config.ArchitectureConfig(
            main=config.MainNetArchitecture(dropout=0.05),
            setup=config.SetupNetArchitecture(dropout=0.05),
        ),
        training=config.TrainingConfig(
            entropy_coef=0.05,
            entropy_coef_final=0.01,
            dropout_final=0.01,
            setup=config.SetupTrainingConfig(
                entropy_coef=0.05,
                entropy_coef_final=0.01,
                dropout_final=0.01,
            ),
        ),
    )


_ANNEAL_POINTS = [(0, 0.05), (50, 0.03), (100, 0.01), (500, 0.01)]


@pytest.mark.parametrize("iteration,expected", _ANNEAL_POINTS)
def test_entropy_coef_at(iteration: int, expected: float):
    assert _annealing_config().entropy_coef_at(iteration) == pytest.approx(expected)


@pytest.mark.parametrize("iteration,expected", _ANNEAL_POINTS)
def test_dropout_p_at(iteration: int, expected: float):
    assert _annealing_config().dropout_p_at(iteration) == pytest.approx(expected)


@pytest.mark.parametrize("iteration,expected", _ANNEAL_POINTS)
def test_setup_entropy_coef_at(iteration: int, expected: float):
    assert _annealing_config().setup_entropy_coef_at(iteration) == pytest.approx(
        expected
    )


@pytest.mark.parametrize("iteration,expected", _ANNEAL_POINTS)
def test_setup_dropout_p_at(iteration: int, expected: float):
    assert _annealing_config().setup_dropout_p_at(iteration) == pytest.approx(expected)


def test_accessors_are_constant_when_final_unset():
    """A config with no *_final set holds its initial value at every iteration."""
    cfg = config.RunConfig(misc=config.MiscConfig(device="cpu"))
    for iteration in (0, 10, 10_000):
        assert cfg.entropy_coef_at(iteration) == pytest.approx(
            cfg.training.entropy_coef
        )
        assert cfg.dropout_p_at(iteration) == pytest.approx(
            cfg.architecture.main.dropout
        )


# ---------------------------------------------------------------------------
# loop_anneal.set_dropout
# ---------------------------------------------------------------------------


def _small_main_arch(
    dropout: float, *, card_dropout: float | None = None
) -> architecture.ModelArchitecture:
    return architecture.ModelArchitecture(
        trunk_layers=(8, 8),
        choice_layers=(8, 8),
        head_layers=(8,),
        value_layers=(8,),
        card_embed_dim=4,
        card_encoder_layers=(8,),
        dropout=dropout,
        card_dropout=card_dropout,
    )


def test_set_dropout_sweeps_every_dropout_module_including_scorers_and_value_head():
    net = model.PolicyValueNet(arch=_small_main_arch(0.05))
    count = loop_anneal.set_dropout(net, 0.2)

    dropouts = [m for m in net.modules() if isinstance(m, torch.nn.Dropout)]
    assert count == len(dropouts)
    assert count > 0
    assert all(m.p == pytest.approx(0.2) for m in dropouts)

    assert any(isinstance(m, torch.nn.Dropout) for m in net.value_head.modules())
    assert any(
        isinstance(m, torch.nn.Dropout)
        for scorer in net.scorers
        for m in scorer.modules()
    )


def test_set_dropout_is_a_no_op_when_no_dropout_modules_exist():
    net = model.PolicyValueNet(arch=_small_main_arch(0.0))
    assert loop_anneal.set_dropout(net, 0.3) == 0
    assert not any(isinstance(m, torch.nn.Dropout) for m in net.modules())


def _dropout_anneal_config(
    *, dropout: float, card_dropout: float | None, dropout_final: float
) -> config.RunConfig:
    return config.RunConfig(
        misc=config.MiscConfig(device="cpu"),
        run=config.RunSettings(target_iterations=10),
        architecture=config.ArchitectureConfig(
            main=config.MainNetArchitecture(dropout=dropout, card_dropout=card_dropout)
        ),
        training=config.TrainingConfig(dropout_final=dropout_final),
    )


def test_anneal_main_net_dropout_ignores_blocks_with_zero_initial():
    """global dropout=0, only card_dropout overridden nonzero: every other
    block has no Dropout module at all and is left untouched."""
    net = model.PolicyValueNet(arch=_small_main_arch(0.0, card_dropout=0.1))
    cfg = _dropout_anneal_config(dropout=0.0, card_dropout=0.1, dropout_final=0.02)
    other_blocks = [net.state_trunk, net.choice_encoder, net.value_head, *net.scorers]

    loop_anneal._anneal_main_net_dropout(net, cfg, iteration=0)
    card_dropouts = [
        m for m in net.card_encoder.modules() if isinstance(m, torch.nn.Dropout)
    ]
    other_dropouts = [
        m
        for block in other_blocks
        for m in block.modules()
        if isinstance(m, torch.nn.Dropout)
    ]
    assert all(m.p == pytest.approx(0.1) for m in card_dropouts)
    assert card_dropouts
    assert other_dropouts == []

    loop_anneal._anneal_main_net_dropout(net, cfg, iteration=10)
    card_dropouts = [
        m for m in net.card_encoder.modules() if isinstance(m, torch.nn.Dropout)
    ]
    assert all(m.p == pytest.approx(0.02) for m in card_dropouts)


def test_anneal_main_net_dropout_each_block_anneals_from_its_own_initial():
    """global dropout=0.05, card_dropout overridden to 0.1: the card block
    tapers from its own 0.1, the rest taper from the global 0.05 — both
    converge on the shared final rather than one snapping to the other's
    initial."""
    net = model.PolicyValueNet(arch=_small_main_arch(0.05, card_dropout=0.1))
    cfg = _dropout_anneal_config(dropout=0.05, card_dropout=0.1, dropout_final=0.01)

    loop_anneal._anneal_main_net_dropout(net, cfg, iteration=0)
    card_dropouts = [
        m for m in net.card_encoder.modules() if isinstance(m, torch.nn.Dropout)
    ]
    trunk_dropouts = [
        m for m in net.state_trunk.modules() if isinstance(m, torch.nn.Dropout)
    ]
    assert all(m.p == pytest.approx(0.1) for m in card_dropouts)
    assert all(m.p == pytest.approx(0.05) for m in trunk_dropouts)
    assert trunk_dropouts

    loop_anneal._anneal_main_net_dropout(net, cfg, iteration=5)
    card_dropouts = [
        m for m in net.card_encoder.modules() if isinstance(m, torch.nn.Dropout)
    ]
    trunk_dropouts = [
        m for m in net.state_trunk.modules() if isinstance(m, torch.nn.Dropout)
    ]
    assert all(m.p == pytest.approx(0.055) for m in card_dropouts)
    assert all(m.p == pytest.approx(0.03) for m in trunk_dropouts)

    loop_anneal._anneal_main_net_dropout(net, cfg, iteration=10)
    card_dropouts = [
        m for m in net.card_encoder.modules() if isinstance(m, torch.nn.Dropout)
    ]
    trunk_dropouts = [
        m for m in net.state_trunk.modules() if isinstance(m, torch.nn.Dropout)
    ]
    assert all(m.p == pytest.approx(0.01) for m in card_dropouts)
    assert all(m.p == pytest.approx(0.01) for m in trunk_dropouts)


def test_set_dropout_sweeps_setup_net_and_frozen_embedders_stay_eval():
    setup_arch = setup_model.SetupArchitecture(
        trunk_layers=(8,),
        choice_layers=(8,),
        head_layers=(8,),
        value_layers=(8,),
        dropout=0.05,
        use_policy_head=True,
    )
    # main_arch dropout=0.0 so the frozen card/hand embedder copies carry no
    # Dropout of their own — every swept module belongs to the setup-owned blocks.
    main_arch = architecture.ModelArchitecture(dropout=0.0)
    net = setup_net.SetupNet(arch=setup_arch, main_arch=main_arch)

    count = loop_anneal.set_dropout(net, 0.25)
    dropouts = [m for m in net.modules() if isinstance(m, torch.nn.Dropout)]
    assert count == len(dropouts)
    assert count > 0
    assert all(m.p == pytest.approx(0.25) for m in dropouts)

    net.train()
    assert net.card_encoder.training is False


# ---------------------------------------------------------------------------
# The effective coefficient reaches the loss
# ---------------------------------------------------------------------------


def test_annealed_entropy_coef_at_target_matches_equivalent_constant_loss():
    """An annealed run evaluated at its target iteration (final=0.05) produces
    the same loss as a constant-0.05 run at iteration 0 — same net, same data,
    same effective coefficient. Dropout=0 throughout for determinism."""
    device = torch.device("cpu")

    torch.manual_seed(0)
    net_a = model.PolicyValueNet(arch=_small_main_arch(0.0))
    torch.manual_seed(0)
    net_b = model.PolicyValueNet(arch=_small_main_arch(0.0))

    rng = random.Random(0)
    records = [collect.play_game(net_a, device, rng, seed=seed) for seed in (1, 2)]

    cfg_a = config.RunConfig(
        misc=config.MiscConfig(device="cpu"),
        architecture=config.ArchitectureConfig(
            main=config.MainNetArchitecture(dropout=0.0)
        ),
        training=config.TrainingConfig(entropy_coef=0.05),
    )
    cfg_b = config.RunConfig(
        misc=config.MiscConfig(device="cpu"),
        architecture=config.ArchitectureConfig(
            main=config.MainNetArchitecture(dropout=0.0)
        ),
        run=config.RunSettings(target_iterations=10),
        training=config.TrainingConfig(entropy_coef=0.5, entropy_coef_final=0.05),
    )
    opt_a = torch.optim.Adam(net_a.parameters(), lr=1e-3)
    opt_b = torch.optim.Adam(net_b.parameters(), lr=1e-3)

    stats_a = learner.update(net_a, opt_a, records, cfg_a, device, iteration=0)
    stats_b = learner.update(net_b, opt_b, records, cfg_b, device, iteration=10)

    assert stats_a.loss == pytest.approx(stats_b.loss, rel=1e-5)


def _make_setup_sample(
    feature_dim: int,
    k: int = 10,
    chosen_idx: int = 0,
    seed: int = 0,
    margin: float = 5.0,
) -> record.SetupSample:
    """A synthetic actor-critic ``SetupSample`` (mirrors test_setup_actor_critic.py's
    ``_make_ac_sample``). Distinct seeds give distinct candidates/margins so the
    batch's whitened advantage has non-zero variance."""
    rng = np.random.default_rng(seed)
    return record.SetupSample(
        features=rng.standard_normal(feature_dim).astype(np.float32),
        margin=margin,
        iteration=0,
        chosen_idx=chosen_idx,
        all_candidates=rng.standard_normal((k, feature_dim)).astype(np.float32),
        own_total=margin,
        opp_total=0.0,
        won=1 if margin > 0 else (-1 if margin < 0 else 0),
    )


def test_annealed_setup_entropy_coef_at_target_matches_equivalent_constant_loss():
    setup_arch = setup_model.SetupArchitecture(
        dropout=0.0, head_layers=(16, 8), use_policy_head=True
    )
    main_arch = architecture.ModelArchitecture(dropout=0.0)

    torch.manual_seed(1)
    net_a = setup_net.SetupNet(arch=setup_arch, main_arch=main_arch)
    torch.manual_seed(1)
    net_b = setup_net.SetupNet(arch=setup_arch, main_arch=main_arch)

    feature_dim = setup_model.SetupEncoding().total_dim
    samples = [
        _make_setup_sample(feature_dim, seed=i, margin=float(i + 1)) for i in range(4)
    ]

    cfg_a = config.RunConfig(
        misc=config.MiscConfig(device="cpu"),
        training=config.TrainingConfig(
            setup=config.SetupTrainingConfig(entropy_coef=0.05)
        ),
    )
    cfg_b = config.RunConfig(
        misc=config.MiscConfig(device="cpu"),
        run=config.RunSettings(target_iterations=10),
        training=config.TrainingConfig(
            setup=config.SetupTrainingConfig(entropy_coef=0.5, entropy_coef_final=0.05)
        ),
    )
    opt_a = torch.optim.Adam(
        [p for p in net_a.parameters() if p.requires_grad], lr=1e-3
    )
    opt_b = torch.optim.Adam(
        [p for p in net_b.parameters() if p.requires_grad], lr=1e-3
    )

    stats_a = setup_learner.actor_critic_update(
        net_a, opt_a, samples, cfg_a, torch.device("cpu"), iteration=0
    )
    stats_b = setup_learner.actor_critic_update(
        net_b, opt_b, samples, cfg_b, torch.device("cpu"), iteration=10
    )

    assert stats_a.loss == pytest.approx(stats_b.loss, rel=1e-5)


# ---------------------------------------------------------------------------
# validate_launchable
# ---------------------------------------------------------------------------


def test_validate_launchable_anneal_without_target_flagged():
    cfg = config.RunConfig(
        misc=config.MiscConfig(device="cpu"),
        training=config.TrainingConfig(entropy_coef_final=0.01),
    )
    problems = config.validate_launchable(cfg)
    assert any("target_iterations" in problem for problem in problems)


def test_validate_launchable_dropout_final_without_base_dropout_is_clean():
    """A global dropout of 0.0 with no per-block override means the anneal has
    no Dropout module anywhere to act on — inert, not a launch rejection."""
    cfg = config.RunConfig(
        misc=config.MiscConfig(device="cpu"),
        run=config.RunSettings(target_iterations=100),
        training=config.TrainingConfig(dropout_final=0.01),
    )
    assert config.validate_launchable(cfg) == []


def test_validate_launchable_dropout_final_with_per_block_override_is_clean():
    """A per-block override no longer conflicts with the anneal — each block
    tapers from its own resolved initial (loop_anneal.apply_dropout_schedules)."""
    cfg = config.RunConfig(
        misc=config.MiscConfig(device="cpu"),
        run=config.RunSettings(target_iterations=100),
        architecture=config.ArchitectureConfig(
            main=config.MainNetArchitecture(dropout=0.05, card_dropout=0.1)
        ),
        training=config.TrainingConfig(dropout_final=0.01),
    )
    assert config.validate_launchable(cfg) == []


def test_validate_launchable_setup_dropout_final_without_base_dropout_is_clean():
    cfg = config.RunConfig(
        misc=config.MiscConfig(device="cpu"),
        run=config.RunSettings(target_iterations=100),
        training=config.TrainingConfig(
            setup=config.SetupTrainingConfig(dropout_final=0.01)
        ),
    )
    assert config.validate_launchable(cfg) == []


def test_validate_launchable_fully_configured_anneal_is_clean():
    assert config.validate_launchable(_annealing_config()) == []


# ---------------------------------------------------------------------------
# The schedule is a pure function of the (round-tripped) config + iteration
# ---------------------------------------------------------------------------


def test_anneal_accessors_survive_dump_and_revalidate_round_trip():
    cfg = _annealing_config()
    round_tripped = config.RunConfig.model_validate(cfg.model_dump())
    for iteration in (0, 25, 50, 100, 250):
        assert cfg.entropy_coef_at(iteration) == pytest.approx(
            round_tripped.entropy_coef_at(iteration)
        )
        assert cfg.dropout_p_at(iteration) == pytest.approx(
            round_tripped.dropout_p_at(iteration)
        )
        assert cfg.setup_entropy_coef_at(iteration) == pytest.approx(
            round_tripped.setup_entropy_coef_at(iteration)
        )
        assert cfg.setup_dropout_p_at(iteration) == pytest.approx(
            round_tripped.setup_dropout_p_at(iteration)
        )
