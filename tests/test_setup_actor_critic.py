"""Tests for the actor-critic setup training path.

Covers: SetupNet two-tower construction/forwarding, actor_critic_update loss and
gradient flow, and play_game_with_setup populating chosen_idx/all_candidates.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from wingspan import setup_model  # noqa: E402
from wingspan.model import core as model_core  # noqa: E402
from wingspan.setup_model import record  # noqa: E402
from wingspan.training import collect, config, setup_learner, setup_net  # noqa: E402

# ---------------------------------------------------------------------------
# SetupNet: policy head construction


def _make_net(use_policy_head: bool) -> setup_net.SetupNet:
    arch = setup_model.SetupArchitecture(
        head_layers=(16, 8), use_policy_head=use_policy_head
    )
    return setup_net.SetupNet(arch=arch)


def test_policy_head_absent_by_default():
    net = _make_net(use_policy_head=False)
    assert net.policy_head is None


def test_policy_head_present_when_enabled():
    net = _make_net(use_policy_head=True)
    assert net.policy_head is not None


def test_forward_unchanged_shape_with_policy_head():
    """forward() still returns (B,) value scalars regardless of policy head."""
    net = _make_net(use_policy_head=True)
    batch = torch.zeros((5, net.feature_dim), dtype=torch.float32)
    out = net(batch)
    assert out.shape == (5,)


def test_policy_and_value_returns_two_tensors():
    net = _make_net(use_policy_head=True)
    batch = torch.zeros((5, net.feature_dim), dtype=torch.float32)
    policy_logits, value_preds = net.policy_and_value(batch)
    assert policy_logits.shape == (5,)
    assert value_preds.shape == (5,)


def test_policy_and_value_heads_differ():
    """Both heads read the shared state encoding, but the policy head also reads the
    choice encoding and has its own MLP, so their outputs diverge."""
    net = _make_net(use_policy_head=True)
    rng = torch.Generator()
    rng.manual_seed(42)
    batch = torch.randn(3, net.feature_dim, generator=rng)
    policy_logits, value_preds = net.policy_and_value(batch)
    assert not torch.allclose(policy_logits, value_preds)


def test_policy_and_value_raises_without_policy_head():
    net = _make_net(use_policy_head=False)
    batch = torch.zeros((2, net.feature_dim), dtype=torch.float32)
    with pytest.raises(RuntimeError, match="use_policy_head"):
        net.policy_and_value(batch)


@pytest.mark.parametrize("split_bonus", [False, True])
def test_value_head_invariant_to_chosen_action(split_bonus: bool):
    """The critic is V(s): its output depends only on the action-independent
    STATE stripes (tray / feeder / goals / bonus-on-offer) — read through the
    shared state trunk — so two candidate vectors that share those stripes but
    differ everywhere else (the keep) receive an identical value. This is the
    property whose absence made the old per-candidate Q(s, a) baseline self-cancel."""
    encoding = setup_model.SetupEncoding(
        split_bonus=split_bonus, split_food=split_bonus
    )
    arch = setup_model.SetupArchitecture(head_layers=(16, 8), use_policy_head=True)
    net = setup_net.SetupNet(encoding=encoding, arch=arch)
    net.eval()
    enc = net.encoding

    rng = np.random.default_rng(7)
    base = rng.standard_normal(net.feature_dim).astype(np.float32)
    variant = rng.standard_normal(net.feature_dim).astype(np.float32)
    # Force the action-independent STATE stripes equal between the two vectors.
    state_slices = [
        slice(enc.off_tray, enc.off_feeder),  # tray
        slice(enc.off_feeder, enc.off_goals),  # birdfeeder
        slice(enc.off_goals, enc.off_bonus_value),  # round goals
    ]
    if enc.split_bonus:
        state_slices.append(
            slice(enc.off_bonus_block, enc.off_bonus_block + enc.bonus_cards_dim)
        )
    for state_slice in state_slices:
        variant[state_slice] = base[state_slice]

    feats = torch.tensor(np.stack([base, variant]), dtype=torch.float32)
    values = net(feats)
    # V(s) ignores the keep -> identical for two same-state candidates.
    assert torch.allclose(values[0], values[1], atol=1e-6)
    # The policy head reads state ⊕ choice, so it still distinguishes them.
    logits = net.policy_logits(feats)
    assert not torch.allclose(logits[0], logits[1])


# ---------------------------------------------------------------------------
# actor_critic_update: loss computation and gradient flow


def _make_config() -> config.TrainConfig:
    return config.RunConfig(
        architecture=config.ArchitectureConfig(
            setup=config.SetupNetArchitecture(
                head_layers=(16, 8),
            ),
        ),
    )


_SAMPLE_FEATURE_DIM = setup_model.SetupEncoding().total_dim


def _make_ac_sample(
    k: int = 10, chosen_idx: int = 0, seed: int = 0, margin: float = 5.0
) -> record.SetupSample:
    """One synthetic SetupSample with all_candidates and return fields populated.

    Distinct ``seed`` values give distinct candidate matrices and margins so the
    learner's per-batch advantage whitening sees non-zero variance (identical
    samples would whiten to a zero advantage and produce no policy gradient)."""
    rng = np.random.default_rng(seed)
    features = rng.standard_normal(_SAMPLE_FEATURE_DIM).astype(np.float32)
    all_candidates = rng.standard_normal((k, _SAMPLE_FEATURE_DIM)).astype(np.float32)
    return record.SetupSample(
        features=features,
        margin=margin,
        iteration=2001,
        chosen_idx=chosen_idx,
        all_candidates=all_candidates,
        own_total=margin,
        opp_total=0.0,
        won=1 if margin > 0 else (-1 if margin < 0 else 0),
    )


def test_actor_critic_update_returns_nonzero_loss():
    cfg = _make_config()
    net = _make_net(use_policy_head=True)
    optimizer = torch.optim.Adam(
        [p for p in net.parameters() if p.requires_grad], lr=1e-3
    )
    samples = [_make_ac_sample(seed=i, margin=float(i + 1)) for i in range(4)]
    stats = setup_learner.actor_critic_update(
        net, optimizer, samples, cfg, torch.device("cpu")
    )
    assert stats.n_samples == 4
    assert np.isfinite(stats.loss)
    assert stats.loss != 0.0


def test_actor_critic_update_empty_samples():
    cfg = _make_config()
    net = _make_net(use_policy_head=True)
    optimizer = torch.optim.Adam(
        [p for p in net.parameters() if p.requires_grad], lr=1e-3
    )
    stats = setup_learner.actor_critic_update(
        net, optimizer, [], cfg, torch.device("cpu")
    )
    assert stats.n_samples == 0
    assert stats.loss == 0.0


def test_actor_critic_update_skips_samples_without_ac_data():
    """Samples with chosen_idx=None are silently skipped."""
    cfg = _make_config()
    net = _make_net(use_policy_head=True)
    optimizer = torch.optim.Adam(
        [p for p in net.parameters() if p.requires_grad], lr=1e-3
    )
    # One valid + two without ac data
    rng = np.random.default_rng(1)
    no_ac = record.SetupSample(
        features=rng.standard_normal(_SAMPLE_FEATURE_DIM).astype(np.float32),
        margin=2.0,
        iteration=2001,
    )
    samples = [_make_ac_sample(), no_ac, no_ac]
    stats = setup_learner.actor_critic_update(
        net, optimizer, samples, cfg, torch.device("cpu")
    )
    assert stats.n_samples == 1


def test_actor_critic_update_gradient_flows_to_policy_head():
    """Gradient reaches policy_head parameters but not the frozen card encoder.

    The samples must differ so the whitened advantage has non-zero variance,
    otherwise the policy gradient is zero by construction."""
    cfg = _make_config()
    net = _make_net(use_policy_head=True)
    optimizer = torch.optim.Adam(
        [p for p in net.parameters() if p.requires_grad], lr=1e-3
    )
    samples = [
        _make_ac_sample(k=8, chosen_idx=3, seed=i, margin=float(i + 1))
        for i in range(3)
    ]
    setup_learner.actor_critic_update(net, optimizer, samples, cfg, torch.device("cpu"))

    # Policy head parameters must have nonzero grad after the update's backward.
    assert net.policy_head is not None
    policy_params = list(net.policy_head.parameters())
    assert any(p.grad is not None and p.grad.abs().max() > 0 for p in policy_params)

    # The value head must also receive gradient (value loss).
    value_params = list(net.value_head.parameters())
    assert any(p.grad is not None and p.grad.abs().max() > 0 for p in value_params)

    # Frozen card encoder parameters must have no grad.
    for param in net.card_encoder.parameters():
        assert param.grad is None or param.grad.abs().max() == 0


def test_actor_critic_update_groups_by_candidate_count():
    """Samples with different K values (504 vs 252) are processed separately."""
    cfg = _make_config()
    net = _make_net(use_policy_head=True)
    optimizer = torch.optim.Adam(
        [p for p in net.parameters() if p.requires_grad], lr=1e-3
    )
    samples_504 = [
        _make_ac_sample(k=504, seed=i, margin=float(i + 1)) for i in range(2)
    ]
    samples_252 = [
        _make_ac_sample(k=252, seed=i + 10, margin=float(i - 1)) for i in range(2)
    ]
    stats = setup_learner.actor_critic_update(
        net, optimizer, samples_504 + samples_252, cfg, torch.device("cpu")
    )
    assert stats.n_samples == 4
    assert np.isfinite(stats.loss)


# ---------------------------------------------------------------------------
# shape_key includes use_policy_head


def test_shape_key_differs_by_policy_head():
    key_off = setup_model.SetupArchitecture(
        head_layers=(64, 32), use_policy_head=False
    ).shape_key
    key_on = setup_model.SetupArchitecture(
        head_layers=(64, 32), use_policy_head=True
    ).shape_key
    assert key_off != key_on


def test_shape_key_same_layers_different_policy_head():
    key_a = setup_model.SetupArchitecture(
        head_layers=(128, 64), use_policy_head=False
    ).shape_key
    key_b = setup_model.SetupArchitecture(
        head_layers=(128, 64), use_policy_head=True
    ).shape_key
    # shape_key = (trunk_layers, choice_layers, head_layers, value_layers, use_policy_head)
    assert key_a[0] == key_b[0]  # trunk_layers same (both default)
    assert key_a[2] == key_b[2]  # head_layers same
    assert key_a[4] != key_b[4]  # use_policy_head differs


def test_state_trunk_shared_choice_trunk_policy_only():
    """The shared state trunk feeds both heads (so a value-only backward still
    reaches it); the choice trunk feeds only the policy head."""
    arch = setup_model.SetupArchitecture(
        trunk_layers=(32,),
        choice_layers=(24,),
        head_layers=(16,),
        use_policy_head=True,
    )
    net = setup_net.SetupNet(arch=arch)
    net.train()
    batch = torch.randn(2, net.feature_dim)
    state_params = list(net.state_trunk.parameters())
    choice_params = list(net.choice_trunk.parameters())
    assert state_params and choice_params

    # A value-only backward reaches the shared state trunk but not the choice trunk.
    net.zero_grad()
    net(batch).sum().backward()  # pyright: ignore[reportUnknownMemberType]
    assert any(p.grad is not None and p.grad.abs().max() > 0 for p in state_params)
    assert all(p.grad is None or p.grad.abs().max() == 0 for p in choice_params)

    # A policy backward reaches both trunks.
    net.zero_grad()
    net.policy_logits(
        batch
    ).sum().backward()  # pyright: ignore[reportUnknownMemberType]
    assert any(p.grad is not None and p.grad.abs().max() > 0 for p in state_params)
    assert any(p.grad is not None and p.grad.abs().max() > 0 for p in choice_params)


# ---------------------------------------------------------------------------
# play_game_with_setup: actor-critic data populated in MODEL_DRIVEN phase


def test_play_game_with_setup_ac_data_in_model_driven():
    """MODEL_DRIVEN + use_actor_critic=True → SetupSamples carry chosen_idx and
    all_candidates."""
    from wingspan import setup_model as sm

    net_cfg = config.RunConfig(
        architecture=config.ArchitectureConfig(
            setup=config.SetupNetArchitecture(
                head_layers=(16, 8),
            )
        ),
    )
    device = torch.device("cpu")

    # The setup net always trains actor-critic.
    net = setup_net.SetupNet(arch=net_cfg.setup_arch)

    main_net = model_core.PolicyValueNet(
        state_dim=net_cfg.state_dim,
        choice_dim=net_cfg.choice_dim,
        arch=net_cfg.arch,
        spec=net_cfg.encoding_spec,
    )

    generator = sm.RandomSetupGenerator(hand_combos=2, food_sets=1)
    spec = collect.SetupGameSpec(
        deal_seed=42,
        continuation_seed=42,
        iteration=2001,
    )

    game_record = collect.play_game_with_setup(
        main_net,
        device,
        spec,
        generator,
        net,
        setup_temperature=1.0,
    )

    assert len(game_record.setup_samples) == 2  # both seats in self-play
    for sample in game_record.setup_samples:
        assert sample.chosen_idx is not None
        assert sample.all_candidates is not None
        assert sample.all_candidates.shape[0] > 0
        assert sample.all_candidates.shape[1] == net.feature_dim
        # The reward-consistency fields are populated for returns.setup_return.
        assert sample.margin == pytest.approx(sample.own_total - sample.opp_total)
        assert sample.won in (-1, 0, 1)
        assert sample.final_timestamp > 0.0
        # One checkpoint per in-game decision the seat made.
        assert len(sample.margin_checkpoints) == len(sample.decision_times)
        assert len(sample.score_checkpoints) == len(sample.decision_times)
