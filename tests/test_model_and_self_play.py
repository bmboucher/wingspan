"""Tests for ``model.PolicyValueNet`` and the self-play loop.

* Forward pass: shapes, masking semantics, and the no-NaN guarantee under
  variable-length padding.
* Self-play: both seats record steps from their own POV; the same network
  drives both sides and the training step runs end-to-end without crashing.
"""

from __future__ import annotations

import os
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

torch = pytest.importorskip("torch")
functional = pytest.importorskip("torch.nn.functional")

from wingspan import encode, model, train


def test_model_forward_shapes_and_mask():
    net = model.PolicyValueNet()
    batch_size, n_choices = 3, 5
    state = torch.zeros(batch_size, encode.state_size())
    choices = torch.randn(batch_size, n_choices, encode.CHOICE_FEATURE_DIM)
    mask = torch.tensor(
        [
            [1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1],
        ],
        dtype=torch.float32,
    )
    logits, value = net(state, choices, mask)
    assert logits.shape == (batch_size, n_choices)
    assert value.shape == (batch_size,)
    # Padding positions are masked to -inf so their softmax contribution
    # vanishes.
    probs = functional.softmax(logits, dim=-1)
    assert torch.allclose(probs[0, 3:], torch.zeros(2))
    assert torch.allclose(probs[1, 2:], torch.zeros(3))
    assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size), atol=1e-5)


def test_model_handles_single_choice_decision():
    net = model.PolicyValueNet()
    state = torch.zeros(1, encode.state_size())
    choices = torch.randn(1, 1, encode.CHOICE_FEATURE_DIM)
    mask = torch.ones(1, 1)
    logits, value = net(state, choices, mask)
    # With only one legal choice, softmax must be exactly 1.0.
    probs = functional.softmax(logits, dim=-1)
    assert torch.allclose(probs, torch.ones(1, 1))
    assert torch.isfinite(value).all()


def test_self_play_records_steps_for_both_players():
    """In self-play both seats should produce recorded steps tagged by
    their own player_id. We're not asserting balance (the network is
    random-init), just that the loop captures both sides."""
    net = model.PolicyValueNet()
    device = torch.device("cpu")
    rng = random.Random(0)
    traj = train.collect_episode(net, device, rng, epsilon=0.0, seed=42)
    assert traj.steps, "expected at least one recorded step"
    seen_ids = {step.player_id for step in traj.steps}
    assert seen_ids == {0, 1}, f"expected both player_ids, got {seen_ids}"


def test_train_step_runs_on_self_play_trajectories():
    """A full mini-cycle: collect a couple of self-play episodes, run a
    REINFORCE update, and confirm no shape errors / NaNs."""
    net = model.PolicyValueNet()
    device = torch.device("cpu")
    rng = random.Random(0)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    trajs = [
        train.collect_episode(net, device, rng, epsilon=0.0, seed=seed)
        for seed in (1001, 1002)
    ]
    stats = train.train_step(net, optimizer, trajs, device)
    assert stats.n_steps > 0
    for name, val in (
        ("loss", stats.loss),
        ("policy_loss", stats.policy_loss),
        ("value_loss", stats.value_loss),
        ("entropy", stats.entropy),
    ):
        assert np.isfinite(val), f"{name} should be finite, got {val}"


def test_policy_responds_to_per_choice_features():
    """Different per-choice features at the *same* state should produce
    different logits — i.e. the network is actually reading the candidate
    embeddings, not just the state."""
    net = model.PolicyValueNet()
    state = torch.zeros(1, encode.state_size())
    mask = torch.ones(1, 2)
    choices_a = torch.zeros(1, 2, encode.CHOICE_FEATURE_DIM)
    choices_b = torch.zeros(1, 2, encode.CHOICE_FEATURE_DIM)
    choices_b[0, 0, 0] = 1.0  # perturb slot 0's kind stripe
    choices_b[0, 1, 6] = 1.0  # perturb slot 1's bird stripe
    la, _ = net(state, choices_a, mask)
    lb, _ = net(state, choices_b, mask)
    assert not torch.allclose(
        la, lb
    ), "policy logits should respond to per-choice features"
