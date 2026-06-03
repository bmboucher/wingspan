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

from wingspan import architecture, cards, decisions, encode, engine, model, state, train


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
    # A mix of family heads exercises the per-family routing in one pass.
    family = torch.tensor([0, 1, 2], dtype=torch.long)
    logits, value = net(state, choices, mask, family)
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
    family = torch.zeros(1, dtype=torch.long)
    logits, value = net(state, choices, mask, family)
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
    # Every recorded step carries a valid judgment-family head index.
    num_families = len(decisions.ALL_DECISION_FAMILIES)
    assert all(0 <= step.family_idx < num_families for step in traj.steps)


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
    family = torch.zeros(1, dtype=torch.long)
    choices_a = torch.zeros(1, 2, encode.CHOICE_FEATURE_DIM)
    choices_b = torch.zeros(1, 2, encode.CHOICE_FEATURE_DIM)
    choices_b[0, 0, 0] = 1.0  # perturb slot 0's kind stripe
    choices_b[0, 1, 6] = 1.0  # perturb slot 1's bird stripe
    la, _ = net(state, choices_a, mask, family)
    lb, _ = net(state, choices_b, mask, family)
    assert not torch.allclose(
        la, lb
    ), "policy logits should respond to per-choice features"


def test_model_routes_distinct_family_heads():
    """The same (state, choices) scored under two different judgment families
    must produce different logits — proof that family_idx selects a distinct
    scoring head — while the value is identical, since the value head is shared
    and reads only the (here-identical) state."""
    net = model.PolicyValueNet()
    n_choices = 4
    state = torch.randn(1, encode.state_size()).repeat(2, 1)
    choices = torch.randn(1, n_choices, encode.CHOICE_FEATURE_DIM).repeat(2, 1, 1)
    mask = torch.ones(2, n_choices)
    family = torch.tensor([0, 1], dtype=torch.long)
    logits, value = net(state, choices, mask, family)
    assert not torch.allclose(
        logits[0], logits[1]
    ), "distinct family heads should score the same candidates differently"
    assert torch.allclose(
        value[0], value[1]
    ), "the value head is shared and family-agnostic"


def test_model_same_family_scores_identically():
    """Two rows with identical inputs and the *same* family must yield
    identical logits — the routing is deterministic per family."""
    net = model.PolicyValueNet()
    n_choices = 3
    state = torch.randn(1, encode.state_size()).repeat(2, 1)
    choices = torch.randn(1, n_choices, encode.CHOICE_FEATURE_DIM).repeat(2, 1, 1)
    mask = torch.ones(2, n_choices)
    family = torch.zeros(2, dtype=torch.long)
    logits, _ = net(state, choices, mask, family)
    assert torch.allclose(logits[0], logits[1])


def test_model_has_one_scorer_head_per_family():
    # A bare net uses the default encoding spec (setup excluded), so it has one
    # head per *active* family — the SETUP head is dropped when setup is delegated
    # to the separate setup model.
    net = model.PolicyValueNet()
    expected = len(decisions.active_decision_families(net.include_setup))
    assert len(net.scorers) == expected
    assert net.num_families == expected


# ---------------------------------------------------------------------------
# Shared card table (the card encoder's per-card output)


def test_card_table_shape():
    """The shared card table has one row per core-set bird plus a padding row
    (index 0), and the padding row is held at zero."""
    net = model.PolicyValueNet(arch=architecture.ModelArchitecture(card_embed_dim=64))
    net.eval()
    table = net.card_table()
    assert table.shape == (encode.HAND_MULTIHOT_DIM + 1, 64)
    assert torch.allclose(table[0], torch.zeros(64))


def test_model_forward_with_custom_architecture():
    """A non-default topology — asymmetric per-block depths, a different
    activation, dropout, and LayerNorm — still produces well-shaped, finite,
    properly-masked outputs (the build is fully data-driven)."""
    arch = architecture.ModelArchitecture(
        trunk_layers=(96, 64, 48),
        choice_layers=(72, 48),
        head_layers=(32,),
        value_layers=(24, 16),
        activation=architecture.ActivationName.GELU,
        dropout=0.1,
        layernorm=True,
        card_embed_dim=48,
    )
    net = model.PolicyValueNet(arch=arch)
    net.eval()  # disable dropout for a deterministic shape check
    assert net.trunk_hidden == 48  # trunk output width M
    batch_size, n_choices = 3, 5
    state_vec = torch.zeros(batch_size, encode.state_size())
    choices = torch.randn(batch_size, n_choices, encode.CHOICE_FEATURE_DIM)
    mask = torch.tensor(
        [[1, 1, 1, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.float32
    )
    family = torch.tensor([0, 1, 2], dtype=torch.long)
    logits, value = net(state_vec, choices, mask, family)
    assert logits.shape == (batch_size, n_choices)
    assert value.shape == (batch_size,)
    probs = functional.softmax(logits, dim=-1)
    assert torch.allclose(probs[0, 3:], torch.zeros(2))
    assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size), atol=1e-5)
    assert torch.isfinite(value).all()


def test_model_accepts_asymmetric_body_widths():
    """Trunk and choice encoder may end at different widths M and N; the scorer
    input is M+N and the model builds and runs without error."""
    arch = architecture.ModelArchitecture(
        trunk_layers=(128, 64),
        choice_layers=(128, 128),
    )
    assert arch.trunk_embed_width == 64
    assert arch.choice_embed_width == 128
    net = model.PolicyValueNet(arch=arch)
    net.eval()
    assert net.trunk_hidden == 64
    state_vec = torch.zeros(2, encode.state_size())
    choices = torch.randn(2, 3, encode.CHOICE_FEATURE_DIM)
    mask = torch.ones(2, 3)
    family = torch.zeros(2, dtype=torch.long)
    logits, value = net(state_vec, choices, mask, family)
    assert logits.shape == (2, 3)
    assert value.shape == (2,)


def test_card_representation_shared_between_board_and_hand():
    """The *same* per-card representation drives a card whether it sits on the
    board or in the hand: perturbing that one card's feature row (its input to the
    shared encoder) moves both the board-state value and the hand-state value —
    proof the card table is shared, not re-learned per position."""
    eng, birds, *_ = engine.Engine.create(seed=51)
    net = model.PolicyValueNet()
    net.eval()
    card = birds[0]
    row = cards.bird_index(card) + 1
    me = eng.state.players[eng.state.current_player]

    me.hand = []
    me.board[cards.Habitat.FOREST] = [state.PlayedBird(bird=card)]
    board_state = torch.tensor(
        encode.encode_state(eng.state), dtype=torch.float32
    ).unsqueeze(0)
    me.board[cards.Habitat.FOREST] = []
    me.hand = [card]
    hand_state = torch.tensor(
        encode.encode_state(eng.state), dtype=torch.float32
    ).unsqueeze(0)

    choices = torch.zeros(1, 1, encode.CHOICE_FEATURE_DIM)
    mask = torch.ones(1, 1)
    family = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        v_board0 = net(board_state, choices, mask, family)[1]
        v_hand0 = net(hand_state, choices, mask, family)[1]
        net.card_features[row] += 5.0
        # The inference path memoizes the card table (card_features is a constant
        # buffer in production); refresh the cache after this probe-only mutation.
        net.eval()
        v_board1 = net(board_state, choices, mask, family)[1]
        v_hand1 = net(hand_state, choices, mask, family)[1]
    assert not torch.allclose(v_board0, v_board1), "board read should use the row"
    assert not torch.allclose(v_hand0, v_hand1), "hand read should use the same row"


def test_choice_candidate_uses_shared_card_representation():
    """A choice candidate carrying a bird is scored through the same shared card
    table: perturbing that bird's feature row shifts the candidate's logit (and
    thus the policy) relative to a skip option."""
    eng, birds, *_ = engine.Engine.create(seed=52)
    net = model.PolicyValueNet()
    net.eval()
    card = birds[0]
    row = cards.bird_index(card) + 1
    me = eng.state.players[eng.state.current_player]

    decision = decisions.BirdPowerTuckFromHandDecision(
        player_id=me.id,
        prompt="x",
        choices=[
            decisions.BirdChoice(label=card.name, bird=card),
            decisions.SkipChoice(label="skip"),
        ],
    )
    choices = torch.tensor(
        encode.encode_choices(decision, eng.state), dtype=torch.float32
    ).unsqueeze(0)
    state_vec = torch.tensor(
        encode.encode_state(eng.state), dtype=torch.float32
    ).unsqueeze(0)
    mask = torch.ones(1, 2)
    family = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        probs0 = functional.softmax(net(state_vec, choices, mask, family)[0], dim=-1)
        net.card_features[row] += 5.0
        # Refresh the memoized inference card table after this probe-only poke at
        # the otherwise-constant card_features buffer (see the board/hand test).
        net.eval()
        probs1 = functional.softmax(net(state_vec, choices, mask, family)[0], dim=-1)
    assert not torch.allclose(probs0, probs1), "candidate should read the card table"
