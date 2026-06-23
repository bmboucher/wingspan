"""Board self-attention layer (``use_board_attention``) unit tests.

Covers the forward pass (including fully-empty boards at game start), parameter
accounting accuracy, ShapeKey/architecture_key incompatibility with the base net,
trunk-input-width invariance, and the save/reload round-trip.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

torch = pytest.importorskip("torch")

from wingspan import architecture, encode
from wingspan.model import core

# ---------------------------------------------------------------------------
# Helpers

_N_CHOICES = 4


def _arch(use_board_attention: bool) -> architecture.ModelArchitecture:
    return architecture.ModelArchitecture(
        trunk_layers=(64, 64),
        choice_layers=(64, 64),
        head_layers=(64,),
        card_embed_dim=64,
        card_encoder_layers=(64,),
        use_distinct_hand_model=True,
        hand_encoder_layers=(64,),
        use_board_attention=use_board_attention,
    )


def _net(use_board_attention: bool) -> core.PolicyValueNet:
    return core.PolicyValueNet(arch=_arch(use_board_attention)).eval()


def _zero_state(batch: int = 2):
    """A batch of all-zero state vectors (empty boards, game start).

    Unannotated: ``torch`` arrives via ``pytest.importorskip`` (an ``Any``)."""
    state_dim = encode.state_size()
    return torch.zeros(batch, state_dim)


def _inputs(batch: int = 2):
    """A minimal (state, choices, mask, family_idx) batch.

    Unannotated: ``torch`` arrives via ``pytest.importorskip`` (an ``Any``)."""
    state = _zero_state(batch)
    choice_dim = encode.CHOICE_FEATURE_DIM
    choices = torch.zeros(batch, _N_CHOICES, choice_dim)
    mask = torch.ones(batch, _N_CHOICES)
    family_idx = torch.zeros(batch, dtype=torch.long)
    return state, choices, mask, family_idx


# ---------------------------------------------------------------------------
# Tests


def test_forward_no_nan_empty_boards() -> None:
    """Forward pass on fully-empty boards (game start) must produce finite outputs."""
    net = _net(use_board_attention=True)
    state, choices, mask, family_idx = _inputs(batch=3)
    with torch.no_grad():
        logits, value = net(state, choices, mask, family_idx)
    assert logits.shape == (3, _N_CHOICES)
    assert value.shape == (3,)
    assert torch.isfinite(logits).all(), "logits contain NaN/inf"
    assert torch.isfinite(value).all(), "value contains NaN/inf"


def test_forward_no_nan_partial_boards() -> None:
    """Forward with a non-trivial state (non-zero card indices) is also NaN-free."""
    net = _net(use_board_attention=True)
    state, choices, mask, family_idx = _inputs(batch=2)
    # Set slot 0 and slot 1 of own board to card indices 1 and 2.
    state[:, encode.OFF_CARD_INDEX + 0] = 1.0
    state[:, encode.OFF_CARD_INDEX + 1] = 2.0
    with torch.no_grad():
        logits, value = net(state, choices, mask, family_idx)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()


def test_base_net_has_no_attention_modules() -> None:
    """With use_board_attention=False the two attention modules must not exist."""
    net = _net(use_board_attention=False)
    assert not hasattr(net, "board_attn_me")
    assert not hasattr(net, "board_attn_opp")


def test_attention_net_has_attention_modules() -> None:
    """With use_board_attention=True both attention modules are registered."""
    net = _net(use_board_attention=True)
    assert hasattr(net, "board_attn_me")
    assert hasattr(net, "board_attn_opp")
    assert isinstance(net.board_attn_me, torch.nn.MultiheadAttention)
    assert isinstance(net.board_attn_opp, torch.nn.MultiheadAttention)


def test_param_count_matches_accounting() -> None:
    """The actual parameter count of the attention net equals count_parameters."""
    arch_attn = _arch(use_board_attention=True)
    net = core.PolicyValueNet(arch=arch_attn)
    actual = sum(p.numel() for p in net.parameters())

    spec = encode.DEFAULT_SPEC
    report = architecture.count_parameters(
        arch_attn,
        card_feat_in=encode.CARD_FEATURE_DIM,
        trunk_in=encode.trunk_input_dim(
            encode.state_size(spec),
            arch_attn.card_embed_dim,
            use_distinct_hand_model=arch_attn.use_distinct_hand_model,
            hand_embed_dim=arch_attn.hand_embed_dim,
            tray_set_embedding=arch_attn.tray_set_embedding,
            n_playable_multihots=encode.N_HAND_PLAYABLE_MULTIHOTS,
        ),
        choice_in=encode.choice_input_dim(
            encode.CHOICE_FEATURE_DIM, arch_attn.card_embed_dim
        ),
        num_families=encode.num_families(spec),
        hand_feat_in=encode.HAND_ENCODER_INPUT_DIM,
        slot_scalar_dim=encode.SLOT_SCALAR_DIM,
    )
    assert actual == report.total, (
        f"actual {actual} != accounting {report.total} "
        f"(board_attention block: {report.board_attention})"
    )


def test_base_param_count_unchanged() -> None:
    """With use_board_attention=False the net's parameter count equals the accounting."""
    arch_base = _arch(use_board_attention=False)
    net = core.PolicyValueNet(arch=arch_base)
    actual = sum(p.numel() for p in net.parameters())

    spec = encode.DEFAULT_SPEC
    report = architecture.count_parameters(
        arch_base,
        card_feat_in=encode.CARD_FEATURE_DIM,
        trunk_in=encode.trunk_input_dim(
            encode.state_size(spec),
            arch_base.card_embed_dim,
            use_distinct_hand_model=arch_base.use_distinct_hand_model,
            hand_embed_dim=arch_base.hand_embed_dim,
            tray_set_embedding=arch_base.tray_set_embedding,
            n_playable_multihots=encode.N_HAND_PLAYABLE_MULTIHOTS,
        ),
        choice_in=encode.choice_input_dim(
            encode.CHOICE_FEATURE_DIM, arch_base.card_embed_dim
        ),
        num_families=encode.num_families(spec),
        hand_feat_in=encode.HAND_ENCODER_INPUT_DIM,
    )
    assert actual == report.total
    assert report.board_attention is None


def test_shape_key_differs() -> None:
    """shape_key must differ between attention-on and attention-off architectures."""
    assert _arch(True).shape_key != _arch(False).shape_key


def test_architecture_key_incompatible() -> None:
    """architecture_key (used by the resume gate) must be incompatible."""
    from wingspan.training import config as train_config

    cfg_base = train_config.RunConfig()
    cfg_attn = train_config.RunConfig(
        architecture=train_config.ArchitectureConfig(
            main=train_config.MainNetArchitecture(use_board_attention=True)
        )
    )
    assert cfg_base.architecture_key != cfg_attn.architecture_key


def test_trunk_input_width_invariant() -> None:
    """The trunk's first-Linear in_features must be identical with/without attention."""
    net_base = core.PolicyValueNet(arch=_arch(False))
    net_attn = core.PolicyValueNet(arch=_arch(True))
    # state_trunk is an nn.Sequential; first child is a Linear.
    first_base = next(net_base.state_trunk.children())
    first_attn = next(net_attn.state_trunk.children())
    assert isinstance(first_base, torch.nn.Linear)
    assert isinstance(first_attn, torch.nn.Linear)
    assert first_base.in_features == first_attn.in_features, (
        f"trunk in_features differ: base={first_base.in_features}, "
        f"attn={first_attn.in_features}"
    )


def test_save_reload_roundtrip() -> None:
    """Save and reload the attention net; flag must survive and state_dict loads."""
    net = core.PolicyValueNet(arch=_arch(True))
    state_dict = net.state_dict()

    # Simulate model_config.json round-trip via architecture Pydantic model.
    arch_json = _arch(True).model_dump()
    arch_loaded = architecture.ModelArchitecture(**arch_json)
    assert arch_loaded.use_board_attention is True
    assert arch_loaded.shape_key == _arch(True).shape_key

    # Rebuild and load weights.
    net2 = core.PolicyValueNet(arch=arch_loaded)
    net2.load_state_dict(state_dict)
    # Forward pass should still work.
    state, choices, mask, family_idx = _inputs()
    with torch.no_grad():
        logits, value = net2.eval()(state, choices, mask, family_idx)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()
