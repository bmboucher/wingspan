# pyright: reportPrivateUsage=false
# (white-box tests of the model's private state-embedding path — they call
# ``_embed_state`` directly to isolate the tray-set block from the trunk)
"""The main net's tray-*set* embedding (``ModelArchitecture.tray_set_embedding``).

With the flag on, ``_embed_state`` appends one hand-encoder embedding of the
face-up tray set, derived in-model from the three tray index columns; the tray's
per-slot card-table lookups are unchanged. These tests lock in the appended
block's exact value, the derived set summary's equivalence with the encoder's
hand summary, empty-slot behavior, the flag-off path's invariance, the M ≠ N
shape flow, and the validator guarding the flag.
"""

from __future__ import annotations

import os
import random
import sys

import numpy as np
import pydantic
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from wingspan import architecture, cards, encode, model, state  # noqa: E402
from wingspan.encode import state_encode  # noqa: E402
from wingspan.model import hand_model  # noqa: E402
from wingspan.training import config  # noqa: E402

_ARCH_OFF = architecture.ModelArchitecture(
    trunk_layers=(32, 32),
    choice_layers=(32, 32),
    card_embed_dim=8,
    use_distinct_hand_model=True,
    tray_set_embedding=False,
)
_ARCH_ON = _ARCH_OFF.model_copy(update={"tray_set_embedding": True})


def _state_with_tray(tray: list[cards.Bird | None]) -> np.ndarray:
    """An encoded state whose face-up tray is exactly ``tray``."""
    birds, bonuses, goals = cards.load_all()
    game_state = state.new_game(
        random.Random(3), list(birds), list(bonuses), list(goals)
    )
    game_state.tray = list(tray)
    return encode.encode_state(game_state)


def _tray_index_columns(state_vec: np.ndarray) -> torch.Tensor:
    """The three tray index columns of an encoded state, as a (1, 3) long tensor."""
    start = encode.OFF_CARD_INDEX + encode.N_BOARD_INDEX_SLOTS
    return torch.tensor(
        state_vec[start : start + state.TRAY_SIZE], dtype=torch.long
    ).unsqueeze(0)


def test_flag_on_appends_exactly_the_tray_set_embedding():
    """Flag on inserts one N-wide block between the slot embeddings and the hand
    embedding — equal to ``hand_encoder([derived multi-hot ⊕ derived summary])``
    — and leaves every other element of the flag-off embedding unchanged."""
    birds = cards.load_all()[0]
    net_off = model.PolicyValueNet(arch=_ARCH_OFF)
    net_on = model.PolicyValueNet(arch=_ARCH_ON)
    net_on.card_encoder.load_state_dict(net_off.card_encoder.state_dict())
    net_on.hand_encoder.load_state_dict(net_off.hand_encoder.state_dict())

    state_vec = _state_with_tray([birds[10], birds[20], birds[30]])
    state_t = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
    emb_on = net_on._embed_state(state_t, net_on.card_table())
    emb_off = net_off._embed_state(state_t, net_off.card_table())

    # Widths: +N overall, matching trunk_input_dim's accounting.
    set_width = _ARCH_ON.hand_embed_width
    assert emb_on.shape[-1] == emb_off.shape[-1] + set_width
    assert emb_on.shape[-1] == encode.trunk_input_dim(
        len(state_vec),
        _ARCH_ON.card_embed_dim,
        use_distinct_hand_model=True,
        tray_set_embedding=True,
        n_playable_multihots=encode.N_HAND_PLAYABLE_MULTIHOTS,
    )

    # The shared prefix (continuous + slot embeddings) is everything before the
    # first set embedding; the suffix (hand + playability embeddings) follows the
    # tray block.  There are 1 + N_HAND_PLAYABLE_MULTIHOTS set-width blocks at the
    # tail of emb_off (hand plus the two playability copies), so the prefix ends
    # that many set-widths before the end.
    prefix = emb_off.shape[-1] - (1 + encode.N_HAND_PLAYABLE_MULTIHOTS) * set_width
    assert torch.equal(emb_on[..., :prefix], emb_off[..., :prefix])
    assert torch.equal(emb_on[..., prefix + set_width :], emb_off[..., prefix:])

    # The inserted block is exactly the hand encoder over the derived inputs.
    tray_idx = _tray_index_columns(state_vec)
    expected = hand_model.embed_card_set(
        net_on.hand_encoder,
        hand_model.multihot_from_indices(tray_idx, encode.HAND_MULTIHOT_DIM),
        hand_model.set_summary_from_indices(tray_idx, net_on.card_summary_matrix),
    )
    assert torch.allclose(emb_on[..., prefix : prefix + set_width], expected)


def test_derived_set_summary_matches_hand_summary_of_equivalent_hand():
    """The in-model summary derived from tray index columns equals the numpy
    encoder's ``_summary_hand`` for a hand holding the same cards."""
    birds, bonuses, goals = cards.load_all()
    tray_birds = [birds[5], birds[50], birds[120]]
    state_vec = _state_with_tray(list(tray_birds))
    tray_idx = _tray_index_columns(state_vec)

    derived = hand_model.set_summary_from_indices(
        tray_idx, torch.tensor(encode.card_summary_matrix())
    )

    game_state = state.new_game(random.Random(0), birds, bonuses, goals)
    game_state.players[0].hand = list(tray_birds)
    expected = state_encode._summary_hand(game_state.players[0])
    assert torch.allclose(derived.squeeze(0), torch.tensor(expected), atol=1e-6)


def test_empty_tray_slots_drop_out():
    """An empty slot contributes a zero card-table row and nothing to the set
    summary / multi-hot (index 0 is the zeroed padding row)."""
    birds = cards.load_all()[0]
    net = model.PolicyValueNet(arch=_ARCH_ON)
    state_vec = _state_with_tray([birds[10], None, None])
    tray_idx = _tray_index_columns(state_vec)

    # Per-slot card-table rows: the empty slots map to the zero padding row.
    table = net.card_table()
    rows = table[tray_idx.squeeze(0)]
    assert torch.all(rows[1] == 0.0) and torch.all(rows[2] == 0.0)
    assert bool(torch.any(rows[0] != 0.0))

    # The derived multi-hot marks exactly the one occupied slot's bird.
    multihot = hand_model.multihot_from_indices(tray_idx, encode.HAND_MULTIHOT_DIM)
    assert float(multihot.sum()) == 1.0
    assert multihot[0, cards.bird_index(birds[10])] == 1.0

    # The derived summary equals the single-card summary (empty slots add 0).
    summary = hand_model.set_summary_from_indices(
        tray_idx, torch.tensor(encode.card_summary_matrix())
    )
    single = state_encode._hand_summary_row(birds[10])
    assert torch.allclose(summary.squeeze(0), torch.tensor(single), atol=1e-6)


def test_mismatched_embed_dims_flow_end_to_end():
    """M ≠ N (hand_embed_dim its own knob): a full forward pass works and the
    trunk consumes the resolved widths."""
    arch = architecture.ModelArchitecture(
        trunk_layers=(32, 32),
        choice_layers=(32, 32),
        card_embed_dim=16,
        use_distinct_hand_model=True,
        hand_embed_dim=24,
        tray_set_embedding=True,
    )
    assert arch.hand_embed_width == 24
    net = model.PolicyValueNet(arch=arch)
    net.eval()
    batch, n_choices = 2, 3
    state_t = torch.randn(batch, encode.state_size())
    choices = torch.randn(batch, n_choices, encode.CHOICE_FEATURE_DIM)
    mask = torch.ones(batch, n_choices)
    family = torch.zeros(batch, dtype=torch.long)
    with torch.no_grad():
        logits, value = net(state_t, choices, mask, family)
    assert logits.shape == (batch, n_choices)
    assert value.shape == (batch,)


def test_none_hand_embed_dim_resolves_to_card_embed_dim():
    """``hand_embed_dim=None`` and an explicit equal value are the same shape —
    the ShapeKey compares the *resolved* width so the two stay compatible."""
    implicit = architecture.ModelArchitecture(use_distinct_hand_model=True)
    explicit = architecture.ModelArchitecture(
        use_distinct_hand_model=True, hand_embed_dim=implicit.card_embed_dim
    )
    assert implicit.hand_embed_width == explicit.hand_embed_width
    assert implicit.shape_key == explicit.shape_key


def test_validator_rejects_tray_set_embedding_without_hand_model():
    with pytest.raises(pydantic.ValidationError):
        architecture.ModelArchitecture(
            use_distinct_hand_model=False, tray_set_embedding=True
        )
    with pytest.raises(pydantic.ValidationError):
        config.RunConfig(
            misc=config.MiscConfig(device="cpu"),
            architecture=config.ArchitectureConfig(
                main=config.MainNetArchitecture(
                    use_distinct_hand_model=False, tray_set_embedding=True
                )
            ),
        )


def test_tray_flag_changes_shape_key_for_fresh_restart():
    """Turning the flag on must register as a weight-incompatible change (the
    documented FRESH mechanism) — old checkpoints then restart cleanly."""
    cfg_off = config.RunConfig(
        misc=config.MiscConfig(device="cpu"),
        architecture=config.ArchitectureConfig(
            main=config.MainNetArchitecture(
                use_distinct_hand_model=True, tray_set_embedding=False
            )
        ),
    )
    cfg_on = config.RunConfig(
        misc=config.MiscConfig(device="cpu"),
        architecture=config.ArchitectureConfig(
            main=config.MainNetArchitecture(
                use_distinct_hand_model=True, tray_set_embedding=True
            )
        ),
    )
    assert cfg_off.architecture_key != cfg_on.architecture_key
