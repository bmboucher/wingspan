"""Tests for configurable hand pooling over the shared card encoder.

Covers:
- ``pool_card_set`` per-mode output widths (M, M+1, 2M+1).
- Correctness: empty hand → 0; max only over selected rows; sum == multihot @ rows.
- ``pooled_hand_width`` property agrees with ``pool_card_set`` output width.
- ``ModelArchitecture.shape_key`` is distinct when ``hand_pooling`` changes.
- Distinct-encoder checkpoints have the same ``shape_key`` slot regardless of
  the (inert) ``hand_pooling`` field.
- End-to-end: building a net at each mode and verifying ``_embed_state`` output
  width matches ``trunk_input_dim``, on both the standard and board-attention paths.
"""

from __future__ import annotations

import torch

from wingspan import architecture
from wingspan.model import hand_model

_CARD_EMBED_DIM = 16  # small M for speed
_N_BIRDS = 10  # small hand space for speed


def _card_rows(m: int = _CARD_EMBED_DIM, n: int = _N_BIRDS) -> torch.Tensor:
    """Deterministic [n, m] card table (rows 1.. of the padded table)."""
    return torch.arange(float(n * m)).reshape(n, m)


def _multihot(selected: list[int], n: int = _N_BIRDS) -> torch.Tensor:
    """[1, n] multi-hot with the given bird indices selected."""
    mh = torch.zeros(1, n)
    for idx in selected:
        mh[0, idx] = 1.0
    return mh


#### Width tests ####


def test_mean_width():
    rows = _card_rows()
    mh = _multihot([0, 2])
    out = hand_model.pool_card_set(mh, rows, architecture.HandPooling.MEAN)
    assert out.shape == (1, _CARD_EMBED_DIM)


def test_sum_width():
    rows = _card_rows()
    mh = _multihot([0, 2])
    out = hand_model.pool_card_set(mh, rows, architecture.HandPooling.SUM)
    assert out.shape == (1, _CARD_EMBED_DIM)


def test_max_width():
    rows = _card_rows()
    mh = _multihot([0, 2])
    out = hand_model.pool_card_set(mh, rows, architecture.HandPooling.MAX)
    assert out.shape == (1, _CARD_EMBED_DIM + 1)


def test_concat_max_sum_width():
    rows = _card_rows()
    mh = _multihot([0, 2])
    out = hand_model.pool_card_set(mh, rows, architecture.HandPooling.CONCAT_MAX_SUM)
    assert out.shape == (1, 2 * _CARD_EMBED_DIM + 1)


#### Correctness tests ####


def test_sum_equals_matmul():
    rows = _card_rows()
    mh = _multihot([1, 3, 5])
    out = hand_model.pool_card_set(mh, rows, architecture.HandPooling.SUM)
    expected = mh @ rows
    assert torch.allclose(out, expected)


def test_mean_equals_sum_over_count():
    rows = _card_rows()
    mh = _multihot([0, 4, 7])
    out = hand_model.pool_card_set(mh, rows, architecture.HandPooling.MEAN)
    expected = (mh @ rows) / 3.0
    assert torch.allclose(out, expected)


def test_max_only_over_selected():
    rows = _card_rows()
    selected = [1, 3]
    mh = _multihot(selected)
    out = hand_model.pool_card_set(mh, rows, architecture.HandPooling.MAX)
    max_part = out[0, :_CARD_EMBED_DIM]
    expected = torch.stack([rows[idx] for idx in selected]).amax(dim=0)
    assert torch.allclose(max_part, expected)


def test_empty_hand_max_is_zero():
    rows = _card_rows()
    mh = _multihot([])
    out = hand_model.pool_card_set(mh, rows, architecture.HandPooling.MAX)
    max_part = out[0, :_CARD_EMBED_DIM]
    assert (max_part == 0).all()


def test_empty_hand_concat_max_sum_is_zero():
    rows = _card_rows()
    mh = _multihot([])
    out = hand_model.pool_card_set(mh, rows, architecture.HandPooling.CONCAT_MAX_SUM)
    assert (out == 0).all()


def test_count_appended_to_max():
    rows = _card_rows()
    selected = [0, 2, 5]
    mh = _multihot(selected)
    out = hand_model.pool_card_set(mh, rows, architecture.HandPooling.MAX)
    count = out[0, _CARD_EMBED_DIM]
    assert count.item() == len(selected)


def test_count_in_concat_max_sum():
    rows = _card_rows()
    selected = [1, 6]
    mh = _multihot(selected)
    out = hand_model.pool_card_set(mh, rows, architecture.HandPooling.CONCAT_MAX_SUM)
    count = out[0, 2 * _CARD_EMBED_DIM]
    assert count.item() == len(selected)


#### pooled_hand_width property ####


def test_pooled_hand_width_mean():
    arch = architecture.ModelArchitecture(
        card_embed_dim=_CARD_EMBED_DIM,
        use_distinct_hand_model=False,
        hand_pooling=architecture.HandPooling.MEAN,
    )
    assert arch.pooled_hand_width == _CARD_EMBED_DIM


def test_pooled_hand_width_sum():
    arch = architecture.ModelArchitecture(
        card_embed_dim=_CARD_EMBED_DIM,
        use_distinct_hand_model=False,
        hand_pooling=architecture.HandPooling.SUM,
    )
    assert arch.pooled_hand_width == _CARD_EMBED_DIM


def test_pooled_hand_width_max():
    arch = architecture.ModelArchitecture(
        card_embed_dim=_CARD_EMBED_DIM,
        use_distinct_hand_model=False,
        hand_pooling=architecture.HandPooling.MAX,
    )
    assert arch.pooled_hand_width == _CARD_EMBED_DIM + 1


def test_pooled_hand_width_concat_max_sum():
    arch = architecture.ModelArchitecture(
        card_embed_dim=_CARD_EMBED_DIM,
        use_distinct_hand_model=False,
        hand_pooling=architecture.HandPooling.CONCAT_MAX_SUM,
    )
    assert arch.pooled_hand_width == 2 * _CARD_EMBED_DIM + 1


#### ShapeKey tests ####


def test_shape_key_distinct_from_pooled():
    arch_distinct = architecture.ModelArchitecture(use_distinct_hand_model=True)
    arch_pooled = architecture.ModelArchitecture(use_distinct_hand_model=False)
    assert arch_distinct.shape_key != arch_pooled.shape_key


def test_shape_key_changes_with_pooling_mode():
    arch_mean = architecture.ModelArchitecture(
        use_distinct_hand_model=False,
        hand_pooling=architecture.HandPooling.MEAN,
    )
    arch_concat = architecture.ModelArchitecture(
        use_distinct_hand_model=False,
        hand_pooling=architecture.HandPooling.CONCAT_MAX_SUM,
    )
    assert arch_mean.shape_key != arch_concat.shape_key


def test_shape_key_distinct_ignores_hand_pooling():
    """Old distinct-encoder checkpoints must not be split by the inert new field."""
    arch_a = architecture.ModelArchitecture(
        use_distinct_hand_model=True,
        hand_pooling=architecture.HandPooling.MEAN,
    )
    arch_b = architecture.ModelArchitecture(
        use_distinct_hand_model=True,
        hand_pooling=architecture.HandPooling.CONCAT_MAX_SUM,
    )
    assert arch_a.shape_key == arch_b.shape_key


#### End-to-end net build and embed tests ####


def _net_for(pooling: architecture.HandPooling, board_attention: bool = False):
    from wingspan.model import core

    arch = architecture.ModelArchitecture(
        card_embed_dim=32,
        trunk_layers=(64,),
        choice_layers=(64,),
        use_distinct_hand_model=False,
        hand_pooling=pooling,
        use_board_attention=board_attention,
    )
    from wingspan.training import config

    run_cfg = config.RunConfig()
    net = core.PolicyValueNet(
        arch=arch,
        state_dim=run_cfg.state_dim,
        choice_dim=run_cfg.choice_dim,
    )
    return net


def _check_embed_width(
    pooling: architecture.HandPooling, board_attention: bool = False
):
    """Build a net with ``pooling`` and verify ``_embed_state`` output == trunk input."""
    import typing

    from wingspan import encode
    from wingspan.training import config

    net = _net_for(pooling, board_attention=board_attention)
    arch = net.arch
    run_cfg = config.RunConfig()

    # Build a zero-valued state batch — we only check shapes, not values.
    batch = torch.zeros(2, run_cfg.state_dim)
    with torch.no_grad():
        ct = net.card_table()
        emb = typing.cast(
            torch.Tensor,
            net._embed_state(batch, ct),  # type: ignore[attr-defined]
        )

    # Offsets fetched after the forward pass (offsets are pure config, no forward needed).
    offsets = net._state_embed_offsets()  # type: ignore[attr-defined]
    n_extra = (
        offsets.decision_type - offsets.hand_multihot
    ) // encode.HAND_MULTIHOT_DIM - 1
    expected_w = encode.trunk_input_dim(
        run_cfg.state_dim,
        arch.card_embed_dim,
        use_distinct_hand_model=False,
        pooled_hand_width=arch.pooled_hand_width,
        n_playable_multihots=n_extra,
    )
    assert emb.shape == (2, expected_w), (
        f"pooling={pooling} board_attention={board_attention}: "
        f"embed width {emb.shape[1]} != trunk_input_dim {expected_w}"
    )


def test_embed_width_mean():
    _check_embed_width(architecture.HandPooling.MEAN)


def test_embed_width_sum():
    _check_embed_width(architecture.HandPooling.SUM)


def test_embed_width_max():
    _check_embed_width(architecture.HandPooling.MAX)


def test_embed_width_concat_max_sum():
    _check_embed_width(architecture.HandPooling.CONCAT_MAX_SUM)


def test_embed_width_board_attention_mean():
    _check_embed_width(architecture.HandPooling.MEAN, board_attention=True)


def test_embed_width_board_attention_concat_max_sum():
    _check_embed_width(architecture.HandPooling.CONCAT_MAX_SUM, board_attention=True)
