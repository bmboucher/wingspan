"""Stateless torch helpers for the shared multi-card *set* embedder.

The hand encoder (``ModelArchitecture.use_distinct_hand_model``) maps a card
set's ``[multi-hot ⊕ 10-dim set summary]`` to one ``hand_embed_width``-wide
vector. Beyond the hand itself, the same encoder embeds every other card set —
the tray (main net, under ``tray_set_embedding``) and the setup model's kept /
tray sets — whose multi-hots and summaries are *derived in-model* rather than
encoded: from index columns (the tray's three slots) or from a multi-hot (the
kept set) plus the constant per-card summary table
(:func:`wingspan.encode.card_summary_matrix`). These helpers are that
derivation, shared by ``model.PolicyValueNet`` and
``training.setup_net.SetupNet`` so the two nets compute set embeddings the same
way. Set reduction follows ``encode.HAND_SUMMARY_SUM_DIMS``: the leading dims
sum over the set, the food-cost flags combine by max (OR — every entry is
non-negative, and padding rows are zero, so empty slots drop out).
"""

from __future__ import annotations

import torch
from torch import nn

from wingspan import architecture, encode


def multihot_from_indices(indices: torch.Tensor, n_birds: int) -> torch.Tensor:
    """Scatter ``(..., K)`` integer card indices (``bird_index + 1``; 0 = empty)
    into a ``(..., n_birds)`` float multi-hot. Empty slots scatter into the
    dropped padding column, so they leave the multi-hot untouched."""
    padded = torch.zeros(
        (*indices.shape[:-1], n_birds + 1),
        dtype=torch.float32,
        device=indices.device,
    )
    padded.scatter_(-1, indices.long(), 1.0)
    return padded[..., 1:]


def set_summary_from_indices(
    indices: torch.Tensor, summary_matrix: torch.Tensor
) -> torch.Tensor:
    """The 10-dim set summary for ``(..., K)`` card indices, gathered from the
    ``[n_birds + 1, HAND_SUMMARY_DIM]`` per-card ``summary_matrix`` (padding row
    0 all-zero) and reduced per ``encode.HAND_SUMMARY_SUM_DIMS``: leading dims
    sum, food flags max. Empty slots hit the zero row, adding nothing and never
    setting a flag."""
    rows = summary_matrix[indices.long()]  # (..., K, HAND_SUMMARY_DIM)
    sum_dims = encode.HAND_SUMMARY_SUM_DIMS
    summed = rows[..., :sum_dims].sum(dim=-2)
    flags = rows[..., sum_dims:].amax(dim=-2)
    return torch.cat([summed, flags], dim=-1)


def set_summary_from_multihot(
    multihot: torch.Tensor, summary_rows: torch.Tensor
) -> torch.Tensor:
    """The 10-dim set summary for a ``(..., n_birds)`` multi-hot, using the
    ``[n_birds, HAND_SUMMARY_DIM]`` per-card ``summary_rows`` (the summary matrix
    *without* its padding row). The leading ``encode.HAND_SUMMARY_SUM_DIMS`` dims
    are the multi-hot's weighted sum; the food flags are a masked max — rows of
    cards outside the set are zeroed before the max, which equals OR since every
    flag is non-negative."""
    sum_dims = encode.HAND_SUMMARY_SUM_DIMS
    summed = multihot @ summary_rows[:, :sum_dims]
    masked = multihot.unsqueeze(-1) * summary_rows[:, sum_dims:]
    flags = masked.amax(dim=-2)
    return torch.cat([summed, flags], dim=-1)


def embed_card_set(
    hand_encoder: nn.Module, multihot: torch.Tensor, summary: torch.Tensor
) -> torch.Tensor:
    """One card set's embedding: the hand encoder applied to its
    ``[multi-hot ⊕ summary]`` concat — the exact input shape the encoder sees for
    the hand itself (``encode.HAND_ENCODER_INPUT_DIM``)."""
    return hand_encoder(torch.cat([multihot, summary], dim=-1))


def pool_card_set(
    multihot: torch.Tensor,
    card_rows: torch.Tensor,
    pooling: architecture.HandPooling,
) -> torch.Tensor:
    """Permutation-invariant pooling of a card set through the shared card table.

    ``multihot`` is ``[B, 180]`` (any selected card is ``1``); ``card_rows``
    is ``card_table[1:]`` — the ``[180, M]`` block that skips the padding row.
    Returns a ``[B, W]`` tensor where ``W`` depends on ``pooling`` mode:
    ``MEAN`` and ``SUM`` → ``M``; ``MAX`` → ``M+1`` (max ⊕ count);
    ``CONCAT_MAX_SUM`` → ``2M+1`` (max ⊕ sum ⊕ count).

    The masked max maps an empty hand to ``0`` on all axes (not ``-inf``),
    so the pooled vector is always well-defined."""
    count = multihot.sum(dim=-1, keepdim=True)  # [B, 1]

    if pooling == architecture.HandPooling.MEAN:
        weighted_sum = multihot @ card_rows  # [B, M]
        return weighted_sum / count.clamp(min=1.0)

    if pooling == architecture.HandPooling.SUM:
        return multihot @ card_rows  # [B, M]

    # MAX and CONCAT_MAX_SUM both need an element-wise masked max.
    max_pool = _masked_max(multihot, card_rows, count)  # [B, M]

    if pooling == architecture.HandPooling.MAX:
        return torch.cat([max_pool, count], dim=-1)  # [B, M+1]

    # CONCAT_MAX_SUM
    weighted_sum = multihot @ card_rows  # [B, M]
    return torch.cat([max_pool, weighted_sum, count], dim=-1)  # [B, 2M+1]


def _masked_max(
    multihot: torch.Tensor, card_rows: torch.Tensor, count: torch.Tensor
) -> torch.Tensor:
    """Element-wise max of the selected cards' vectors; empty hand → ``0``.

    Expands ``multihot`` to ``[B, 180, M]``, masks unselected rows to ``-inf``,
    takes the max over the card axis, then zeros out the result for empty sets."""
    mask = multihot.bool()  # [B, 180]
    finfo = torch.finfo(card_rows.dtype)
    # [B, 180, M]: broadcast card_rows and mask unselected cards to -inf.
    expanded = card_rows.unsqueeze(0).expand(multihot.shape[0], -1, -1)
    masked = torch.where(
        mask.unsqueeze(-1), expanded, torch.full_like(expanded, finfo.min)
    )
    max_vals = masked.max(dim=1).values  # [B, M]
    # Empty hand: every entry is -inf after masking; reset to 0.
    return torch.where(count > 0, max_vals, torch.zeros_like(max_vals))
