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

from wingspan import encode


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
