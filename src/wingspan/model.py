"""PyTorch model: pointer-style actor-critic with per-choice features.

The network scores each candidate at a decision point individually rather
than emitting a fixed-slot policy head. At each forward pass the caller
provides:

* ``state``   — ``(B, state_dim)`` POV-aware game features
* ``choices`` — ``(B, K, choice_dim)`` per-choice features (padded)
* ``mask``    — ``(B, K)`` 1.0 for real choices, 0.0 for padding

The trunk reads the state, a separate MLP reads each choice, the two are
concatenated and passed through a small scorer to produce one logit per
candidate. Padding rows get ``-inf`` so they never receive probability
mass. The value head reads only the state.

This is the structural change called out in the trainability review: action
slots no longer carry positional semantics; the network has to look at the
candidate's features to decide.
"""

from __future__ import annotations

import torch
from torch import nn

from wingspan import encode


class PolicyValueNet(nn.Module):
    """Actor-critic over (state, choice-set) decisions.

    A two-layer state trunk feeds (a) a single state-context vector used to
    rescore every candidate and (b) the value head. A two-layer per-choice
    encoder consumes the per-choice features. The score for choice ``i`` is
    an MLP over ``concat(state_ctx, choice_emb[i])``.
    """

    def __init__(
        self,
        state_dim: int | None = None,
        choice_dim: int = encode.CHOICE_FEATURE_DIM,
        hidden: int = 128,
    ):
        super().__init__()
        if state_dim is None:
            state_dim = encode.state_size()
        self.state_dim = state_dim
        self.choice_dim = choice_dim
        self.hidden = hidden

        self.state_trunk = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.choice_encoder = nn.Sequential(
            nn.Linear(choice_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.value_head = nn.Linear(hidden, 1)

    def forward(
        self,
        state: torch.Tensor,
        choices: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score every candidate at every decision in the batch.

        Args:
            state:   ``(B, state_dim)``
            choices: ``(B, K, choice_dim)``  — pad rows are arbitrary
            mask:    ``(B, K)`` with 1.0 on real choices, 0.0 on padding.

        Returns:
            logits: ``(B, K)`` — masked rows are set to ``-inf``
            value:  ``(B,)``
        """
        # State trunk produces both the per-decision context and the value.
        h = self.state_trunk(state)  # (B, H)
        value = self.value_head(h).squeeze(-1)  # (B,)

        # Per-choice MLP. choices is (B, K, F); the Linear layers broadcast
        # across the K dimension naturally.
        ce = self.choice_encoder(choices)  # (B, K, H)
        K = ce.shape[1]
        s_exp = h.unsqueeze(1).expand(-1, K, -1)  # (B, K, H)
        combined = torch.cat([s_exp, ce], dim=-1)  # (B, K, 2H)
        scores = self.scorer(combined).squeeze(-1)  # (B, K)

        # Mask out padding. Use very-negative rather than -inf to avoid NaN
        # if a row turns out to be entirely padded (defensive — shouldn't
        # happen in practice). For all-real rows -inf is fine; we use a
        # large finite number so softmax stays numerically clean either way.
        neg_inf = torch.full_like(scores, float("-inf"))
        logits = torch.where(mask > 0.5, scores, neg_inf)
        # If a whole row is masked (no real choices), fall back to a zero
        # row so downstream log_softmax doesn't produce NaN. Caller should
        # never feed an empty decision.
        any_legal = mask.sum(dim=-1, keepdim=True) > 0
        logits = torch.where(any_legal, logits, torch.zeros_like(logits))
        return logits, value
