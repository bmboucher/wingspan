"""PyTorch model: pointer-style actor-critic with per-family policy heads.

The network scores each candidate at a decision point individually rather
than emitting a fixed-slot policy head. At each forward pass the caller
provides:

* ``state``      — ``(B, state_dim)`` POV-aware game features
* ``choices``    — ``(B, K, choice_dim)`` per-choice features (padded)
* ``mask``       — ``(B, K)`` 1.0 for real choices, 0.0 for padding
* ``family_idx`` — ``(B,)`` long; the judgment-family head index for each
  decision (``decisions.family_index_for``)

The trunk reads the state, a separate MLP reads each choice, the two are
concatenated and passed through the scoring head **for that decision's
judgment family** to produce one logit per candidate. The trunk and the
per-choice encoder are shared across all families; only the final scorer
specializes (one head per ``decisions.DecisionFamily``), and the value head
is shared because position value is a property of the board, not of the
decision being asked. Padding rows get ``-inf`` so they never receive
probability mass.

This realizes DECISIONS.md §0's "shared trunk + per-family heads": the single
monolithic scorer is replaced by a family-routed bank of scorers, turning
"one model conditioned on a decision-type one-hot" into "one model per kind
of choice" without multiplying the trunk or starving the shared critic.
"""

from __future__ import annotations

import torch
from torch import nn

from wingspan import decisions, encode


class PolicyValueNet(nn.Module):
    """Actor-critic over (state, choice-set) decisions with per-family heads.

    A two-layer state trunk feeds (a) a state-context vector used to rescore
    every candidate and (b) the shared value head. A two-layer per-choice
    encoder consumes the per-choice features. The score for choice ``i`` is an
    MLP over ``concat(state_ctx, choice_emb[i])``, selected per decision by its
    judgment family (``decisions.ALL_DECISION_FAMILIES``) so different kinds of
    choice are scored by different, specialized heads.
    """

    def __init__(
        self,
        state_dim: int | None = None,
        choice_dim: int = encode.CHOICE_FEATURE_DIM,
        hidden: int = 128,
        num_families: int = len(decisions.ALL_DECISION_FAMILIES),
    ):
        super().__init__()
        if state_dim is None:
            state_dim = encode.state_size()
        self.state_dim = state_dim
        self.choice_dim = choice_dim
        self.hidden = hidden
        self.num_families = num_families

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
        # One scoring head per judgment family. The trunk + choice-encoder are
        # shared; specialization lives here. ``family_idx`` routes each
        # decision to its head in ``forward``.
        self.scorers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden * 2, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )
            for _ in range(num_families)
        )
        self.value_head = nn.Linear(hidden, 1)

    def forward(
        self,
        state: torch.Tensor,
        choices: torch.Tensor,
        mask: torch.Tensor,
        family_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score every candidate at every decision in the batch.

        Args:
            state:      ``(B, state_dim)``
            choices:    ``(B, K, choice_dim)``  — pad rows are arbitrary
            mask:       ``(B, K)`` with 1.0 on real choices, 0.0 on padding.
            family_idx: ``(B,)`` long — judgment-family head index per decision
                (``decisions.family_index_for``); each value in
                ``[0, num_families)``.

        Returns:
            logits: ``(B, K)`` — masked rows are set to ``-inf``
            value:  ``(B,)``
        """
        # State trunk produces both the per-decision context and the value.
        state_ctx = self.state_trunk(state)  # (B, H)
        value = self.value_head(state_ctx).squeeze(-1)  # (B,)

        # Per-choice MLP. choices is (B, K, F); the Linear layers broadcast
        # across the K dimension naturally.
        ce = self.choice_encoder(choices)  # (B, K, H)
        num_choices = ce.shape[1]
        s_exp = state_ctx.unsqueeze(1).expand(-1, num_choices, -1)  # (B, K, H)
        combined = torch.cat([s_exp, ce], dim=-1)  # (B, K, 2H)

        # Route each decision through its judgment family's scoring head. Every
        # candidate in a decision shares one head (family is a property of the
        # decision, not the candidate), so we slice the batch by family, score
        # each slice with its head, and scatter the logits back. Disjoint row
        # sets cover the whole batch, so every row is scored exactly once.
        scores = combined.new_zeros(combined.shape[:2])  # (B, K)
        for family in range(self.num_families):
            rows = family_idx == family
            if not bool(rows.any()):
                continue
            scores = scores.index_copy(
                0,
                rows.nonzero(as_tuple=True)[0],
                self.scorers[family](combined[rows]).squeeze(-1),
            )

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
