"""PyTorch model — a simple actor-critic-style network with action masking.

This is intentionally small. The state vector is ~90 dims and the action
space is ~80 slots; a 2-layer MLP is plenty for a first training cycle.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encode import TOTAL_ACTION_SLOTS, state_size


class PolicyValueNet(nn.Module):
    def __init__(self, state_dim: int = None, action_dim: int = TOTAL_ACTION_SLOTS, hidden: int = 128):
        super().__init__()
        if state_dim is None:
            state_dim = state_size()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, action_dim)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, state: torch.Tensor, mask: torch.Tensor):
        h = self.trunk(state)
        logits = self.policy_head(h)
        # Apply legality mask: -inf on illegal slots
        logits = logits.masked_fill(mask < 0.5, float("-inf"))
        # Guard against entirely-illegal rows (shouldn't happen but safe)
        all_illegal = mask.sum(dim=-1, keepdim=True) == 0
        logits = torch.where(all_illegal, torch.zeros_like(logits), logits)
        value = self.value_head(h).squeeze(-1)
        return logits, value
