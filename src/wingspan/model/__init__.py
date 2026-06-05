"""PyTorch policy-value network and supporting MLP/set-encoder blocks.

Re-exports ``PolicyValueNet`` so all existing ``model.PolicyValueNet``
consumers continue to work without change after the module became a package.
"""

from __future__ import annotations

from wingspan.model.core import PolicyValueNet

__all__ = ["PolicyValueNet"]
