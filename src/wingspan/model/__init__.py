"""PyTorch policy-value network and supporting MLP/set-encoder blocks.

Re-exports ``PolicyValueNet`` (and the ``StateEmbedOffsets`` seam its compat
subclasses override) so all existing ``model.PolicyValueNet`` consumers continue
to work without change after the module became a package.
"""

from __future__ import annotations

from wingspan.model.core import PolicyValueNet, StateEmbedOffsets

__all__ = ["PolicyValueNet", "StateEmbedOffsets"]
