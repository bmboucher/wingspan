"""The setup model network: a small MLP value-regressor over a setup candidate.

``SetupNet`` consumes one
:func:`wingspan.setup_model.encode.encode_setup_candidate` feature vector and
emits a single scalar — its predicted end-of-game score margin (normalized by
``score_norm`` during training). The setup policy is a softmax over the predicted
margins of all 504 candidate keeps for a dealt hand
(``setup_model.select_by_margins``).

This is deliberately plain: no shared card embedding (the encoder is multi-hot),
no per-family heads — just ``Linear → activation → … → Linear(·, 1)`` built from a
:class:`wingspan.setup_model.SetupArchitecture`, so the whole network is a few
tens of thousands of parameters and runs comfortably on CPU.
"""

from __future__ import annotations

import typing

import torch
from torch import nn

from wingspan import architecture, setup_model

if typing.TYPE_CHECKING:
    from wingspan.training import setup_runmeta

# The selectable activations, keyed by the descriptor enum (same set the main
# net offers; kept local so this module needs nothing private from ``model``).
_ACTIVATIONS: dict[architecture.ActivationName, typing.Callable[[], nn.Module]] = {
    architecture.ActivationName.RELU: nn.ReLU,
    architecture.ActivationName.GELU: nn.GELU,
    architecture.ActivationName.TANH: nn.Tanh,
    architecture.ActivationName.SILU: nn.SiLU,
    architecture.ActivationName.LEAKY_RELU: nn.LeakyReLU,
}


class SetupNet(nn.Module):
    """A scalar-output MLP scoring one setup candidate (see module docstring)."""

    def __init__(
        self,
        *,
        feature_dim: int = setup_model.SETUP_FEATURE_DIM,
        arch: setup_model.SetupArchitecture | None = None,
    ):
        super().__init__()
        if arch is None:
            arch = setup_model.SetupArchitecture()
        self.feature_dim = feature_dim
        self.arch = arch
        modules: list[nn.Module] = []
        prev = feature_dim
        for width in arch.hidden_layers:
            modules.append(nn.Linear(prev, width))
            modules.append(_ACTIVATIONS[arch.activation]())
            if arch.dropout > 0.0:
                modules.append(nn.Dropout(arch.dropout))
            prev = width
        modules.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*modules)

    @classmethod
    def from_setup_config(cls, descriptor: "setup_runmeta.SetupConfig") -> "SetupNet":
        """Rebuild a net matching a saved ``setup_config.json`` descriptor — fresh
        weights in the saved shape, ready for ``load_state_dict``."""
        return cls(feature_dim=descriptor.setup_feature_dim, arch=descriptor.setup_arch)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Score a batch of setup candidates: ``(B, feature_dim)`` -> ``(B,)``."""
        return self.mlp(features).squeeze(-1)
