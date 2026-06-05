"""Shared MLP block builders for the policy net and the setup net.

The two networks build their layer stacks from the same two recipes — a *body*
block (``Linear`` → optional ``LayerNorm`` → activation → optional ``Dropout``
per layer) and a scalar *readout* block (hidden ``Linear`` + activation layers,
then a bare ``Linear(·, 1)``). They are factored here, taking **scalar**
``activation`` / ``dropout`` / ``layernorm`` handles rather than a full
:class:`wingspan.architecture.ModelArchitecture`, so the setup net (whose own
descriptor is a :class:`wingspan.setup_model.SetupArchitecture`) builds
byte-identical ``nn.Sequential`` stacks — identical module indices, so
``load_state_dict`` syncs a frozen copy exactly.
"""

from __future__ import annotations

import typing

from torch import nn

from wingspan import architecture

# The selectable activation functions, keyed by their descriptor enum. Each maps
# to a zero-argument ``nn.Module`` factory.
ACTIVATIONS: dict[architecture.ActivationName, typing.Callable[[], nn.Module]] = {
    architecture.ActivationName.RELU: nn.ReLU,
    architecture.ActivationName.GELU: nn.GELU,
    architecture.ActivationName.TANH: nn.Tanh,
    architecture.ActivationName.SILU: nn.SiLU,
    architecture.ActivationName.LEAKY_RELU: nn.LeakyReLU,
}


def build_body(
    in_dim: int,
    widths: architecture.Widths,
    *,
    activation: architecture.ActivationName,
    dropout: float,
    layernorm: bool,
    final_activation: bool,
) -> tuple[nn.Sequential, int]:
    """Build a body MLP — a trunk, the choice encoder, or a card/hand encoder —
    and return it with its output width. Each layer is ``Linear`` → (optional)
    ``LayerNorm`` → activation → (optional) ``Dropout``; the activation + dropout
    on the final layer are emitted only when ``final_activation`` (the trunk keeps
    a trailing activation, the encoders do not). LayerNorm — when enabled — is
    applied to these body blocks; the readout heads omit it."""
    modules: list[nn.Module] = []
    prev = in_dim
    last_index = len(widths) - 1
    for index, width in enumerate(widths):
        modules.append(nn.Linear(prev, width))
        if layernorm:
            modules.append(nn.LayerNorm(width))
        if final_activation or index != last_index:
            modules.append(ACTIVATIONS[activation]())
            if dropout > 0.0:
                modules.append(nn.Dropout(dropout))
        prev = width
    return nn.Sequential(*modules), prev


def build_readout(
    in_dim: int,
    widths: architecture.Widths,
    *,
    activation: architecture.ActivationName,
    dropout: float,
) -> nn.Sequential:
    """Build a scalar-readout MLP (a scorer head, the value head, or the setup
    net's MLP): the hidden ``widths`` as ``Linear`` → activation → (optional)
    ``Dropout`` blocks, then a final ``Linear(·, 1)`` with no activation. Empty
    ``widths`` collapses to a single ``Linear(in_dim, 1)``."""
    modules: list[nn.Module] = []
    prev = in_dim
    for width in widths:
        modules.append(nn.Linear(prev, width))
        modules.append(ACTIVATIONS[activation]())
        if dropout > 0.0:
            modules.append(nn.Dropout(dropout))
        prev = width
    modules.append(nn.Linear(prev, 1))
    return nn.Sequential(*modules)
