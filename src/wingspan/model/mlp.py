"""Shared MLP block builders for the policy net and the setup net.

The two networks build their layer stacks from the same two recipes — a *body*
block (``Linear`` → optional ``LayerNorm`` → activation → optional ``Dropout``
per layer) and a scalar *readout* block (hidden ``Linear`` + between-activation
layers, then a final ``Linear(·, 1)`` optionally followed by a final activation).
They are factored here, taking **scalar** activation / dropout / layernorm handles
rather than a full :class:`wingspan.architecture.ModelArchitecture`, so the setup
net (whose own descriptor is a :class:`wingspan.setup_model.SetupArchitecture`)
builds byte-identical ``nn.Sequential`` stacks — identical module indices, so
``load_state_dict`` syncs a frozen copy exactly.

``ActivationName.NONE`` on any activation slot causes that module to be skipped
entirely (no ``nn.Identity`` inserted — skipping keeps module indices matching
those of checkpoints where the activation was absent by ``final_activation=False``
in the pre-between/final API).
"""

from __future__ import annotations

import typing

from torch import nn

from wingspan import architecture

# The selectable activation functions, keyed by their descriptor enum. Each maps
# to a zero-argument ``nn.Module`` factory. NONE is intentionally absent — an
# accidental ``ACTIVATIONS[NONE]`` raises ``KeyError`` loudly.
ACTIVATIONS: dict[architecture.ActivationName, typing.Callable[[], nn.Module]] = {
    architecture.ActivationName.RELU: nn.ReLU,
    architecture.ActivationName.GELU: nn.GELU,
    architecture.ActivationName.TANH: nn.Tanh,
    architecture.ActivationName.SILU: nn.SiLU,
    architecture.ActivationName.LEAKY_RELU: nn.LeakyReLU,
}

_NONE = architecture.ActivationName.NONE


def build_body(
    in_dim: int,
    widths: architecture.Widths,
    *,
    between_activation: architecture.ActivationName,
    final_activation: architecture.ActivationName,
    dropout: float,
    layernorm: bool,
) -> tuple[nn.Sequential, int]:
    """Build a body MLP — a trunk, the choice encoder, or a card/hand encoder —
    and return it with its output width.

    Each layer is ``Linear`` → (optional) ``LayerNorm`` → activation → (optional)
    ``Dropout``. Non-final layers use ``between_activation``; the last layer uses
    ``final_activation``. Either activation may be ``NONE`` to skip that module
    (and the paired ``Dropout``) entirely. LayerNorm — when enabled — is applied
    on every layer independent of activation."""
    modules: list[nn.Module] = []
    prev = in_dim
    last_index = len(widths) - 1
    for index, width in enumerate(widths):
        modules.append(nn.Linear(prev, width))
        if layernorm:
            modules.append(nn.LayerNorm(width))
        act = final_activation if index == last_index else between_activation
        if act is not _NONE:
            modules.append(ACTIVATIONS[act]())
            if dropout > 0.0:
                modules.append(nn.Dropout(dropout))
        prev = width
    return nn.Sequential(*modules), prev


def build_readout(
    in_dim: int,
    widths: architecture.Widths,
    *,
    between_activation: architecture.ActivationName,
    final_activation: architecture.ActivationName,
    dropout: float,
) -> nn.Sequential:
    """Build a scalar-readout MLP (a scorer head, the value head, or the setup
    net's MLP).

    Hidden ``widths`` layers are ``Linear`` → ``between_activation`` →
    (optional) ``Dropout``; a ``NONE`` between-activation skips the activation
    and its paired dropout. The block ends with a final ``Linear(·, 1)`` followed
    by ``final_activation`` when not ``NONE``. Empty ``widths`` collapses to a
    single ``Linear(in_dim, 1)``."""
    modules: list[nn.Module] = []
    prev = in_dim
    for width in widths:
        modules.append(nn.Linear(prev, width))
        if between_activation is not _NONE:
            modules.append(ACTIVATIONS[between_activation]())
            if dropout > 0.0:
                modules.append(nn.Dropout(dropout))
        prev = width
    modules.append(nn.Linear(prev, 1))
    if final_activation is not _NONE:
        modules.append(ACTIVATIONS[final_activation]())
    return nn.Sequential(*modules)
