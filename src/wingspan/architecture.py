"""The network topology descriptor: the editable *shape* of ``PolicyValueNet``.

:class:`ModelArchitecture` is the single, torch-free record of the network's
configurable topology — the per-block hidden-layer widths plus the activation,
dropout, LayerNorm, and shared card-embedding handles. It is the one vehicle
that:

* the configurator edits (via the flat fields it mirrors on
  :class:`wingspan.training.config.TrainConfig`),
* :class:`model.PolicyValueNet` builds itself from, and
* ``model_config.json`` serializes so a run's full topology can be read at a
  glance and the network reconstituted later.

It lives at the package top level (beside ``model`` / ``encode`` / ``decisions``)
so the top-level ``model`` module can import it without dragging in the training
package; kept torch-free (only ``pydantic`` / ``enum``) so ``config`` and
``runmeta`` import it without pulling in torch. The four blocks it sizes are the
state trunk, the per-choice encoder, the per-family scorer heads, and the value
head; the trunk and the choice encoder must end at the same width (the embedding
``H`` that is concatenated and fed to the scorers), which is the descriptor's one
genuine cross-field invariant.
"""

from __future__ import annotations

import enum
import typing

import pydantic

# A per-block list of hidden-layer widths (each width >= 1). The body blocks
# (trunk / choice encoder) additionally require at least one layer; the head
# blocks may be empty (a direct linear readout).
type Widths = tuple[typing.Annotated[int, pydantic.Field(ge=1)], ...]

# The weight-shape signature of an architecture: the parts that, if changed,
# make previously-trained weights unloadable (activation / dropout are excluded —
# they leave every tensor shape intact, so a resumed run may change them).
type ShapeKey = tuple[
    tuple[int, ...],  # trunk_layers
    tuple[int, ...],  # choice_layers
    tuple[int, ...],  # head_layers
    tuple[int, ...],  # value_layers
    bool,  # layernorm
    int,  # card_embed_dim
]


class ActivationName(enum.StrEnum):
    """The activation functions selectable for every MLP block."""

    RELU = "relu"
    GELU = "gelu"
    TANH = "tanh"
    SILU = "silu"
    LEAKY_RELU = "leaky_relu"


class ModelArchitecture(pydantic.BaseModel):
    """The full, reconstitutable topology of a :class:`model.PolicyValueNet`.

    Width lists are ordered input-to-output: ``trunk_layers=(256, 128)`` is a
    two-layer trunk that projects to 128. ``trunk_layers`` and ``choice_layers``
    must share a final width — both produce the embedding ``H`` that is
    concatenated (``2H``) and scored — which is checked after construction.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    trunk_layers: typing.Annotated[Widths, pydantic.Field(min_length=1)] = (128, 128)
    choice_layers: typing.Annotated[Widths, pydantic.Field(min_length=1)] = (128, 128)
    head_layers: Widths = (128,)
    value_layers: Widths = ()
    activation: ActivationName = ActivationName.RELU
    dropout: typing.Annotated[float, pydantic.Field(ge=0.0, lt=1.0)] = 0.0
    layernorm: bool = False
    card_embed_dim: typing.Annotated[int, pydantic.Field(ge=1)] = 64

    @pydantic.model_validator(mode="after")
    def _check_embedding_width(self) -> ModelArchitecture:
        """Trunk and choice encoder must end at the same width: their outputs are
        concatenated into the ``2H`` vector each scorer head reads."""
        if self.choice_layers[-1] != self.trunk_layers[-1]:
            raise ValueError(
                "choice_layers must end at the same width as trunk_layers "
                f"(got {self.choice_layers[-1]} vs {self.trunk_layers[-1]}) — both "
                "produce the embedding width H that is concatenated for scoring"
            )
        return self

    @property
    def embed_width(self) -> int:
        """The embedding width ``H`` the heads consume (the trunk / choice-encoder
        output width)."""
        return self.trunk_layers[-1]

    @property
    def shape_key(self) -> ShapeKey:
        """The weight-compatibility signature (everything that changes a tensor
        shape). Two architectures load each other's weights iff these agree."""
        return (
            self.trunk_layers,
            self.choice_layers,
            self.head_layers,
            self.value_layers,
            self.layernorm,
            self.card_embed_dim,
        )
