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
head. The trunk and the choice encoder may end at different widths ``M`` and
``N``; their outputs are concatenated to ``M+N`` before the scorer heads.
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
    two-layer trunk that projects to 128. The trunk ends at width ``M``
    (``trunk_embed_width``) and the choice encoder ends at width ``N``
    (``choice_embed_width``); both are independent and their outputs are
    concatenated to ``M+N`` before the scorer heads.
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

    @property
    def trunk_embed_width(self) -> int:
        """The trunk's output width ``M`` — what the scorer and value heads consume."""
        return self.trunk_layers[-1]

    @property
    def choice_embed_width(self) -> int:
        """The choice encoder's output width ``N`` — concatenated with ``M`` for
        scoring."""
        return self.choice_layers[-1]

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


class LayerParam(pydantic.BaseModel):
    """The parameter count of one ``Linear`` layer and the ``LayerNorm`` that may
    follow it. ``linear`` is ``in*out + out`` (weight + bias); ``norm`` is
    ``2*out`` when a LayerNorm follows (its affine weight + bias), else 0. They are
    tracked apart so the diagram can annotate the Linear row with the dominant
    ``linear`` cost while ``params`` still rolls the LayerNorm into the block total."""

    in_features: int
    out_features: int
    linear: int
    norm: int = 0

    @property
    def params(self) -> int:
        """Total trainable parameters in this layer (Linear + any LayerNorm)."""
        return self.linear + self.norm


class BlockParam(pydantic.BaseModel):
    """One network block's parameter breakdown: its per-layer counts plus a flat
    ``extra`` (the embedding table), scaled by ``multiplier`` — the scorer bank
    instantiates one identical readout head per decision family."""

    label: str
    layers: tuple[LayerParam, ...] = ()
    multiplier: int = 1
    extra: int = 0

    @property
    def total(self) -> int:
        """The block's trainable-parameter count, including the family multiplier."""
        return self.multiplier * (
            sum(layer.params for layer in self.layers) + self.extra
        )


class ParamReport(pydantic.BaseModel):
    """The whole network's parameter accounting, block by block — the display
    source for the configurator's per-layer / per-block / total counts. Built by
    :func:`count_parameters`; its block totals sum to ``sum(p.numel())`` of the
    equivalent :class:`model.PolicyValueNet`."""

    embed: BlockParam
    trunk: BlockParam
    choice: BlockParam
    scorer: BlockParam
    value: BlockParam

    @property
    def blocks(self) -> tuple[BlockParam, ...]:
        """The five blocks in flow order (embed, trunk, choice, scorer, value)."""
        return (self.embed, self.trunk, self.choice, self.scorer, self.value)

    @property
    def total(self) -> int:
        """The network's total trainable-parameter count."""
        return sum(block.total for block in self.blocks)


def count_parameters(
    arch: ModelArchitecture,
    *,
    trunk_in: int,
    choice_in: int,
    embed_rows: int,
    num_families: int,
) -> ParamReport:
    """Analytic per-block parameter accounting for the network ``arch`` describes.

    Torch-free — it reproduces the exact layer shapes ``model._build_body`` /
    ``_build_readout`` would create, so the returned counts equal
    ``sum(p.numel())`` of the built net. The effective post-embedding input widths
    (``trunk_in`` / ``choice_in``, from ``encode.{trunk,choice}_input_dim``) and the
    embedding-table row count are passed in to keep this module free of the
    encoder / torch.
    """
    # The trunk ends at width M and the choice encoder at width N; the scorer
    # heads read the M+N concat and the value head reads the trunk's M alone,
    # mirroring ``model.PolicyValueNet.__init__``.
    trunk_m = arch.trunk_embed_width
    scorer_in = arch.trunk_embed_width + arch.choice_embed_width
    return ParamReport(
        embed=BlockParam(label="EMBED", extra=embed_rows * arch.card_embed_dim),
        trunk=BlockParam(
            label="TRUNK", layers=_body_layers(trunk_in, arch.trunk_layers, arch)
        ),
        choice=BlockParam(
            label="CHOICE", layers=_body_layers(choice_in, arch.choice_layers, arch)
        ),
        scorer=BlockParam(
            label="SCORER",
            layers=_readout_layers(scorer_in, arch.head_layers, arch),
            multiplier=num_families,
        ),
        value=BlockParam(
            label="VALUE", layers=_readout_layers(trunk_m, arch.value_layers, arch)
        ),
    )


def _linear_params(in_features: int, out_features: int) -> int:
    """Parameter count of ``nn.Linear(in, out)`` — the weight plus the bias."""
    return in_features * out_features + out_features


def _body_layers(
    in_dim: int, widths: Widths, arch: ModelArchitecture
) -> tuple[LayerParam, ...]:
    """Per-layer counts for a body block (trunk / choice encoder): each width is a
    ``Linear`` followed — when ``arch.layernorm`` — by a ``LayerNorm`` on every
    layer, mirroring ``model._build_body`` (activation / dropout add no params)."""
    layers: list[LayerParam] = []
    prev = in_dim
    for width in widths:
        norm = 2 * width if arch.layernorm else 0
        layers.append(
            LayerParam(
                in_features=prev,
                out_features=width,
                linear=_linear_params(prev, width),
                norm=norm,
            )
        )
        prev = width
    return tuple(layers)


def _readout_layers(
    in_dim: int, widths: Widths, arch: ModelArchitecture
) -> tuple[LayerParam, ...]:
    """Per-layer counts for a readout block (scorer head / value head): the hidden
    ``widths`` as bare ``Linear`` layers (readouts never LayerNorm) then a final
    ``Linear(prev, 1)``, mirroring ``model._build_readout``."""
    layers: list[LayerParam] = []
    prev = in_dim
    for width in widths:
        layers.append(
            LayerParam(
                in_features=prev, out_features=width, linear=_linear_params(prev, width)
            )
        )
        prev = width
    layers.append(
        LayerParam(in_features=prev, out_features=1, linear=_linear_params(prev, 1))
    )
    return tuple(layers)
