"""The setup network's topology descriptor — the editable *shape* of ``SetupNet``.

:class:`SetupArchitecture` is the small, torch-free analogue of
:class:`wingspan.architecture.ModelArchitecture` for the separately-trained setup
model: the readout MLP that scores one setup candidate at a time. The setup model
supports two training modes:

* **Value-only** (``use_policy_head=False``, default): a single scalar output
  trained with MSE regression against realized score margin.
* **Actor-critic** (``use_policy_head=True``): two scalar outputs — a value head
  (MSE critic) and a policy head (REINFORCE actor); candidate selection at
  collection time uses policy-head logits.

The network's card-embedding blocks are *not* described here — they are frozen
copies of the main net's shared embedders, so their shapes come from the main
:class:`~wingspan.architecture.ModelArchitecture` (threaded into
:func:`setup_readout_input_dim` / :func:`count_setup_parameters`).

Kept torch-free (only ``pydantic`` / ``enum``) so ``config`` and ``setup_runmeta``
can import it without pulling in torch, mirroring why ``ModelArchitecture`` lives
at the package top level.
"""

from __future__ import annotations

import typing

import pydantic

from wingspan import architecture, cards, encode, state

# A setup-net shape signature: the hidden-layer widths plus whether the policy
# head is present. Activation / dropout are excluded — they leave tensor shapes
# intact and a resumed run may change them without invalidating the saved weights.
type SetupShapeKey = tuple[tuple[int, ...], bool]


class SetupArchitecture(pydantic.BaseModel):
    """The reconstitutable topology of a :class:`wingspan.training.setup_net.SetupNet`'s
    readout MLP.

    ``hidden_layers`` is ordered input-to-output: ``(128, 64)`` is a two-layer
    MLP projecting to 64 before the final scalar readout. Reuses
    :data:`wingspan.architecture.Widths` and
    :class:`wingspan.architecture.ActivationName` so the configurator's layer /
    activation editors apply unchanged.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    hidden_layers: typing.Annotated[
        architecture.Widths, pydantic.Field(min_length=1)
    ] = (
        128,
        64,
    )
    activation: architecture.ActivationName = architecture.ActivationName.RELU
    dropout: typing.Annotated[float, pydantic.Field(ge=0.0, lt=1.0)] = 0.0
    # When True the net has two scalar readout MLPs (value head + policy head)
    # of identical shape, enabling actor-critic training. Included in the shape
    # key because it adds a second set of Linear layers to the network.
    use_policy_head: bool = False

    @property
    def shape_key(self) -> SetupShapeKey:
        """The readout MLP's weight-compatibility signature (everything that
        changes one of *its* tensor shapes). The embedder copies' shapes ride the
        main architecture and are keyed separately
        (``TrainConfig.setup_architecture_key``)."""
        return (self.hidden_layers, self.use_policy_head)


def setup_readout_input_dim(
    feature_dim: int, main_arch: architecture.ModelArchitecture
) -> int:
    """The setup readout MLP's first-``Linear`` input width: the raw
    ``feature_dim`` vector with the kept-cards multi-hot replaced by one
    ``N``-wide set embedding and the tray index columns replaced by
    ``TRAY_SIZE`` ``M``-wide card-table rows plus one more ``N``-wide tray-set
    embedding (``M = card_embed_dim``, ``N = hand_embed_width``). The single
    source of truth shared by ``SetupNet`` and the parameter accounting."""
    passthrough = feature_dim - cards.n_birds() - state.TRAY_SIZE
    return (
        passthrough
        + 2 * main_arch.hand_embed_width
        + state.TRAY_SIZE * main_arch.card_embed_dim
    )


def count_setup_parameters(
    setup_arch: SetupArchitecture,
    *,
    feature_dim: int,
    main_arch: architecture.ModelArchitecture | None = None,
) -> architecture.BlockParam:
    """Analytic per-layer parameter accounting for the ``SetupNet`` that
    ``setup_arch`` describes.

    The setup net is a readout MLP (``setup_readout_input_dim → hidden… → 1``,
    Linears only, no LayerNorm) over the in-net embedding of the raw features,
    plus its two frozen embedder copies — the card encoder and the set (hand)
    encoder, whose shapes come from ``main_arch`` (``None`` = a bare
    :class:`~wingspan.architecture.ModelArchitecture`, matching ``SetupNet``'s
    default). The readout's per-layer counts are
    :func:`wingspan.architecture.readout_layers`; the embedder copies are rolled
    into ``extra`` (frozen parameters still count in ``numel``). Returns one
    :class:`wingspan.architecture.BlockParam` whose ``total`` equals
    ``sum(p.numel())`` of the built ``SetupNet`` — the architecture diagram's
    per-op and Σ source for the separate setup model.
    """
    if main_arch is None:
        main_arch = architecture.ModelArchitecture()
    embedder_params = sum(
        layer.params
        for layer in architecture.body_layers(
            encode.CARD_FEATURE_DIM,
            main_arch.card_encoder_layers + (main_arch.card_embed_dim,),
            main_arch,
        )
    ) + sum(
        layer.params
        for layer in architecture.body_layers(
            encode.HAND_ENCODER_INPUT_DIM,
            main_arch.hand_encoder_layers + (main_arch.hand_embed_width,),
            main_arch,
        )
    )
    readout_in = setup_readout_input_dim(feature_dim, main_arch)
    # When the policy head is present there are two readout MLPs of identical
    # shape (value + policy), so their parameter count doubles.
    readout = architecture.readout_layers(readout_in, setup_arch.hidden_layers)
    n_heads = 2 if setup_arch.use_policy_head else 1
    return architecture.BlockParam(
        label="SETUP",
        layers=readout * n_heads,
        extra=embedder_params,
    )
