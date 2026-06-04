"""The setup network's topology descriptor — the editable *shape* of ``SetupNet``.

:class:`SetupArchitecture` is the small, torch-free analogue of
:class:`wingspan.architecture.ModelArchitecture` for the separately-trained setup
model: the readout MLP that scores one setup candidate at a time. The setup model
is a value-regression contextual bandit (it predicts the expected end-of-game
score margin a setup leads to), so the descriptor only needs the hidden-layer
widths plus the activation and dropout handles. The network's card-embedding
blocks are *not* described here — they are frozen copies of the main net's
shared embedders, so their shapes come from the main
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

# A setup-net shape signature: the hidden-layer widths. Activation / dropout are
# excluded for the same reason ``ModelArchitecture.shape_key`` excludes them —
# they leave every tensor shape intact, so a resumed run may change them without
# invalidating the saved weights.
type SetupShapeKey = tuple[int, ...]


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

    @property
    def shape_key(self) -> SetupShapeKey:
        """The readout MLP's weight-compatibility signature (everything that
        changes one of *its* tensor shapes). The embedder copies' shapes ride the
        main architecture and are keyed separately
        (``TrainConfig.setup_architecture_key``)."""
        return self.hidden_layers


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
    return architecture.BlockParam(
        label="SETUP",
        layers=architecture.readout_layers(readout_in, setup_arch.hidden_layers),
        extra=embedder_params,
    )
