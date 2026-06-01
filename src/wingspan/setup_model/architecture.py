"""The setup network's topology descriptor — the editable *shape* of ``SetupNet``.

:class:`SetupArchitecture` is the small, torch-free analogue of
:class:`wingspan.architecture.ModelArchitecture` for the separately-trained setup
model: a plain MLP that scores one setup candidate at a time. The setup model is
a value-regression contextual bandit (it predicts the expected end-of-game score
margin a setup leads to), so the descriptor only needs the hidden-layer widths
plus the activation and dropout handles — there is no per-card embedding (the
setup encoder uses plain multi-hot inputs) and no second block to keep widths in
sync with.

Kept torch-free (only ``pydantic`` / ``enum``) so ``config`` and ``setup_runmeta``
can import it without pulling in torch, mirroring why ``ModelArchitecture`` lives
at the package top level.
"""

from __future__ import annotations

import typing

import pydantic

from wingspan import architecture

# A setup-net shape signature: the hidden-layer widths. Activation / dropout are
# excluded for the same reason ``ModelArchitecture.shape_key`` excludes them —
# they leave every tensor shape intact, so a resumed run may change them without
# invalidating the saved weights.
type SetupShapeKey = tuple[int, ...]


class SetupArchitecture(pydantic.BaseModel):
    """The full, reconstitutable topology of a :class:`wingspan.training.setup_net.SetupNet`.

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
        """The weight-compatibility signature (everything that changes a tensor
        shape). Two setup nets load each other's weights iff these agree."""
        return self.hidden_layers
