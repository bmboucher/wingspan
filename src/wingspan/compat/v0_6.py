# pyright: reportPrivateUsage=false
# (the v0.6 shim reads the live layout's package-private offsets and dims to
# freeze the card encoder at the pre-0.7 geometry -- a deliberate compat
# coupling, pinned by the import-time layout-contract assertions below)
"""Frozen v0.6 card-feature encoding: the shim that keeps pre-0.7 artifacts playable.

Artifact version 0.7 added an ``or_cost`` flag to every bird's attribute vector,
growing ``CARD_FEATURE_DIM`` by 1 (224 → 225). The flag is 1.0 for OR-cost birds
(pay 1 accepted food OR 2 non-accepted) and 0.0 for AND-cost birds. State and
choice vector widths are unchanged; only the card encoder's first linear input
grew.

Nets trained before 0.7 have a 224-wide input to their card encoder; this module
keeps them loadable:

* :func:`card_feature_matrix_v06` rebuilds the ``[181, 224]`` feature table
  without the trailing ``or_cost`` flag — the exact matrix the old card encoder
  first-layer weights expect.
* :func:`_install_v06_card_encoder_main` wires the frozen 224-wide encoder into a
  :class:`~wingspan.model.core.PolicyValueNet` instance (called from
  :meth:`~wingspan.model.core.PolicyValueNet._build_card_encoder` overrides in
  this and the earlier shims).
* :func:`_install_v06_card_encoder_setup` does the same for
  :class:`~wingspan.training.setup_net.SetupNet` instances.
* :class:`PolicyValueNetV06` overrides :meth:`_build_card_encoder` to keep the
  224-wide input for 0.6 artifacts, and also overrides :meth:`encode_choices`
  to restore the v0.7 eggs-included ``becomes_playable`` food semantics (v0.6
  artifacts predate the 0.8 food-encoding fix, exactly as 0.7 artifacts do).
* :class:`SetupNetV06` does the same for the separately-trained setup net.
* :func:`uses_v0_6_card_feature_encoding` identifies which artifact versions need
  this shim (0.2 through 0.6, i.e. post-v0.1-card-feature-reshape, pre-0.7).

Per the compatibility policy (``CLAUDE.md``), this shim lives until a MAJOR
``MODEL_VERSION`` bump deletes it together with the pre-0.7 fixture set.
"""

from __future__ import annotations

import numpy as np
import torch

from wingspan import architecture, cards, decisions, encode, state, version
from wingspan.encode import layout
from wingspan.model import core, mlp
from wingspan.training import setup_net as setup_net_module

CARD_FEATURE_CHANGED_IN = "0.7"
"""The artifact version whose card-feature growth this module undoes."""

# ---------------------------------------------------------------------------
# Frozen v0.2–v0.6 card-feature geometry.
#
# v0.7 appends a single ``or_cost`` flag at the end of the per-card attribute
# block. Every pre-0.7 checkpoint therefore has a card encoder whose first linear
# maps 224 inputs (not 225) to its hidden width. The constants below are frozen
# literals so the frozen matrix is stable even if the live dims change again.

_OR_COST_FLAG_DIM = 1  # the stripe added in v0.7
_BIRD_ATTR_DIM_V06 = layout._BIRD_ATTR_DIM - _OR_COST_FLAG_DIM  # 44
_BIRD_ID_DIM_V06 = cards.n_birds()  # 180

CARD_FEATURE_DIM_V06 = layout.CARD_FEATURE_DIM - _OR_COST_FLAG_DIM  # 224
"""The pre-0.7 per-card feature vector width (224 = 44 attr dims + 180 identity dims)."""


def uses_v0_6_card_feature_encoding(artifact_version: str) -> bool:
    """Whether ``artifact_version`` uses the pre-0.7 224-wide card encoder and
    therefore needs this module's frozen geometry to load and play.

    Covers v0.2 through v0.6: the era after the 229→224 card-feature reshape
    (v0.2) and before the 224→225 or-cost addition (v0.7). Pre-0.2 artifacts
    are caught earlier in the routing chain by ``v0_1.uses_v0_1_card_feature_encoding``
    (229-wide encoder) and never reach this check."""
    parsed = version.parse_version(artifact_version)
    changed = version.parse_version(CARD_FEATURE_CHANGED_IN)
    card_reshaped = version.parse_version("0.2")
    return (parsed.major, parsed.minor) >= (
        card_reshaped.major,
        card_reshaped.minor,
    ) and (parsed.major, parsed.minor) < (changed.major, changed.minor)


def card_feature_matrix_v06() -> np.ndarray:
    """The frozen ``[181, 224]`` per-card feature table for pre-0.7 checkpoints.

    Row 0 is all zeros (padding / empty slot). Row ``bird_index + 1`` is the
    44-dim attribute vector (identical to the live vector minus the trailing
    ``or_cost`` flag) concatenated with the 180-wide bird-identity one-hot.
    Because the ``or_cost`` flag is the last attribute dim, the first 44 columns
    of each live row equal the v0.6 attribute block exactly — the flag column
    is simply omitted and the identity one-hot is placed at position 44 instead
    of 45."""
    rows = _BIRD_ID_DIM_V06 + 1
    matrix = np.zeros((rows, CARD_FEATURE_DIM_V06), dtype=np.float32)
    live = encode.card_feature_matrix()  # (181, 225) in v0.7
    for bird in cards.load_all()[0]:
        idx = cards.bird_index(bird)
        row = idx + 1
        # Copy the 44-dim attr block (the live first 44 columns exclude or_cost).
        matrix[row, :_BIRD_ATTR_DIM_V06] = live[row, :_BIRD_ATTR_DIM_V06]
        # Rebuild identity one-hot at the v0.6 position (44, not 45).
        matrix[row, _BIRD_ATTR_DIM_V06 + idx] = 1.0
    return matrix


def _install_v06_card_encoder_main(
    net: core.PolicyValueNet, arch: architecture.ModelArchitecture
) -> None:
    """Install the frozen 224-wide card encoder on a main-net instance.

    Called from ``_build_card_encoder`` overrides in every v0.2–v0.6
    :class:`~wingspan.model.core.PolicyValueNet` compat subclass to keep the
    card encoder first-linear weight shape at 224 inputs."""
    net.card_encoder, _ = mlp.build_body(
        CARD_FEATURE_DIM_V06,
        arch.card_encoder_layers + (arch.card_embed_dim,),
        between_activation=arch.card_between_activation_resolved,
        final_activation=arch.card_final_activation_resolved,
        dropout=arch.card_dropout_resolved,
        layernorm=arch.card_layernorm_resolved,
    )
    net.register_buffer(
        "card_features",
        torch.tensor(card_feature_matrix_v06(), dtype=torch.float32),
        persistent=False,
    )
    pad_mask = torch.ones(encode.HAND_MULTIHOT_DIM + 1, 1)
    pad_mask[0] = 0.0
    net.register_buffer("card_pad_mask", pad_mask, persistent=False)


def _install_v06_card_encoder_setup(
    net: setup_net_module.SetupNet, arch: architecture.ModelArchitecture
) -> None:
    """Install the frozen 224-wide card encoder on a setup-net instance.

    Same as :func:`_install_v06_card_encoder_main` but freezes the encoder
    weights (``requires_grad_(False)``)."""
    net.card_encoder, _ = mlp.build_body(
        CARD_FEATURE_DIM_V06,
        arch.card_encoder_layers + (arch.card_embed_dim,),
        between_activation=arch.card_between_activation_resolved,
        final_activation=arch.card_final_activation_resolved,
        dropout=arch.card_dropout_resolved,
        layernorm=arch.card_layernorm_resolved,
    )
    net.card_encoder.requires_grad_(False)
    net.register_buffer(
        "card_features",
        torch.tensor(card_feature_matrix_v06(), dtype=torch.float32),
        persistent=False,
    )
    pad_mask = torch.ones(encode.HAND_MULTIHOT_DIM + 1, 1)
    pad_mask[0] = 0.0
    net.register_buffer("card_pad_mask", pad_mask, persistent=False)


class PolicyValueNetV06(core.PolicyValueNet):
    """A :class:`~wingspan.model.core.PolicyValueNet` frozen to the v0.6 card-
    feature geometry, for checkpoints written before artifact version 0.7.

    The card encoder's first linear was trained against 224-wide input rows;
    this subclass overrides :meth:`_build_card_encoder` to keep that width and
    register the frozen v0.6 feature table. State encoding uses the 1155-dim
    pre-0.9 geometry (same as v0.8 — state has been 1155-dim since v0.6):
    :meth:`encode_state` and :meth:`_state_embed_offsets` delegate to the
    ``v0_8`` module so no state-encoding logic is duplicated here.

    :meth:`encode_choices` is also overridden to restore the v0.7 eggs-included
    ``becomes_playable`` food semantics: v0.6 artifacts predate the 0.8
    food-encoding fix and must compute the same bits as v0.7 checkpoints.

    Constructed by the version-routing loaders (``PolicyValueNet.from_model_config``,
    ``players.loaders.load_policy_net``) — never by the training pipeline.
    """

    def _build_card_encoder(self, arch: architecture.ModelArchitecture) -> None:
        """Register ``card_encoder`` at the frozen 224-wide input and
        ``card_features`` from the v0.6 feature table."""
        _install_v06_card_encoder_main(self, arch)

    def encode_state(
        self,
        game_state: state.GameState,
        decision: decisions.Decision[decisions.Choice],
    ) -> np.ndarray:
        """Featurize ``game_state`` with the 1155-dim pre-0.9 state geometry.

        Delegates to ``v0_8.encode_state_v08``: state has been 1155-dim since
        v0.6 (the playability-stripe bump) and the v0.8 shim reconstructs that
        exact vector."""
        import wingspan.compat.v0_8 as v0_8_module  # local: avoids import cycle

        return v0_8_module.encode_state_v08(game_state, decision, self.spec)

    def _state_embed_offsets(self) -> core.StateEmbedOffsets:
        """Frozen slice offsets for the 1155-dim pre-0.9 state vector."""
        import wingspan.compat.v0_8 as v0_8_module  # local: avoids import cycle

        return v0_8_module.state_embed_offsets_v08()

    def encode_choices(
        self,
        decision: decisions.Decision[decisions.Choice],
        game_state: state.GameState,
    ) -> np.ndarray:
        """Featurize all choices with eggs-included food ``becomes_playable``
        (v0.7 semantics), alongside the frozen 224-wide card encoder geometry."""
        import wingspan.compat.v0_7 as v0_7_module  # local: avoids import cycle

        return v0_7_module.encode_choices_v07(decision, game_state, self.spec)


class SetupNetV06(setup_net_module.SetupNet):
    """A :class:`~wingspan.training.setup_net.SetupNet` frozen to the v0.6 card-
    feature geometry, for setup checkpoints written before artifact version 0.7.

    The card encoder's first linear was trained against 224-wide input rows;
    this subclass overrides :meth:`_build_card_encoder` to keep that width and
    register the frozen v0.6 feature table. Constructed by
    :func:`~wingspan.players.loaders.load_setup_net` when the checkpoint version
    is in the v0.2–v0.6 range — never directly.
    """

    def _build_card_encoder(self, main_arch: architecture.ModelArchitecture) -> None:
        """Register ``card_encoder`` at the frozen 224-wide input and
        ``card_features`` from the v0.6 feature table."""
        _install_v06_card_encoder_setup(self, main_arch)


###### PRIVATE #######


def _assert_live_layout_contract() -> None:
    """Import-time pins for the invariants the shim relies on.

    The shim omits the ``or_cost`` stripe that v0.7 appends to the per-card
    attribute block. The frozen geometry is correct only while:

    1. ``_OR_COST_FLAG_DIM`` is 1 (exactly one new attr dim added in v0.7).
    2. The ``or_cost`` stripe is the LAST attr stripe (so dims 0..43 of the
       live attr block are identical to the v0.6 attr block, and the identity
       one-hot sits at position 44 in v0.6 vs 45 in v0.7).
    3. The bird catalog size is unchanged (180 birds).
    """
    assert layout._OR_COST_FLAG_DIM == 1, (
        f"v0.6 shim expects _OR_COST_FLAG_DIM == 1, "
        f"but found {layout._OR_COST_FLAG_DIM}; update the shim"
    )
    assert layout._OFF_ATTR_OR_COST == _BIRD_ATTR_DIM_V06, (
        f"v0.6 shim assumes or_cost is the last attr stripe (offset {_BIRD_ATTR_DIM_V06}), "
        f"but _OFF_ATTR_OR_COST is {layout._OFF_ATTR_OR_COST}; update the shim"
    )
    assert cards.n_birds() == _BIRD_ID_DIM_V06, (
        f"v0.6 shim freezes bird catalog at {_BIRD_ID_DIM_V06} birds, "
        f"but current catalog has {cards.n_birds()}; update the shim"
    )


_assert_live_layout_contract()
