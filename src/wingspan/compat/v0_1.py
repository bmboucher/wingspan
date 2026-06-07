# pyright: reportPrivateUsage=false
# (the v0.1 -> v0.2 transform reads the live layout's package-private stripe
# offsets and overrides the net's private builder -- a deliberate compat
# coupling, pinned by the import-time layout-contract assertions below)
"""Frozen v0.1 card-feature encoding: the shim that keeps pre-0.2 artifacts playable.

Artifact version 0.2 redesigned the card feature vector (``CARD_FEATURE_DIM``
229 → 224): ``bonus_categories`` was trimmed from 26 dims (all bonus cards) to
7 curated intrinsic-property categories, a new ``caches_food`` flag was added,
and a 13-dim ``power_exchange`` stripe was appended. Nets trained before 0.2
have a 229-wide input to their card encoder, so this module keeps them loadable:

* :func:`card_feature_matrix` rebuilds the [181, 229] feature table using the
  frozen v0.1 geometry: the unchanged leading 23-dim attribute prefix (points
  through plays-another-bird) concatenated with the 26-wide bonus-categories
  multi-hot (one bit per bonus card, keyed to ``cards.bonus_index()``), then
  the 180-wide bird-identity one-hot.
* :class:`PolicyValueNetV01` overrides :meth:`_build_card_encoder` so the card
  encoder's first linear uses the frozen 229-wide input and loads the v0.1
  feature table. State and choice encoding are identical to the live era.

Per the compatibility policy (``CLAUDE.md``), this shim lives until a MAJOR
``MODEL_VERSION`` bump deletes it together with the v0.1 fixture set.
"""

from __future__ import annotations

import numpy as np
import torch

from wingspan import architecture, cards, encode, version
from wingspan.encode import layout
from wingspan.model import core, mlp
from wingspan.training import setup_net as setup_net_module

CARD_FEATURE_CHANGED_IN = "0.2"
"""The artifact version whose card-feature reshape this module undoes."""

# ---------------------------------------------------------------------------
# Frozen v0.1 card-feature geometry — literal copies of the attribute layout
# every pre-0.2 checkpoint was trained against. The bird-identity one-hot
# always follows at position _BIRD_ATTR_DIM_V01. Catalog-derived sizes are
# frozen as literals deliberately: the v0.1 row format must not move even if
# the live catalog or layout does.

_ATTR_LEADING_DIM = 23  # unchanged prefix: points through plays-another-bird
_BONUS_ID_DIM_V01 = 26  # cards.n_bonus_cards() at v0.1 — one-hot per bonus card
_BIRD_ATTR_DIM_V01 = _ATTR_LEADING_DIM + _BONUS_ID_DIM_V01  # 49
_BIRD_ID_DIM_V01 = 180  # cards.n_birds() at v0.1
CARD_FEATURE_DIM_V01 = _BIRD_ATTR_DIM_V01 + _BIRD_ID_DIM_V01  # 229

# The v0.1 bonus-categories stripe mapped each bonus card's printed name to
# its dense ``cards.bonus_index()`` position (0..25). Frozen here so the
# feature rebuild does not depend on the live (pruned) ``_BONUS_NAME_TO_INDEX``.
_BONUS_INDEX_V01: dict[str, int] = {
    bonus_card.name: cards.bonus_index(bonus_card) for bonus_card in cards.load_all()[1]
}


def uses_v0_1_card_feature_encoding(artifact_version: str) -> bool:
    """Whether ``artifact_version`` predates the 0.2 card-feature reshape and
    therefore needs this module's frozen geometry to load and play."""
    parsed = version.parse_version(artifact_version)
    changed = version.parse_version(CARD_FEATURE_CHANGED_IN)
    return (parsed.major, parsed.minor) < (changed.major, changed.minor)


def card_feature_matrix() -> np.ndarray:
    """The frozen v0.1 ``[181, 229]`` feature table for the card encoder.

    Row 0 is all zeros (padding / empty slot). Row ``bird_index + 1`` is the
    unchanged 23-dim attribute prefix (points through plays-another-bird),
    followed by the 26-wide bonus-categories multi-hot keyed to all 26 bonus
    cards via ``cards.bonus_index()``, then the 180-wide bird-identity one-hot.
    This matches the exact on-disk geometry of every pre-0.2 checkpoint's
    card-encoder first-layer weights."""
    rows = _BIRD_ID_DIM_V01 + 1
    matrix = np.zeros((rows, CARD_FEATURE_DIM_V01), dtype=np.float32)
    live_matrix = encode.card_feature_matrix()
    for bird in cards.load_all()[0]:
        idx = cards.bird_index(bird)
        row = idx + 1

        # The leading 23 dims (points through plays-another-bird) are identical
        # between v0.1 and v0.2; copy them from the live feature matrix.
        matrix[row, :_ATTR_LEADING_DIM] = live_matrix[row, :_ATTR_LEADING_DIM]

        # Rebuild the old 26-wide bonus-categories multi-hot.
        for category in bird.bonus_categories:
            bonus_idx = _BONUS_INDEX_V01.get(category)
            if bonus_idx is not None:
                matrix[row, _ATTR_LEADING_DIM + bonus_idx] = 1.0

        # Bird-identity one-hot at the fixed offset.
        matrix[row, _BIRD_ATTR_DIM_V01 + idx] = 1.0

    return matrix


class PolicyValueNetV01(core.PolicyValueNet):
    """A :class:`~wingspan.model.core.PolicyValueNet` frozen to the v0.1 card-
    feature geometry, for checkpoints written before artifact version 0.2.

    The card encoder's first linear was trained against 229-wide input rows;
    this subclass overrides :meth:`_build_card_encoder` to keep that width and
    register the frozen v0.1 feature table. State and choice encoding, the
    family-head ordering, and the choice-encoder shape are all identical to the
    live era — only the card-encoder input differs. Constructed by the
    version-routing loaders (``PolicyValueNet.from_model_config``,
    ``players.loaders.load_policy_net``) — never by the training pipeline.
    """

    def _build_card_encoder(self, arch: architecture.ModelArchitecture) -> None:
        """Register ``card_encoder`` at the frozen 229-wide input, and
        ``card_features`` from the v0.1 feature table."""
        self.card_encoder, _ = mlp.build_body(
            CARD_FEATURE_DIM_V01,
            arch.card_encoder_layers + (arch.card_embed_dim,),
            activation=arch.activation,
            dropout=arch.dropout,
            layernorm=arch.layernorm,
            final_activation=arch.encoder_final_activation,
        )
        self.register_buffer(
            "card_features",
            torch.tensor(card_feature_matrix(), dtype=torch.float32),
            persistent=False,
        )
        pad_mask = torch.ones(encode.HAND_MULTIHOT_DIM + 1, 1)
        pad_mask[0] = 0.0
        self.register_buffer("card_pad_mask", pad_mask, persistent=False)


class SetupNetV01(setup_net_module.SetupNet):
    """A :class:`~wingspan.training.setup_net.SetupNet` frozen to the v0.1
    card-feature geometry, for setup checkpoints written before artifact version
    0.2.

    The card encoder was trained against 229-wide input rows; this subclass
    overrides :meth:`_build_card_encoder` to keep that width and register the
    frozen v0.1 feature table. Constructed by :func:`~wingspan.players.loaders.load_setup_net`
    when the checkpoint version predates 0.2 — never directly.
    """

    def _build_card_encoder(self, main_arch: architecture.ModelArchitecture) -> None:
        """Register ``card_encoder`` at the frozen 229-wide input, and
        ``card_features`` from the v0.1 feature table."""
        self.card_encoder, _ = mlp.build_body(
            CARD_FEATURE_DIM_V01,
            main_arch.card_encoder_layers + (main_arch.card_embed_dim,),
            activation=main_arch.activation,
            dropout=main_arch.dropout,
            layernorm=main_arch.layernorm,
            final_activation=main_arch.encoder_final_activation,
        )
        self.card_encoder.requires_grad_(False)
        self.register_buffer(
            "card_features",
            torch.tensor(card_feature_matrix(), dtype=torch.float32),
            persistent=False,
        )
        pad_mask = torch.ones(encode.HAND_MULTIHOT_DIM + 1, 1)
        pad_mask[0] = 0.0
        self.register_buffer("card_pad_mask", pad_mask, persistent=False)


###### PRIVATE #######


def _assert_live_layout_contract() -> None:
    """Import-time pins for the block-copy invariants the shim relies on.

    The shim copies the first ``_ATTR_LEADING_DIM`` dims from the live feature
    matrix, which requires the live encoding to be identical to v0.1 for those
    positions. An incompatible future live-layout change fails loudly here
    instead of silently mis-copying."""
    # The unchanged prefix ends right before the new caches_food flag: the first
    # dim the 0.2 reshape added sits at _OFF_ATTR_CACHES_FOOD = _ATTR_LEADING_DIM.
    assert layout._OFF_ATTR_CACHES_FOOD == _ATTR_LEADING_DIM, (
        f"v0.1 shim assumes the unchanged attribute prefix spans "
        f"dims 0..{_ATTR_LEADING_DIM - 1}, but _OFF_ATTR_CACHES_FOOD is "
        f"{layout._OFF_ATTR_CACHES_FOOD}; extend the v0.1 shim"
    )
    # Catalog size must not have changed since v0.1.
    assert cards.n_birds() == _BIRD_ID_DIM_V01, (
        f"v0.1 shim freezes bird catalog at {_BIRD_ID_DIM_V01} birds, "
        f"but current catalog has {cards.n_birds()}; update the shim"
    )
    assert cards.n_bonus_cards() == _BONUS_ID_DIM_V01, (
        f"v0.1 shim freezes bonus catalog at {_BONUS_ID_DIM_V01} cards, "
        f"but current catalog has {cards.n_bonus_cards()}; update the shim"
    )


_assert_live_layout_contract()
