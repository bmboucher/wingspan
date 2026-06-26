# pyright: reportPrivateUsage=false
# (the v0.7 shim calls the live choice encoder with a flag to restore
# the eggs-included becomes_playable semantics that pre-0.8 checkpoints
# were trained against -- a deliberate compat coupling)
"""Frozen v0.7 ``becomes_playable`` food encoding: the shim that keeps pre-0.8 artifacts playable.

Artifact version 0.8 changed the food-gain ``becomes_playable`` stripe so the
egg-cost gate is dropped: a hand bird is flagged as "becomes playable" whenever
gaining the offered food meets its food cost AND an open slot exists, regardless
of whether the egg cost is also met. The egg-gain path is unchanged.

This is a **code-carried FRESH change** — no tensor widths change, but the
computed value of ``becomes_playable`` bits on food-gain rows differs between
v0.7 and v0.8. Pre-0.8 checkpoints were trained with the eggs-included
semantics; this module restores them:

* :func:`encode_choices_v07` calls the live ``choice_encode.encode_choices``
  with ``food_playable_ignores_eggs=False``, reproducing the eggs-included
  ``becomes_playable`` bits a v0.7 checkpoint expects.
* :func:`uses_v0_7_becomes_playable_encoding` identifies artifact versions that
  need this shim (exactly 0.7 — v0.6 artifacts are caught by ``v0_6``'s own
  ``encode_choices`` override which delegates here).
* :class:`PolicyValueNetV07` overrides :meth:`encode_choices` to delegate to
  :func:`encode_choices_v07`. State encoding and card features are identical to
  the live era (no shape changes in 0.8).

Per the compatibility policy (``CLAUDE.md``), this shim lives until a MAJOR
``MODEL_VERSION`` bump deletes it together with the pre-0.8 fixture set.
"""

from __future__ import annotations

import typing

import numpy as np

from wingspan import decisions, state, version
from wingspan.encode import choice_encode, layout
from wingspan.model import core

FOOD_BECOMES_PLAYABLE_CHANGED_IN = "0.8"
"""The artifact version whose food ``becomes_playable`` semantics this module undoes."""


def uses_v0_7_becomes_playable_encoding(artifact_version: str) -> bool:
    """Whether ``artifact_version`` uses the pre-0.8 eggs-included ``becomes_playable``
    food encoding and therefore needs this module to restore it for inference.

    Covers exactly 0.7 — v0.6 artifacts have their own card-encoder shim in
    ``v0_6`` which also gains the delegating ``encode_choices`` override."""
    parsed = version.parse_version(artifact_version)
    changed = version.parse_version(FOOD_BECOMES_PLAYABLE_CHANGED_IN)
    return (parsed.major, parsed.minor) == (changed.major, changed.minor - 1)


def encode_choices_v07(
    decision: decisions.Decision[typing.Any],
    game_state: state.GameState,
    spec: layout.EncodingSpec = layout.DEFAULT_SPEC,
) -> np.ndarray:
    """Featurize all choices with eggs-included ``becomes_playable`` semantics.

    Delegates to the live ``choice_encode.encode_choices`` with
    ``food_playable_ignores_eggs=False``, restoring the v0.7 behaviour where
    the egg-cost gate is included in the food-gain path."""
    return choice_encode.encode_choices(
        decision, game_state, spec, food_playable_ignores_eggs=False
    )


class PolicyValueNetV07(core.PolicyValueNet):
    """A :class:`~wingspan.model.core.PolicyValueNet` frozen to the v0.7
    ``becomes_playable`` food encoding, for checkpoints written before
    artifact version 0.8.

    v0.9 compacted the state vector (1155→1119 dims), so this shim also
    overrides :meth:`encode_state` and :meth:`_state_embed_offsets` to keep the
    1155-dim pre-0.9 geometry — delegating to ``v0_8`` (which owns that frozen
    vector). Only the ``becomes_playable`` food bits differ from v0.8:
    :meth:`encode_choices` overrides to ``encode_choices_v07`` (eggs-included
    food path).

    Constructed by the version-routing loaders (``PolicyValueNet.from_model_config``,
    ``players.loaders.load_policy_net``) — never by the training pipeline.
    """

    def encode_state(
        self,
        game_state: state.GameState,
        decision: decisions.Decision[decisions.Choice],
    ) -> np.ndarray:
        """Featurize ``game_state`` with the 1155-dim pre-0.9 state geometry."""
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
        """Featurize all choices with eggs-included food ``becomes_playable``."""
        return encode_choices_v07(decision, game_state, self.spec)
