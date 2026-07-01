"""v1.3 artifact compat shim: strip the ``resets_feeder`` choice stripe added in v1.4.

**What changed in v1.4 (encoding).** A 1-dim ``resets_feeder`` stripe was appended as
the last *base* choice-feature stripe (immediately after ``becomes_unplayable``, before
the conditional setup stripes). It is set on a ``combine_gain_food`` ``FoodSubsetChoice``
whose selection rerolls the birdfeeder — a partial take that commits to a reset, or a
full take that empties the feeder. v1.0–1.3 choice vectors lack this stripe, so the shim
strips it after encoding and narrows the choice encoder's first ``Linear`` to match.

**Shim strategy.** :class:`PolicyValueNetV1_3` is a thin ``PolicyValueNet`` subclass
that overrides:

* ``_build_choice_encoder`` — builds the encoder at the v1.3 (one narrower) width.
* ``encode_choices`` — calls the live encoder (full v1.4 vector), then ``np.delete``s
  the ``resets_feeder`` column.
* ``_choice_embed_offsets`` — the new stripe follows ``bird_id`` / ``becomes_playable``
  / ``becomes_unplayable`` (their offsets are unchanged) but precedes the trailing
  ``kept_multihot``, so only ``kept_multihot`` shifts left by
  ``CHOICE_RESETS_FEEDER_DIM``.

The main net's topology is otherwise identical between v1.1–1.3 and live, so there is no
trunk override. The v1.0 shim (:mod:`wingspan.compat.v1_0`) inherits this class to
compose the ``resets_feeder`` strip with its own ``becomes_unplayable`` strip and
trunk-final-activation fallback.

**Fixture note.** Following the v1.0 precedent, the round-trip is exercised via a
freshly-built weight tensor (``tests/test_compat_v1_3.py``) rather than a saved LFS
checkpoint.
"""

from __future__ import annotations

import typing

import numpy as np

from wingspan import architecture, decisions, encode, state
from wingspan.model import core


class PolicyValueNetV1_3(core.PolicyValueNet):
    """``PolicyValueNet`` with v1.1–1.3 choice geometry: the ``resets_feeder`` stripe
    removed from choice encoding."""

    def _build_choice_encoder(
        self,
        choice_dim: int,
        arch: architecture.ModelArchitecture,
    ) -> None:
        """Build the choice encoder at the v1.3 (narrower) input width.

        The v1.1–1.3 choice vectors lack the ``resets_feeder`` stripe, so the
        encoder's first linear must match the post-strip width."""
        super()._build_choice_encoder(
            choice_dim - encode.CHOICE_RESETS_FEEDER_DIM, arch
        )

    def encode_choices(
        self,
        decision: decisions.Decision[typing.Any],
        game_state: state.GameState,
    ) -> np.ndarray:
        """Encode choices at live v1.4 dims, then strip the ``resets_feeder`` column.

        The live encoder writes the full v1.4 row including the new stripe;
        ``np.delete`` removes its column so the result matches the width the v1.3
        choice encoder (built without that stripe) expects."""
        full = super().encode_choices(decision, game_state)
        start = encode.CHOICE_RESETS_FEEDER_OFFSET
        end = start + encode.CHOICE_RESETS_FEEDER_DIM
        return np.delete(full, slice(start, end), axis=1)

    def _choice_embed_offsets(self) -> core.ChoiceEmbedOffsets:
        """Return v1.3 era offsets: ``bird_id`` / ``becomes_playable`` /
        ``becomes_unplayable`` are unchanged (they precede the new stripe);
        ``kept_multihot`` shifts left by ``CHOICE_RESETS_FEEDER_DIM`` because the
        stripe was never in the v1.3 stored choice vectors."""
        live = super()._choice_embed_offsets()
        kept = live.kept_multihot
        if kept is not None:
            kept = kept - encode.CHOICE_RESETS_FEEDER_DIM
        return core.ChoiceEmbedOffsets(
            bird_id=live.bird_id,
            becomes_playable=live.becomes_playable,
            becomes_unplayable=live.becomes_unplayable,
            kept_multihot=kept,
        )
