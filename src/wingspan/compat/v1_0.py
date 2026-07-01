"""v1.0 artifact compat shim: restore old trunk-final-activation fallback and
strip the ``becomes_unplayable`` choice stripe added in v1.1.

**What changed in v1.1 (architecture).** ``ModelArchitecture.trunk_final_activation_resolved``
was changed to fall back to ``final_activation`` (the universal rule for every
other block) instead of ``between_activation`` (the old trunk-specific exception).
Under v1.0 code, any artifact with ``trunk_final_activation=null`` in its config
resolved the trunk's final layer to ``between_activation`` (typically ``relu``).
Under v1.1 code it resolves to ``final_activation`` (typically ``none``), which
would silently change the computed output for any rehydrated v1.0 checkpoint.

**What changed in v1.1 (encoding).** The ``becomes_unplayable`` 180-dim multi-hot
stripe was appended to the base choice feature vector immediately after
``becomes_playable``. v1.0 choice vectors lack this stripe; the shim strips it
after encoding so the truncated vector matches the width the v1.0 choice encoder
was built for.

**What changed in v1.4 (encoding).** v1.4 appended two food-unlock **state**
stripes and a 1-dim ``resets_feeder`` **choice** stripe (the last base choice
stripe, after ``becomes_unplayable``). v1.0 artifacts predate all three, so this
class inherits :class:`wingspan.compat.v1_3.PolicyValueNetV1_3` — which already
strips the state stripes and ``resets_feeder`` — and composes the additional
``becomes_unplayable`` strip via ``super()`` chaining. The choice compose is exact
because ``becomes_unplayable`` precedes ``resets_feeder``: stripping the trailing
``resets_feeder`` first leaves the ``becomes_unplayable`` offset unchanged.

**Shim strategy.** :class:`PolicyValueNetV1_0` is a thin ``PolicyValueNetV1_3``
subclass that overrides:

* ``_build_trunk`` — reproduces the v1.0 trunk-final-activation fallback, at the
  inherited pre-1.4 (narrow) state width (``_true_state_dim``).
* ``_true_choice_dim`` — narrows the inherited v1.3 true choice width by a further
  ``CHOICE_BECOMES_UNPLAYABLE_DIM``; the inherited
  ``PolicyValueNetV1_3._build_choice_encoder`` reads it polymorphically, so v1.0
  never re-implements the builder.
* ``encode_choices`` / ``_choice_embed_offsets`` — call ``super()`` (the v1_3 shim,
  which has already handled ``resets_feeder``) and then additionally strip
  ``becomes_unplayable``: ``encode_choices`` ``np.delete``s its columns,
  ``_choice_embed_offsets`` returns ``becomes_unplayable=None`` and shifts
  ``kept_multihot`` left by a further ``CHOICE_BECOMES_UNPLAYABLE_DIM`` (all three
  stripes were absent from v1.0 vectors).

When ``trunk_final_activation`` is explicit (non-``None``) the two trunk branches
are identical, so the shim is correct for all v1.0 artifacts.

**Fixture note.** An LFS checkpoint fixture at ``tests/data/compat/v1.0/`` is
deferred: the only in-production v1.0 artifacts at bump time had
``trunk_final_activation=null`` and were intentionally discarded in favour of a
fresh training run with the corrected semantics.  A round-trip load test is in
``tests/test_compat_v1_0.py``; it exercises the shim via a freshly-built weight
tensor rather than a saved checkpoint.
"""

from __future__ import annotations

import typing

import numpy as np

from wingspan import architecture, decisions, encode, state
from wingspan.compat import v1_3
from wingspan.model import core, mlp


class PolicyValueNetV1_0(v1_3.PolicyValueNetV1_3):
    """``PolicyValueNet`` with v1.0 semantics: the old trunk-final-activation
    fallback plus every stripe added since v1.0 removed from encoding.

    Inherits :class:`~wingspan.compat.v1_3.PolicyValueNetV1_3`, so v1.0 artifacts
    strip the two food-unlock **state** stripes (v1.4) and the ``resets_feeder``
    **choice** stripe (v1.4) via the inherited shim, and use its frozen pre-1.4
    state-embed offsets. This class adds the ``becomes_unplayable`` choice-stripe
    removal (v1.1) and restores the old trunk-final-activation fallback — v1.0
    predates all three stripes.
    """

    def _build_trunk(
        self, state_dim: int, arch: architecture.ModelArchitecture
    ) -> None:
        """Build trunk with v1.0 final-activation semantics at the pre-1.4 width.

        When ``arch.trunk_final_activation`` is ``None`` the fallback is
        ``arch.trunk_between_activation_resolved`` (the old rule); otherwise
        the resolved value matches the live behaviour identically. The trunk
        width is ``_true_state_dim`` (inherited from ``PolicyValueNetV1_3``), the
        width ``encode_state`` emits after stripping the food-unlock stripes,
        derived from ``self.spec`` rather than the passed ``state_dim``."""
        offsets = self._state_embed_offsets()
        n_extra = (
            offsets.decision_type - offsets.hand_multihot
        ) // encode.HAND_MULTIHOT_DIM - 1
        hand_summary_in_state = offsets.hand_summary_end > offsets.hand_summary
        trunk_in_dim = encode.trunk_input_dim(
            self._true_state_dim(),
            arch.card_embed_dim,
            use_distinct_hand_model=arch.use_distinct_hand_model,
            hand_summary_in_state=hand_summary_in_state,
            hand_embed_dim=arch.hand_embed_dim,
            pooled_hand_width=arch.pooled_hand_width,
            tray_set_embedding=arch.tray_set_embedding,
            n_playable_multihots=n_extra,
        )
        # v1.0 fallback: None trunk_final_activation inherits between_activation.
        trunk_final = (
            arch.trunk_final_activation
            if arch.trunk_final_activation is not None
            else arch.trunk_between_activation_resolved
        )
        self.state_trunk, _ = mlp.build_body(
            trunk_in_dim,
            arch.trunk_layers,
            between_activation=arch.trunk_between_activation_resolved,
            final_activation=trunk_final,
            dropout=arch.trunk_dropout_resolved,
            layernorm=arch.trunk_layernorm_resolved,
        )

    def _true_choice_dim(self) -> int:
        """The choice width v1.0's ``encode_choices`` produces: the inherited v1.3
        true width (live minus ``resets_feeder``) narrowed by a further
        ``CHOICE_BECOMES_UNPLAYABLE_DIM``, because v1.0 predates both stripes.

        The choice encoder is built (in the inherited ``PolicyValueNetV1_3._build_choice_encoder``)
        at this width, read polymorphically — so v1.0 never re-implements the
        builder. Spec-derived, so it is correct whether the constructor was handed
        live or era dims."""
        return super()._true_choice_dim() - encode.CHOICE_BECOMES_UNPLAYABLE_DIM

    def encode_choices(
        self,
        decision: decisions.Decision[typing.Any],
        game_state: state.GameState,
    ) -> np.ndarray:
        """Encode choices at live v1.1 dims, then strip the ``becomes_unplayable`` block.

        The live encoder writes the full v1.1 row including the new stripe;
        ``np.delete`` removes its columns so the result matches the width the
        v1.0 choice encoder (built without that stripe) expects."""
        full = super().encode_choices(decision, game_state)
        start = encode.CHOICE_BECOMES_UNPLAYABLE_OFFSET
        end = start + encode.CHOICE_BECOMES_UNPLAYABLE_DIM
        return np.delete(full, slice(start, end), axis=1)

    def _choice_embed_offsets(self) -> core.ChoiceEmbedOffsets:
        """Return v1.0 era offsets: ``becomes_unplayable=None``; ``kept_multihot``
        shifted left by ``CHOICE_BECOMES_UNPLAYABLE_DIM`` because the stripe was
        never in the v1.0 stored choice vectors."""
        live = super()._choice_embed_offsets()
        kept = live.kept_multihot
        if kept is not None:
            kept = kept - encode.CHOICE_BECOMES_UNPLAYABLE_DIM
        return core.ChoiceEmbedOffsets(
            bird_id=live.bird_id,
            becomes_playable=live.becomes_playable,
            becomes_unplayable=None,
            kept_multihot=kept,
        )
