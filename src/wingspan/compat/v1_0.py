"""v1.0 artifact compat shim: restore old trunk-final-activation fallback.

**What changed in v1.1.** ``ModelArchitecture.trunk_final_activation_resolved``
was changed to fall back to ``final_activation`` (the universal rule for every
other block) instead of ``between_activation`` (the old trunk-specific exception).
Under v1.0 code, any artifact with ``trunk_final_activation=null`` in its config
resolved the trunk's final layer to ``between_activation`` (typically ``relu``).
Under v1.1 code it resolves to ``final_activation`` (typically ``none``), which
would silently change the computed output for any rehydrated v1.0 checkpoint.

**Shim strategy.** :class:`PolicyValueNetV1_0` is a thin ``PolicyValueNet``
subclass that overrides ``_build_trunk`` to reproduce the v1.0 resolution:
when ``arch.trunk_final_activation`` is ``None``, use
``arch.trunk_between_activation_resolved`` (= old fallback) in place of
``arch.trunk_final_activation_resolved`` (= new fallback). When
``trunk_final_activation`` is explicit (non-``None``) the two branches are
identical, so the shim class is correct for all v1.0 artifacts.

**Fixture note.** An LFS checkpoint fixture at ``tests/data/compat/v1.0/`` is
deferred: the only in-production v1.0 artifacts at bump time had
``trunk_final_activation=null`` and were intentionally discarded in favour of a
fresh training run with the corrected semantics.  A round-trip load test is in
``tests/test_compat_v1_0.py``; it exercises the shim via a freshly-built weight
tensor rather than a saved checkpoint.
"""

from __future__ import annotations

from wingspan import architecture, encode
from wingspan.model import core, mlp


class PolicyValueNetV1_0(core.PolicyValueNet):
    """``PolicyValueNet`` with the v1.0 trunk-final-activation fallback.

    Identical to the live net except that when ``trunk_final_activation`` is
    ``None``, the trunk's final layer uses ``between_activation`` (the v1.0
    rule) rather than ``final_activation`` (the v1.1 rule).
    """

    def _build_trunk(
        self, state_dim: int, arch: architecture.ModelArchitecture
    ) -> None:
        """Build trunk with v1.0 final-activation semantics.

        When ``arch.trunk_final_activation`` is ``None`` the fallback is
        ``arch.trunk_between_activation_resolved`` (the old rule); otherwise
        the resolved value matches the live behaviour identically."""
        offsets = self._state_embed_offsets()
        n_extra = (
            offsets.decision_type - offsets.hand_multihot
        ) // encode.HAND_MULTIHOT_DIM - 1
        hand_summary_in_state = offsets.hand_summary_end > offsets.hand_summary
        trunk_in_dim = encode.trunk_input_dim(
            state_dim,
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
