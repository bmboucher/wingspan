# pyright: reportPrivateUsage=false
# (mirrors compat.v1_3's package-private layout-constant coupling)
"""Pre-1.8 artifact compat shim: strip the per-opponent ``known_hand_opp``
state stripe (added in v1.8).

**What changed in v1.8 (encoding).** A 180-wide identity multi-hot per
opponent — the publicly-known subset of that opponent's hand
(``state.Player.known_hand``, maintained by ``engine.ledger``: cards seen
entering the hand via a tray draw or a face-up draft draw, wholesale-cleared
on any face-down departure) — was appended to the state vector's multi-hot
region, immediately after ``hand_playable_eggs_me`` and before the trailing
``decision_type`` one-hot. Every compat-shimmed era is 2-player only (the
dims router refuses ``spec.num_players != 2`` outright — see
``compat.encoding_dims_for_era``), so this shim only ever strips the single
``known_hand_opp`` stripe, never a suffixed N>=3 replica. Pre-1.8 state
vectors are 180 dims narrower, and their trailing ``decision_type`` offset
sits 180 columns to the left; ``card_index`` and ``hand_multihot`` precede
the new stripe and are UNCHANGED — the opposite shape from every prior
state-dims shim (``compat.v1_3``'s food-unlock strip precedes ``card_index``
and shifts all three offsets).

**No choice-side change.** v1.8 touches state encoding only; the choice
vector and the setup model are untouched, so this shim overrides no choice
seam and there is no ``SetupNetV1_7`` — a 1.7-era setup artifact keeps
routing through ``SetupNet.class_for_version`` exactly as before this bump
(its cascade already sends era 1.7 to the live ``SetupNet``, and v1.8 adds
no setup branch).

**Shim strategy.** Mirrors ``compat.v1_3``'s narrowing shape, applied at the
opposite end of the state vector:

* ``encode_state`` — call the live encoder (full v1.8 vector), then
  ``np.delete`` the appended ``known_hand_opp`` columns so the result matches
  the width the pre-1.8 trunk was built for.
* ``_state_embed_offsets`` — shift ONLY ``decision_type`` left by
  ``STATE_KNOWN_HAND_OPP_DIM``; ``card_index`` and ``hand_multihot`` are
  returned unchanged (they sit upstream of the new stripe).
* ``_build_trunk`` — build at the width its own encoder actually produces,
  derived from ``self.spec`` via ``_true_state_dim`` rather than the passed
  ``state_dim`` (the same load-path-vs-test-default reasoning as
  ``compat.v1_3``: never subtract a stripe width the passed dim may or may
  not already exclude). The dynamic extra-multihot-block count
  ``core.PolicyValueNet._build_trunk`` derives —
  ``(offsets.decision_type - offsets.hand_multihot) // HAND_MULTIHOT_DIM - 1``
  — comes out ``2`` automatically once ``decision_type`` is shifted back (the
  playability pair only, no known-hand block), so this shim needs no
  ``n_playable_multihots`` override of its own; the trunk input width follows
  from the shifted offsets alone (Stage 2's threshold-gated embed rules,
  ``encode.stripes.embed_rules.state_embed_rules``, are built for exactly
  this: passing the playability-pair count reproduces the pre-1.8 layout).

**Why the spec-derived width.** ``encode_state`` always calls the live
encoder and strips a fixed number of columns, so the block input is fixed at
``live - STATE_KNOWN_HAND_OPP_DIM`` regardless of the ``state_dim`` the
constructor was handed. Load paths pass the era's (narrow) dims
(``players.loaders.load_policy_net`` / ``core.from_model_config`` derive them
from ``encoding_dims_for_era``); tests may pass the live defaults. Deriving
from ``self.spec`` makes the shim correct under both.

**Routing.** ``PolicyValueNet.class_for_version`` routes era 1.7 here.
``compat.v1_6.PolicyValueNetV1_6`` subclasses this net (re-chained from
``core.PolicyValueNet``), so every era <= 1.6 strips the stripe too, on top
of its own choice-side value refills — composed via ``super()`` chaining,
the same shape the v1_3 -> v1_4 chain already uses for the food-unlock
stripes.

**A future state-narrowing era.** This class computes ``_true_state_dim``
absolutely from ``self.spec`` — it is the shim closest to live, mirroring
the rationale ``compat.v1_3`` originally used before the v1.6 bump converted
that method to compose via ``super()`` (see ``compat.v1_3._true_state_dim``'s
docstring). A future MINOR bump that narrows the state vector again must
convert this method to the same composing form
(``super()._true_state_dim() - ...``), so a further narrowing stacks instead
of silently overwriting this one.

**Fixture note.** A committed LFS checkpoint fixture is deferred (as for
every prior era): ``tests/test_compat_v1_7.py`` builds a v1.7-era net, saves
it with a v1.7 stamp, and round-trip-loads it through the production
``players.loaders.load_policy_net`` path.
"""

from __future__ import annotations

import typing

import numpy as np

from wingspan import architecture, decisions, encode, state
from wingspan.encode import stripes
from wingspan.model import core

# Layout name of the state stripe this shim strips (offset/width:
# encode.STATE_KNOWN_HAND_OPP_*). Every compat-shimmed era is 2-player only
# (encoding_dims_for_era refuses spec.num_players != 2 up front), so there is
# never a suffixed known_hand_opp2 / known_hand_opp3 replica to strip here.
_V1_8_STATE_STRIPE_NAMES = ("known_hand_opp",)


class PolicyValueNetV1_7(core.PolicyValueNet):
    """``PolicyValueNet`` with pre-1.8 geometry: the per-opponent
    ``known_hand_opp`` state stripe removed from state encoding, with the
    frozen pre-1.8 ``StateEmbedOffsets`` (only ``decision_type`` shifts).
    No choice-side override — v1.8 changed state encoding only — and no
    ``SetupNet`` shim, since v1.8 leaves setup encoding entirely untouched
    (a 1.7-era setup artifact keeps routing through the live ``SetupNet``
    exactly as it did before this bump)."""

    def _true_state_dim(self) -> int:
        """The state width this shim's ``encode_state`` actually produces —
        the live width minus the ``known_hand_opp`` stripe — derived from
        ``self.spec`` so it is independent of the ``state_dim`` passed to
        ``__init__``. Absolute form: this is the state-narrowing shim
        closest to live, so there is no ancestor era to compose through (see
        the module docstring's "future state-narrowing era" note)."""
        return encode.state_size(self.spec) - encode.STATE_KNOWN_HAND_OPP_DIM

    def _build_trunk(
        self, state_dim: int, arch: architecture.ModelArchitecture
    ) -> None:
        """Build the trunk at the pre-1.8 (narrow) state width.

        Ignores the passed ``state_dim`` in favour of ``_true_state_dim`` so
        the trunk matches what ``encode_state`` emits whether the
        constructor was handed live or era dims."""
        super()._build_trunk(self._true_state_dim(), arch)

    def encode_state(
        self,
        game_state: state.GameState,
        decision: decisions.Decision[typing.Any],
    ) -> np.ndarray:
        """Encode at live v1.8 dims, then strip the ``known_hand_opp`` stripe.

        The live encoder writes the full v1.8 vector including the new
        stripe; ``np.delete`` removes its contiguous columns so the result
        matches the width the pre-1.8 trunk expects."""
        full = super().encode_state(game_state, decision)
        start = encode.STATE_KNOWN_HAND_OPP_OFFSET
        end = start + encode.STATE_KNOWN_HAND_OPP_DIM
        return np.delete(full, slice(start, end), axis=0)

    def _state_embed_offsets(self) -> core.StateEmbedOffsets:
        """Return pre-1.8 offsets: ``decision_type`` shifted left by
        ``STATE_KNOWN_HAND_OPP_DIM`` because the stripe was never in the
        pre-1.8 state vector. ``card_index`` and ``hand_multihot`` are
        UNCHANGED — the new stripe is appended after both playability
        multi-hots, well past either offset, so neither moves (the opposite
        shift shape from ``compat.v1_3``, whose food-unlock stripe precedes
        ``card_index`` and so shifts all three)."""
        live = super()._state_embed_offsets()
        return core.StateEmbedOffsets(
            card_index=live.card_index,
            hand_multihot=live.hand_multihot,
            decision_type=live.decision_type - encode.STATE_KNOWN_HAND_OPP_DIM,
            hand_summary=live.hand_summary,
            hand_summary_end=live.hand_summary_end,
        )

    def raw_state_stripe_layout(self) -> stripes.VectorLayout:
        """The live state layout minus ``known_hand_opp`` — the layout of
        the (narrow) vector this shim's ``encode_state`` actually emits, so
        consumers that decode recorded vectors (the game-log encoding
        viewer) name the era's columns instead of the live ones."""
        return (
            super().raw_state_stripe_layout().without_stripes(_V1_8_STATE_STRIPE_NAMES)
        )
