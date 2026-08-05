"""Pre-1.4 artifact compat shim: strip the two food-unlock state stripes and the
``resets_feeder`` choice stripe (both added in v1.4).

**What changed in v1.4 (encoding).** v1.4 folded two independent encoding changes
into one era, so this shim reverses both:

* Two 5-wide pass-through **state** stripes — ``hand_food_unlock_me`` and
  ``tray_food_unlock_me`` (per food, the smallest count that would newly unlock a
  hand / tray bird; see ``engine.playability.min_food_to_unlock``) — were appended
  to the continuous state prefix immediately after ``food_opp``. Pre-1.4 state
  vectors are 10 dims narrower, and their ``card_index`` / ``hand_multihot`` /
  ``decision_type`` offsets sit 10 columns to the left. This is the first
  same-MAJOR change to alter the state width.
* A 1-dim ``resets_feeder`` **choice** stripe was appended as the last *base*
  choice-feature stripe (immediately after ``becomes_unplayable``, before the
  conditional setup stripes). It is set on a ``combine_gain_food``
  ``FoodSubsetChoice`` whose selection rerolls the birdfeeder. v1.0–1.3 choice
  vectors lack it, so the shim strips it and shifts only the trailing
  ``kept_multihot``.

**Shim strategy.** :class:`PolicyValueNetV1_3` overrides the state and choice
seams independently:

* ``encode_state`` / ``encode_choices`` — call the live encoder (full v1.4
  vector), then ``np.delete`` the added columns so each result matches the width
  the pre-1.4 blocks were built for.
* ``_state_embed_offsets`` — return the frozen pre-1.4 offsets (``card_index`` /
  ``hand_multihot`` / ``decision_type`` shifted left by the two state stripes'
  width) so ``_embed_state`` slices the narrower vector correctly.
* ``_choice_embed_offsets`` — keep ``bird_id`` / ``becomes_playable`` /
  ``becomes_unplayable`` (they precede ``resets_feeder``) and shift ``kept_multihot``
  left by ``CHOICE_RESETS_FEEDER_DIM``.
* ``_build_trunk`` / ``_build_choice_encoder`` — build each block at the width its
  own encoder actually produces, derived from ``self.spec`` via ``_true_state_dim``
  / ``_true_choice_dim``.

**Why the spec-derived widths.** ``encode_state`` / ``encode_choices`` always call
the live encoder and strip a fixed number of columns, so the block inputs are fixed
at ``live - stripes`` regardless of the ``state_dim`` / ``choice_dim`` the
constructor was handed. Load paths pass the era's (narrow) dims
(``players.loaders.load_policy_net`` / ``core.from_model_config`` derive them from
``encoding_dims_for_era``); tests may pass the live defaults. Deriving from
``self.spec`` makes the shim correct under both — never subtracting a stripe width
that the passed dim may or may not already exclude.

**Routing.** ``PolicyValueNet.class_for_version`` routes eras 1.1-1.3 here.
``compat.v1_0.PolicyValueNetV1_0`` subclasses this net so v1.0 artifacts strip the
state stripes and ``resets_feeder`` too, on top of their own ``becomes_unplayable``
choice-stripe removal and the old trunk-final-activation fallback. This net in
turn subclasses ``compat.v1_4.PolicyValueNetV1_4``, so every pre-1.4 era also
freezes the pre-1.5 habitat-agnostic play-bird ``goal_delta`` pricing — the
value refill runs at live column offsets inside the ``super().encode_choices``
chain, before this shim's column strips shift anything.

**Fixture note.** A committed LFS checkpoint fixture is deferred (as for v1.0): no
in-production v1.3 checkpoint was preserved. Instead
``tests/test_compat_v1_3.py`` builds a v1.3-era net, saves it with a v1.3 stamp,
and round-trip-loads it through the production ``players.loaders.load_policy_net``
path — exercising the narrow-dims load path (``encoding_dims_for_era`` ->
constructor -> ``load_state_dict`` -> forward) so any double-subtraction of a stripe
width would fail the test.
"""

from __future__ import annotations

import typing

import numpy as np

from wingspan import architecture, decisions, encode, state
from wingspan.compat import v1_4
from wingspan.encode import stripes
from wingspan.model import core

# Layout names of the stripes this shim strips (their offsets/widths are the
# ``encode.STATE_*`` / ``encode.CHOICE_RESETS_FEEDER_*`` constants used below).
_V1_4_STATE_STRIPE_NAMES = ("hand_food_unlock_me", "tray_food_unlock_me")
_V1_4_CHOICE_STRIPE_NAME = "resets_feeder"


class PolicyValueNetV1_3(v1_4.PolicyValueNetV1_4):
    """``PolicyValueNet`` with pre-1.4 geometry: the two food-unlock state stripes
    removed from state encoding and the ``resets_feeder`` stripe removed from choice
    encoding, with the frozen pre-1.4 embed offsets for both. Inherits the pre-1.5
    habitat-agnostic play-bird ``goal_delta`` pricing from
    :class:`wingspan.compat.v1_4.PolicyValueNetV1_4` and, through it, the pre-1.6
    ``goal_delta_ignoring_eggs`` choice-stripe removal from
    :class:`wingspan.compat.v1_5.PolicyValueNetV1_5` — so this era's true choice
    width is live minus both the v1.6 stripe and this era's own ``resets_feeder``."""

    # --- state: strip the two food-unlock stripes ---

    def _true_state_dim(self) -> int:
        """The state width this shim's ``encode_state`` actually produces — the
        live width minus the two food-unlock stripes — derived from ``self.spec``
        so it is independent of the ``state_dim`` passed to ``__init__``."""
        return encode.state_size(self.spec) - 2 * encode.STATE_FOOD_UNLOCK_DIM

    def _build_trunk(
        self, state_dim: int, arch: architecture.ModelArchitecture
    ) -> None:
        """Build the trunk at the pre-1.4 (narrow) state width.

        Ignores the passed ``state_dim`` in favour of ``_true_state_dim`` so the
        trunk matches what ``encode_state`` emits whether the constructor was
        handed live or era dims."""
        super()._build_trunk(self._true_state_dim(), arch)

    def encode_state(
        self,
        game_state: state.GameState,
        decision: decisions.Decision[typing.Any],
    ) -> np.ndarray:
        """Encode at live v1.4 dims, then strip the two food-unlock stripes.

        The live encoder writes the full v1.4 vector including the new stripes;
        ``np.delete`` removes their contiguous columns so the result matches the
        width the pre-1.4 trunk expects."""
        full = super().encode_state(game_state, decision)
        start = encode.STATE_HAND_FOOD_UNLOCK_OFFSET
        end = start + 2 * encode.STATE_FOOD_UNLOCK_DIM
        return np.delete(full, slice(start, end), axis=0)

    def _state_embed_offsets(self) -> core.StateEmbedOffsets:
        """Return pre-1.4 offsets: ``card_index`` / ``hand_multihot`` /
        ``decision_type`` shifted left by the two stripes' total width, because
        those stripes were never in the pre-1.4 state vector."""
        live = super()._state_embed_offsets()
        shift = 2 * encode.STATE_FOOD_UNLOCK_DIM
        return core.StateEmbedOffsets(
            card_index=live.card_index - shift,
            hand_multihot=live.hand_multihot - shift,
            decision_type=live.decision_type - shift,
            hand_summary=live.hand_summary,
            hand_summary_end=live.hand_summary_end,
        )

    def raw_state_stripe_layout(self) -> stripes.VectorLayout:
        """The live state layout minus the two food-unlock stripes — the layout
        of the (narrow) vector this shim's ``encode_state`` actually emits, so
        consumers that decode recorded vectors (the game-log encoding viewer)
        name the era's columns instead of the live ones."""
        return (
            super().raw_state_stripe_layout().without_stripes(_V1_4_STATE_STRIPE_NAMES)
        )

    # --- choice: strip the resets_feeder stripe ---

    def _true_choice_dim(self) -> int:
        """The choice width this shim's ``encode_choices`` actually produces — the
        parent (``v1_4`` -> ``v1_5``) true choice width, which already excludes the
        v1.6 ``goal_delta_ignoring_eggs`` stripe, minus this era's own
        ``resets_feeder`` stripe. Composed via ``super()`` rather than recomputed
        absolutely from ``self.spec``, so a further tail-stripe narrowing an
        ancestor era applies here automatically instead of being silently dropped.
        Subclasses (``v1_0``) narrow further by overriding this the same way."""
        return super()._true_choice_dim() - encode.CHOICE_RESETS_FEEDER_DIM

    def _build_choice_encoder(
        self,
        choice_dim: int,
        arch: architecture.ModelArchitecture,
    ) -> None:
        """Build the choice encoder at the pre-1.4 (narrow) input width.

        Ignores the passed ``choice_dim`` in favour of ``_true_choice_dim`` — which
        is polymorphic, so a ``v1_0`` instance builds at its own (further narrowed)
        width through this same method."""
        super()._build_choice_encoder(self._true_choice_dim(), arch)

    def encode_choices(
        self,
        decision: decisions.Decision[typing.Any],
        game_state: state.GameState,
    ) -> np.ndarray:
        """Encode choices at live dims, then strip the ``resets_feeder`` column.

        ``super().encode_choices`` (the v1_4 shim) writes the full-width row —
        live encoding plus the pre-1.5 play-bird ``goal_delta`` refill —
        including the v1.4 stripe; ``np.delete`` removes its column so the
        result matches the width the pre-1.4 choice encoder (built without
        that stripe) expects."""
        full = super().encode_choices(decision, game_state)
        start = encode.CHOICE_RESETS_FEEDER_OFFSET
        end = start + encode.CHOICE_RESETS_FEEDER_DIM
        return np.delete(full, slice(start, end), axis=1)

    def _choice_embed_offsets(self) -> core.ChoiceEmbedOffsets:
        """Return pre-1.4 offsets: ``bird_id`` / ``becomes_playable`` /
        ``becomes_unplayable`` are unchanged (they precede the new stripe);
        ``kept_multihot`` shifts left by ``CHOICE_RESETS_FEEDER_DIM`` because the
        stripe was never in the pre-1.4 stored choice vectors."""
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

    def raw_choice_stripe_layout(self) -> stripes.VectorLayout:
        """The live choice layout minus ``resets_feeder`` — the layout of the
        (narrow) rows this shim's ``encode_choices`` actually emits (see
        :meth:`raw_state_stripe_layout`)."""
        return (
            super()
            .raw_choice_stripe_layout()
            .without_stripes((_V1_4_CHOICE_STRIPE_NAME,))
        )
