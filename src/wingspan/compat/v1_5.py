"""Pre-1.6 artifact compat shim: strip the ``goal_delta_ignoring_eggs`` choice
stripe (added in v1.6), and freeze the pre-1.6 static setup ``goal_affinity``
pricing.

**What changed in v1.6, main-net half (encoding).** An 8-dim
``goal_delta_ignoring_eggs`` choice stripe — per round goal, a
``(count_delta, vp_delta)`` pair pricing the hypothesis that this row's bird
is eventually played (a slot must be open) and egg-populated to whatever
level best advances the goal — was appended as the new last base choice
stripe (immediately after ``resets_feeder``). Pre-1.6 choice vectors lack it;
the shim strips it and shifts only the trailing ``kept_multihot`` region.

**Shim strategy.** :class:`PolicyValueNetV1_5` mirrors ``compat.v1_3``'s
narrowing shape, applied to one stripe instead of two:

* ``encode_choices`` — call the live encoder (full v1.6 vector), then
  ``np.delete`` the appended columns so the result matches the width the
  pre-1.6 choice encoder was built for.
* ``_choice_embed_offsets`` — keep ``bird_id`` / ``becomes_playable`` /
  ``becomes_unplayable`` (they precede the stripe) and shift ``kept_multihot``
  left by ``CHOICE_GOAL_DELTA_IGNORING_EGGS_DIM``.
* ``_build_choice_encoder`` — build at the width its own encoder actually
  produces, derived from ``self.spec`` via ``_true_choice_dim`` rather than
  the passed ``choice_dim`` (the same load-path-vs-test-default reasoning as
  ``compat.v1_3``: never subtract a stripe width the passed dim may or may not
  already exclude).

**No state-side change (main net).** v1.6's only main-net encoding change is
this one choice stripe; the state vector and ``_state_embed_offsets`` are
untouched, so this shim overrides nothing on the main net's state side.

**Routing (main net).** ``PolicyValueNet.class_for_version`` routes era 1.5
here. ``compat.v1_4.PolicyValueNetV1_4`` now **inherits** this net (re-chained
from ``core.PolicyValueNet``), so eras 1.0-1.4 strip the tail stripe too, on
top of their own stripes and value refills — composed via ``super()``
chaining and ``_true_choice_dim`` narrowing exactly like the v1_3 -> v1_4
chain already did for the food-unlock / ``resets_feeder`` stripes.

**What changed in v1.6, setup half (behavior only).** The setup encoder's
``goal_affinity`` stripe (:mod:`wingspan.setup_model.encode`, step 7) switched
from :func:`wingspan.engine.scoring.goal_count_delta_for_bird` — zero for
every egg-driven category, since a freshly played bird starts with no eggs —
to :func:`wingspan.engine.scoring.goal_affinity_for_kept`, the
played-and-optimally-egg-populated hand-level bound. No dims change (the
stripe stays 4 scalars at unchanged offsets), so this is a REGIME-shaped,
FRESH-behavior change exactly like v1.5's main-net ``goal_delta`` value
freeze — and the same doctrine applies: a pre-1.6 setup net trained against
the egg-blind pricing must keep seeing it on rehydration.

**Shim strategy (setup net).** :class:`SetupNetV1_5` overrides only
``encode_candidate``: call the live encoder (via ``super()``), then
:func:`wingspan.setup_model.encode.refill_goal_affinity_static` overwrites
the ``goal_affinity`` stripe with the pre-1.6 static pricing, in place. No
geometry override — ``SetupNetV1_5`` asserts nothing shape-related and joins
no dims-router branch, since v1.6's setup change touches values only. This is
the first :class:`wingspan.training.setup_net.SetupNet` compat shim — the
``compat/INDEX.md`` "PolicyValueNet/SetupNet subclass" hook, now realized.

**Routing (setup net).** ``SetupNet.class_for_version`` routes eras <= 1.5
here.

**Fixture note.** A committed LFS checkpoint fixture is deferred (as for
v1.0 / v1.3 / v1.4), for both halves: ``tests/test_compat_v1_5.py`` builds
v1.5-era nets (main and setup), saves them with a v1.5 stamp, and round-trip-
loads them through the production ``players.loaders.load_policy_net`` /
``load_setup_net`` paths.
"""

from __future__ import annotations

import typing

import numpy as np

from wingspan import architecture, decisions, encode, setup_model, state
from wingspan.encode import stripes
from wingspan.model import core
from wingspan.setup_model import encode as setup_encode
from wingspan.training import setup_net

# Layout name of the v1.6 choice stripe this shim strips (offset/width:
# ``encode.CHOICE_GOAL_DELTA_IGNORING_EGGS_*``).
_V1_6_CHOICE_STRIPE_NAME = "goal_delta_ignoring_eggs"


class PolicyValueNetV1_5(core.PolicyValueNet):
    """``PolicyValueNet`` with pre-1.6 geometry: the ``goal_delta_ignoring_eggs``
    stripe removed from choice encoding, with the frozen pre-1.6 choice-embed
    offsets. Era <=1.5 artifacts predate the played-and-egg-populated goal
    pricing entirely — their choice vectors are 8 dims narrower than live."""

    def _true_choice_dim(self) -> int:
        """The choice width this shim's ``encode_choices`` actually produces —
        the live width minus ``goal_delta_ignoring_eggs`` — derived from
        ``self.spec`` so it is independent of the ``choice_dim`` passed to
        ``__init__``. Subclasses (``v1_3``, reached via ``v1_4``) narrow
        further by overriding this and composing via ``super()`` (the v1_3
        pattern), rather than recomputing the width absolutely."""
        return (
            encode.choice_feature_dim(self.spec)
            - encode.CHOICE_GOAL_DELTA_IGNORING_EGGS_DIM
        )

    def _build_choice_encoder(
        self,
        choice_dim: int,
        arch: architecture.ModelArchitecture,
    ) -> None:
        """Build the choice encoder at the pre-1.6 (narrow) input width.

        Ignores the passed ``choice_dim`` in favour of ``_true_choice_dim`` —
        which is polymorphic, so a subclass instance builds at its own
        (further narrowed) width through this same method."""
        super()._build_choice_encoder(self._true_choice_dim(), arch)

    def encode_choices(
        self,
        decision: decisions.Decision[typing.Any],
        game_state: state.GameState,
    ) -> np.ndarray:
        """Encode choices at live dims, then strip the
        ``goal_delta_ignoring_eggs`` column block.

        ``super().encode_choices`` writes the full live-width row; ``np.delete``
        removes its columns so the result matches the width the pre-1.6 choice
        encoder (built without that stripe) expects."""
        full = super().encode_choices(decision, game_state)
        start = encode.CHOICE_GOAL_DELTA_IGNORING_EGGS_OFFSET
        end = start + encode.CHOICE_GOAL_DELTA_IGNORING_EGGS_DIM
        return np.delete(full, slice(start, end), axis=1)

    def _choice_embed_offsets(self) -> core.ChoiceEmbedOffsets:
        """Return pre-1.6 offsets: ``bird_id`` / ``becomes_playable`` /
        ``becomes_unplayable`` are unchanged (they precede the new stripe);
        ``kept_multihot`` shifts left by ``CHOICE_GOAL_DELTA_IGNORING_EGGS_DIM``
        because the stripe was never in the pre-1.6 stored choice vectors."""
        live = super()._choice_embed_offsets()
        kept = live.kept_multihot
        if kept is not None:
            kept = kept - encode.CHOICE_GOAL_DELTA_IGNORING_EGGS_DIM
        return core.ChoiceEmbedOffsets(
            bird_id=live.bird_id,
            becomes_playable=live.becomes_playable,
            becomes_unplayable=live.becomes_unplayable,
            kept_multihot=kept,
        )

    def raw_choice_stripe_layout(self) -> stripes.VectorLayout:
        """The live choice layout minus ``goal_delta_ignoring_eggs`` — the
        layout of the (narrow) rows this shim's ``encode_choices`` actually
        emits (the game-log encoding viewer's decode seam; see
        ``docs/VERSIONING.md``)."""
        return (
            super()
            .raw_choice_stripe_layout()
            .without_stripes((_V1_6_CHOICE_STRIPE_NAME,))
        )


class SetupNetV1_5(setup_net.SetupNet):
    """``SetupNet`` with pre-1.6 ``goal_affinity`` pricing: overrides only
    ``encode_candidate`` to overwrite the ``goal_affinity`` stripe with the
    static, egg-blind pre-1.6 computation after live encoding.

    Geometry is UNCHANGED — v1.6's setup-side change is behavior-only (the
    setup twin of v1.5's main-net ``goal_delta`` value freeze), so this class
    asserts nothing shape-related and joins no dims-router branch: every
    other block of the candidate vector, and every submodule shape, is
    identical to the live ``SetupNet``.

    The first ``SetupNet`` compat shim — the ``compat/INDEX.md``
    "PolicyValueNet/SetupNet subclass" hook, now realized. Routed by
    :meth:`wingspan.training.setup_net.SetupNet.class_for_version` for eras
    <= 1.5."""

    def encode_candidate(
        self,
        candidate: setup_model.SetupCandidate,
        context: setup_model.SetupContext,
    ) -> np.ndarray:
        """Encode at live geometry, then overwrite ``goal_affinity`` with the
        pre-1.6 static pricing (see
        :func:`wingspan.setup_model.encode.refill_goal_affinity_static`)."""
        vec = super().encode_candidate(candidate, context)
        setup_encode.refill_goal_affinity_static(vec, candidate, context, self.encoding)
        return vec
