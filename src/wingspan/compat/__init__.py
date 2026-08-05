"""Version-specific artifact-compatibility shims.

**v1.0 shim** (see :mod:`wingspan.compat.v1_0`): two things changed in v1.1:

1. *Architecture* — the trunk's final-layer activation fallback changed.
   ``trunk_final_activation=null`` now inherits ``final_activation`` (the
   universal rule) instead of ``between_activation`` (the old trunk-specific
   exception). The shim class :class:`wingspan.compat.v1_0.PolicyValueNetV1_0`
   restores the old fallback for v1.0 artifacts, routed from
   ``model.PolicyValueNet.class_for_version``.

2. *Encoding* — the ``becomes_unplayable`` 180-dim multi-hot stripe was appended
   to the base choice feature vector immediately after ``becomes_playable``.
   v1.0 choice vectors lack this stripe, so ``encoding_dims_for_era`` returns a
   ``choice_dim`` that is 180 less than the live width for v1.0 artifacts.  The
   shim's ``encode_choices`` override strips the stripe after live encoding, and
   ``_choice_embed_offsets`` returns ``becomes_unplayable=None``.

**v1_3 shim** (see :mod:`wingspan.compat.v1_3`): v1.4 landed two independent
encoding changes together, so its shim strips both from pre-1.4 vectors:

* Two 5-wide food-distance-to-playable **state** stripes (``hand_food_unlock_me``,
  ``tray_food_unlock_me``) appended to the continuous state prefix — so
  ``encoding_dims_for_era`` returns a ``state_dim`` 10 less than live for every
  pre-1.4 same-MAJOR era (its first state-dim branch), and
  :class:`wingspan.compat.v1_3.PolicyValueNetV1_3` strips them after live
  ``encode_state`` and freezes the pre-1.4 ``_state_embed_offsets``.
* A 1-dim ``resets_feeder`` **choice** stripe appended after ``becomes_unplayable``
  — so ``encoding_dims_for_era`` returns a ``choice_dim`` 1 less for every era with
  minor ≤ 3, and the shim strips that column after live ``encode_choices`` and
  shifts only ``kept_multihot``.

The shim derives both narrow encoder widths from ``self.spec`` (``_true_state_dim``
/ ``_true_choice_dim``), not the passed dims, so it is correct whether the
constructor is handed live dims (tests) or the era's already-narrow dims (the load
path). It routes for eras 1.1-1.3; ``PolicyValueNetV1_0`` **inherits** it, so v1.0
loads strip the state stripes and ``resets_feeder`` too, on top of their own
``becomes_unplayable`` strip and trunk-final-activation fix.

**v1_4 shim** (see :mod:`wingspan.compat.v1_4`): v1.5 was a behavior-only era —
no tensor shape changed of its own. The live ``PlayBirdChoice`` featurizer now
prices the ``goal_delta`` stripe at the row's committed landing habitat;
pre-1.5 rows priced the bird's *card* habitats (a two-habitat bird advanced a
``birds_<habitat>`` goal on both rows). :class:`wingspan.compat.v1_4.PolicyValueNetV1_4`
overrides only ``encode_choices``: after its parent's encoding it re-fills each
play-bird row's ``goal_delta`` with the habitat-agnostic pricing
(``choice_encode.refill_goal_delta_habitat_agnostic``). It routes for era 1.4;
``PolicyValueNetV1_3`` **inherits** it, so every pre-1.4 era freezes the old
pricing too (the refill runs at live offsets before their column strips). Since
the v1.6 bump this class subclasses ``v1_5`` (below) rather than the live net,
so era 1.4's choice geometry is no longer dims-equal-live.

**v1_5 shim** (see :mod:`wingspan.compat.v1_5`): v1.6 appends an 8-dim
``goal_delta_ignoring_eggs`` choice stripe as the new last base stripe
(immediately after ``resets_feeder``) — per round goal, a ``(count_delta,
vp_delta)`` pair pricing the hypothesis that this row's bird is eventually
played and egg-populated optimally. Pre-1.6 choice vectors lack it, so
``encoding_dims_for_era`` returns a ``choice_dim`` 8 less for every era with
minor <= 5. :class:`wingspan.compat.v1_5.PolicyValueNetV1_5` strips the column
after live ``encode_choices`` and shifts only ``kept_multihot``. It routes for
era 1.5; ``PolicyValueNetV1_4`` now **inherits** it, so every pre-1.6 era
strips the tail stripe too, on top of its own stripes and value refills.

The pre-1.0 shims (``v0_0`` … ``v0_7``) were dropped at the 1.0 MAJOR bump; no
0.x artifact loads under 1.x code. Each module is version-number-specific —
never a config flag — and the whole package is deleted again at the next MAJOR
bump.

:func:`encoding_dims_for_era` is the package-level dims router: given an artifact
version it returns the raw state/choice vector widths that era's encoders
produce, so an era-pinned ``RunConfig`` derives the dims its checkpoints actually
carry.
"""

from wingspan import encode, version

__all__ = ["encoding_dims_for_era"]


def encoding_dims_for_era(
    artifact_version: str, spec: encode.EncodingSpec
) -> tuple[int, int]:
    """The raw ``(state_dim, choice_dim)`` an era's encoders produce under ``spec``.

    v1.6 added the ``goal_delta_ignoring_eggs`` **choice** stripe, so every era
    with minor ≤ 5 predates it: its ``choice_dim`` is
    ``CHOICE_GOAL_DELTA_IGNORING_EGGS_DIM`` (8) less than the live width (the
    newest, and so broadest, narrowing branch). v1.4 added both the two
    food-unlock **state** stripes and the ``resets_feeder`` **choice** stripe, so
    every era with minor ≤ 3 additionally predates both: its ``state_dim`` is a
    further ``2 * STATE_FOOD_UNLOCK_DIM`` (10) less (the first same-MAJOR era to
    narrow the state dim) and its ``choice_dim`` a further
    ``CHOICE_RESETS_FEEDER_DIM`` (1) less. v1.0 additionally predates the v1.1
    ``becomes_unplayable`` choice stripe, so its ``choice_dim`` drops a further
    ``CHOICE_BECOMES_UNPLAYABLE_DIM`` (180). Era 1.5 has no state branch and no
    further choice branch beyond the v1.6 one: v1.5 itself changed only stripe
    *values* (the play-bird ``goal_delta`` pricing), not shape. Raises
    ``ValueError`` for a malformed version string, and
    :class:`wingspan.version.IncompatibleArtifactError` when
    ``spec.num_players != 2`` — every superseded (pre-live) era predates N-player
    support by definition: ``num_players`` is a config-carried, default-2 field
    that was introduced alongside the live encoder, so no compat-era spec can
    honestly claim a different seat count (see the ``num_players`` entry in
    ``docs/VERSIONING.md``)."""
    if spec.num_players != 2:
        raise version.IncompatibleArtifactError(
            f"encoding era {artifact_version!r} predates N-player support "
            f"(spec.num_players={spec.num_players}) — compat-shimmed eras are "
            "2-player only; N>=3 encodings are fresh-net-only, never shimmed"
        )
    parsed = version.parse_version(artifact_version)
    state_dim = encode.state_size(spec)
    choice_dim = encode.choice_feature_dim(spec)
    # Every era with minor <= 5 predates the v1.6 goal_delta_ignoring_eggs stripe.
    if parsed.major == 1 and parsed.minor <= 5:
        choice_dim -= encode.CHOICE_GOAL_DELTA_IGNORING_EGGS_DIM
    # Every era with minor <= 3 predates the v1.4 stripes: the two food-unlock
    # state stripes and the resets_feeder choice stripe.
    if parsed.major == 1 and parsed.minor <= 3:
        state_dim -= 2 * encode.STATE_FOOD_UNLOCK_DIM
        choice_dim -= encode.CHOICE_RESETS_FEEDER_DIM
    # v1.0 additionally predates the v1.1 becomes_unplayable choice stripe.
    if parsed.major == 1 and parsed.minor == 0:
        choice_dim -= encode.CHOICE_BECOMES_UNPLAYABLE_DIM
    return (state_dim, choice_dim)
