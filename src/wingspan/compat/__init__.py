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

**v1_4 shim** (see :mod:`wingspan.compat.v1_4`): pre-1.5 geometry + behavior,
both nets — the merge of what were four provisionally-numbered eras (1.5,
1.6, 1.7, 1.8) that landed on main in sequence but never trained a run past
1.4, so they were collapsed into one era before any of them trained (the
fold-in rule; see ``docs/VERSIONING.md``). Four changes, all frozen by
:class:`wingspan.compat.v1_4.PolicyValueNetV1_4` /
:class:`wingspan.compat.v1_4.SetupNetV1_4`:

* *State* — the per-opponent ``known_hand_opp`` 180-dim multi-hot (appended
  after both playability multi-hots, before ``decision_type``), so
  ``encoding_dims_for_era`` returns a ``state_dim`` 180 less than live for
  every era with minor ≤ 4. Stripped after live ``encode_state``; only
  ``decision_type`` shifts left in ``_state_embed_offsets`` — ``card_index``
  / ``hand_multihot`` precede the new stripe and are unchanged (the
  opposite shift shape from ``v1_3``'s food-unlock strip, which precedes
  both and shifts all three).
* *Choice geometry* — an 8-dim ``goal_delta_ignoring_eggs`` tail stripe, so
  ``encoding_dims_for_era`` returns a ``choice_dim`` a further 8 less for
  every era with minor ≤ 4. Stripped after live ``encode_choices``; only
  ``kept_multihot`` shifts.
* *Choice/setup values* — the ``PlayBirdChoice`` featurizer's ``goal_delta``
  stripe habitat-conditioned (main net) and the setup ``goal_affinity``
  stripe egg-aware (setup net); both frozen to their pre-1.5 pricing via
  refill after live encoding.
* *Choice/setup values* — the bonus *potential* counters egg-optimistic on
  both nets, and (main net only) spend-decision ``FoodChoice`` rows routed
  to ``pay_food`` instead of ``gain_food``; both frozen to their pre-1.7
  pricing via refill after live encoding.

Both geometry seams (``_true_state_dim`` / ``_true_choice_dim``) derive their
narrow width absolutely from ``self.spec`` rather than composing via
``super()`` — this is the shim closest to live. It routes for era 1.4 on
both nets; ``PolicyValueNetV1_3`` **inherits** :class:`PolicyValueNetV1_4`,
so every pre-1.4 era freezes all four changes too, on top of its own
stripes and value refills.

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

    v1.5 added both the per-opponent ``known_hand_opp`` **state** stripe and
    the ``goal_delta_ignoring_eggs`` **choice** stripe (the merge of what
    were four provisionally-numbered eras, 1.5-1.8 — see
    ``compat.v1_4``'s module docstring), so every era with minor ≤ 4
    predates both: its ``state_dim`` is ``STATE_KNOWN_HAND_OPP_DIM`` (180)
    less than the live width and its ``choice_dim`` is
    ``CHOICE_GOAL_DELTA_IGNORING_EGGS_DIM`` (8) less. v1.4 added both the two
    food-unlock **state** stripes and the ``resets_feeder`` **choice**
    stripe, so every era with minor ≤ 3 additionally predates both: its
    ``state_dim`` a further ``2 * STATE_FOOD_UNLOCK_DIM`` (10) less and its
    ``choice_dim`` a further ``CHOICE_RESETS_FEEDER_DIM`` (1) less. v1.0
    additionally predates the v1.1 ``becomes_unplayable`` choice stripe, so
    its ``choice_dim`` drops a further ``CHOICE_BECOMES_UNPLAYABLE_DIM``
    (180). Raises ``ValueError`` for a malformed version string, and
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
    # Every era with minor <= 4 predates v1.5: the known_hand_opp state
    # stripe and the goal_delta_ignoring_eggs choice stripe.
    if parsed.major == 1 and parsed.minor <= 4:
        state_dim -= encode.STATE_KNOWN_HAND_OPP_DIM
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
