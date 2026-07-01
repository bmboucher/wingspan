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

**v1_3 shim** (see :mod:`wingspan.compat.v1_3`): v1.4 appended a 1-dim
``resets_feeder`` stripe as the last base choice stripe (after
``becomes_unplayable``), marking a ``combine_gain_food`` ``FoodSubsetChoice`` whose
selection rerolls the birdfeeder. v1.1–1.3 choice vectors lack it, so
``PolicyValueNetV1_3`` strips it after live encoding and shifts only
``kept_multihot``. The v1.0 shim inherits ``PolicyValueNetV1_3`` to compose that
strip with its own ``becomes_unplayable`` strip (v1.0 lacks both). Routed from
``model.PolicyValueNet.class_for_version`` (1.1–1.3 → v1_3, 1.0 → v1_0).

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

    Each superseded era's ``choice_dim`` is the live width minus every stripe added
    after it: every era with minor ≤ 3 predates the v1.4 ``resets_feeder`` stripe
    (1), and v1.0 additionally predates the v1.1 ``becomes_unplayable`` stripe (180).
    The state dim is unchanged across all same-MAJOR eras. Raises ``ValueError`` for
    a malformed version string."""
    parsed = version.parse_version(artifact_version)
    state_dim = encode.state_size(spec)
    choice_dim = encode.choice_feature_dim(spec)
    # Every era with minor <= 3 predates the v1.4 resets_feeder stripe.
    if parsed.major == 1 and parsed.minor <= 3:
        choice_dim -= encode.CHOICE_RESETS_FEEDER_DIM
    # v1.0 additionally predates the v1.1 becomes_unplayable stripe.
    if parsed.major == 1 and parsed.minor == 0:
        choice_dim -= encode.CHOICE_BECOMES_UNPLAYABLE_DIM
    return (state_dim, choice_dim)
