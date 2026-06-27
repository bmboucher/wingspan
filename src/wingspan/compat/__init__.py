"""Version-specific artifact-compatibility shims.

**v1.0 shim** (see :mod:`wingspan.compat.v1_0`): the trunk's final-layer
activation fallback changed in v1.1 — ``trunk_final_activation=null`` now
inherits ``final_activation`` (the universal rule) instead of
``between_activation`` (the old trunk-specific exception). The shim class
:class:`wingspan.compat.v1_0.PolicyValueNetV1_0` restores the old fallback for
v1.0 artifacts, routed from ``model.PolicyValueNet.class_for_version``. The
pre-1.0 shims (``v0_0`` … ``v0_7``) were dropped at the 1.0 MAJOR bump; no 0.x
artifact loads under 1.x code.

Each module is version-number-specific — never a config flag — and the whole
package is deleted again at the next MAJOR bump.

:func:`encoding_dims_for_era` is the package-level dims router: given an artifact
version it returns the raw state/choice vector widths that era's encoders
produce, so an era-pinned ``RunConfig`` derives the dims its checkpoints actually
carry. The v1.0 encoding is identical to v1.1 (this is a model-architecture
change, not an encoding shape change), so the router falls through to live widths
for all same-MAJOR artifacts; a future encoding-shape FRESH change branches it.
"""

from wingspan import encode, version

__all__ = ["encoding_dims_for_era"]


def encoding_dims_for_era(
    artifact_version: str, spec: encode.EncodingSpec
) -> tuple[int, int]:
    """The raw ``(state_dim, choice_dim)`` an era's encoders produce under ``spec``.

    The v1.0 → v1.1 change was a model-architecture change (trunk final activation
    fallback), not an encoding shape change — both eras produce the same dims.
    The router falls through to live widths for all same-MAJOR artifacts;
    a future encoding-shape FRESH change branches it on its era predicate.
    Raises ``ValueError`` for a malformed version string."""
    version.parse_version(artifact_version)  # reject malformed; eras branch here later
    return (encode.state_size(spec), encode.choice_feature_dim(spec))
