"""Version-specific artifact-compatibility shims.

Currently **empty**: the pre-1.0 shims (``v0_0`` … ``v0_7``) were dropped at the
1.0 MAJOR bump, per the compat policy in ``docs/VERSIONING.md`` (a MAJOR bump
deletes the accumulated shims and old fixture sets wholesale). No 0.x artifact
loads under 1.0 code — the loaders refuse a different-MAJOR artifact via
``version.check_artifact_compatible``.

The package is kept as the documented home for the seam: the next MINOR FRESH
change adds one module here (``v1_<N>``) that regenerates the older same-MAJOR
shape, routed by ``model.PolicyValueNet.class_for_version`` and by
:func:`encoding_dims_for_era`. Each module is version-number-specific — never a
config flag — and the whole package is deleted again at the next MAJOR bump.

:func:`encoding_dims_for_era` is the package-level dims router: given an artifact
version it returns the raw state/choice vector widths that era's encoders
produce, so an era-pinned ``RunConfig`` derives the dims its checkpoints actually
carry. With no shims present it returns the live widths for every (compatible)
artifact; a future MINOR FRESH change branches it on its era predicate.
"""

from wingspan import encode, version

__all__ = ["encoding_dims_for_era"]


def encoding_dims_for_era(
    artifact_version: str, spec: encode.EncodingSpec
) -> tuple[int, int]:
    """The raw ``(state_dim, choice_dim)`` an era's encoders produce under ``spec``.

    No pre-1.0 shims remain, so every same-MAJOR artifact uses the live widths;
    a future MINOR FRESH change adds the era branch here (see the package
    docstring). Raises ``ValueError`` for a malformed version string."""
    version.parse_version(artifact_version)  # reject malformed; eras branch here later
    return (encode.state_size(spec), encode.choice_feature_dim(spec))
