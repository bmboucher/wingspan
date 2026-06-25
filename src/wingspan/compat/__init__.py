"""Version-specific artifact-compatibility shims.

Each module here keeps one older artifact era loadable and playable under the
current code, per the "Checkpoint compatibility policy" in ``CLAUDE.md``: a
MINOR ``MODEL_VERSION`` bump that reshapes an encoding ships a shim regenerating
the older shape for same-MAJOR artifacts. Shims are version-number-specific —
never config flags — and the whole package is deleted wholesale at a MAJOR bump.

* ``v0_0`` — the pre-0.1 choice-vector geometry (habitat stripe, 180-wide
  ``bird_id`` one-hot / setup multi-hot, no landing-slot marks).
* ``v0_1`` — the pre-0.2 card-feature geometry (229-wide input: 23 attr dims +
  26-wide bonus-categories multi-hot + 180 bird-identity one-hot).
* ``v0_2`` — the pre-0.3 misc-scalar state geometry (round ÷ 3, cubes ÷ 8;
  771-dim state vector before the one-hot round + cube stripes).
* ``v0_3`` — the pre-0.4 state geometry (one-hot round + cubes in misc_scalars,
  no leading turn_state stripe; 790-dim state vector).
* ``v0_4`` — the pre-0.6 state + choice geometry (no hand-playability multi-hot
  stripes in state; no ``becomes_playable`` stripe in choices; 795-dim state).
* ``v0_6`` — the pre-0.7 card-feature geometry (224-wide card encoder input;
  no ``or_cost`` flag in the per-card attribute vector).

:func:`encoding_dims_for_era` is the package-level dims router: given an
artifact version it returns the raw state/choice vector widths that era's
encoders produce, so an era-pinned ``TrainConfig`` (training resume across a
FRESH change — see ``docs/VERSIONING.md``) derives the dims its checkpoints
actually carry instead of the live ones. The v0.7 card-feature change does not
affect state/choice vector widths, so no new branch is needed here for v0.6
artifacts.
"""

from wingspan import encode
from wingspan.compat import v0_0, v0_1, v0_2, v0_3, v0_4, v0_6

__all__ = ["encoding_dims_for_era", "v0_0", "v0_1", "v0_2", "v0_3", "v0_4", "v0_6"]


def encoding_dims_for_era(
    artifact_version: str, spec: encode.EncodingSpec
) -> tuple[int, int]:
    """The raw ``(state_dim, choice_dim)`` an era's encoders produce under ``spec``.

    Routes each axis through the shim that froze it: pre-0.3 artifacts carry the
    771-dim misc-scalar state vector (``v0_2``), pre-0.4 artifacts carry the
    790-dim one-hot state vector (``v0_3``), pre-0.6 artifacts carry the 795-dim
    no-playability state vector and the narrower pre-0.6 choice rows (``v0_4``),
    and pre-0.1 artifacts carry the reshaped-away choice geometry (``v0_0``).
    Current-era artifacts get the live widths.
    Raises ``ValueError`` for a malformed version string."""
    if v0_2.uses_v0_2_state_encoding(artifact_version):
        state_dim = v0_2.state_feature_dim_v02(spec)
    elif v0_3.uses_v0_3_state_encoding(artifact_version):
        state_dim = v0_3.state_feature_dim_v03(spec)
    elif v0_4.uses_v0_4_encoding(artifact_version):
        state_dim = v0_4.state_feature_dim_v04(spec)
    else:
        state_dim = encode.state_size(spec)
    if v0_0.uses_v0_0_choice_encoding(artifact_version):
        choice_dim = v0_0.choice_feature_dim(spec)
    elif _uses_pre_v06_choice_encoding(artifact_version):
        # v0.1 through v0.5: same pre-0.6 choice format (no becomes_playable stripe).
        # v0_4.uses_v0_4_encoding covers 0.4–0.5; the elif here catches 0.1–0.3
        # which also predate the v0.6 becomes_playable addition.
        choice_dim = v0_4.choice_feature_dim_v04(spec)
    else:
        choice_dim = encode.choice_feature_dim(spec)
    return (state_dim, choice_dim)


def _uses_pre_v06_choice_encoding(artifact_version: str) -> bool:
    """True for artifact versions that predate the v0.6 ``becomes_playable`` choice stripe.

    Covers versions 0.1 through 0.5 — the versions not caught by
    ``v0_0.uses_v0_0_choice_encoding`` (which handles pre-0.1) that nonetheless
    predate the ``becomes_playable`` stripe added in 0.6."""
    from wingspan import version  # local: avoids top-level import of version

    parsed = version.parse_version(artifact_version)
    becomes_playable_added = version.parse_version(v0_4.PLAYABILITY_STRIPES_ADDED_IN)
    return (parsed.major, parsed.minor) < (
        becomes_playable_added.major,
        becomes_playable_added.minor,
    )
