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
  no ``or_cost`` flag in the per-card attribute vector; also restores v0.7
  eggs-included food ``becomes_playable`` semantics and the 1155-dim pre-0.9
  state geometry via delegating overrides).
* ``v0_7`` — the pre-0.8 eggs-included food ``becomes_playable`` semantics
  (food-gain rows gate on both food AND eggs; 0.8 drops the egg-cost gate);
  also carries the 1155-dim pre-0.9 state geometry via delegating overrides.
* ``v0_8`` — the pre-0.9 state geometry (1155-dim; compacted to 1119 in v0.9)
  AND the pre-0.9 choice board geometry (``board_target`` 120 dims with
  per-type cached food; ``board_idx`` 15-slot embedded block); the first
  board-geometry change since 0.1, so all earlier shims route through v0_8.

:func:`encoding_dims_for_era` is the package-level dims router: given an
artifact version it returns the raw state/choice vector widths that era's
encoders produce, so an era-pinned ``TrainConfig`` (training resume across a
FRESH change — see ``docs/VERSIONING.md``) derives the dims its checkpoints
actually carry instead of the live ones.
"""

from wingspan import encode
from wingspan.compat import v0_0, v0_1, v0_2, v0_3, v0_4, v0_6, v0_7, v0_8

__all__ = [
    "encoding_dims_for_era",
    "v0_0",
    "v0_1",
    "v0_2",
    "v0_3",
    "v0_4",
    "v0_6",
    "v0_7",
    "v0_8",
]


def encoding_dims_for_era(
    artifact_version: str, spec: encode.EncodingSpec
) -> tuple[int, int]:
    """The raw ``(state_dim, choice_dim)`` an era's encoders produce under ``spec``.

    Routes each axis through the shim that froze it: pre-0.3 artifacts carry the
    771-dim misc-scalar state vector (``v0_2``), pre-0.4 carry the 790-dim one-hot
    state vector (``v0_3``), pre-0.6 carry the 795-dim no-playability state vector
    and narrower pre-0.6 choice rows (``v0_4``), 0.6–0.8 carry the 1155-dim
    pre-compaction state vector (``v0_8``), 0.1–0.8 carry the frozen board choice
    geometry (``v0_8``), pre-0.1 carry the reshaped-away choice geometry (``v0_0``),
    and current-era artifacts get the live widths.
    Raises ``ValueError`` for a malformed version string."""
    if v0_2.uses_v0_2_state_encoding(artifact_version):
        state_dim = v0_2.state_feature_dim_v02(spec)
    elif v0_3.uses_v0_3_state_encoding(artifact_version):
        state_dim = v0_3.state_feature_dim_v03(spec)
    elif v0_4.uses_v0_4_encoding(artifact_version):
        state_dim = v0_4.state_feature_dim_v04(spec)
    elif _uses_1155_dim_state(artifact_version):
        # Covers 0.6/0.7/0.8 — all share the 1155-dim state geometry that v0.9
        # compacted. The earlier branches already handle ≤ 0.5.
        state_dim = v0_8.state_feature_dim_v08(spec)
    else:
        state_dim = encode.state_size(spec)
    if v0_0.uses_v0_0_choice_encoding(artifact_version):
        choice_dim = v0_0.choice_feature_dim(spec)
    elif _uses_pre_v06_choice_encoding(artifact_version):
        # v0.1 through v0.5: pre-0.6 choice format (no becomes_playable stripe)
        # AND pre-0.9 board geometry (board_target 120, board_idx 15).
        choice_dim = v0_8.choice_feature_dim_v08(spec, has_becomes_playable=False)
    elif v0_8.uses_v0_8_choice_encoding(artifact_version):
        # v0.6 through v0.8: has becomes_playable stripe but pre-0.9 board geometry.
        choice_dim = v0_8.choice_feature_dim_v08(spec, has_becomes_playable=True)
    else:
        choice_dim = encode.choice_feature_dim(spec)
    return (state_dim, choice_dim)


def _uses_1155_dim_state(artifact_version: str) -> bool:
    """True for artifact versions 0.6–0.8 that use the 1155-dim pre-0.9 state vector.

    State has been 1155-dim since the v0.6 playability-stripe bump; v0.9 compacted
    it to 1119. Versions ≤ 0.5 are caught earlier in the routing chain."""
    from wingspan import version  # local: avoids top-level import of version

    parsed = version.parse_version(artifact_version)
    v09 = version.parse_version("0.9")
    v06 = version.parse_version("0.6")
    return (
        (v06.major, v06.minor)
        <= (parsed.major, parsed.minor)
        < (
            v09.major,
            v09.minor,
        )
    )


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
