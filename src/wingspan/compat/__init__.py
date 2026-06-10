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

:func:`encoding_dims_for_era` is the package-level dims router: given an
artifact version it returns the raw state/choice vector widths that era's
encoders produce, so an era-pinned ``TrainConfig`` (training resume across a
FRESH change — see ``docs/VERSIONING.md``) derives the dims its checkpoints
actually carry instead of the live ones.
"""

from wingspan import encode
from wingspan.compat import v0_0, v0_1, v0_2

__all__ = ["encoding_dims_for_era", "v0_0", "v0_1", "v0_2"]


def encoding_dims_for_era(
    artifact_version: str, spec: encode.EncodingSpec
) -> tuple[int, int]:
    """The raw ``(state_dim, choice_dim)`` an era's encoders produce under ``spec``.

    Routes each axis through the shim that froze it: pre-0.3 artifacts carry the
    771-dim misc-scalar state vector (``v0_2``), and pre-0.1 artifacts carry the
    reshaped-away choice geometry (``v0_0``). Current-era artifacts get the live
    widths. Raises ``ValueError`` for a malformed version string."""
    state_dim = (
        v0_2.state_feature_dim_v02(spec)
        if v0_2.uses_v0_2_state_encoding(artifact_version)
        else encode.state_size(spec)
    )
    choice_dim = (
        v0_0.choice_feature_dim(spec)
        if v0_0.uses_v0_0_choice_encoding(artifact_version)
        else encode.choice_feature_dim(spec)
    )
    return (state_dim, choice_dim)
