"""Version-specific artifact-compatibility shims.

Each module here keeps one older artifact era loadable and playable under the
current code, per the "Checkpoint compatibility policy" in ``CLAUDE.md``: a
MINOR ``MODEL_VERSION`` bump that reshapes an encoding ships a shim regenerating
the older shape for same-MAJOR artifacts. Shims are version-number-specific —
never config flags — and the whole package is deleted wholesale at a MAJOR bump.

* ``v0_0`` — the pre-0.1 choice-vector geometry (habitat stripe, 180-wide
  ``bird_id`` one-hot / setup multi-hot, no landing-slot marks).
"""

from wingspan.compat import v0_0

__all__ = ["v0_0"]
