"""The artifact-compatibility version and its load-time enforcement.

Every persisted training artifact (the ``model_config.json`` /
``setup_config.json`` sidecars and the ``*.pt`` checkpoint payloads) is stamped
with :data:`MODEL_VERSION`, a ``MAJOR.MINOR`` string that is bumped whenever the
encoding or network architecture changes shape. The compatibility contract:

* **Same MAJOR, artifact MINOR <= current MINOR** — the artifact must load and
  play games (inference / eval / tournament). Older-minor artifacts are kept
  loadable via version-specific shims (see :func:`adapt_encoding_for_version`),
  never via per-change config flags.
* **Different MAJOR, or artifact MINOR > current MINOR** — the loaders refuse
  with :class:`IncompatibleArtifactError`. A MAJOR bump is the deliberate
  escape hatch that deletes the accumulated shims and old test fixtures.

Training *resume* honors the same eras via pinning: a run carries its
``TrainConfig.encoding_version`` and keeps training at that era's frozen
geometry under newer same-MAJOR code (``loop_resume.adopt_checkpoint_era``,
``docs/VERSIONING.md``), stamping every artifact it writes with its own era.
The resume gate still refuses any genuine ``architecture_key`` mismatch and
starts fresh — and every *fresh* launch is re-keyed at the live
:data:`MODEL_VERSION`, so a new run never trains at a stale era.

This is distinct from the *package release* version
(``wingspan.__version__``): that tracks the codebase, this tracks the on-disk
artifact format. Kept torch-free and dependency-free (stdlib + pydantic only)
so every loader module can import it without cycles.
"""

from __future__ import annotations

import re

import pydantic

MODEL_VERSION = "1.0"
"""The current artifact-compatibility version (the only place it is defined).

1.0 is the clean-break baseline. It was a MAJOR bump that dropped the accumulated
pre-1.0 compat shims (``wingspan.compat.v0_0`` … ``v0_8``), deleted the old
fixture sets, and removed the dead code paths those shims existed to support. No
0.x artifact loads under 1.0 code: ``check_artifact_compatible`` refuses any
different-MAJOR artifact. The per-version 0.1–0.8 changelog that used to live
here is recoverable from git history and summarized in ``docs/VERSIONING.md``.

The live encoding 1.0 ships is the geometry main reached at 0.9 — the state
vector compacted to 1119 dims and the choice vector to 328 (board-target
compression; the ``board_hab`` / ``board_col`` habitat + column one-hots
replacing the embedded ``board_idx`` block) — with no compat path back to any
earlier shape.

The versioning *machinery* is intact, just empty: a future MINOR FRESH change
adds its ``wingspan.compat.v1_<N>`` module and routes through the same seams
(``model.PolicyValueNet.class_for_version``, ``compat.encoding_dims_for_era``)
that currently fall straight through to the live encoders."""

PRE_VERSIONING_VERSION = "0.0"
"""The version assigned to artifacts that predate the ``version`` field.

Files lacking the field were by definition written before versioning existed,
so they read as the original era — this stays pinned at ``"0.0"`` forever while
:data:`MODEL_VERSION` advances."""

_VERSION_PATTERN = re.compile(r"^(\d+)\.(\d+)$")


class Version(pydantic.BaseModel):
    """A parsed ``MAJOR.MINOR`` artifact version."""

    model_config = pydantic.ConfigDict(frozen=True)

    major: int
    minor: int

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}"


class IncompatibleArtifactError(Exception):
    """Raised when a persisted artifact's version cannot be loaded by this code."""


def parse_version(raw: str) -> Version:
    """Parse a ``MAJOR.MINOR`` string into a :class:`Version`.

    Raises ``ValueError`` for anything that is not exactly two dot-separated
    integers (``"1"``, ``"1.2.3"``, ``"abc"``)."""
    match = _VERSION_PATTERN.match(raw)
    if match is None:
        raise ValueError(
            f"Invalid artifact version {raw!r}: expected 'MAJOR.MINOR' "
            "(two dot-separated integers, e.g. '0.0')."
        )
    return Version(major=int(match.group(1)), minor=int(match.group(2)))


def check_artifact_compatible(artifact_version: str, *, what: str) -> None:
    """Refuse an artifact this code does not guarantee to load.

    ``what`` is a short label naming the artifact (e.g. ``"model_config.json at
    <dir>"``) folded into the error message. Passes silently when the artifact
    shares the current MAJOR and its MINOR is at most the current MINOR; raises
    :class:`IncompatibleArtifactError` otherwise."""
    artifact = parse_version(artifact_version)
    current = parse_version(MODEL_VERSION)
    if artifact.major != current.major:
        raise IncompatibleArtifactError(
            f"{what} has artifact version {artifact} but this code is version "
            f"{current}: different MAJOR versions are not loadable. Use a "
            f"codebase from the {artifact.major}.x line, or retrain."
        )
    if artifact.minor > current.minor:
        raise IncompatibleArtifactError(
            f"{what} has artifact version {artifact} but this code is version "
            f"{current}: the artifact is newer than this code understands. "
            "Update the codebase to load it."
        )


def adapt_encoding_for_version(artifact_version: str) -> None:
    """The seam where version-specific encoding shims are documented.

    The ``wingspan.compat`` package is currently empty — the pre-1.0 shims were
    dropped at the 1.0 MAJOR bump. The next MINOR FRESH change re-introduces one:
    a ``compat.v1_<N>`` module keyed on ``parse_version(artifact_version)``
    older-than-the-change, regenerating the prior shape for same-MAJOR artifacts,
    routed by the loaders (``model.PolicyValueNet.from_model_config`` →
    ``class_for_version``, ``players.loaders``).

    This function itself stays a validating no-op (this module is torch-free
    and must not import the shims); it remains so a future caller that only
    needs the validation keeps a stable seam.
    """
    parse_version(artifact_version)
