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

Training *resume* is not covered by this contract — the resume gate keeps its
strict ``architecture_key`` comparison and starts fresh on any mismatch.

This is distinct from the *package release* version
(``wingspan.__version__``): that tracks the codebase, this tracks the on-disk
artifact format. Kept torch-free and dependency-free (stdlib + pydantic only)
so every loader module can import it without cycles.
"""

from __future__ import annotations

import re

import pydantic

MODEL_VERSION = "0.0"
"""The current artifact-compatibility version (the only place it is defined)."""

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
    """The seam where version-specific encoding shims land.

    When a MINOR bump changes the encoding (e.g. version 1.3 adds a stripe),
    the shim that regenerates the *older* encoding for a same-major artifact
    belongs here — the shape is ``if parse_version(artifact_version) older
    than the change: encode without the new field``. Such a shim must also
    satisfy (or route around) the live-layout checks at
    ``selfplay._encoding_key`` and
    ``tournament.participants._descriptor_encoding_key``, which currently
    require an exact match with the live encoder.

    At version 0.0 nothing has changed yet, so this is an identity no-op kept
    as the documented hook; promote it to a dedicated ``compat`` module when
    the first real shim arrives.
    """
    parse_version(artifact_version)
