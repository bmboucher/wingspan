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

MODEL_VERSION = "0.2"
"""The current artifact-compatibility version (the only place it is defined).

0.2 makes the setup input vector dynamic: ``kept_foods`` is omitted when
``split_setup_food=True``; ``kept_bonus`` + ``kept_bonus_value`` are replaced by
``bonus_cards`` (multi-hot of available bonuses) + ``bonus_card_affinity``
(min/max qualifier counts, 2 dims) when ``split_setup_bonus=True``.  The vector
size changes with the flags (308 / 303 / 306 / 301 depending on config).  Pre-0.2
setup artifacts load as ``SetupEncoding(split_food=False, split_bonus=False)``
(the old 308-dim layout) via Pydantic defaults — no explicit shim needed.

0.1 reshaped the choice vector (landing-slot placement encoding, the single
``bird_id`` index column, the dedicated ``kept_multihot`` stripe); pre-0.1
artifacts load and play through the ``wingspan.compat.v0_0`` shim."""

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

    The first real shim landed with 0.1 and lives, as this docstring always
    promised, in the dedicated ``wingspan.compat`` package:
    ``compat.v0_0`` regenerates the pre-0.1 choice encoding for same-major
    artifacts (``compat.v0_0.encode_choices`` + ``PolicyValueNetV00``), routed
    by the loaders (``model.PolicyValueNet.from_model_config``,
    ``players.loaders.load_policy_net``) and by the era-aware
    expected-encoding keys in ``players.loaders``. Future MINOR encoding
    changes follow the same shape: a ``compat.v<X_Y>`` module keyed on
    ``parse_version(artifact_version)`` older-than-the-change.

    This function itself stays a validating no-op (this module is torch-free
    and must not import the shims); it remains so a future caller that only
    needs the validation keeps a stable seam.
    """
    parse_version(artifact_version)
