"""Bundled-model resolution for the pip-installed ``wingspan-rl`` package.

The wheel ships one stripped trained artifact under ``wingspan/data/release/``
in the standard run-directory layout (``last.pt``, ``setup.pt``,
``run_config_*.json``, ``manifest.json``), so ``wingspan play`` works with no
checkout and no training. This module is the single home for locating it. It
imports only the stdlib and pydantic (never torch, ``players`` or ``training``)
so ``players.spec`` can depend on it without import cycles.

The package version (``wingspan.__version__``) is distinct from the artifact
era (``version.MODEL_VERSION``); the manifest records the latter.
"""

from __future__ import annotations

import importlib.resources
import pathlib

import pydantic

RELEASE_SUBDIR = "data/release"  # relative to the ``wingspan`` package directory
MANIFEST_FILE = "manifest.json"

# Mirrors ``training.artifacts.LAST_CKPT``; duplicated here so this module stays
# free of the heavy training imports.
_LAST_CKPT = "last.pt"


class ReleaseManifest(pydantic.BaseModel):
    """Provenance of the bundled checkpoint, written by ``release_prep``."""

    era: str  # artifact MODEL_VERSION stamped in the bundled checkpoint
    run_name: str
    source_file: str  # e.g. "last.pt"
    iteration: int | None = None
    git_sha: str | None = None
    created: str  # UTC ISO timestamp


def release_dir() -> pathlib.Path:
    """The bundled release directory as a concrete path (it may not exist when
    the package was built without a bundle). Wheel installs are always unzipped
    directories; anything else is an error."""
    traversable = importlib.resources.files("wingspan")
    for part in RELEASE_SUBDIR.split("/"):
        traversable = traversable / part
    path = pathlib.Path(str(traversable))
    if path.exists() and not path.is_dir():
        raise RuntimeError(f"Bundled release path is not a directory: {path}")
    return path


def has_release() -> bool:
    """Whether a bundled model (a ``last.pt``) is present."""
    return (release_dir() / _LAST_CKPT).is_file()


def load_manifest() -> ReleaseManifest | None:
    """Parse the bundled ``manifest.json``; ``None`` when absent."""
    path = release_dir() / MANIFEST_FILE
    if not path.is_file():
        return None
    return ReleaseManifest.model_validate_json(path.read_text(encoding="utf-8"))
