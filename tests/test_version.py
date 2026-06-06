"""Tests for the artifact-compatibility version machinery (``wingspan.version``).

Pure-logic coverage of parsing and the load guarantee — same MAJOR with an
older-or-equal MINOR loads, everything else refuses — plus the explicit-version
path of the config descriptors. The fixture-driven proof that real v0.0
artifacts load and play lives in ``test_compat_v0_0.py``.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import architecture, version  # noqa: E402
from wingspan.training import runmeta  # noqa: E402


def test_parse_version_accepts_major_minor():
    parsed = version.parse_version("0.0")
    assert (parsed.major, parsed.minor) == (0, 0)
    assert str(parsed) == "0.0"
    assert (
        version.parse_version("12.34").major,
        version.parse_version("12.34").minor,
    ) == (12, 34)


def test_parse_version_rejects_garbage():
    for raw in ("abc", "1", "1.2.3", "1.x", "", "1.", ".2", "-1.0"):
        with pytest.raises(ValueError):
            version.parse_version(raw)


def test_model_version_constant_parses():
    """The constant itself must always be a well-formed MAJOR.MINOR string."""
    version.parse_version(version.MODEL_VERSION)


def test_check_artifact_compatible_same_version_passes():
    version.check_artifact_compatible(version.MODEL_VERSION, what="x")


def test_check_artifact_compatible_refuses_different_major():
    current = version.parse_version(version.MODEL_VERSION)
    newer_major = f"{current.major + 1}.0"
    with pytest.raises(version.IncompatibleArtifactError):
        version.check_artifact_compatible(newer_major, what="x")


def test_check_artifact_compatible_refuses_newer_minor():
    current = version.parse_version(version.MODEL_VERSION)
    newer_minor = f"{current.major}.{current.minor + 1}"
    with pytest.raises(version.IncompatibleArtifactError):
        version.check_artifact_compatible(newer_minor, what="x")


def test_check_artifact_compatible_error_names_the_artifact():
    current = version.parse_version(version.MODEL_VERSION)
    with pytest.raises(version.IncompatibleArtifactError, match="model_config"):
        version.check_artifact_compatible(
            f"{current.major + 1}.0", what="model_config.json at /some/run"
        )


def test_adapt_encoding_seam_validates_its_input():
    """The (currently no-op) shim seam still rejects malformed versions so a
    future shim never has to re-validate."""
    version.adapt_encoding_for_version(version.MODEL_VERSION)
    with pytest.raises(ValueError):
        version.adapt_encoding_for_version("not-a-version")


def test_model_config_explicit_version_round_trips():
    """A descriptor carrying an explicit version keeps it through JSON."""
    descriptor = runmeta.ModelConfig(
        run_name="explicit",
        state_dim=4,
        choice_dim=3,
        family_order=("main_action",),
        architecture=architecture.ModelArchitecture(),
        include_setup=False,
        version="0.0",
    )
    reread = runmeta.ModelConfig.model_validate_json(descriptor.model_dump_json())
    assert reread.version == "0.0"


def test_model_config_defaults_missing_version():
    """A descriptor JSON that predates the field reads as the pre-versioning
    era ("0.0") — the sanctioned new-field-default mechanism."""
    descriptor = runmeta.ModelConfig(
        run_name="legacy",
        state_dim=4,
        choice_dim=3,
        family_order=("main_action",),
        architecture=architecture.ModelArchitecture(),
        include_setup=False,
    )
    raw = descriptor.model_dump_json(exclude={"version"})
    assert '"version"' not in raw
    assert runmeta.ModelConfig.model_validate_json(raw).version == "0.0"
