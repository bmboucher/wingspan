"""Backwards-compatibility smoke tests for the pinned v0.0 artifacts.

Loads the real run snapshot committed under ``tests/data/compat/v0.0/`` (see
its README for provenance; the checkpoints are gzip-compressed and LFS-tracked)
through the production loaders and proves the artifact-version contract:
same-MAJOR artifacts must load and play games. The fixture files deliberately
predate the ``version`` field, so every load here also exercises the
default-to-"0.0" path real legacy artifacts take.

Heavy (a ~13 MB checkpoint load plus a full self-play game), so the nets are
loaded once per module and the file is front-loaded via ``_HEAVY_TEST_FILES``.
"""

from __future__ import annotations

import gzip
import io
import os
import pathlib
import random
import sys
import typing

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import encode, model, version  # noqa: E402
from wingspan.training import collect, runmeta, setup_net, setup_runmeta  # noqa: E402

FIXTURE_DIR = pathlib.Path(__file__).parent / "data" / "compat" / "v0.0"

_DEVICE = "cpu"


def _load_gzipped_checkpoint(filename: str) -> dict[str, typing.Any]:
    """Load a fixture checkpoint stored gzip-compressed (and LFS-tracked).

    Decompressed fully into memory first — ``torch.load`` needs a seekable
    stream, which a ``gzip`` file object only fakes by re-decompressing."""
    raw = gzip.decompress((FIXTURE_DIR / filename).read_bytes())
    return typing.cast(
        "dict[str, typing.Any]",
        torch.load(io.BytesIO(raw), map_location=_DEVICE, weights_only=False),
    )


@pytest.fixture(scope="module")
def main_payload() -> dict[str, typing.Any]:
    """The fixture run's ``last.pt`` payload, loaded once for the module."""
    return _load_gzipped_checkpoint("last.pt.gz")


@pytest.fixture(scope="module")
def loaded_net(main_payload: dict[str, typing.Any]) -> model.PolicyValueNet:
    """The fixture run's main net: descriptor-reconstructed, real weights."""
    descriptor = runmeta.read_model_config(str(FIXTURE_DIR))
    net = model.PolicyValueNet.from_model_config(descriptor)
    net.load_state_dict(typing.cast("dict[str, torch.Tensor]", main_payload["model"]))
    net.eval()
    return net


@pytest.fixture(scope="module")
def setup_payload() -> dict[str, typing.Any]:
    """The fixture run's ``setup.pt`` payload, loaded once for the module."""
    return _load_gzipped_checkpoint("setup.pt.gz")


def test_model_config_loads_with_default_version():
    descriptor = runmeta.read_model_config(str(FIXTURE_DIR))
    assert descriptor.version == version.PRE_VERSIONING_VERSION
    # The pinned descriptor must stay compatible until a deliberate MAJOR bump.
    version.check_artifact_compatible(descriptor.version, what="v0.0 fixture")


def test_setup_config_loads_with_default_version():
    descriptor = setup_runmeta.read_setup_config(str(FIXTURE_DIR))
    assert descriptor.version == version.PRE_VERSIONING_VERSION
    version.check_artifact_compatible(descriptor.version, what="v0.0 fixture")


def test_policy_net_loads_state_dict(
    loaded_net: model.PolicyValueNet, main_payload: dict[str, typing.Any]
):
    """The pinned weights drop into the descriptor-reconstructed net exactly,
    and the payload — which predates the version stamp — passes the check via
    the pre-versioning default."""
    assert "version" not in main_payload
    version.check_artifact_compatible(
        str(main_payload.get("version", version.PRE_VERSIONING_VERSION)),
        what="v0.0 fixture last.pt",
    )
    # Strict mode (the default) raises on any missing or unexpected key, so a
    # clean load *is* the exact-key-match assertion.
    state_dict = typing.cast("dict[str, torch.Tensor]", main_payload["model"])
    loaded_net.load_state_dict(state_dict)


def test_setup_net_loads_state_dict(setup_payload: dict[str, typing.Any]):
    descriptor = setup_runmeta.read_setup_config(str(FIXTURE_DIR))
    net = setup_net.SetupNet.from_setup_config(descriptor)
    net.load_state_dict(
        typing.cast("dict[str, torch.Tensor]", setup_payload["setup_model"])
    )
    net.eval()
    assert "version" not in setup_payload


def test_forward_pass(loaded_net: model.PolicyValueNet):
    """A batch of freshly-encoded-shape inputs flows through the loaded
    weights to finite logits and value."""
    spec = encode.EncodingSpec(include_setup=loaded_net.include_setup)
    batch_size, n_choices = 2, 4
    state_vec = torch.zeros(batch_size, encode.state_size(spec))
    choices = torch.randn(batch_size, n_choices, encode.choice_feature_dim(spec))
    mask = torch.ones(batch_size, n_choices)
    family = torch.zeros(batch_size, dtype=torch.long)
    with torch.no_grad():
        logits, value = loaded_net(state_vec, choices, mask, family)
    assert logits.shape == (batch_size, n_choices)
    assert value.shape == (batch_size,)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()


def test_loaded_net_plays_a_game(loaded_net: model.PolicyValueNet):
    """The v0.0 net drives a full self-play game through the production
    collector — the load-and-play guarantee end to end."""
    record = collect.play_game(
        loaded_net, torch.device(_DEVICE), random.Random(0), seed=20260605
    )
    assert record.steps, "expected at least one recorded step"
    assert record.winner in (-1, 0, 1)
    assert all(score >= 0 for score in record.scores)
