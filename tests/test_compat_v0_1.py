"""Backwards-compatibility smoke tests for the pinned v0.1 artifacts.

Loads the real run snapshot committed under ``tests/data/compat/v0.1/`` (see
its README for provenance; the checkpoints are gzip-compressed and LFS-tracked)
through the production loaders and proves the artifact-version contract:
same-MAJOR artifacts must load and play games. Unlike the v0.0 set, these
files carry an explicit ``version: "0.1"`` stamp, so every load here exercises
the stamped-version path — and the nets reconstruct as the *live* era (the
choice-vector reshape that 0.1 introduced), never the ``compat.v0_0`` shim.

Heavy (a ~12 MB checkpoint load plus a full self-play game), so the nets are
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
from wingspan.compat import v0_0  # noqa: E402
from wingspan.training import collect, runmeta, setup_net, setup_runmeta  # noqa: E402

FIXTURE_DIR = pathlib.Path(__file__).parent / "data" / "compat" / "v0.1"

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


def test_model_config_carries_the_explicit_version():
    descriptor = runmeta.read_model_config(str(FIXTURE_DIR))
    assert descriptor.version == "0.1"
    # The pinned descriptor must stay compatible until a deliberate MAJOR bump.
    version.check_artifact_compatible(descriptor.version, what="v0.1 fixture")


def test_setup_config_carries_the_explicit_version():
    descriptor = setup_runmeta.read_setup_config(str(FIXTURE_DIR))
    assert descriptor.version == "0.1"
    version.check_artifact_compatible(descriptor.version, what="v0.1 fixture")


def test_v0_1_net_is_the_live_era(loaded_net: model.PolicyValueNet):
    """A 0.1 descriptor reconstructs as the live net at the live choice dims —
    the compat shim is exclusively for artifacts older than the reshape."""
    assert not isinstance(loaded_net, v0_0.PolicyValueNetV00)
    assert loaded_net.choice_dim == encode.choice_feature_dim(loaded_net.spec)


def test_policy_net_loads_state_dict(
    loaded_net: model.PolicyValueNet, main_payload: dict[str, typing.Any]
):
    """The pinned weights drop into the descriptor-reconstructed net exactly,
    and the payload's explicit version stamp passes the check."""
    assert main_payload["version"] == "0.1"
    version.check_artifact_compatible(
        str(main_payload["version"]), what="v0.1 fixture last.pt"
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
    assert setup_payload["version"] == "0.1"


def test_forward_pass(loaded_net: model.PolicyValueNet):
    """A batch of freshly-encoded-shape inputs flows through the loaded
    weights to finite logits and value."""
    batch_size, n_choices = 2, 4
    state_vec = torch.zeros(batch_size, loaded_net.state_dim)
    choices = torch.randn(batch_size, n_choices, loaded_net.choice_dim)
    mask = torch.ones(batch_size, n_choices)
    family = torch.zeros(batch_size, dtype=torch.long)
    with torch.no_grad():
        logits, value = loaded_net(state_vec, choices, mask, family)
    assert logits.shape == (batch_size, n_choices)
    assert value.shape == (batch_size,)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()


def test_loaded_net_plays_a_game(loaded_net: model.PolicyValueNet):
    """The v0.1 net drives a full self-play game through the production
    collector — the load-and-play guarantee end to end."""
    record = collect.play_game(
        loaded_net, torch.device(_DEVICE), random.Random(0), seed=20260606
    )
    assert record.steps, "expected at least one recorded step"
    assert record.winner in (-1, 0, 1)
    assert all(score >= 0 for score in record.scores)


def test_param_report_matches_the_loaded_net(loaded_net: model.PolicyValueNet):
    """The era-routed parameter report equals ``sum(p.numel())`` of the
    reconstituted net — the inspect / report surfaces describe the pinned
    checkpoint exactly."""
    descriptor = runmeta.read_model_config(str(FIXTURE_DIR))
    report = runmeta.param_report_for(descriptor)
    assert report.total == sum(p.numel() for p in loaded_net.parameters())


def test_choice_layout_routes_to_the_live_registry():
    """``choice_layout_for`` on a 0.1 descriptor is the live stripe table —
    no resurrected habitat stripe — at the live choice-encoder input width."""
    descriptor = runmeta.read_model_config(str(FIXTURE_DIR))
    layout = runmeta.choice_layout_for(descriptor)
    names = [stripe.name for stripe in layout.stripes]
    assert "habitat" not in names
    expected_input = encode.choice_input_dim(
        descriptor.choice_dim,
        descriptor.architecture.card_embed_dim,
        include_setup=descriptor.include_setup,
    )
    assert layout.total_size == expected_input
    assert runmeta.choice_input_dim_for(descriptor) == expected_input
    assert runmeta.choice_extra_for(descriptor) == encode.choice_passthrough_dim(
        descriptor.choice_dim, include_setup=descriptor.include_setup
    )
