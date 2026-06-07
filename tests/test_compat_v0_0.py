"""Backwards-compatibility smoke tests for the pinned v0.0 artifacts.

Loads the real run snapshot committed under ``tests/data/compat/v0.0/`` (see
its README for provenance; the checkpoints are gzip-compressed and LFS-tracked)
through the production loaders and proves the artifact-version contract:
same-MAJOR artifacts must load and play games. The fixture files deliberately
predate the ``version`` field, so every load here also exercises the
default-to-"0.0" path real legacy artifacts take — and, since the 0.1
choice-vector reshape, the ``compat.v0_0`` shim: the descriptor routes to the
frozen-era ``PolicyValueNetV00``, whose game drives the v0.0 row transform
end to end.

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

from wingspan import model, version  # noqa: E402
from wingspan.compat import v0_0, v0_1  # noqa: E402
from wingspan.training import collect, runmeta, setup_runmeta  # noqa: E402

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
    # v0.0 artifacts have a 229-wide card encoder (CARD_FEATURE_DIM was 229
    # before the 0.2 change), so reconstruct via the frozen v0.1 shim.
    net = v0_1.SetupNetV01.from_setup_config(descriptor)
    net.load_state_dict(
        typing.cast("dict[str, torch.Tensor]", setup_payload["setup_model"])
    )
    net.eval()
    assert "version" not in setup_payload


def test_pre_0_1_net_is_the_frozen_compat_subclass(loaded_net: model.PolicyValueNet):
    """The version-less (v0.0) descriptor routes to the frozen-era compat net,
    whose choice geometry matches the pinned weights — not the live encoder."""
    assert isinstance(loaded_net, v0_0.PolicyValueNetV00)
    assert loaded_net.choice_dim == v0_0.choice_feature_dim(loaded_net.spec)


def test_forward_pass(loaded_net: model.PolicyValueNet):
    """A batch of inputs in the net's own (frozen v0.0) shape flows through
    the loaded weights to finite logits and value."""
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
    """The v0.0 net drives a full self-play game through the production
    collector — the load-and-play guarantee end to end."""
    record = collect.play_game(
        loaded_net, torch.device(_DEVICE), random.Random(0), seed=20260605
    )
    assert record.steps, "expected at least one recorded step"
    assert record.winner in (-1, 0, 1)
    assert all(score >= 0 for score in record.scores)


def test_param_report_matches_the_loaded_net(loaded_net: model.PolicyValueNet):
    """The era-routed parameter report equals ``sum(p.numel())`` of the
    reconstituted net — what ``wingspan inspect`` and the run reports show for
    this directory describes the pinned checkpoint, not the live encoder."""
    descriptor = runmeta.read_model_config(str(FIXTURE_DIR))
    report = runmeta.param_report_for(descriptor)
    assert report.total == sum(p.numel() for p in loaded_net.parameters())


def test_choice_layout_routes_to_the_frozen_registry():
    """``choice_layout_for`` on the v0.0 descriptor returns the frozen-era
    stripe table — the habitat one-hot is back, there is no kept_multihot
    stripe, and the post-embedding total is the v0.0 choice-encoder input
    width (matching the loaded net's first ``Linear``)."""
    descriptor = runmeta.read_model_config(str(FIXTURE_DIR))
    layout = runmeta.choice_layout_for(descriptor)
    names = [stripe.name for stripe in layout.stripes]
    assert "habitat" in names
    assert "kept_multihot" not in names
    expected_input = v0_0.choice_input_dim(
        descriptor.choice_dim, descriptor.architecture.card_embed_dim
    )
    assert layout.total_size == expected_input
    assert runmeta.choice_input_dim_for(descriptor) == expected_input
    assert runmeta.choice_extra_for(descriptor) == v0_0.choice_passthrough_dim(
        descriptor.choice_dim
    )
