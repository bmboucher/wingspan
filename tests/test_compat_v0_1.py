"""Backwards-compatibility smoke tests for the pinned v0.1 artifacts.

Loads the real run snapshot committed under ``tests/data/compat/v0.1/`` (see
its README for provenance; the checkpoints are gzip-compressed and LFS-tracked)
through the production loaders and proves the artifact-version contract:
same-MAJOR artifacts must load and play games. These files carry an explicit
``version: "0.1"`` stamp; since artifact version 0.2 reshaped the card feature
vector (CARD_FEATURE_DIM 229 → 224), v0.1 nets now reconstruct as
``compat.v0_1.PolicyValueNetV01`` (frozen 229-wide card encoder) — not as the
live era's net. Choice encoding and state encoding are unchanged between 0.1 and
0.2, so game play still uses the live encoders for those paths.

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
from wingspan.compat import v0_0, v0_1  # noqa: E402
from wingspan.training import collect, runmeta, setup_runmeta  # noqa: E402

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


def test_v0_1_net_uses_compat_shim(loaded_net: model.PolicyValueNet):
    """A 0.1 descriptor reconstructs as PolicyValueNetV01 (frozen 229-wide card
    encoder) — not the live net (which has a 225-wide card encoder since 0.7).
    The frozen choice_dim from the descriptor is preserved (v0.6 added the
    becomes_playable stripe to the live choice row, but v0.1 predates it)."""
    assert not isinstance(loaded_net, v0_0.PolicyValueNetV00)
    assert isinstance(loaded_net, v0_1.PolicyValueNetV01)
    descriptor = runmeta.read_model_config(str(FIXTURE_DIR))
    assert loaded_net.choice_dim == descriptor.choice_dim
    # The v0.6+ live row is wider by becomes_playable.
    assert loaded_net.choice_dim != encode.choice_feature_dim(loaded_net.spec)


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
    # v0.1 artifacts have a 229-wide card encoder (CARD_FEATURE_DIM changed in
    # 0.2), so reconstruct via the frozen shim.
    net = v0_1.SetupNetV01.from_setup_config(descriptor)
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
    no resurrected habitat stripe — with a choice-encoder input width matching
    the v0.8 frozen formula (board_idx embedded, no becomes_playable).

    v0.1 artifacts predate both the v0.6 becomes_playable stripe and the v0.9
    board simplification. ``choice_input_dim_for`` routes through
    ``v0_8.choice_input_dim_v08(has_becomes_playable=False)`` which includes the
    15-slot board-index embedding that the live formula dropped in v0.9."""
    from wingspan.compat import v0_8

    descriptor = runmeta.read_model_config(str(FIXTURE_DIR))
    layout = runmeta.choice_layout_for(descriptor)
    names = [stripe.name for stripe in layout.stripes]
    assert "habitat" not in names
    # The pre-0.9 encoder includes the board-index embedding (15 slots).
    expected_input = v0_8.choice_input_dim_v08(
        descriptor.choice_dim,
        descriptor.architecture.card_embed_dim,
        include_setup=descriptor.include_setup,
        has_becomes_playable=False,
    )
    assert runmeta.choice_input_dim_for(descriptor) == expected_input
    assert runmeta.choice_extra_for(descriptor) == v0_8.choice_passthrough_dim_v08(
        descriptor.choice_dim,
        include_setup=descriptor.include_setup,
        has_becomes_playable=False,
    )
