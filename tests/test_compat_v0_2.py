# pyright: reportPrivateUsage=false
# (tests call the private _summary_misc_scalars to check one-hot structure;
#  matches the convention in test_compat_shim_v0_0.py and state_encode.py)
"""Backwards-compatibility smoke tests for the pinned v0.2 artifacts.

Loads the real run snapshot committed under ``tests/data/compat/v0.2/`` (see
its README for provenance; the checkpoints are gzip-compressed and LFS-tracked)
through the production loaders and proves the artifact-version contract:
same-MAJOR artifacts must load and play games. These files carry an explicit
``version: "0.2"`` stamp; since artifact version 0.3 replaced scalar round/cube
encoding with one-hot vectors (state vector 771 → 790 dims), v0.2 nets now
reconstruct as ``compat.v0_2.PolicyValueNetV02`` (frozen 7-scalar misc stripe)
— not as the live era's net. Card and choice encoding are unchanged between 0.2
and 0.3, so game play still uses the live encoders for those paths.

Heavy (a ~12 MB checkpoint load plus a full self-play game), so the nets are
loaded once per module and the file is front-loaded via ``_HEAVY_TEST_FILES``.
"""

from __future__ import annotations

import gzip
import io
import math
import os
import pathlib
import random
import sys
import typing

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import encode, model, version  # noqa: E402
from wingspan.compat import v0_1, v0_2  # noqa: E402
from wingspan.training import collect, runmeta, setup_runmeta  # noqa: E402

FIXTURE_DIR = pathlib.Path(__file__).parent / "data" / "compat" / "v0.2"

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
    assert descriptor.version == "0.2"
    # The pinned descriptor must stay compatible until a deliberate MAJOR bump.
    version.check_artifact_compatible(descriptor.version, what="v0.2 fixture")


def test_setup_config_carries_the_explicit_version():
    descriptor = setup_runmeta.read_setup_config(str(FIXTURE_DIR))
    assert descriptor.version == "0.2"
    version.check_artifact_compatible(descriptor.version, what="v0.2 fixture")


def test_v0_2_net_uses_compat_shim(loaded_net: model.PolicyValueNet):
    """A 0.2 descriptor reconstructs as PolicyValueNetV02 (frozen 7-scalar misc
    stripe) — not the live net (which has a 26-dim one-hot misc stripe since 0.3).
    The v0.2 net's state_dim is the old 771, not the live 790. Card and choice
    encoding are unchanged between eras so choice_dim matches the live encoder."""
    assert not isinstance(loaded_net, v0_1.PolicyValueNetV01)
    assert isinstance(loaded_net, v0_2.PolicyValueNetV02)
    # state_dim must be the frozen 771, not the live 790
    assert loaded_net.state_dim == 771
    assert loaded_net.state_dim != encode.state_size(loaded_net.spec)
    # choice_dim is unchanged: still matches the live encoder
    assert loaded_net.choice_dim == encode.choice_feature_dim(loaded_net.spec)


def test_policy_net_loads_state_dict(
    loaded_net: model.PolicyValueNet, main_payload: dict[str, typing.Any]
):
    """The pinned weights drop into the descriptor-reconstructed net exactly,
    and the payload's explicit version stamp passes the check."""
    assert main_payload["version"] == "0.2"
    version.check_artifact_compatible(
        str(main_payload["version"]), what="v0.2 fixture last.pt"
    )
    # Strict mode (the default) raises on any missing or unexpected key, so a
    # clean load *is* the exact-key-match assertion.
    state_dict = typing.cast("dict[str, torch.Tensor]", main_payload["model"])
    loaded_net.load_state_dict(state_dict)


def test_setup_net_loads_state_dict(setup_payload: dict[str, typing.Any]):
    descriptor = setup_runmeta.read_setup_config(str(FIXTURE_DIR))
    # v0.2 artifacts have a 224-wide card encoder (same as current) — no card-
    # feature shim needed; only the state encoding changed in 0.3.
    from wingspan.training import setup_net as setup_net_module

    net = setup_net_module.SetupNet.from_setup_config(descriptor)
    net.load_state_dict(
        typing.cast("dict[str, torch.Tensor]", setup_payload["setup_model"])
    )
    net.eval()
    assert setup_payload["version"] == "0.2"


def test_forward_pass(loaded_net: model.PolicyValueNet):
    """A batch of freshly-v0.2-encoded-shape inputs flows through the loaded
    weights to finite logits and value — the frozen encode_state produces the
    771-dim vector the trunk expects."""
    batch_size, n_choices = 2, 4
    state_vec = torch.zeros(batch_size, loaded_net.state_dim)  # 771
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
    """The v0.2 net drives a full self-play game through the production
    collector — the load-and-play guarantee end to end."""
    record = collect.play_game(
        loaded_net, torch.device(_DEVICE), random.Random(0), seed=20260609
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


def test_state_layout_routes_to_the_v02_registry():
    """``state_layout_for`` on a 0.2 descriptor returns the frozen 7-dim misc-
    scalar stripe (not the live 26-dim one-hot version), and the total matches
    the descriptor's 771 state_dim (pre-embedding)."""
    descriptor = runmeta.read_model_config(str(FIXTURE_DIR))
    layout = runmeta.state_layout_for(descriptor)
    # Find the misc_scalars stripe and confirm it has the frozen 7-dim size.
    misc_stripe = next(
        (stripe for stripe in layout.stripes if stripe.name == "misc_scalars"), None
    )
    assert misc_stripe is not None
    assert (
        misc_stripe.size == 7
    ), f"Expected 7-dim scalar misc stripe for v0.2, got {misc_stripe.size}"
    # The frozen misc stripe must not have the one-hot sub-fields.
    sub_field_names = [sub.name for sub in misc_stripe.sub_fields]
    assert "round_index" in sub_field_names
    assert "my_action_cubes" in sub_field_names
    assert sub_field_names[0] == "round_index"  # first sub-field is round scalar


class _Approx:
    """Tolerant float comparator (pytest.approx is untyped under strict pyright)."""

    def __init__(self, expected: float) -> None:
        self.expected = expected

    def __eq__(self, other: object) -> bool:
        return isinstance(other, (int, float)) and math.isclose(
            float(other), self.expected, rel_tol=1e-6, abs_tol=1e-9
        )


def test_misc_scalars_one_hot_structure():
    """The live ``_summary_misc_scalars`` output is valid one-hot in the round
    and cube positions: exactly one 1.0 per one-hot window, rest 0.0."""
    from wingspan.encode import layout, state_encode
    from wingspan.engine import core as engine_core

    eng, *_ = engine_core.Engine.create(seed=42)
    pov = 0
    me = eng.state.players[pov]
    opp = eng.state.players[1 - pov]
    vec = state_encode._summary_misc_scalars(eng.state, me, opp)

    # Round one-hot: exactly one 1.0 in dims [0..N_ROUNDS-1]
    round_hot = vec[: layout.N_ROUNDS]
    assert float(round_hot.sum()) == _Approx(
        1.0
    ), f"round one-hot sum={round_hot.sum()}"
    assert set(round_hot.tolist()) == {0.0, 1.0}, f"round one-hot values: {round_hot}"

    # Cube-me one-hot: exactly one 1.0 in dims [N_ROUNDS..N_ROUNDS+MAX_CUBES]
    cube_me_start = layout.N_ROUNDS
    cube_me_end = layout.N_ROUNDS + layout.MAX_ACTION_CUBES + 1
    cube_me_hot = vec[cube_me_start:cube_me_end]
    assert float(cube_me_hot.sum()) == _Approx(
        1.0
    ), f"cube-me one-hot sum={cube_me_hot.sum()}"
    assert set(cube_me_hot.tolist()) == {0.0, 1.0}, f"cube-me one-hot: {cube_me_hot}"

    # Cube-opp one-hot
    cube_opp_start = cube_me_end
    cube_opp_end = cube_opp_start + layout.MAX_ACTION_CUBES + 1
    cube_opp_hot = vec[cube_opp_start:cube_opp_end]
    assert float(cube_opp_hot.sum()) == _Approx(
        1.0
    ), f"cube-opp one-hot sum={cube_opp_hot.sum()}"
    assert set(cube_opp_hot.tolist()) == {0.0, 1.0}, f"cube-opp one-hot: {cube_opp_hot}"

    # Total vector length must be 26
    assert len(vec) == 26, f"Expected 26 dims, got {len(vec)}"

    # Verify the one-hot position matches the actual game state value
    assert vec[eng.state.round_idx] == 1.0
    assert vec[cube_me_start + me.action_cubes_left] == 1.0
    assert vec[cube_opp_start + opp.action_cubes_left] == 1.0


def test_choice_layout_routes_to_the_live_registry():
    """``choice_layout_for`` on a 0.2 descriptor uses the live stripe table
    (no habitat stripe) at the live choice-encoder input width."""
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
