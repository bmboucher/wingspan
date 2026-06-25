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

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import decisions, encode, engine, model, version  # noqa: E402
from wingspan.compat import v0_1, v0_2  # noqa: E402
from wingspan.encode import state_encode  # noqa: E402
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
    # state_dim must be the frozen 771, not the live 1155
    assert loaded_net.state_dim == 771
    assert loaded_net.state_dim != encode.state_size(loaded_net.spec)
    # choice_dim is the frozen pre-0.6 format (no becomes_playable stripe)
    descriptor = runmeta.read_model_config(str(FIXTURE_DIR))
    assert loaded_net.choice_dim == descriptor.choice_dim
    # The v0.6+ live row is wider by becomes_playable.
    assert loaded_net.choice_dim != encode.choice_feature_dim(loaded_net.spec)


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
    # v0.2 artifacts have a 224-wide card encoder; the live SetupNet uses
    # 225-wide (v0.7 added the or_cost flag), so the v0.6 shim is needed here.
    from wingspan.compat import v0_6

    net = v0_6.SetupNetV06.from_setup_config(descriptor)
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


def test_state_embed_offsets_are_frozen_to_v02(loaded_net: model.PolicyValueNet):
    """The v0.2 net must slice its 771-dim state vector at the frozen v0.2
    offsets, not the live 1155-dim ones.

    Three deltas apply selectively by stripe position:

    * ``card_index``, ``hand_multihot``: shifted by -24 (the v0.4
      turn_state/misc delta — 27 dims absent turn_state minus 3 from misc
      shrink = -24). These stripes sit BEFORE the v0.6 playability stripes.
    * ``decision_type``: additionally shifted by -360 for the two v0.6
      playability multi-hot stripes (180 × 2 = 360 dims). decision_type is
      derived directly as hand_multihot + HAND_MULTIHOT_DIM so the three
      stripes remain contiguous in the frozen vector.
    * ``hand_summary``: shifted only by the absent turn_state stripe width
      (not -24, since it precedes misc_scalars — the 2026-06-14 regression).

    Because slice widths coincide, slicing live would corrupt silently."""
    frozen = loaded_net._state_embed_offsets()
    assert frozen == v0_2.state_embed_offsets_v02()
    assert frozen == model.StateEmbedOffsets(
        card_index=encode.OFF_CARD_INDEX + v0_2._MISC_DIM_DELTA,
        hand_multihot=encode.OFF_HAND_MULTIHOT + v0_2._MISC_DIM_DELTA,
        # decision_type is derived from hand_multihot so the stripes are contiguous
        # in the v0.2 vector (no playability stripes between them).
        decision_type=encode.OFF_HAND_MULTIHOT
        + v0_2._MISC_DIM_DELTA
        + encode.HAND_MULTIHOT_DIM,
        hand_summary=encode.HAND_SUMMARY_OFFSET + v0_2._HAND_SUMMARY_DIM_DELTA,
    )
    # card_index → hand_multihot → decision_type must remain contiguous.
    assert frozen.card_index + encode.N_CARD_INDEX_SLOTS == frozen.hand_multihot
    assert frozen.hand_multihot + encode.HAND_MULTIHOT_DIM == frozen.decision_type
    assert loaded_net.state_dim - frozen.decision_type == (
        encode.state_size(loaded_net.spec) - encode.OFF_DECISION_TYPE
    )


def test_embed_state_reads_card_index_block_on_board_filled_state(
    loaded_net: model.PolicyValueNet,
):
    """Board-filled regression: the ``torch.zeros`` forward pass cannot catch the
    offset bug (an empty card-index block reads the same garbage either way), so
    pin behavior on a real late-game state.

    ``_embed_state`` must gather one card-table row per occupied board/tray slot
    from the card-index block at the *frozen* v0.2 offset. Slicing 19 columns
    right (the live offset) reads the wrong region — this asserts the slot
    embeddings match a direct lookup at the frozen offset."""
    # Play a game and take the most-populated recorded state (fullest board/tray).
    record = collect.play_game(
        loaded_net, torch.device(_DEVICE), random.Random(1), seed=20260610
    )
    offsets = v0_2.state_embed_offsets_v02()
    off_index = offsets.card_index
    off_decision = offsets.decision_type
    n_slots = encode.N_CARD_INDEX_SLOTS

    def board_fill(step: typing.Any) -> int:
        block = step.state[off_index : off_index + n_slots]
        return int((block > 0).sum())

    fullest = max(record.steps, key=board_fill)
    assert board_fill(fullest) > 0, "expected at least one occupied board/tray slot"

    state_vec = torch.tensor(fullest.state, dtype=torch.float32).unsqueeze(0)
    card_table = loaded_net.card_table()
    embed_dim = card_table.shape[1]
    with torch.no_grad():
        embedded = loaded_net._embed_state(state_vec, card_table)

    # _embed_state emits [continuous | slot_emb | (tray_set_emb) | hand_emb]; the
    # slot block starts right after the continuous prefix, which drops the 10-dim
    # hand summary only when a distinct hand model is active.
    cont_width = off_index + (loaded_net.state_dim - off_decision)
    if loaded_net.arch.use_distinct_hand_model:
        cont_width -= encode.HAND_SUMMARY_DIM
    slot_region = embedded[0, cont_width : cont_width + n_slots * embed_dim]

    card_idx = (
        state_vec[0, off_index : off_index + n_slots]
        .long()
        .clamp_(0, encode.HAND_MULTIHOT_DIM)
    )
    expected = card_table[card_idx].reshape(-1)
    assert torch.allclose(slot_region, expected), (
        "slot embeddings do not match the card-index block at the frozen v0.2 "
        "offset — _embed_state is slicing the state vector at the wrong columns"
    )


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


def test_v02_hand_summary_offset_lands_on_hand_summary_stripe():
    """Regression (2026-06-14): the v0.2 net must slice the 10-dim hand-summary
    stripe at the frozen column its 771-dim vector wrote it to — the live offset
    minus the 27-dim turn_state stripe added in v0.4.

    ``_embed_state`` reads ``_state_embed_offsets().hand_summary``; before the fix
    it used the live ``encode.HAND_SUMMARY_OFFSET`` constant, so a pre-0.4
    checkpoint had its hand summary read 27 columns too far right — the same class
    of silent forward-pass corruption as the card-index offset regression."""
    offsets = v0_2.state_embed_offsets_v02()
    assert (
        offsets.hand_summary
        == encode.HAND_SUMMARY_OFFSET + v0_2._HAND_SUMMARY_DIM_DELTA
    )
    assert offsets.hand_summary != encode.HAND_SUMMARY_OFFSET

    # On a real post-setup state the frozen offset must land exactly on the
    # hand-summary stripe encode_state_v02 wrote (the live offset would read a
    # different stripe entirely). Random play populates the hand along the way.
    captured: dict[str, np.ndarray] = {}

    def agent[C: decisions.Choice](
        game_engine: engine.Engine, decision: decisions.Decision[C]
    ) -> C:
        # Skip the opening SetupDecision: the main net never encodes it when
        # include_setup is off (the setup model handles the opening), and its
        # type index is out of range for the include_setup=False encoding.
        if not decisions.is_setup_decision(decision) and "vec" not in captured:
            me = game_engine.state.players[decision.player_id]
            captured["vec"] = v0_2.encode_state_v02(
                game_engine.state,
                typing.cast("decisions.Decision[decisions.Choice]", decision),
            )
            captured["hand_summary"] = state_encode._summary_hand(me)
        rng = game_engine.state.rng
        return decision.choices[rng.randrange(len(decision.choices))]

    eng, *_ = engine.Engine.create(seed=3)
    engine.Engine.play_one_game(eng.state, (agent, agent))
    assert "vec" in captured, "expected at least one non-setup decision"
    sliced = captured["vec"][
        offsets.hand_summary : offsets.hand_summary + encode.HAND_SUMMARY_DIM
    ]
    assert np.array_equal(sliced, captured["hand_summary"])


class _Approx:
    """Tolerant float comparator (pytest.approx is untyped under strict pyright)."""

    def __init__(self, expected: float) -> None:
        self.expected = expected

    def __eq__(self, other: object) -> bool:
        return isinstance(other, (int, float)) and math.isclose(
            float(other), self.expected, rel_tol=1e-6, abs_tol=1e-9
        )


def test_misc_scalars_scalar_structure():
    """The live ``_summary_misc_scalars`` output is 4 normalized scalars.

    As of v0.4 the round/cube one-hots moved to the new ``turn_state`` stripe;
    ``misc_scalars`` now carries only the 4 trailing scalars from v0.3.
    Verified against ``encode_misc_scalars_v03`` to confirm the scalars are
    consistent between the live and frozen encoders."""
    from wingspan.compat import v0_3
    from wingspan.encode import state_encode
    from wingspan.engine import core as engine_core

    eng, *_ = engine_core.Engine.create(seed=42)
    pov = 0
    me = eng.state.players[pov]
    opp = eng.state.players[1 - pov]
    vec = state_encode._summary_misc_scalars(eng.state, me, opp)

    # v0.4 misc_scalars is exactly 4 dims
    assert len(vec) == 4, f"Expected 4 dims, got {len(vec)}"

    # All values are normalized scalars in [0, ~1.5] range (not one-hots)
    assert all(
        v >= 0.0 for v in vec.tolist()
    ), f"Negative values in misc_scalars: {vec}"

    # The 4 scalars should match the trailing 4 dims of the frozen v0.3 stripe
    frozen_vec = v0_3.encode_misc_scalars_v03(eng.state, me, opp)
    assert len(frozen_vec) == 26
    # Last 4 dims of the frozen v0.3 stripe are the same scalars
    assert all(
        float(vec[i]) == _Approx(float(frozen_vec[22 + i])) for i in range(4)
    ), f"live misc={vec}, frozen v0.3 trailing 4={frozen_vec[22:]}"


def test_misc_scalars_v03_one_hot_structure():
    """The frozen v0.3 ``encode_misc_scalars_v03`` output is valid one-hot in the
    round and cube positions: exactly one 1.0 per one-hot window, rest 0.0."""
    from wingspan.compat import v0_3
    from wingspan.encode import layout
    from wingspan.engine import core as engine_core

    eng, *_ = engine_core.Engine.create(seed=42)
    pov = 0
    me = eng.state.players[pov]
    opp = eng.state.players[1 - pov]
    vec = v0_3.encode_misc_scalars_v03(eng.state, me, opp)

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

    # Total vector length must be 26 (v0.3 frozen dim)
    assert len(vec) == 26, f"Expected 26 dims, got {len(vec)}"

    # Verify the one-hot position matches the actual game state value
    assert vec[eng.state.round_idx] == 1.0
    assert vec[cube_me_start + me.action_cubes_left] == 1.0
    assert vec[cube_opp_start + opp.action_cubes_left] == 1.0


def test_choice_layout_routes_to_the_live_registry():
    """``choice_layout_for`` on a 0.2 descriptor uses the live stripe table
    (no habitat stripe) with the pre-0.6 choice-encoder input width (no
    becomes_playable embedding)."""
    descriptor = runmeta.read_model_config(str(FIXTURE_DIR))
    layout = runmeta.choice_layout_for(descriptor)
    names = [stripe.name for stripe in layout.stripes]
    assert "habitat" not in names
    # The pre-0.6 encoder width omits the becomes_playable embedding.
    expected_input = encode.choice_input_dim(
        descriptor.choice_dim,
        descriptor.architecture.card_embed_dim,
        include_setup=descriptor.include_setup,
        has_becomes_playable=False,
    )
    assert runmeta.choice_input_dim_for(descriptor) == expected_input
    assert runmeta.choice_extra_for(descriptor) == encode.choice_passthrough_dim(
        descriptor.choice_dim,
        include_setup=descriptor.include_setup,
        has_becomes_playable=False,
    )
