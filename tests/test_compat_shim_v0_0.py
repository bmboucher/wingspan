# pyright: reportPrivateUsage=false
# (asserts against the shim's package-private frozen offsets and the live
# layout's stripe constants, and calls the net's private ``_embed_choices``)
"""Unit tests for the ``wingspan.compat.v0_0`` choice-encoding shim.

The end-to-end proof that a real pre-0.1 checkpoint loads and plays lives in
``test_compat_v0_0.py`` (the pinned fixture); these tests pin the shim's
geometry directly: the frozen dims match the fixture era, the live→v0.0
transform regenerates the reshaped placement / card-identity stripes and
copies every shared stripe verbatim, and ``PolicyValueNetV00`` builds and
embeds in the frozen geometry.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import (
    architecture,
    cards,
    decisions,
    encode,
    engine,
    model,
    state,
    version,
)
from wingspan.compat import v0_0
from wingspan.encode import layout
from wingspan.training import runmeta

_SMALL = architecture.ModelArchitecture(
    trunk_layers=(8, 8),
    choice_layers=(8, 8),
    head_layers=(),
    value_layers=(),
    card_embed_dim=4,
)

# The v0.0 fixture's published shape (tests/data/compat/v0.0/model_config.json):
# choice_dim 397 at card_embed_dim 64, whose choice encoder reads 1226 inputs.
_FIXTURE_CHOICE_DIM = 397
_FIXTURE_EMBED_DIM = 64
_FIXTURE_CHOICE_IN = 1226


def _habitat_one_hot(row: np.ndarray) -> dict[int, float]:
    """The nonzero entries of the v0.0 habitat stripe, keyed by habitat index."""
    stripe = row[v0_0._OFF_HAB : v0_0._OFF_HAB + v0_0._HABITAT_DIM]
    return {index: float(value) for index, value in enumerate(stripe) if value != 0.0}


def _bird_one_hot_bits(row: np.ndarray) -> set[int]:
    """The set bit positions of the v0.0 bird-identity stripe."""
    stripe = row[v0_0._OFF_BIRD_ID : v0_0._OFF_BIRD_ID + v0_0._BIRD_ID_DIM]
    return {index for index, value in enumerate(stripe) if value != 0.0}


def _board_idx_block(row: np.ndarray) -> np.ndarray:
    return row[v0_0._OFF_BOARD_IDX : v0_0._OFF_BOARD_IDX + v0_0._BOARD_IDX_SLOTS]


def _assert_shared_stripes_match(v00_row: np.ndarray, live_row: np.ndarray) -> None:
    """Every stripe both eras share must copy verbatim: kind + gain_food, the
    pay_food..exchange run, and the bonus_id..bonus_value tail."""
    assert np.array_equal(
        v00_row[v0_0._OFF_KIND : v0_0._OFF_HAB],
        live_row[layout._OFF_KIND : layout._OFF_PAY],
    )
    assert np.array_equal(
        v00_row[v0_0._OFF_PAY : v0_0._OFF_BOARD_IDX],
        live_row[layout._OFF_PAY : layout._OFF_BOARD_IDX],
    )
    # Stop before layout.CHOICE_BECOMES_PLAYABLE_OFFSET: the becomes_playable
    # stripe was added in v0.6 after bonus_value; it is not present in v0.0 rows.
    assert np.array_equal(
        v00_row[v0_0._OFF_BONUS_ID : v0_0._OFF_SETUP],
        live_row[layout._OFF_BONUS_ID : layout.CHOICE_BECOMES_PLAYABLE_OFFSET],
    )


# ---------------------------------------------------------------------------
# Frozen dims and the version predicate


def test_frozen_dims_match_the_pinned_fixture_era():
    assert v0_0.choice_feature_dim() == _FIXTURE_CHOICE_DIM
    assert (
        v0_0.choice_feature_dim(encode.EncodingSpec(include_setup=True))
        == _FIXTURE_CHOICE_DIM + v0_0._SETUP_DIM
    )
    assert (
        v0_0.choice_input_dim(_FIXTURE_CHOICE_DIM, _FIXTURE_EMBED_DIM)
        == _FIXTURE_CHOICE_IN
    )


def test_version_predicate_selects_only_pre_0_1():
    assert v0_0.uses_v0_0_choice_encoding("0.0")
    assert not v0_0.uses_v0_0_choice_encoding(v0_0.CHOICE_ENCODING_CHANGED_IN)
    assert not v0_0.uses_v0_0_choice_encoding("0.2")
    assert not v0_0.uses_v0_0_choice_encoding("1.0")


# ---------------------------------------------------------------------------
# The live -> v0.0 row transform


def test_play_bird_rows_regenerate_the_v0_0_placement_stripes():
    """A play-bird row carries the v0.0 habitat one-hot and bird one-hot, with
    the board-index block bare (the landing-slot mark is the 0.1 change being
    undone); every shared stripe copies verbatim from the live row."""
    eng, birds, *_ = engine.Engine.create(seed=3)
    bird = next(candidate for candidate in birds if len(candidate.habitats) >= 2)
    decision = decisions.PlayBirdDecision(
        player_id=0,
        prompt="play",
        choices=[
            decisions.PlayBirdChoice(
                label="first", bird=bird, habitat=bird.habitats[0]
            ),
            decisions.PlayBirdChoice(
                label="second", bird=bird, habitat=bird.habitats[1]
            ),
        ],
    )
    rows = v0_0.encode_choices(decision, eng.state)
    live_rows = encode.encode_choices(decision, eng.state)

    assert rows.shape == (2, _FIXTURE_CHOICE_DIM)
    habitat_indices = list(cards.ALL_HABITATS)
    for row, live_row, choice in zip(rows, live_rows, decision.choices):
        assert _habitat_one_hot(row) == {habitat_indices.index(choice.habitat): 1.0}
        assert not _board_idx_block(row).any()
        assert _bird_one_hot_bits(row) == {cards.bird_index(bird)}
        _assert_shared_stripes_match(row, live_row)


def test_payment_rows_regenerate_the_committed_play_context():
    """Payment rows carry the committed bird's one-hot plus the destination
    habitat from the decision context, board-index block bare."""
    eng, birds, *_ = engine.Engine.create(seed=5)
    bird = birds[0]
    payment = state.FoodPool()
    payment[cards.Food.SEED] = 1
    decision = decisions.PayBirdFoodDecision(
        player_id=0,
        prompt="pay",
        choices=[decisions.FoodPaymentChoice(label="seed", payment=payment)],
        bird=bird,
        habitat=bird.habitats[0],
    )
    row = v0_0.encode_choices(decision, eng.state)[0]
    live_row = encode.encode_choices(decision, eng.state)[0]

    habitat_indices = list(cards.ALL_HABITATS)
    assert _habitat_one_hot(row) == {habitat_indices.index(bird.habitats[0]): 1.0}
    assert not _board_idx_block(row).any()
    assert _bird_one_hot_bits(row) == {cards.bird_index(bird)}
    _assert_shared_stripes_match(row, live_row)


def test_move_rows_regenerate_destination_habitats():
    """Move-bird destination rows (including the stay row) carry the v0.0
    habitat one-hot and the moving bird's one-hot, board-index block bare."""
    eng, birds, *_ = engine.Engine.create(seed=7)
    mover = state.PlayedBird(bird=birds[0])
    eng.state.players[0].board[cards.Habitat.FOREST].append(mover)
    decision = decisions.BirdPowerPickHabitatDecision(
        player_id=0,
        prompt="move",
        choices=[
            decisions.HabitatChoice(label="stay", habitat=cards.Habitat.FOREST),
            decisions.HabitatChoice(label="go", habitat=cards.Habitat.GRASSLAND),
        ],
        moving_bird=mover,
        from_habitat=cards.Habitat.FOREST,
    )
    rows = v0_0.encode_choices(decision, eng.state)
    live_rows = encode.encode_choices(decision, eng.state)

    habitat_indices = list(cards.ALL_HABITATS)
    for row, live_row, choice in zip(rows, live_rows, decision.choices):
        assert _habitat_one_hot(row) == {habitat_indices.index(choice.habitat): 1.0}
        assert not _board_idx_block(row).any()
        assert _bird_one_hot_bits(row) == {cards.bird_index(birds[0])}
        _assert_shared_stripes_match(row, live_row)


def test_board_target_rows_keep_their_occupancy_indices():
    """Board-target rows wrote the board-index occupancy in both eras, so the
    transform copies it (only placement rows clear the block)."""
    eng, birds, *_ = engine.Engine.create(seed=9)
    eng.state.players[0].board[cards.Habitat.FOREST].append(
        state.PlayedBird(bird=birds[0])
    )
    decision = decisions.LayEggDecision(
        player_id=0,
        prompt="lay",
        choices=[
            decisions.BoardTargetChoice(
                label="forest 0", habitat=cards.Habitat.FOREST, slot=0
            )
        ],
    )
    row = v0_0.encode_choices(decision, eng.state)[0]
    live_row = encode.encode_choices(decision, eng.state)[0]

    forest_slot = list(cards.ALL_HABITATS).index(cards.Habitat.FOREST) * state.ROW_SLOTS
    assert _board_idx_block(row)[forest_slot] == cards.bird_index(birds[0]) + 1
    live_block = live_row[layout._OFF_BOARD_IDX : layout._OFF_BIRD_ID]
    assert np.array_equal(_board_idx_block(row), live_block)
    assert _habitat_one_hot(row) == {}
    _assert_shared_stripes_match(row, live_row)


def test_setup_rows_map_the_keep_back_onto_the_bird_stripe():
    """An include_setup row is 401 wide: the kept multi-hot lands back on the
    v0.0 bird-identity stripe and the setup_agg stripe copies verbatim."""
    eng, birds, *_ = engine.Engine.create(seed=13)
    spec = encode.EncodingSpec(include_setup=True)
    kept = (birds[0], birds[2])
    decision = decisions.SetupDecision(
        player_id=0,
        prompt="keep",
        choices=[
            decisions.SetupChoice(
                label="keep two",
                kept_cards=kept,
                kept_foods=tuple(cards.ALL_FOODS[:3]),
                bonus_card=None,
            )
        ],
        dealt_cards=list(birds[:5]),
        dealt_bonus=[],
    )
    row = v0_0.encode_choices(decision, eng.state, spec)[0]
    live_row = encode.encode_choices(decision, eng.state, spec)[0]

    assert row.shape == (v0_0.choice_feature_dim(spec),)
    assert _bird_one_hot_bits(row) == {cards.bird_index(bird) for bird in kept}
    assert np.array_equal(
        row[v0_0._OFF_SETUP : v0_0._OFF_SETUP + v0_0._SETUP_DIM],
        live_row[layout._OFF_SETUP : layout._OFF_SETUP + layout._SETUP_DIM],
    )
    _assert_shared_stripes_match(row, live_row)


# ---------------------------------------------------------------------------
# The frozen-era net


def test_v00_net_builds_and_encodes_in_the_frozen_geometry():
    """``PolicyValueNetV00`` defaults to the frozen row width and its
    ``encode_choices`` runs the transform, never the live encoder."""
    net = v0_0.PolicyValueNetV00(arch=_SMALL)
    assert net.choice_dim == _FIXTURE_CHOICE_DIM

    eng, birds, *_ = engine.Engine.create(seed=3)
    decision = decisions.PlayBirdDecision(
        player_id=0,
        prompt="play",
        choices=[
            decisions.PlayBirdChoice(
                label="play", bird=birds[0], habitat=birds[0].habitats[0]
            )
        ],
    )
    feats = net.encode_choices(decision, eng.state)
    assert feats.shape == (1, _FIXTURE_CHOICE_DIM)
    assert np.array_equal(feats, v0_0.encode_choices(decision, eng.state))


def test_v00_embedding_uses_the_frozen_card_regions():
    """The frozen ``_embed_choices``: a bird one-hot embeds to that card's
    table row, a multi-hot to the sum of its cards' rows, at the v0.0 offsets."""
    net = v0_0.PolicyValueNetV00(arch=_SMALL)
    net.eval()
    card_table = net.card_table()
    embed_dim = _SMALL.card_embed_dim

    choices = torch.zeros(1, 2, net.choice_dim)
    choices[0, 0, v0_0._OFF_BIRD_ID + 7] = 1.0
    for kept_index in (0, 5, 17):
        choices[0, 1, v0_0._OFF_BIRD_ID + kept_index] = 1.0
    embedded = net._embed_choices(choices, card_table)

    assert embedded.shape[-1] == v0_0.choice_input_dim(net.choice_dim, embed_dim)
    rest_width = net.choice_dim - v0_0._BIRD_ID_DIM - v0_0._BOARD_IDX_SLOTS
    candidate_slice = embedded[..., rest_width : rest_width + embed_dim]
    assert torch.allclose(candidate_slice[0, 0], card_table[8])
    expected_sum = torch.stack(
        [card_table[kept_index + 1] for kept_index in (0, 5, 17)]
    ).sum(dim=0)
    assert torch.allclose(candidate_slice[0, 1], expected_sum)


def test_from_model_config_routes_by_artifact_version():
    """A pre-0.1 descriptor (including the version-less default) reconstructs
    as the frozen-era subclass; a current-version one as the live net."""
    old = runmeta.ModelConfig(
        run_name="routing-old",
        state_dim=encode.state_size(),
        choice_dim=_FIXTURE_CHOICE_DIM,
        family_order=("main_action",),
        architecture=_SMALL,
        include_setup=False,
    )
    assert old.version == version.PRE_VERSIONING_VERSION
    old_net = model.PolicyValueNet.from_model_config(old)
    assert isinstance(old_net, v0_0.PolicyValueNetV00)

    current = runmeta.ModelConfig(
        run_name="routing-current",
        state_dim=encode.state_size(),
        choice_dim=encode.choice_feature_dim(),
        family_order=("main_action",),
        architecture=_SMALL,
        include_setup=False,
        version=version.MODEL_VERSION,
    )
    current_net = model.PolicyValueNet.from_model_config(current)
    assert not isinstance(current_net, v0_0.PolicyValueNetV00)
