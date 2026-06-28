# pyright: reportPrivateUsage=false
# (slices choice rows by the layout's package-private stripe constants, and
# calls the model's private ``_embed_choices`` to pin its embedding contract)
"""Tests for the landing-slot choice encoding and the candidate index column.

Placement rows (play-bird, its food payment, move-bird destinations) carry the
bird's exact resulting location as a habitat one-hot (``board_hab``) and a
column one-hot (``board_col``); the candidate ``bird_id`` stripe is a single
integer index column the model looks up (masked to zero when no bird); and a
setup pick's kept set rides the trailing ``kept_multihot`` stripe, summed
through the shared card table.
"""

from __future__ import annotations

import numpy as np
import torch

from wingspan import architecture, cards, decisions, encode, engine, model, state
from wingspan.encode import layout

_SMALL = architecture.ModelArchitecture(
    trunk_layers=(8, 8),
    choice_layers=(8, 8),
    head_layers=(),
    value_layers=(),
    card_embed_dim=4,
)


def _get_board_hab(row: np.ndarray) -> int | None:
    """The marked habitat index (0-2) in the ``board_hab`` one-hot, or None."""
    block = row[
        layout.CHOICE_BOARD_HAB_OFFSET : layout.CHOICE_BOARD_HAB_OFFSET
        + layout.CHOICE_BOARD_HAB_DIM
    ]
    for index, value in enumerate(block):
        if value != 0.0:
            return index
    return None


def _get_board_col(row: np.ndarray) -> int | None:
    """The marked column index (0-4) in the ``board_col`` one-hot, or None."""
    block = row[
        layout.CHOICE_BOARD_COL_OFFSET : layout.CHOICE_BOARD_COL_OFFSET
        + layout.CHOICE_BOARD_COL_DIM
    ]
    for index, value in enumerate(block):
        if value != 0.0:
            return index
    return None


def _board_hab_index(habitat: cards.Habitat) -> int:
    return list(cards.ALL_HABITATS).index(habitat)


def _zero_board_location(row: np.ndarray) -> np.ndarray:
    """Return a copy of ``row`` with ``board_hab`` and ``board_col`` zeroed."""
    cleared = row.copy()
    cleared[
        layout.CHOICE_BOARD_HAB_OFFSET : layout.CHOICE_BOARD_HAB_OFFSET
        + layout.CHOICE_BOARD_HAB_DIM
    ] = 0
    cleared[
        layout.CHOICE_BOARD_COL_OFFSET : layout.CHOICE_BOARD_COL_OFFSET
        + layout.CHOICE_BOARD_COL_DIM
    ] = 0
    return cleared


# ---------------------------------------------------------------------------
# Featurizer side: the landing slot and the index column


def test_play_bird_rows_mark_the_landing_slot():
    """A bird playable in two habitats produces rows that differ exactly at the
    landing slot — board_hab + board_col pointing at the destination."""
    eng, birds, *_ = engine.Engine.create(seed=3)
    bird = next(candidate for candidate in birds if len(candidate.habitats) >= 2)
    first_habitat, second_habitat = bird.habitats[0], bird.habitats[1]
    # Occupy one slot of the first habitat so the two landing columns differ.
    eng.state.players[0].board[first_habitat].append(state.PlayedBird(bird=birds[0]))

    decision = decisions.PlayBirdDecision(
        player_id=0,
        prompt="play",
        choices=[
            decisions.PlayBirdChoice(label="first", bird=bird, habitat=first_habitat),
            decisions.PlayBirdChoice(label="second", bird=bird, habitat=second_habitat),
        ],
    )
    first_row, second_row = encode.encode_choices(decision, eng.state)

    bird_column = float(cards.bird_index(bird) + 1)
    assert first_row[layout._OFF_BIRD_ID] == bird_column
    assert second_row[layout._OFF_BIRD_ID] == bird_column

    # First habitat: column 1 (slot occupied by birds[0]).
    assert _get_board_hab(first_row) == _board_hab_index(first_habitat)
    assert _get_board_col(first_row) == 1
    # Second habitat: column 0 (empty).
    assert _get_board_hab(second_row) == _board_hab_index(second_habitat)
    assert _get_board_col(second_row) == 0

    # The landing location (board_hab + board_col) is the rows' only difference.
    assert np.array_equal(
        _zero_board_location(first_row), _zero_board_location(second_row)
    )


def test_payment_rows_carry_bird_and_landing_slot():
    """Every payment row shares the committed play as context: the bird's index
    column plus its landing slot (the payment is asked before placement)."""
    eng, birds, *_ = engine.Engine.create(seed=5)
    bird = birds[0]
    habitat = bird.habitats[0]
    payment_a = state.FoodPool()
    payment_a[cards.Food.SEED] = 1
    payment_b = state.FoodPool()
    payment_b[cards.Food.FRUIT] = 1
    decision = decisions.PayBirdFoodDecision(
        player_id=0,
        prompt="pay",
        choices=[
            decisions.FoodPaymentChoice(label="seed", payment=payment_a),
            decisions.FoodPaymentChoice(label="fruit", payment=payment_b),
        ],
        bird=bird,
        habitat=habitat,
    )
    rows = encode.encode_choices(decision, eng.state)

    bird_column = float(cards.bird_index(bird) + 1)
    expected_hab = _board_hab_index(habitat)
    for row in rows:
        assert row[layout._OFF_BIRD_ID] == bird_column
        assert _get_board_hab(row) == expected_hab
        assert _get_board_col(row) == 0  # first slot (board is empty)


def test_setup_rows_use_the_kept_multihot_stripe():
    """A setup pick's kept set is a multi-hot on the trailing kept_multihot
    stripe; the single-candidate bird_id column stays zero (a keep is a set,
    not one bird)."""
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
    row = encode.encode_choices(decision, eng.state, spec)[0]

    assert row.shape == (encode.choice_feature_dim(spec),)
    assert row[layout._OFF_BIRD_ID] == 0.0
    kept_block = row[layout._OFF_KEPT_MULTIHOT :]
    expected_bits = {cards.bird_index(bird) for bird in kept}
    assert {i for i, bit in enumerate(kept_block) if bit != 0.0} == expected_bits


# ---------------------------------------------------------------------------
# Model side: the embedding contract for the new card regions


def test_model_candidate_embedding_is_a_masked_lookup():
    """The candidate index column embeds to that card's table row, and index 0
    (no bird) embeds to an exact zero vector."""
    net = model.PolicyValueNet(arch=_SMALL)
    net.eval()
    card_table = net.card_table()
    embed_dim = _SMALL.card_embed_dim

    choices = torch.zeros(1, 2, net.choice_dim)
    bird_index = 7  # any in-range card index
    choices[0, 1, encode.CHOICE_BIRD_ID_OFFSET] = float(bird_index + 1)
    embedded = net._embed_choices(choices, card_table)

    assert embedded.shape[-1] == encode.choice_input_dim(
        net.choice_dim, embed_dim, pooled_hand_width=net.arch.pooled_hand_width
    )
    # ``_embed_choices`` concatenates [rest, cand_emb, becomes_emb].
    # ``rest`` is the passthrough slice (everything except bird-id and
    # becomes_playable columns), so ``cand_emb`` starts at passthrough width.
    rest_width = encode.choice_passthrough_dim(net.choice_dim)
    candidate_slice = embedded[..., rest_width : rest_width + embed_dim]
    assert torch.all(candidate_slice[0, 0] == 0.0)
    assert torch.allclose(candidate_slice[0, 1], card_table[bird_index + 1])


def test_model_kept_set_embedding_sums_card_vectors():
    """The trailing kept multi-hot embeds to the sum of the kept cards' table
    rows — the same vector the old bird_id multi-hot matmul produced."""
    spec = encode.EncodingSpec(include_setup=True)
    net = model.PolicyValueNet(spec=spec, arch=_SMALL)
    net.eval()
    card_table = net.card_table()
    embed_dim = _SMALL.card_embed_dim

    kept_indices = (0, 5, 17)
    choices = torch.zeros(1, 1, net.choice_dim)
    for kept_index in kept_indices:
        choices[0, 0, encode.CHOICE_KEPT_MULTIHOT_OFFSET + kept_index] = 1.0
    embedded = net._embed_choices(choices, card_table)

    assert embedded.shape[-1] == encode.choice_input_dim(
        net.choice_dim,
        embed_dim,
        include_setup=True,
        pooled_hand_width=net.arch.pooled_hand_width,
    )
    kept_slice = embedded[..., -embed_dim:]
    expected = torch.stack(
        [card_table[kept_index + 1] for kept_index in kept_indices]
    ).sum(dim=0)
    assert torch.allclose(kept_slice[0, 0], expected)
