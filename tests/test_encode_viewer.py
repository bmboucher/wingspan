"""Tests for the encoding-viewer stripe extractor.

Covers :func:`extract_state_stripes` and :func:`extract_choice_stripes` in
:mod:`wingspan.reporting.encode_viewer`.  Tests use real encoder output
(not hand-crafted vectors) wherever possible so that layout-change regressions
are caught here rather than silently at runtime.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import cards, decisions, encode, engine, state
from wingspan.reporting import encode_viewer, game_log_html

# ---------------------------------------------------------------------------
# Helpers


def _make_engine(seed: int = 1) -> tuple[engine.Engine, list[cards.Bird]]:
    """Return (engine, all_birds) for a fresh two-player game."""
    eng, birds, *_ = engine.Engine.create(seed=seed)
    return eng, birds


def _main_action_decision(player_id: int = 0) -> decisions.MainActionDecision:
    return decisions.MainActionDecision(
        player_id=player_id,
        prompt="",
        choices=[
            decisions.MainActionChoice(
                label="lay_eggs", action=decisions.MainAction.LAY_EGGS
            )
        ],
    )


# ---------------------------------------------------------------------------
# extract_state_stripes — smoke tests


def test_extract_state_stripes_nonempty_for_real_state():
    """A real game-state vector produces at least one non-zero EncodedStripe."""
    eng, _ = _make_engine(seed=2)
    vec = encode.encode_state(eng.state).tolist()
    result = encode_viewer.extract_state_stripes(vec, include_setup=False)
    assert isinstance(result, list)
    assert len(result) > 0


def test_extract_state_stripes_zero_vector_empty():
    """A fully zero vector produces an empty stripe list."""
    vec = [0.0] * encode.state_size()
    result = encode_viewer.extract_state_stripes(vec, include_setup=False)
    assert result == []


def test_extract_state_stripes_structure():
    """Each returned stripe has a name, description, and at least one sub_field."""
    eng, _ = _make_engine(seed=4)
    vec = encode.encode_state(eng.state).tolist()
    result = encode_viewer.extract_state_stripes(vec, include_setup=False)
    for stripe in result:
        assert isinstance(stripe, game_log_html.EncodedStripe)
        assert stripe.name
        assert stripe.description
        assert stripe.sub_fields


def test_extract_state_stripes_sub_field_structure():
    """Every EncodedSubField has name, description, encoding, value_range."""
    eng, _ = _make_engine(seed=5)
    vec = encode.encode_state(eng.state).tolist()
    result = encode_viewer.extract_state_stripes(vec, include_setup=False)
    for stripe in result:
        for sub in stripe.sub_fields:
            assert isinstance(sub, game_log_html.EncodedSubField)
            assert sub.name
            assert sub.description
            assert sub.encoding
            assert sub.value_range


# ---------------------------------------------------------------------------
# extract_state_stripes — bird index decode


def test_card_idx_board_decoded_for_played_bird():
    """A bird on the board produces a card_idx_board sub_field with decoded_label."""
    eng, birds = _make_engine(seed=6)
    # Place the first bird directly onto player 0's forest row.
    bird = birds[0]
    eng.state.players[0].board[cards.Habitat.FOREST].append(
        state.PlayedBird(bird=bird, eggs=0)
    )
    vec = encode.encode_state(eng.state).tolist()
    result = encode_viewer.extract_state_stripes(vec, include_setup=False)

    board_stripes = [s for s in result if s.name == "card_idx_board"]
    assert board_stripes, "card_idx_board stripe missing from result"
    decoded_labels = [
        sub.decoded_label
        for sub in board_stripes[0].sub_fields
        if sub.decoded_label is not None
    ]
    assert any(
        bird.name in label for label in decoded_labels
    ), f"Expected '{bird.name}' in decoded labels, got: {decoded_labels}"


def test_card_idx_tray_decoded_for_tray_bird():
    """A bird in the tray produces a card_idx_tray sub_field with decoded_label."""
    eng, birds = _make_engine(seed=7)
    eng.state.tray = [birds[5], birds[10]]
    vec = encode.encode_state(eng.state).tolist()
    result = encode_viewer.extract_state_stripes(vec, include_setup=False)

    tray_stripes = [s for s in result if s.name == "card_idx_tray"]
    assert tray_stripes, "card_idx_tray stripe missing"
    decoded_labels = [
        sub.decoded_label
        for sub in tray_stripes[0].sub_fields
        if sub.decoded_label is not None
    ]
    assert any(
        birds[5].name in label for label in decoded_labels
    ), f"Expected '{birds[5].name}' in decoded labels, got: {decoded_labels}"


# ---------------------------------------------------------------------------
# extract_state_stripes — multi-hot collapse


def test_hand_multihot_collapsed_to_one_row():
    """Birds in hand collapse to a single sub_field row in hand_multihot."""
    eng, birds = _make_engine(seed=8)
    eng.state.players[0].hand = [birds[3], birds[7], birds[12]]
    vec = encode.encode_state(eng.state).tolist()
    result = encode_viewer.extract_state_stripes(vec, include_setup=False)

    hand_stripes = [s for s in result if s.name == "hand_multihot"]
    assert hand_stripes, "hand_multihot stripe missing"
    # Multi-hot stripes collapse to exactly one sub_field.
    assert len(hand_stripes[0].sub_fields) == 1
    label = hand_stripes[0].sub_fields[0].decoded_label
    assert label is not None
    # All three bird names appear somewhere in the collapsed label.
    for bird in [birds[3], birds[7], birds[12]]:
        assert bird.name in label, f"'{bird.name}' missing from hand multihot: {label}"


# ---------------------------------------------------------------------------
# extract_state_stripes — decision-type decode


def test_decision_type_decoded_with_class_name():
    """The decision_type one-hot contains the decision class name."""
    eng, _ = _make_engine(seed=9)
    decision = _main_action_decision()
    vec = encode.encode_state(eng.state, decision).tolist()
    result = encode_viewer.extract_state_stripes(vec, include_setup=False)

    dtype_stripes = [s for s in result if s.name == "decision_type"]
    assert dtype_stripes, "decision_type stripe missing"
    sub = dtype_stripes[0].sub_fields[0]
    assert sub.decoded_label is not None
    assert (
        "MainActionDecision" in sub.decoded_label
    ), f"Expected 'MainActionDecision' in: {sub.decoded_label}"


# ---------------------------------------------------------------------------
# extract_state_stripes — denormalization


def test_normalized_scalars_have_integer_decoded_labels():
    """Any sub_field whose notes contain '÷ N' has an integer decoded_label."""
    eng, _ = _make_engine(seed=10)
    vec = encode.encode_state(eng.state).tolist()
    result = encode_viewer.extract_state_stripes(vec, include_setup=False)

    # Find any scalar sub_field with a ÷ in notes that has a decoded_label.
    denorm_fields: list[game_log_html.EncodedSubField] = []
    for stripe in result:
        for sub in stripe.sub_fields:
            if sub.notes and "÷" in sub.notes and sub.decoded_label is not None:
                denorm_fields.append(sub)

    # If there are denormalized fields (there should be), verify they look integer.
    for sub in denorm_fields:
        assert sub.decoded_label is not None
        try:
            int(sub.decoded_label)
        except ValueError:
            raise AssertionError(
                f"decoded_label '{sub.decoded_label}' for '{sub.name}' is not an integer"
            )


# ---------------------------------------------------------------------------
# extract_choice_stripes — smoke tests


def test_extract_choice_stripes_nonempty():
    """A real choice vector produces at least one non-zero EncodedStripe."""
    eng, _ = _make_engine(seed=11)
    decision = _main_action_decision()
    choice_mat = encode.encode_choices(decision, eng.state)
    choice_vec = choice_mat[0].tolist()
    result = encode_viewer.extract_choice_stripes(choice_vec, include_setup=False)
    assert isinstance(result, list)
    assert len(result) > 0


def test_extract_choice_stripes_zero_vector_empty():
    """A zero choice vector returns an empty list."""
    from wingspan.encode import layout

    spec = layout.DEFAULT_SPEC
    choice_dim = encode.choice_feature_dim(spec)
    vec = [0.0] * choice_dim
    result = encode_viewer.extract_choice_stripes(vec, include_setup=False)
    assert result == []


def test_extract_choice_stripes_structure():
    """Every returned choice stripe has name, description, and sub_fields."""
    eng, _ = _make_engine(seed=12)
    decision = _main_action_decision()
    choice_mat = encode.encode_choices(decision, eng.state)
    choice_vec = choice_mat[0].tolist()
    result = encode_viewer.extract_choice_stripes(choice_vec, include_setup=False)
    for stripe in result:
        assert stripe.name
        assert stripe.description
        assert stripe.sub_fields


# ---------------------------------------------------------------------------
# extract_choice_stripes — bird index decode for bird choices


def test_choice_bird_id_decoded():
    """A PlayBird choice's bird_id stripe decodes to the bird name."""
    eng, birds = _make_engine(seed=13)
    # Give player a bird in hand that can be played.
    bird = birds[0]
    eng.state.players[0].hand = [bird]
    eng.state.players[0].food[cards.Food.SEED] = 5
    eng.state.players[0].food[cards.Food.FISH] = 5
    eng.state.players[0].food[cards.Food.FRUIT] = 5
    eng.state.players[0].food[cards.Food.INVERTEBRATE] = 5
    eng.state.players[0].food[cards.Food.RODENT] = 5

    play_choice = decisions.PlayBirdChoice(
        label="play",
        bird=bird,
        habitat=cards.Habitat.FOREST,
    )
    decision = decisions.PlayBirdDecision(player_id=0, prompt="", choices=[play_choice])
    choice_mat = encode.encode_choices(decision, eng.state)
    choice_vec = choice_mat[0].tolist()
    result = encode_viewer.extract_choice_stripes(choice_vec, include_setup=False)

    # The bird_id one-hot stripe should decode to the bird name.
    bird_id_stripes = [s for s in result if s.name == "bird_id"]
    if bird_id_stripes:
        sub = bird_id_stripes[0].sub_fields[0]
        assert sub.decoded_label is not None
        assert (
            bird.name in sub.decoded_label
        ), f"Expected '{bird.name}' in bird_id label: {sub.decoded_label}"


# ---------------------------------------------------------------------------
# extract_card_attr_stripes


def test_extract_card_attr_stripes_returns_one_stripe():
    """extract_card_attr_stripes returns exactly one stripe (bird_attrs) for any bird."""
    bird = cards.birds_ordered()[0]
    result = encode_viewer.extract_card_attr_stripes(bird)
    assert len(result) == 1
    assert result[0].name == "bird_attrs"


def test_extract_card_attr_stripes_no_bird_identity():
    """The bird_identity one-hot is never included in the returned stripes."""
    for bird in cards.birds_ordered():
        result = encode_viewer.extract_card_attr_stripes(bird)
        names = [stripe.name for stripe in result]
        assert "bird_identity" not in names, f"bird_identity found for {bird.name}"


def test_extract_card_attr_stripes_habitats_decoded():
    """The habitats sub-field has a decoded_label listing the bird's habitat names."""
    # Find a bird with exactly two habitats.
    two_hab = next(bird for bird in cards.birds_ordered() if len(bird.habitats) == 2)
    result = encode_viewer.extract_card_attr_stripes(two_hab)
    assert result
    sub_fields = result[0].sub_fields
    hab_fields = [sf for sf in sub_fields if sf.name == "habitats"]
    assert hab_fields, "habitats sub-field missing"
    decoded = hab_fields[0].decoded_label
    assert decoded is not None
    for habitat in two_hab.habitats:
        assert (
            habitat.value in decoded
        ), f"Habitat {habitat.value} missing from label: {decoded}"


def test_extract_card_attr_stripes_food_cost_decoded():
    """The food_cost sub-field has a decoded_label like '1 seed, 2 fruit'."""
    # Find a bird with a non-trivial AND food cost.
    bird_with_food = next(
        bird
        for bird in cards.birds_ordered()
        if not bird.food_cost.is_or_cost and sum(bird.food_cost.specific) >= 1
    )
    result = encode_viewer.extract_card_attr_stripes(bird_with_food)
    assert result
    sub_fields = result[0].sub_fields
    food_fields = [sf for sf in sub_fields if sf.name == "food_cost"]
    assert food_fields, f"food_cost missing for {bird_with_food.name}"
    decoded = food_fields[0].decoded_label
    assert (
        decoded is not None
    ), f"food_cost decoded_label is None for {bird_with_food.name}"


def test_extract_card_attr_stripes_color_decoded():
    """The color sub-field has a decoded_label matching the bird's power color name."""
    # Find a brown bird.
    brown_bird = next(
        bird
        for bird in cards.birds_ordered()
        if bird.power.color == cards.PowerColor.BROWN
    )
    result = encode_viewer.extract_card_attr_stripes(brown_bird)
    assert result
    color_fields = [sf for sf in result[0].sub_fields if sf.name == "color"]
    assert color_fields, "color sub-field missing for brown bird"
    assert color_fields[0].decoded_label == "brown"


def test_extract_card_attr_stripes_sub_fields_all_decoded():
    """Every non-zero sub-field should have a non-None decoded_label."""
    for bird in cards.birds_ordered():
        result = encode_viewer.extract_card_attr_stripes(bird)
        if not result:
            continue
        for sub in result[0].sub_fields:
            assert (
                sub.decoded_label is not None
            ), f"decoded_label is None for sub-field '{sub.name}' on bird '{bird.name}'"
