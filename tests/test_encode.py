"""Tests for the per-choice encoder + POV-aware state encoder.

These cover the four structural changes called out in the RL trainability
review:

1. Per-choice features distinguish candidates that differ only in identity
   (two states identical in aggregate but with different candidate cards
   should produce different choice features).
2. The ``DecisionType`` one-hot stripe in the state vector flips when the
   decision type changes.
3. The state encoder rotates POV when the asking player changes.
4. Choice-count truncation no longer silently drops options — a sanity-cap
   assert fires if a decision balloons past encode.MAX_CHOICES_HARD.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# Make ``import wingspan`` work whether pytest is run from repo root or the
# tests/ directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import cards, decisions, encode, engine

# ---------------------------------------------------------------------------
# State encoder


def test_state_size_matches_encoder_output():
    eng, *_ = engine.Engine.create(seed=1)
    vec = encode.encode_state(eng.state)
    assert vec.shape == (encode.state_size(),)
    assert vec.dtype == np.float32


def test_state_encoder_pov_rotates_with_decision_player():
    """Player 0's POV state should differ from player 1's POV state once
    each player has a non-trivial board. We force the difference by giving
    one player a card in hand and not the other."""
    eng, birds, *_ = engine.Engine.create(seed=7)
    # Re-seed the players' hands so they differ deterministically.
    eng.state.players[0].hand = [birds[0], birds[1]]
    eng.state.players[1].hand = []

    d0 = decisions.MainActionDecision(
        player_id=0,
        prompt="p0",
        choices=[
            decisions.MainActionChoice(label="x", action=decisions.MainAction.GAIN_FOOD)
        ],
    )
    d1 = decisions.MainActionDecision(
        player_id=1,
        prompt="p1",
        choices=[
            decisions.MainActionChoice(label="x", action=decisions.MainAction.GAIN_FOOD)
        ],
    )
    v0 = encode.encode_state(eng.state, d0)
    v1 = encode.encode_state(eng.state, d1)
    assert not np.array_equal(v0, v1), "POV rotation should flip me/opp features"


def test_state_encoder_decision_type_one_hot_flips():
    eng, *_ = engine.Engine.create(seed=2)
    d_main = decisions.MainActionDecision(
        player_id=0,
        prompt="x",
        choices=[
            decisions.MainActionChoice(label="a", action=decisions.MainAction.GAIN_FOOD)
        ],
    )
    d_lay = decisions.LayEggPickBirdDecision(
        player_id=0,
        prompt="x",
        choices=[
            decisions.BoardTargetChoice(
                label="a", habitat=cards.Habitat.GRASSLAND, slot=0
            )
        ],
    )
    v_main = encode.encode_state(eng.state, d_main)
    v_lay = encode.encode_state(eng.state, d_lay)
    # The non-decision-type portion is identical; only the type stripe
    # should differ.
    head = encode.state_size() - encode.DECISION_TYPE_DIM
    assert np.array_equal(v_main[:head], v_lay[:head])
    assert not np.array_equal(v_main[head:], v_lay[head:])


# ---------------------------------------------------------------------------
# Per-choice featurization


def test_choice_features_distinguish_candidate_birds():
    """Two PLAY_BIRD_PICK_CARD choices on the same state but with different
    candidate birds must produce different feature rows. This is the
    headline trainability fix — positional slots used to make these
    indistinguishable to the network."""
    eng, birds, *_ = engine.Engine.create(seed=3)
    # Pick two birds with different point values + costs so feature rows
    # plausibly differ.
    first_bird, second_bird = _two_distinct_birds(birds)
    decision = decisions.PlayBirdPickCardDecision(
        player_id=0,
        prompt="x",
        choices=[
            decisions.BirdChoice(label=first_bird.name, bird=first_bird),
            decisions.BirdChoice(label=second_bird.name, bird=second_bird),
        ],
    )
    feats = encode.encode_choices(decision, eng.state)
    assert feats.shape == (2, encode.CHOICE_FEATURE_DIM)
    assert not np.array_equal(feats[0], feats[1])


def test_choice_features_food_one_hot_is_set():
    eng, *_ = engine.Engine.create(seed=4)
    decision = decisions.GainFoodPickDieDecision(
        player_id=0,
        prompt="x",
        choices=[
            decisions.FoodChoice(label="seed", food=cards.Food.SEED),
            decisions.FoodChoice(label="fish", food=cards.Food.FISH),
        ],
    )
    feats = encode.encode_choices(decision, eng.state)
    # Same decision type → identical kind stripe, differing food stripe.
    assert feats.shape == (2, encode.CHOICE_FEATURE_DIM)
    assert feats[0].sum() > 0
    assert not np.array_equal(feats[0], feats[1])


def test_choice_features_board_target_reflects_dynamic_state():
    """A LAY_EGG_PICK_BIRD choice surfaces the target bird's current eggs
    and capacity. Changing the bird's eggs must change its feature row."""
    from wingspan import state

    eng, birds, *_ = engine.Engine.create(seed=5)
    # Plant a single bird on player 0's grassland and inspect feature flip
    # when eggs change.
    eng.state.players[0].board[cards.Habitat.GRASSLAND] = []

    pb = state.PlayedBird(bird=birds[0])
    eng.state.players[0].board[cards.Habitat.GRASSLAND].append(pb)

    decision = decisions.LayEggPickBirdDecision(
        player_id=0,
        prompt="x",
        choices=[
            decisions.BoardTargetChoice(
                label="x", habitat=cards.Habitat.GRASSLAND, slot=0
            )
        ],
    )
    f0 = encode.encode_choices(decision, eng.state)[0].copy()
    pb.eggs = min(1, pb.bird.egg_limit)
    f1 = encode.encode_choices(decision, eng.state)[0].copy()
    if pb.bird.egg_limit >= 1:
        assert not np.array_equal(
            f0, f1
        ), "board-target features should reflect dynamic egg count"


def test_choice_features_skip_flag_for_skip_choice():
    eng, birds, *_ = engine.Engine.create(seed=6)

    decision = decisions.BirdPowerTuckFromHandDecision(
        player_id=0,
        prompt="x",
        choices=[
            decisions.BirdChoice(label="tuck", bird=birds[0]),
            decisions.SkipChoice(label="skip"),
        ],
    )
    feats = encode.encode_choices(decision, eng.state)
    # Bird choice should set the bird kind; skip should set the special-skip
    # flag. We don't tie this to a specific column index so the test is
    # robust to layout edits, but distinct feature rows are required.
    assert not np.array_equal(feats[0], feats[1])


# ---------------------------------------------------------------------------
# Truncation behavior


def test_encode_choices_asserts_on_absurd_cardinality():
    """The hard cap protects against runaway choice generation (a sign of
    a bug, not normal play)."""
    eng, _birds, *_ = engine.Engine.create(seed=8)
    too_many = [
        decisions.FoodChoice(label=f"c{i}", food=cards.Food.SEED)
        for i in range(encode.MAX_CHOICES_HARD + 1)
    ]
    decision = decisions.GainFoodPickDieDecision(
        player_id=0,
        prompt="x",
        choices=too_many,
    )
    with pytest.raises(AssertionError):
        encode.encode_choices(decision, eng.state)


def test_encode_choices_does_not_truncate_under_soft_threshold():
    """The old encoder silently capped at e.g. MAX_HAND_PICKS=10. The new
    one returns every choice as long as it's under the hard cap."""
    eng, birds, *_ = engine.Engine.create(seed=9)
    n_birds = 25  # comfortably above the old hand-pick cap of 10
    decision = decisions.PlayBirdPickCardDecision(
        player_id=0,
        prompt="x",
        choices=[
            decisions.BirdChoice(label=bird.name, bird=bird) for bird in birds[:n_birds]
        ],
    )
    feats = encode.encode_choices(decision, eng.state)
    assert feats.shape == (n_birds, encode.CHOICE_FEATURE_DIM)


###### PRIVATE #######


def _two_distinct_birds(birds: list[cards.Bird]) -> tuple[cards.Bird, cards.Bird]:
    """Return two birds with at least one differing attribute among
    (points, total_food_cost, color)."""
    first_bird = birds[0]
    for bird in birds[1:]:
        if (bird.points, bird.food_cost.total, bird.color) != (
            first_bird.points,
            first_bird.food_cost.total,
            first_bird.color,
        ):
            return first_bird, bird
    raise AssertionError("could not find two distinct birds in the catalog")
