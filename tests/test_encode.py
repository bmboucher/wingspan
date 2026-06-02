# pyright: reportPrivateUsage=false
# (a few tests reach into ``state_encode`` internals — e.g. ``_bird_attr_vector``,
# the card encoder's per-card input builder — to isolate attribute encoding.)
"""Tests for the per-choice encoder + POV-aware state encoder.

These cover the four structural changes called out in the RL trainability
review:

1. Per-choice features distinguish candidates that differ only in identity
   (two states identical in aggregate but with different candidate cards
   should produce different choice features).
2. The ``DecisionType`` one-hot stripe in the state vector flips when the
   decision type changes.
3. The state encoder rotates POV when the asking player changes.
4. Choice-count truncation no longer silently drops options, and an over-wide
   decision is non-fatal — it logs a warning past
   encode.RUNAWAY_CHOICE_THRESHOLD and still featurizes every choice.
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Make ``import wingspan`` work whether pytest is run from repo root or the
# tests/ directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import cards, decisions, encode, engine, state
from wingspan.encode import state_encode

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


def test_card_identity_offsets_align_with_encoder_output():
    """``model._embed_state`` splits the flat state at ``OFF_CARD_INDEX`` /
    ``OFF_HAND_MULTIHOT`` / ``OFF_DECISION_TYPE`` to pull out the card-index
    block, hand multi-hot, and decision-type stripe. Those offsets derive from
    ``_CONT_PREFIX_DIM``, a separate sum from ``state_size`` and from the actual
    ``encode_state`` parts — and once drifted out of sync (the birdfeeder stripe
    was counted as 5, not 6), shifting the model's window one column off the real
    blocks. This guards the alignment without hard-coding any absolute offset."""
    eng, birds, *_ = engine.Engine.create(seed=51)
    me = eng.state.players[0]
    # Fill the tray so the final card-index slot carries a real bird index > 1;
    # under an off-by-one split it would leak into the hand multi-hot block.
    eng.state.tray = [birds[10], birds[20], birds[30]]
    me.hand = [birds[40], birds[41]]
    decision = decisions.MainActionDecision(
        player_id=0,
        prompt="x",
        choices=[
            decisions.MainActionChoice(label="a", action=decisions.MainAction.GAIN_FOOD)
        ],
    )
    vec = encode.encode_state(eng.state, decision)

    card_block = vec[encode.OFF_CARD_INDEX : encode.OFF_HAND_MULTIHOT]
    hand_block = vec[encode.OFF_HAND_MULTIHOT : encode.OFF_DECISION_TYPE]
    decision_block = vec[encode.OFF_DECISION_TYPE :]

    # Card-index entries are whole numbers (``bird_index + 1``, or 0 for empty);
    # a leaked birdfeeder scalar would be fractional.
    assert np.array_equal(card_block, np.floor(card_block))
    # The hand multi-hot is strictly 0/1; a leaked card index would break it.
    assert set(np.unique(hand_block)).issubset({0.0, 1.0})
    # The decision-type stripe is a clean one-hot of exactly the declared width.
    assert decision_block.shape == (encode.DECISION_TYPE_DIM,)
    assert decision_block.sum() == 1.0
    # The three tray birds occupy the last TRAY_SIZE card-index slots by identity.
    tray_slots = card_block[-state.TRAY_SIZE :]
    expected = sorted(cards.bird_index(bird) + 1 for bird in eng.state.tray)
    assert sorted(int(round(value)) for value in tray_slots) == expected


def test_state_encoder_decision_type_one_hot_flips():
    eng, *_ = engine.Engine.create(seed=2)
    d_main = decisions.MainActionDecision(
        player_id=0,
        prompt="x",
        choices=[
            decisions.MainActionChoice(label="a", action=decisions.MainAction.GAIN_FOOD)
        ],
    )
    d_lay = decisions.LayEggDecision(
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
    """Two bird choices on the same state but with different candidate birds
    must produce different feature rows. This is the headline trainability fix
    — positional slots used to make these indistinguishable to the network."""
    eng, birds, *_ = engine.Engine.create(seed=3)
    # Pick two birds with different point values + costs so feature rows
    # plausibly differ.
    first_bird, second_bird = _two_distinct_birds(birds)
    decision = decisions.BirdPowerPickBirdFromHandDecision(
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
    decision = decisions.GainFoodDecision(
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

    decision = decisions.LayEggDecision(
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


def test_pay_cost_features_distinguish_exchanges():
    """An AcceptExchange ``PayCostChoice`` surfaces its trade terms, so two
    different exchanges (egg→card vs food→tucks) produce different feature rows,
    and both differ from a skip — closing the old "PayCostChoice is featureless"
    gap (DECISIONS.md §4.3)."""
    eng, *_ = engine.Engine.create(seed=11)
    decision = decisions.AcceptExchangeDecision(
        player_id=0,
        prompt="x",
        choices=[
            decisions.PayCostChoice(
                label="egg->card", paid_egg_count=1, gained_card_count=1
            ),
            decisions.PayCostChoice(
                label="food->tucks", paid_food=cards.Food.SEED, gained_tuck_count=2
            ),
            decisions.SkipChoice(label="skip"),
        ],
    )
    feats = encode.encode_choices(decision, eng.state)
    assert feats.shape == (3, encode.CHOICE_FEATURE_DIM)
    assert not np.array_equal(feats[0], feats[1]), "distinct exchanges must differ"
    assert not np.array_equal(feats[0], feats[2]), "an exchange must differ from skip"
    assert feats[0].sum() != 0.0, "the egg->card trade terms should be featurized"


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
# Card-identity stripes (concatenated identity + attributes)


def test_every_bird_has_a_distinct_feature_row():
    """With the bird-identity one-hot, no two distinct birds collapse to the
    same per-choice features — even two birds with identical attributes are told
    apart (the per-card embedding signal #2 adds)."""
    eng, birds, *_ = engine.Engine.create(seed=12)
    decision = decisions.BirdPowerPickBirdFromHandDecision(
        player_id=0,
        prompt="x",
        choices=[decisions.BirdChoice(label=bird.name, bird=bird) for bird in birds],
    )
    feats = encode.encode_choices(decision, eng.state)
    distinct_rows = {row.tobytes() for row in feats}
    assert len(distinct_rows) == len(birds)


def test_every_bonus_card_has_a_distinct_feature_row():
    """The bonus-card identity one-hot distinguishes every bonus card, fixing
    the old ``id % 16`` hash that collapsed cards 16 apart (#2)."""
    eng, _birds, bonuses, _ = engine.Engine.create(seed=14)
    decision = decisions.BirdPowerPickBonusCardDecision(
        player_id=0,
        prompt="x",
        choices=[
            decisions.BonusCardChoice(label=bonus.name, bonus_card=bonus)
            for bonus in bonuses
        ],
    )
    feats = encode.encode_choices(decision, eng.state)
    distinct_rows = {row.tobytes() for row in feats}
    assert len(distinct_rows) == len(bonuses)


def test_setup_kept_set_changes_the_feature_row():
    """The setup pick's kept-card multi-hot makes different kept sets featurize
    differently, so the setup head can see *which* cards were kept (§3.1)."""
    eng, birds, *_ = engine.Engine.create(seed=13)
    kept_foods = tuple(cards.ALL_FOODS[: len(cards.ALL_FOODS) - 1])  # keep 1 card
    decision = decisions.SetupDecision(
        player_id=0,
        prompt="x",
        choices=[
            decisions.SetupChoice(
                label="a",
                kept_cards=(birds[0],),
                kept_foods=kept_foods,
                bonus_card=None,
            ),
            decisions.SetupChoice(
                label="b",
                kept_cards=(birds[1],),
                kept_foods=kept_foods,
                bonus_card=None,
            ),
        ],
        dealt_cards=[birds[0], birds[1]],
        dealt_bonus=[],
    )
    feats = encode.encode_choices(decision, eng.state)
    assert not np.array_equal(feats[0], feats[1])


def test_hand_identity_distinguishes_equal_size_hands():
    """My hand is encoded as an identity multi-hot in the state, so two hands of
    the same size but different cards yield different state vectors (#2)."""
    eng, birds, *_ = engine.Engine.create(seed=15)
    eng.state.players[0].hand = [birds[0], birds[1]]
    vec_a = encode.encode_state(eng.state)
    eng.state.players[0].hand = [birds[2], birds[3]]
    vec_b = encode.encode_state(eng.state)
    assert not np.array_equal(vec_a, vec_b)


def test_bonus_progress_changes_state_vector():
    """Held bonus cards and progress toward them are encoded in the state: a
    held card's identity bit distinguishes it from not holding it (even at 0
    progress), and adding qualifying birds moves the linear / stepped channels.
    Without this stripe the model could not see its bonus cards during play."""
    eng, birds, bonuses, _ = engine.Engine.create(seed=21)
    me = eng.state.players[eng.state.current_player]
    bird_feeder = next(bonus for bonus in bonuses if bonus.name == "Bird Feeder")
    seed_birds = [bird for bird in birds if "Bird Feeder" in bird.bonus_categories][:6]

    me.bonus_cards = []
    me.board[cards.Habitat.FOREST] = []
    vec_base = encode.encode_state(eng.state)

    me.bonus_cards = [bird_feeder]
    vec_held = encode.encode_state(eng.state)

    me.board[cards.Habitat.FOREST] = [
        state.PlayedBird(bird=bird) for bird in seed_birds
    ]
    vec_progress = encode.encode_state(eng.state)

    assert not np.array_equal(vec_base, vec_held)  # identity bit flips
    assert not np.array_equal(vec_held, vec_progress)  # value channels move


def test_multiple_held_bonus_cards_each_appear_in_state():
    """A player can hold more than one bonus card, and each held card is
    encoded: adding a second card to the held set changes the state vector."""
    eng, _birds, bonuses, _ = engine.Engine.create(seed=22)
    me = eng.state.players[eng.state.current_player]
    me.board[cards.Habitat.FOREST] = []

    me.bonus_cards = [bonuses[0]]
    vec_one = encode.encode_state(eng.state)
    me.bonus_cards = [bonuses[0], bonuses[1]]
    vec_two = encode.encode_state(eng.state)
    assert not np.array_equal(vec_one, vec_two)


# ---------------------------------------------------------------------------
# Full per-slot board / tray / round-goal state stripes


def test_board_birds_appear_in_state():
    """Birds in play are encoded per slot: an empty forest differs from one with
    a bird, and two different birds in the same slot yield different vectors
    (the per-slot identity index the model embeds — no aggregate captures *which*
    bird)."""
    eng, birds, *_ = engine.Engine.create(seed=31)
    me = eng.state.players[eng.state.current_player]
    me.board[cards.Habitat.FOREST] = []
    vec_empty = encode.encode_state(eng.state)
    me.board[cards.Habitat.FOREST] = [state.PlayedBird(bird=birds[0])]
    vec_a = encode.encode_state(eng.state)
    me.board[cards.Habitat.FOREST] = [state.PlayedBird(bird=birds[1])]
    vec_b = encode.encode_state(eng.state)
    assert not np.array_equal(vec_empty, vec_a)  # a bird now occupies the slot
    assert not np.array_equal(vec_a, vec_b)  # identity distinguishes which bird


def test_per_slot_eggs_move_state():
    """A bird's egg count is encoded per slot."""
    eng, birds, *_ = engine.Engine.create(seed=32)
    me = eng.state.players[eng.state.current_player]
    egg_bird = next(bird for bird in birds if bird.egg_limit >= 1)
    pb = state.PlayedBird(bird=egg_bird)
    me.board[cards.Habitat.GRASSLAND] = [pb]
    vec0 = encode.encode_state(eng.state)
    pb.eggs = 1
    vec1 = encode.encode_state(eng.state)
    assert not np.array_equal(vec0, vec1)


def test_per_slot_cached_food_by_type_moves_state():
    """The headline of the per-type cached-food change: a bird caching 1 SEED
    versus 1 FISH has the *same* total cached, so every aggregate stripe is
    identical — yet the per-slot cached-by-type block makes the two states
    differ. This was impossible under the old scalar ``cached_food``."""
    eng, birds, *_ = engine.Engine.create(seed=33)
    me = eng.state.players[eng.state.current_player]
    pb = state.PlayedBird(bird=birds[0])
    me.board[cards.Habitat.FOREST] = [pb]
    pb.cached_food[cards.Food.SEED] = 1
    vec_seed = encode.encode_state(eng.state)
    pb.cached_food[cards.Food.SEED] = 0
    pb.cached_food[cards.Food.FISH] = 1
    vec_fish = encode.encode_state(eng.state)
    assert pb.cached_food.total() == 1  # same total in both states
    assert not np.array_equal(vec_seed, vec_fish)  # only the per-type block differs


def test_tray_contents_encoded_and_order_invariant():
    """The public bird tray is encoded by identity (its card-index slots): different
    tray contents move the vector, but reordering the same cards does not (the
    ``bird_index`` sort makes the interchangeable tray slots order-invariant)."""
    eng, birds, *_ = engine.Engine.create(seed=34)
    eng.state.tray = [birds[0], birds[1]]
    vec_ab = encode.encode_state(eng.state)
    eng.state.tray = [birds[1], birds[0]]
    vec_ba = encode.encode_state(eng.state)
    eng.state.tray = [birds[2], birds[3]]
    vec_cd = encode.encode_state(eng.state)
    assert np.array_equal(vec_ab, vec_ba)  # order-invariant
    assert not np.array_equal(vec_ab, vec_cd)  # different cards move the vector


def test_round_goal_stripe_encodes_all_four_rounds():
    """All four round goals are encoded, not just the current one: changing a
    *future* round's goal moves the vector even while sitting in round 0. The old
    encoder, which only saw ``round_goals[round_idx]``, could not."""
    eng, _birds, _bonuses, goals = engine.Engine.create(seed=35)
    eng.state.round_idx = 0
    vec_before = encode.encode_state(eng.state)
    current = eng.state.round_goals[3]
    replacement = next(goal for goal in goals if goal.category != current.category)
    eng.state.round_goals[3] = replacement
    vec_after = encode.encode_state(eng.state)
    assert not np.array_equal(vec_before, vec_after)


def test_opponent_bonus_count_changes_pov_state():
    """The opponent's bonus-card *count* is observable (a single scalar); their
    identities are not. Holding one more opponent bonus card moves my POV vector
    even though my own bonus stripes are untouched."""
    eng, _birds, bonuses, _ = engine.Engine.create(seed=38)
    pov = eng.state.current_player
    opp = eng.state.players[1 - pov]
    opp.bonus_cards = [bonuses[0]]
    vec_one = encode.encode_state(eng.state)
    opp.bonus_cards = [bonuses[0], bonuses[1]]
    vec_two = encode.encode_state(eng.state)
    assert not np.array_equal(vec_one, vec_two)


# ---------------------------------------------------------------------------
# Card attributes ride the card encoder's per-card feature vector, not the state
# vector (a board/tray slot carries only identity + mutable state). Each test
# holds bird identity fixed (``model_copy`` keeps ``id``) and varies one
# attribute, asserting that ``_bird_attr_vector`` — the attribute half of the card
# encoder's input — changes, so the card table can distinguish that attribute.


def test_board_bird_nest_attribute_encoded():
    """A bird's nest is encoded in its attribute vector: a star-nest bird and a
    bowl-nest bird (same identity) yield different ``_bird_attr_vector`` outputs."""
    _eng, birds, *_ = engine.Engine.create(seed=41)
    star = birds[0].model_copy(update={"nest": cards.NestType.STAR})
    bowl = birds[0].model_copy(update={"nest": cards.NestType.BOWL})
    assert not np.array_equal(
        state_encode._bird_attr_vector(star), state_encode._bird_attr_vector(bowl)
    )


def test_board_bird_food_cost_encoded():
    """A bird's food cost is encoded in its attribute vector."""
    _eng, birds, *_ = engine.Engine.create(seed=42)
    cheap = birds[0].model_copy(
        update={"food_cost": cards.BirdCost.from_specific({cards.Food.SEED: 1})}
    )
    pricey = birds[0].model_copy(
        update={"food_cost": cards.BirdCost.from_specific({cards.Food.FISH: 3}, wild=1)}
    )
    assert not np.array_equal(
        state_encode._bird_attr_vector(cheap), state_encode._bird_attr_vector(pricey)
    )


def test_board_bird_bonus_category_test_flags_encoded():
    """A bird's static bonus-card qualifications (the "test" flags such as 'named
    after a person') are encoded in its attribute vector — the multi-hot the card
    encoder reads to learn bonus-relevant card value."""
    _eng, birds, *_ = engine.Engine.create(seed=43)
    plain = birds[0].model_copy(update={"bonus_categories": ()})
    tagged = birds[0].model_copy(update={"bonus_categories": ("Bird Feeder",)})
    assert not np.array_equal(
        state_encode._bird_attr_vector(plain), state_encode._bird_attr_vector(tagged)
    )


# ---------------------------------------------------------------------------
# Truncation behavior


def test_encode_choices_warns_but_does_not_abort_on_absurd_cardinality():
    """A decision wider than the runaway threshold is non-fatal: the encoder
    warns and still featurizes every choice rather than asserting, so an
    unattended training run is never killed by a single wide decision."""
    eng, _birds, *_ = engine.Engine.create(seed=8)
    n_choices = encode.RUNAWAY_CHOICE_THRESHOLD + 1
    too_many: list[decisions.FoodChoice | decisions.SkipChoice] = [
        decisions.FoodChoice(label=f"c{i}", food=cards.Food.SEED)
        for i in range(n_choices)
    ]
    decision = decisions.GainFoodDecision(
        player_id=0,
        prompt="x",
        choices=too_many,
    )
    feats = encode.encode_choices(decision, eng.state)
    assert feats.shape == (n_choices, encode.CHOICE_FEATURE_DIM)


def test_encode_choices_does_not_truncate_under_soft_threshold():
    """The old encoder silently capped at e.g. MAX_HAND_PICKS=10. The new
    one returns every choice regardless of count."""
    eng, birds, *_ = engine.Engine.create(seed=9)
    n_birds = 25  # comfortably above the old hand-pick cap of 10
    decision = decisions.BirdPowerPickBirdFromHandDecision(
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
