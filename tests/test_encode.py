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

import numpy as np

from wingspan import cards, decisions, encode, engine, state
from wingspan.encode import layout, state_encode, stripes
from wingspan.engine import scoring

# Make ``import wingspan`` work whether pytest is run from repo root or the
# tests/ directory.


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
    # The three tray birds occupy the last TRAY_SIZE card-index slots, one per
    # positional slot (0 = empty, bird_index+1 = occupied).
    tray_slots = card_block[-state.TRAY_SIZE :]
    expected = [
        cards.bird_index(bird) + 1 for bird in eng.state.tray if bird is not None
    ]
    assert sorted(int(round(value)) for value in tray_slots if value > 0) == sorted(
        expected
    )


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


def test_state_encoder_birdfeeder_reset_flag_tracks_single_face():
    """The birdfeeder stripe's trailing flag mirrors
    ``Birdfeeder.reset_available()``: 1 when every die shows one face (a single
    food, or all dice on the choice face), 0 on a mixed feeder. The stripe is
    located through the registry so the test survives layout growth."""
    eng, *_ = engine.Engine.create(seed=4)
    feeder = eng.state.birdfeeder
    feeder_stripe = next(
        stripe
        for stripe in stripes.state_stripe_layout().stripes
        if stripe.name == "birdfeeder"
    )
    flag_index = feeder_stripe.offset + feeder_stripe.size - 1

    def flag() -> float:
        return float(encode.encode_state(eng.state)[flag_index])

    feeder.counts.zero()
    feeder.choice_dice = 0
    feeder.counts[cards.Food.SEED] = 3  # one single-food face -> reset on offer
    assert flag() == 1.0

    feeder.counts[cards.Food.FISH] = 1  # a second face -> no reset
    assert flag() == 0.0

    feeder.counts.zero()
    feeder.choice_dice = 5  # all dice on the choice face is still one face
    assert flag() == 1.0


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
    """A LayEgg board-target choice fills the 15-slot board block from the
    deciding player's board: the targeted slot's add flag plus every slot's
    cached food and tucked count. Changing the bird's tucked count must change
    its feature row."""
    from wingspan import state

    eng, birds, *_ = engine.Engine.create(seed=5)
    # Plant a single bird on player 0's grassland and inspect the feature flip
    # when its (per-slot) tucked count changes.
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
    pb.tucked_cards += 1
    f1 = encode.encode_choices(decision, eng.state)[0].copy()
    assert not np.array_equal(
        f0, f1
    ), "board-target features should reflect the targeted slot's dynamic state"


def test_pay_cost_features_distinguish_exchanges():
    """An AcceptExchange ``PayCostChoice`` surfaces its trade terms, so two
    different exchanges (egg→card vs food→tucks) produce different feature rows,
    and both differ from a skip — closing the old "PayCostChoice is featureless"
    gap (DECISIONS.md §2.8)."""
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
    eng, *_ = engine.Engine.create(seed=6)

    # ActivateTuckDecision is the canonical [activate | skip] decision — the
    # activate row (TuckActivateChoice) and skip row must encode differently.
    decision = decisions.ActivateTuckDecision(
        player_id=0,
        prompt="x",
        choices=[
            decisions.TuckActivateChoice(label="tuck 1 card", cards_to_tuck=1),
            decisions.SkipChoice(label="skip"),
        ],
    )
    feats = encode.encode_choices(decision, eng.state)
    # Activate choice should set the exchange stripe; skip should set the
    # special-skip flag. Rows must be distinct.
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
    # Setup choices are scored by the main net only when it carries setup, so
    # featurize with that spec (the trailing setup_agg stripe is otherwise absent).
    feats = encode.encode_choices(
        decision, eng.state, encode.EncodingSpec(include_setup=True)
    )
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


def test_tray_contents_encoded_positionally():
    """The bird tray is encoded per-slot: position matters (swapping birds in the
    same two slots moves the vector), and different card sets also move it."""
    eng, birds, *_ = engine.Engine.create(seed=34)
    eng.state.tray = [birds[0], birds[1], None]
    vec_ab = encode.encode_state(eng.state)
    eng.state.tray = [birds[1], birds[0], None]
    vec_ba = encode.encode_state(eng.state)
    eng.state.tray = [birds[2], birds[3], None]
    vec_cd = encode.encode_state(eng.state)
    assert not np.array_equal(vec_ab, vec_ba)  # position-sensitive: swap moves vector
    assert not np.array_equal(vec_ab, vec_cd)  # different cards also move the vector


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
    """A bird's static bonus-card qualifications (the 'test' flags such as 'named
    after a person') are encoded in its attribute vector — the 7-wide multi-hot
    of curated intrinsic-property categories the card encoder reads.
    Uses 'Photographer' (a kept category) to verify the bit is set."""
    _eng, birds, *_ = engine.Engine.create(seed=43)
    plain = birds[0].model_copy(update={"bonus_categories": ()})
    tagged = birds[0].model_copy(update={"bonus_categories": ("Photographer",)})
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
    too_many: list[
        decisions.FoodChoice | decisions.FoodSubsetChoice | decisions.SkipChoice
    ] = [
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


def test_gain_food_choice_die_uses_distinct_slots():
    """Taking the invertebrate/seed choice die is encoded in its own gain_food
    slot, apart from a plain invertebrate die — so the model scores burning the
    flexible die separately."""
    eng, *_ = engine.Engine.create(seed=2)
    decision = decisions.GainFoodDecision(
        player_id=0,
        prompt="x",
        choices=[
            decisions.FoodChoice(label="inv", food=cards.Food.INVERTEBRATE),
            decisions.FoodChoice(
                label="inv*", food=cards.Food.INVERTEBRATE, from_choice_die=True
            ),
        ],
    )
    plain, combo = encode.encode_choices(decision, eng.state)
    assert plain[layout._OFF_GAIN_FOOD + 0] == 1.0  # invertebrate plain slot
    assert combo[layout._OFF_GAIN_FOOD + layout._GAIN_FOOD_CHOICE_INV] == 1.0
    assert combo[layout._OFF_GAIN_FOOD + 0] == 0.0
    assert not np.array_equal(plain, combo)


def test_food_subset_choice_fills_gain_food_as_count_vector():
    """A combined FoodSubsetChoice gain fills the gain_food stripe as a count
    vector (not a one-hot): a 2-fish supply subset puts 2.0 in the fish slot."""
    eng, *_ = engine.Engine.create(seed=2)
    fish = cards.food_index(cards.Food.FISH)
    seed = cards.food_index(cards.Food.SEED)
    decision = decisions.GainFoodDecision(
        player_id=0,
        prompt="x",
        choices=[
            decisions.FoodSubsetChoice(
                plain=state.FoodPool.from_dict({cards.Food.FISH: 2})
            ),
            decisions.FoodSubsetChoice(
                plain=state.FoodPool.from_dict({cards.Food.FISH: 1, cards.Food.SEED: 1})
            ),
        ],
    )
    two_fish, fish_seed = encode.encode_choices(decision, eng.state)
    assert two_fish[layout._OFF_GAIN_FOOD + fish] == 2.0
    assert fish_seed[layout._OFF_GAIN_FOOD + fish] == 1.0
    assert fish_seed[layout._OFF_GAIN_FOOD + seed] == 1.0


def test_food_subset_choice_die_slots_are_distinct_from_plain():
    """Choice-die resolutions fill the dedicated choice-die slots, not the plain
    invertebrate / seed slots."""
    eng, *_ = engine.Engine.create(seed=2)
    decision = decisions.GainFoodDecision(
        player_id=0,
        prompt="x",
        choices=[
            decisions.FoodSubsetChoice(
                plain=state.FoodPool(), choice_inv=1, choice_seed=1
            )
        ],
    )
    (row,) = encode.encode_choices(decision, eng.state)
    assert row[layout._OFF_GAIN_FOOD + layout._GAIN_FOOD_CHOICE_INV] == 1.0
    assert row[layout._OFF_GAIN_FOOD + layout._GAIN_FOOD_CHOICE_SEED] == 1.0
    assert row[layout._OFF_GAIN_FOOD + cards.food_index(cards.Food.INVERTEBRATE)] == 0.0
    assert row[layout._OFF_GAIN_FOOD + cards.food_index(cards.Food.SEED)] == 0.0


def test_food_subset_single_unit_byte_identical_to_food_choice():
    """A single-unit FoodSubsetChoice encodes byte-identically to the matching
    FoodChoice one-hot — the combine_gain_food regime is REGIME, not a
    shape/scale change, so N==1 produces the exact same row."""
    eng, *_ = engine.Engine.create(seed=2)
    subset = decisions.GainFoodDecision(
        player_id=0,
        prompt="x",
        choices=[
            decisions.FoodSubsetChoice(
                plain=state.FoodPool.from_dict({cards.Food.FISH: 1})
            )
        ],
    )
    one_hot = decisions.GainFoodDecision(
        player_id=0,
        prompt="x",
        choices=[decisions.FoodChoice(label="fish", food=cards.Food.FISH)],
    )
    (subset_row,) = encode.encode_choices(subset, eng.state)
    (one_hot_row,) = encode.encode_choices(one_hot, eng.state)
    assert np.array_equal(subset_row, one_hot_row)


def test_main_action_choice_is_one_hot_in_stable_order():
    """A MainActionChoice is a one-hot over the four actions (never an index)."""
    eng, *_ = engine.Engine.create(seed=3)
    decision = decisions.MainActionDecision(
        player_id=0,
        prompt="x",
        choices=[
            decisions.MainActionChoice(label=action.value, action=action)
            for action in layout._MAIN_ACTION_ORDER
        ],
    )
    feats = encode.encode_choices(decision, eng.state)
    off, dim = layout._OFF_MAIN_ACTION, layout._MAIN_ACTION_DIM
    for i, row in enumerate(feats):
        stripe = row[off : off + dim]
        assert stripe.sum() == 1.0  # exactly one action bit
        assert stripe[i] == 1.0  # in the stable MAIN_ACTION order
    assert len({tuple(row) for row in feats}) == 4  # all four rows distinct


def test_hand_summary_is_size_habitat_counts_and_food_multihot():
    """The 10-dim hand summary: size, per-habitat counts, and a food+wild
    multi-hot keyed on each card's food cost."""
    eng, birds, *_ = engine.Engine.create(seed=4)
    bird = birds[0]
    eng.state.players[0].hand = [bird]
    summary = state_encode._summary_hand(eng.state.players[0])
    assert summary.shape == (10,)
    assert summary[0] == 1.0 / layout._HAND_SIZE_SCALE  # hand_size = 1
    assert summary[1:4].sum() > 0.0  # at least one habitat count
    for i in range(layout._FOOD_COST_VEC_DIM):  # 5 foods + wild
        expected = 1.0 if bird.food_cost.counts[i] > 0 else 0.0
        assert summary[4 + i] == expected


def test_board_target_lay_vs_pay_flag_and_card_index():
    """A lay-egg board target sets lay_eggs on the slot; a remove-egg target
    sets pay_eggs; both mark the target slot in board_hab/board_col and write
    the occupant bird into the bird_id column."""
    eng, birds, *_ = engine.Engine.create(seed=5)
    eng.state.players[0].board[cards.Habitat.GRASSLAND] = [
        state.PlayedBird(bird=birds[0])
    ]
    target = decisions.BoardTargetChoice(
        label="x", habitat=cards.Habitat.GRASSLAND, slot=0
    )
    lay = decisions.LayEggDecision(player_id=0, prompt="x", choices=[target])
    remove = decisions.RemoveEggDecision(player_id=0, prompt="x", choices=[target])
    lay_row = encode.encode_choices(lay, eng.state)[0]
    rem_row = encode.encode_choices(remove, eng.state)[0]

    slot_index = cards.ALL_HABITATS.index(cards.Habitat.GRASSLAND) * state.ROW_SLOTS
    base = layout._OFF_BOARD + slot_index * layout._BT_SLOT_SCALARS
    assert lay_row[base + layout._BT_LAY_EGGS] == 1.0
    assert lay_row[base + layout._BT_PAY_EGGS] == 0.0
    assert rem_row[base + layout._BT_PAY_EGGS] == 1.0
    assert rem_row[base + layout._BT_LAY_EGGS] == 0.0
    # Occupant lands in bird_id (not board_idx block — that's gone).
    assert lay_row[layout._OFF_BIRD_ID] == cards.bird_index(birds[0]) + 1
    # Target location marked in board_hab (GRASSLAND=0) and board_col (slot=0).
    assert (
        lay_row[
            layout._OFF_BOARD_HAB + cards.ALL_HABITATS.index(cards.Habitat.GRASSLAND)
        ]
        == 1.0
    )
    assert lay_row[layout._OFF_BOARD_COL + 0] == 1.0


# ---------------------------------------------------------------------------
# Choice encoder: the bonus_delta stripe (candidate's contribution to the
# deciding player's held bonus cards)


def test_bonus_delta_zero_for_non_qualifying_candidate():
    """A candidate that qualifies for no held bonus card leaves the stripe
    all-zero; a qualifying candidate fills the qual_count slot."""
    eng, *_ = engine.Engine.create(seed=3)
    birds, bonuses, _ = cards.load_all()
    bird_feeder = _named_bonus(bonuses, "Bird Feeder")
    me = eng.state.players[0]
    me.bonus_cards = [bird_feeder]

    qualifying = next(bird for bird in birds if "Bird Feeder" in bird.bonus_categories)
    non_qualifying = next(
        bird for bird in birds if "Bird Feeder" not in bird.bonus_categories
    )
    decision = decisions.PlayBirdDecision(
        player_id=0,
        prompt="x",
        choices=[
            _play_choice(qualifying),
            _play_choice(non_qualifying),
        ],
    )
    feats = encode.encode_choices(decision, eng.state)
    assert (
        feats[0][layout._OFF_BONUS_DELTA + layout._BONUS_DELTA_QUAL]
        == 1.0 / layout._BONUS_COUNT_SCALE
    )
    assert np.all(_bonus_delta_slice(feats[1]) == 0.0)


def test_bonus_delta_per_bird_card():
    """A held per-bird card pays exactly ``per_bird_vp`` for the +1 qualifying
    bird, so stepped and linear deltas agree at ``per_bird_vp / scale``."""
    eng, *_ = engine.Engine.create(seed=4)
    birds, bonuses, _ = cards.load_all()
    bird_counter = _named_bonus(bonuses, "Bird Counter")
    assert bird_counter.per_bird_vp is not None
    me = eng.state.players[0]
    me.bonus_cards = [bird_counter]

    candidate = next(bird for bird in birds if "Bird Counter" in bird.bonus_categories)
    decision = decisions.PlayBirdDecision(
        player_id=0, prompt="x", choices=[_play_choice(candidate)]
    )
    row = encode.encode_choices(decision, eng.state)[0]
    expected = bird_counter.per_bird_vp / layout._BONUS_VALUE_SCALE
    assert np.isclose(
        row[layout._OFF_BONUS_DELTA + layout._BONUS_DELTA_STEPPED], expected
    )
    assert np.isclose(
        row[layout._OFF_BONUS_DELTA + layout._BONUS_DELTA_LINEAR], expected
    )


def test_bonus_delta_threshold_step_vs_slope():
    """A tiered card prices the +1 differently per channel: between thresholds
    the stepped delta is zero while the linear slope is positive; one bird
    below a threshold the stepped delta jumps by the step's VP."""
    eng, *_ = engine.Engine.create(seed=6)
    birds, bonuses, _ = cards.load_all()
    bird_feeder = _named_bonus(bonuses, "Bird Feeder")  # anchors (5, 3), (8, 7)
    seed_birds = [bird for bird in birds if "Bird Feeder" in bird.bonus_categories]
    assert len(seed_birds) >= 7
    me = eng.state.players[0]
    me.bonus_cards = [bird_feeder]
    candidate = seed_birds[6]
    decision = decisions.PlayBirdDecision(
        player_id=0, prompt="x", choices=[_play_choice(candidate)]
    )

    # Count 5 sits on the 3-VP plateau (next step at 8): the +1 crosses no
    # threshold, so stepped is zero while the linear slope toward 8 is positive.
    me.board[cards.Habitat.FOREST] = [
        state.PlayedBird(bird=board_bird) for board_bird in seed_birds[:5]
    ]
    row = encode.encode_choices(decision, eng.state)[0]
    assert row[layout._OFF_BONUS_DELTA + layout._BONUS_DELTA_STEPPED] == 0.0
    assert row[layout._OFF_BONUS_DELTA + layout._BONUS_DELTA_LINEAR] > 0.0

    # Count 4 is one bird below the (5, 3) threshold: the +1 crosses the step.
    me.board[cards.Habitat.FOREST] = [
        state.PlayedBird(bird=board_bird) for board_bird in seed_birds[:4]
    ]
    row = encode.encode_choices(decision, eng.state)[0]
    assert np.isclose(
        row[layout._OFF_BONUS_DELTA + layout._BONUS_DELTA_STEPPED],
        3.0 / layout._BONUS_VALUE_SCALE,
    )


def test_bonus_delta_zero_for_played_bird_and_skip():
    """Birds already in play (and skip rows) never fill the stripe — their
    qualifying contribution is already counted on the board."""
    eng, *_ = engine.Engine.create(seed=8)
    birds, bonuses, _ = cards.load_all()
    bird_feeder = _named_bonus(bonuses, "Bird Feeder")
    qualifying = next(bird for bird in birds if "Bird Feeder" in bird.bonus_categories)
    me = eng.state.players[0]
    me.bonus_cards = [bird_feeder]
    pb = state.PlayedBird(bird=qualifying)
    me.board[cards.Habitat.FOREST] = [pb]

    played = decisions.BirdPowerPickPlayedBirdDecision(
        player_id=0,
        prompt="x",
        choices=[decisions.PlayedBirdChoice(label="x", played_bird=pb)],
    )
    skip = decisions.ActivateTuckDecision(
        player_id=0, prompt="x", choices=[decisions.SkipChoice(label="s")]
    )
    assert np.all(
        _bonus_delta_slice(encode.encode_choices(played, eng.state)[0]) == 0.0
    )
    assert np.all(_bonus_delta_slice(encode.encode_choices(skip, eng.state)[0]) == 0.0)


def test_bonus_delta_filled_for_draw_source_tray_bird():
    """A tray draw-source candidate carries its bonus contribution; the blind
    deck draw leaves the stripe zero."""
    eng, *_ = engine.Engine.create(seed=9)
    birds, bonuses, _ = cards.load_all()
    bird_feeder = _named_bonus(bonuses, "Bird Feeder")
    qualifying = next(bird for bird in birds if "Bird Feeder" in bird.bonus_categories)
    me = eng.state.players[0]
    me.bonus_cards = [bird_feeder]
    eng.state.tray[0] = qualifying

    decision = decisions.DrawCardsPickSourceDecision(
        player_id=0,
        prompt="x",
        choices=[
            decisions.DrawSourceChoice(
                label="t", source="tray", tray_index=0, bird=qualifying
            ),
            decisions.DrawSourceChoice(label="d", source="deck"),
        ],
    )
    tray_row, deck_row = encode.encode_choices(decision, eng.state)
    assert tray_row[layout._OFF_BONUS_DELTA + layout._BONUS_DELTA_QUAL] > 0.0
    assert np.all(_bonus_delta_slice(deck_row) == 0.0)


def test_bonus_delta_reflects_currently_held_cards():
    """The stripe reads the held bonus cards at decision time, so a card drawn
    mid-game (DRAW_BONUS powers) raises a matching candidate's qual_count."""
    eng, *_ = engine.Engine.create(seed=11)
    birds, bonuses, _ = cards.load_all()
    by_name = {bonus.name: bonus for bonus in bonuses}
    candidate = next(
        bird
        for bird in birds
        if len([name for name in bird.bonus_categories if name in by_name]) >= 2
    )
    matching = [name for name in candidate.bonus_categories if name in by_name]
    me = eng.state.players[0]
    decision = decisions.PlayBirdDecision(
        player_id=0, prompt="x", choices=[_play_choice(candidate)]
    )

    me.bonus_cards = [by_name[matching[0]]]
    first = encode.encode_choices(decision, eng.state)[0]
    me.bonus_cards.append(by_name[matching[1]])
    second = encode.encode_choices(decision, eng.state)[0]

    qual = layout._OFF_BONUS_DELTA + layout._BONUS_DELTA_QUAL
    assert np.isclose(first[qual], 1.0 / layout._BONUS_COUNT_SCALE)
    assert np.isclose(second[qual], 2.0 / layout._BONUS_COUNT_SCALE)


def test_goal_delta_nonzero_for_advancing_bird():
    """A bird that advances a round goal gets a non-zero goal_delta stripe;
    one that cannot affect the goal stays all-zero for those slots."""
    eng, *_ = engine.Engine.create(seed=1)
    _, _, all_goals = cards.load_all()

    # Install a birds_forest goal as round 0 and a total_birds goal as round 1.
    forest_goal = next(g for g in all_goals if g.category == "birds_forest")
    total_goal = next(g for g in all_goals if g.category == "total_birds")
    orig_goals = eng.state.round_goals
    eng.state.round_goals = [forest_goal, total_goal, orig_goals[2], orig_goals[3]]

    # Find a forest bird and a wetland-only bird for contrast.
    all_birds, *_ = cards.load_all()
    forest_bird = next(
        bird for bird in all_birds if cards.Habitat.FOREST in bird.habitats
    )
    wetland_only_bird = next(
        bird for bird in all_birds if bird.habitats == (cards.Habitat.WETLAND,)
    )

    decision_forest = decisions.PlayBirdDecision(
        player_id=0, prompt="x", choices=[_play_choice(forest_bird)]
    )
    forest_row = encode.encode_choices(decision_forest, eng.state)[0]

    decision_wetland = decisions.PlayBirdDecision(
        player_id=0, prompt="x", choices=[_play_choice(wetland_only_bird)]
    )
    wetland_row = encode.encode_choices(decision_wetland, eng.state)[0]

    # Forest bird should have count_delta > 0 for goal slot 0 (birds_forest).
    slot0_count = (
        layout._OFF_GOAL_DELTA
        + 0 * layout._GOAL_DELTA_SLOT_DIM
        + layout._GOAL_DELTA_COUNT
    )
    assert forest_row[slot0_count] > 0.0, "forest bird should advance birds_forest goal"

    # Wetland-only bird should be zero for goal slot 0 (birds_forest).
    assert (
        wetland_row[slot0_count] == 0.0
    ), "wetland-only bird should not advance birds_forest goal"

    # Both birds advance total_birds (goal slot 1): count_delta should be nonzero.
    slot1_count = (
        layout._OFF_GOAL_DELTA
        + 1 * layout._GOAL_DELTA_SLOT_DIM
        + layout._GOAL_DELTA_COUNT
    )
    assert forest_row[slot1_count] > 0.0, "forest bird should advance total_birds goal"
    assert (
        wetland_row[slot1_count] > 0.0
    ), "wetland-only bird should also advance total_birds goal"


# ---------------------------------------------------------------------------
# Choice encoder: the bonus_value stripe (the candidate bonus CARD's value to
# the deciding player — board standing plus hand/tray potential)


def test_bonus_value_board_trio_prices_the_candidate_card():
    """A candidate bonus card's row carries its standing board value: the
    qualifying count in play and the stepped / linear VP that count pays,
    matching the scoring functions exactly."""
    eng, *_ = engine.Engine.create(seed=21)
    birds, bonuses, _ = cards.load_all()
    bird_feeder = _named_bonus(bonuses, "Bird Feeder")
    seed_birds = [bird for bird in birds if "Bird Feeder" in bird.bonus_categories]
    assert len(seed_birds) >= 6
    me = eng.state.players[0]
    me.board[cards.Habitat.FOREST] = [
        state.PlayedBird(bird=board_bird) for board_bird in seed_birds[:5]
    ]
    me.board[cards.Habitat.GRASSLAND] = [state.PlayedBird(bird=seed_birds[5])]

    row = encode.encode_choices(_pick_bonus_decision(bird_feeder), eng.state)[0]
    count = scoring.bonus_qualifying_count(me, bird_feeder)
    assert count == 6
    base = layout._OFF_BONUS_VALUE
    assert np.isclose(
        row[base + layout._BONUS_VALUE_QUAL], count / layout._BONUS_COUNT_SCALE
    )
    assert np.isclose(
        row[base + layout._BONUS_VALUE_STEPPED],
        scoring.bonus_score_for_count(bird_feeder, count) / layout._BONUS_VALUE_SCALE,
    )
    assert np.isclose(
        row[base + layout._BONUS_VALUE_LINEAR],
        scoring.bonus_linear_value_for_count(bird_feeder, count)
        / layout._BONUS_VALUE_SCALE,
    )


def test_bonus_value_stepped_vs_linear_on_tiered_card():
    """Between a tiered card's thresholds the stepped channel sits on the lower
    plateau while the linear channel is strictly between the two payouts."""
    eng, *_ = engine.Engine.create(seed=22)
    birds, bonuses, _ = cards.load_all()
    bird_feeder = _named_bonus(bonuses, "Bird Feeder")  # anchors (5, 3), (8, 7)
    seed_birds = [bird for bird in birds if "Bird Feeder" in bird.bonus_categories]
    me = eng.state.players[0]
    # Count 6 sits between the (5, 3) and (8, 7) anchors.
    me.board[cards.Habitat.FOREST] = [
        state.PlayedBird(bird=board_bird) for board_bird in seed_birds[:5]
    ]
    me.board[cards.Habitat.GRASSLAND] = [state.PlayedBird(bird=seed_birds[5])]

    row = encode.encode_choices(_pick_bonus_decision(bird_feeder), eng.state)[0]
    base = layout._OFF_BONUS_VALUE
    plateau = 3.0 / layout._BONUS_VALUE_SCALE
    ceiling = 7.0 / layout._BONUS_VALUE_SCALE
    assert np.isclose(row[base + layout._BONUS_VALUE_STEPPED], plateau)
    assert plateau < row[base + layout._BONUS_VALUE_LINEAR] < ceiling


def test_bonus_value_hand_potential_counts_hand_birds_midgame():
    """An in-game pick (BirdPowerPickBonusCardDecision) counts the qualifying
    birds currently in hand; with an empty board the trio stays zero."""
    eng, *_ = engine.Engine.create(seed=23)
    birds, bonuses, _ = cards.load_all()
    bird_feeder = _named_bonus(bonuses, "Bird Feeder")
    seed_birds = [bird for bird in birds if "Bird Feeder" in bird.bonus_categories]
    non_qualifying = next(
        bird for bird in birds if "Bird Feeder" not in bird.bonus_categories
    )
    me = eng.state.players[0]
    me.hand = [*seed_birds[:3], non_qualifying]

    row = encode.encode_choices(_pick_bonus_decision(bird_feeder), eng.state)[0]
    base = layout._OFF_BONUS_VALUE
    assert np.isclose(
        row[base + layout._BONUS_VALUE_HAND], 3.0 / layout._BONUS_COUNT_SCALE
    )
    assert row[base + layout._BONUS_VALUE_QUAL] == 0.0
    assert row[base + layout._BONUS_VALUE_STEPPED] == 0.0
    assert row[base + layout._BONUS_VALUE_LINEAR] == 0.0


def test_bonus_value_setup_choice_uses_kept_cards():
    """A setup pick's hand potential counts the candidate's kept subset, NOT the
    full dealt hand — at the setup ask the hand still holds every dealt card,
    so counting it would credit birds this pick discards."""
    eng, *_ = engine.Engine.create(seed=24)
    birds, bonuses, _ = cards.load_all()
    bird_feeder = _named_bonus(bonuses, "Bird Feeder")
    seed_birds = [bird for bird in birds if "Bird Feeder" in bird.bonus_categories]
    non_qualifying = next(
        bird for bird in birds if "Bird Feeder" not in bird.bonus_categories
    )
    dealt = [*seed_birds[:4], non_qualifying]  # 4 qualifying birds dealt...
    me = eng.state.players[0]
    me.hand = list(dealt)

    decision = decisions.SetupDecision(
        player_id=0,
        prompt="x",
        choices=[
            decisions.SetupChoice(
                kept_cards=(seed_birds[0], seed_birds[1]),  # ...but only 2 kept
                kept_foods=tuple(cards.ALL_FOODS[:3]),
                bonus_card=bird_feeder,
            )
        ],
        dealt_cards=dealt,
        dealt_bonus=[bird_feeder],
    )
    row = encode.encode_choices(
        decision, eng.state, encode.EncodingSpec(include_setup=True)
    )[0]
    base = layout._OFF_BONUS_VALUE
    assert np.isclose(
        row[base + layout._BONUS_VALUE_HAND], 2.0 / layout._BONUS_COUNT_SCALE
    )
    assert row[base + layout._BONUS_VALUE_QUAL] == 0.0  # setup board is empty


def test_bonus_value_tray_potential_counts_qualifying_tray_birds():
    """The tray potential counts qualifying face-up tray birds only — empty
    slots and non-qualifying birds add nothing."""
    eng, *_ = engine.Engine.create(seed=25)
    birds, bonuses, _ = cards.load_all()
    bird_feeder = _named_bonus(bonuses, "Bird Feeder")
    seed_birds = [bird for bird in birds if "Bird Feeder" in bird.bonus_categories]
    non_qualifying = next(
        bird for bird in birds if "Bird Feeder" not in bird.bonus_categories
    )
    eng.state.tray = [seed_birds[0], non_qualifying, None]

    row = encode.encode_choices(_pick_bonus_decision(bird_feeder), eng.state)[0]
    assert np.isclose(
        row[layout._OFF_BONUS_VALUE + layout._BONUS_VALUE_TRAY],
        1.0 / layout._BONUS_COUNT_SCALE,
    )


def test_bonus_value_zero_for_non_bonus_choice_kinds():
    """Bird-candidate and skip rows never fill the stripe — it prices offered
    bonus CARDS only (bird candidates carry bonus_delta instead)."""
    eng, *_ = engine.Engine.create(seed=26)
    birds, bonuses, _ = cards.load_all()
    bird_feeder = _named_bonus(bonuses, "Bird Feeder")
    qualifying = next(bird for bird in birds if "Bird Feeder" in bird.bonus_categories)
    me = eng.state.players[0]
    me.bonus_cards = [bird_feeder]  # bonus_delta WILL fill; bonus_value must not

    play = decisions.PlayBirdDecision(
        player_id=0, prompt="x", choices=[_play_choice(qualifying)]
    )
    skip = decisions.ActivateTuckDecision(
        player_id=0, prompt="x", choices=[decisions.SkipChoice(label="s")]
    )
    play_row = encode.encode_choices(play, eng.state)[0]
    assert np.any(_bonus_delta_slice(play_row) != 0.0)
    assert np.all(_bonus_value_slice(play_row) == 0.0)
    assert np.all(_bonus_value_slice(encode.encode_choices(skip, eng.state)[0]) == 0.0)


###### PRIVATE #######


def _named_bonus(bonuses: list[cards.BonusCard], name: str) -> cards.BonusCard:
    """The bonus card with printed ``name`` from the loaded catalog."""
    return next(bonus for bonus in bonuses if bonus.name == name)


def _play_choice(bird: cards.Bird) -> decisions.PlayBirdChoice:
    """A play candidate for ``bird`` in one of its printed habitats."""
    return decisions.PlayBirdChoice(
        label=bird.name, bird=bird, habitat=next(iter(bird.habitats))
    )


def _bonus_delta_slice(row: np.ndarray) -> np.ndarray:
    """The bonus_delta stripe of one encoded choice row."""
    return row[
        layout._OFF_BONUS_DELTA : layout._OFF_BONUS_DELTA + layout._BONUS_DELTA_DIM
    ]


def _bonus_value_slice(row: np.ndarray) -> np.ndarray:
    """The bonus_value stripe of one encoded choice row."""
    return row[
        layout._OFF_BONUS_VALUE : layout._OFF_BONUS_VALUE + layout._BONUS_VALUE_DIM
    ]


def _pick_bonus_decision(
    bonus_card: cards.BonusCard,
) -> decisions.BirdPowerPickBonusCardDecision:
    """A one-candidate in-game bonus pick offering ``bonus_card``."""
    return decisions.BirdPowerPickBonusCardDecision(
        player_id=0,
        prompt="x",
        choices=[
            decisions.BonusCardChoice(label=bonus_card.name, bonus_card=bonus_card)
        ],
    )


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
