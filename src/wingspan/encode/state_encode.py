# pyright: reportPrivateUsage=false
# (this encoder reads the shared, package-private layout constants in
# ``layout`` -- a deliberate intra-package coupling, not a privacy break)
"""The state encoder: ``encode_state`` builds the fixed-size game-state
feature vector (POV of the deciding player) by concatenating a fixed,
checkpoint-aligned sequence of per-aspect summary stripes; ``state_size``
reports its length. All the per-aspect summary helpers live below.
"""

from __future__ import annotations

import numpy as np

from wingspan import cards, state
from wingspan.encode import layout


def encode_state(
    state: state.GameState,
    decision: layout._AnyDecision | None = None,
    spec: layout.EncodingSpec = layout.DEFAULT_SPEC,
) -> np.ndarray:
    """Encode the game from the perspective of ``decision.player_id``.

    If ``decision`` is ``None`` we fall back to ``state.current_player`` and
    leave the decision-type stripe zero — useful for value-only inference or
    tests. ``spec`` selects the config-driven shape; only the trailing
    decision-type one-hot's width varies with it. Returns a float32 array of
    length ``state_size(spec)``.
    """
    pov = decision.player_id if decision is not None else state.current_player
    me = state.players[pov]
    opp = state.players[1 - pov] if len(state.players) > 1 else me

    parts: list[np.ndarray] = [
        _summary_turn_state(
            state, me
        ),  # layout.N_PLAYER_TURNS + 1 (turn one-hot + first-player flag)
        _summary_food(me),  # 5
        _summary_food(opp),  # 5
        _board_slots_continuous(
            me
        ),  # layout._BOARD_CONT_STRIPE_DIM — per-slot mutable state
        _board_slots_continuous(
            opp
        ),  # layout._BOARD_CONT_STRIPE_DIM — opponent (public)
        _summary_board(me),  # 18 — kept aggregate
        _summary_board(opp),  # 18 — kept aggregate
        _summary_hand(me),  # 10
        _bonus_progress(
            me
        ),  # 4 * layout._BONUS_ID_DIM — held + count + stepped + linear
        _opp_bonus_count(opp),  # 1 — opponent bonus-card count (hidden identity)
        np.array([len(opp.hand) / layout._HAND_SIZE_SCALE], dtype=np.float32),
        _summary_birdfeeder(state),  # 7 (5 food faces + choice dice + reset flag)
        _summary_misc_scalars(state, me, opp),  # 4 (goal pts ×2, tray size, deck size)
        _round_goals_all_rounds(state, me),  # layout._ROUND_GOALS_STRIPE_DIM
        _card_index_block(me, opp, state),  # layout.N_CARD_INDEX_SLOTS — board+tray ids
        _hand_identity(me),  # layout.HAND_MULTIHOT_DIM — multi-hot of my hand
        _hand_playable(me),  # layout.HAND_MULTIHOT_DIM — playable right now
        _hand_playable_eggs(me),  # layout.HAND_MULTIHOT_DIM — egg-blocked but ready
        _encode_decision_type(
            decision, spec
        ),  # layout.decision_type_dim(spec) (stays last)
    ]
    return np.concatenate(parts).astype(np.float32)


def state_size(spec: layout.EncodingSpec = layout.DEFAULT_SPEC) -> int:
    """Total length of the vector returned by ``encode_state`` for ``spec``.

    Delegates to ``layout.state_feature_dim`` (the single source of truth for the
    cumulative offset chain); only the trailing decision-type one-hot's width is
    spec-dependent."""
    return layout.state_feature_dim(spec)


###### PRIVATE #######

#### State summary helpers ####


def _summary_food(player: state.Player) -> np.ndarray:
    return np.array(
        [player.food[food] / layout._FOOD_INVENTORY_SCALE for food in cards.ALL_FOODS],
        dtype=np.float32,
    )


def _summary_board(player: state.Player) -> np.ndarray:
    parts: list[np.ndarray] = []
    for habitat in cards.ALL_HABITATS:
        row = player.board[habitat]
        parts.append(
            np.array(
                [
                    len(row) / layout._ROW_SLOTS_SCALE,
                    sum(pb.eggs for pb in row) / layout._EGG_COUNT_SCALE,
                    sum(pb.bird.points for pb in row)
                    / (layout._POINTS_SCALE * layout._ROW_SLOTS_SCALE),
                    sum(pb.tucked_cards for pb in row) / layout._TUCKED_SCALE,
                    sum(pb.cached_food.total() for pb in row)
                    / layout._CACHED_FOOD_SCALE,
                    sum(1 for pb in row if pb.bird.color == cards.PowerColor.BROWN)
                    / layout._ROW_SLOTS_SCALE,
                ],
                dtype=np.float32,
            )
        )
    return np.concatenate(parts)


def _summary_hand(player: state.Player) -> np.ndarray:
    """Compact hand summary (10 dims): hand size, per-habitat bird counts, and a
    food+wild multi-hot.

    A bird is counted once per habitat it lives in, so a dual-habitat bird adds to
    two of the three habitat counts. The 6-wide multi-hot flags, for each of the
    five foods and wild, whether *any* card in hand carries that token in its food
    cost — a cheap "can my hand pay this kind of cost?" signal. The specific cards
    held ride the separate hand identity multi-hot (``_hand_identity``).

    Built by combining each card's :func:`_hand_summary_row` — the leading
    ``HAND_SUMMARY_SUM_DIMS`` dims by summation, the food flags by max (OR) — the
    same reduction the model applies to ``card_summary_matrix`` rows, so the
    encoder and the in-model set-summary derivation cannot drift apart."""
    vec = np.zeros(layout.HAND_SUMMARY_DIM, dtype=np.float32)
    sum_dims = layout.HAND_SUMMARY_SUM_DIMS
    for bird in player.hand:
        row = _hand_summary_row(bird)
        vec[:sum_dims] += row[:sum_dims]
        vec[sum_dims:] = np.maximum(vec[sum_dims:], row[sum_dims:])
    return vec


def _hand_summary_row(bird: cards.Bird) -> np.ndarray:
    """One card's contribution to the 10-dim hand/set summary: the 1/scale
    set-size increment, the per-habitat 1/scale increments, then the food-cost
    flags (5 foods + wild). A set of cards combines these rows by summing the
    leading ``HAND_SUMMARY_SUM_DIMS`` dims and max-ing (OR) the flags."""
    row = np.zeros(layout.HAND_SUMMARY_DIM, dtype=np.float32)
    row[0] = 1.0 / layout._HAND_SIZE_SCALE
    for i, habitat in enumerate(cards.ALL_HABITATS):
        if habitat in bird.habitats:
            row[1 + i] = 1.0 / layout._HAND_SIZE_SCALE
    for food_idx in range(layout._FOOD_COST_VEC_DIM):  # 5 foods + wild
        if bird.food_cost.counts[food_idx] > 0:
            row[layout.HAND_SUMMARY_SUM_DIMS + food_idx] = 1.0
    return row


def _hand_identity(player: state.Player) -> np.ndarray:
    """Multi-hot over all core-set birds marking which are in ``player``'s hand.

    Pairs with ``_summary_hand``'s aggregate stats (identity + attributes) so
    every scoring head and the value head can read the *specific* cards held,
    not just their summary. Opponent hands are hidden information, so only the
    POV player's hand is encoded by identity (the opponent contributes its
    size only)."""
    vec = np.zeros(layout._BIRD_ID_DIM, dtype=np.float32)
    for bird in player.hand:
        vec[cards.bird_index(bird)] = 1.0
    return vec


def _hand_playable(player: state.Player) -> np.ndarray:
    """Multi-hot of hand birds playable right now (food, slot, and eggs all met)."""
    from wingspan.engine import playability

    vec = np.zeros(layout._BIRD_ID_DIM, dtype=np.float32)
    playable_now, _ = playability.classify_hand_playability(player)
    for bird in playable_now:
        vec[cards.bird_index(bird)] = 1.0
    return vec


def _hand_playable_eggs(player: state.Player) -> np.ndarray:
    """Multi-hot of hand birds where food is affordable and a slot is open, but
    the egg cost is not yet met."""
    from wingspan.engine import playability

    vec = np.zeros(layout._BIRD_ID_DIM, dtype=np.float32)
    _, playable_if_eggs = playability.classify_hand_playability(player)
    for bird in playable_if_eggs:
        vec[cards.bird_index(bird)] = 1.0
    return vec


def _bonus_progress(player: state.Player) -> np.ndarray:
    """Four POV-only stripes over all core-set bonus cards, keyed by
    ``cards.bonus_index``: which cards ``player`` holds (multi-hot), each held
    card's raw qualifying-bird count, its stepped payoff, and its dense linear
    value. Identity is kept separate so a held card at 0 progress is
    distinguishable from a card not held (both have value 0), letting the value
    head plan toward a card it has not yet begun to fill. The count channel gives
    a gradient even below the first plateau, where stepped and linear are still
    0. Opponent bonus cards are hidden information, so only the POV player's are
    encoded."""
    # Lazy import keeps encode's module-level deps pure (cards / decisions /
    # state) rather than pulling in the whole engine package at import time;
    # this runs once per encode_state (not per candidate), so the cost is nil.
    from wingspan.engine import scoring

    held = np.zeros(layout._BONUS_ID_DIM, dtype=np.float32)
    count = np.zeros(layout._BONUS_ID_DIM, dtype=np.float32)
    stepped = np.zeros(layout._BONUS_ID_DIM, dtype=np.float32)
    linear = np.zeros(layout._BONUS_ID_DIM, dtype=np.float32)
    for bonus_card in player.bonus_cards:
        idx = cards.bonus_index(bonus_card)
        held[idx] = 1.0
        count[idx] = (
            scoring.bonus_qualifying_count(player, bonus_card)
            / layout._BONUS_COUNT_SCALE
        )
        stepped[idx] = (
            scoring.bonus_score(player, bonus_card) / layout._BONUS_VALUE_SCALE
        )
        linear[idx] = (
            scoring.bonus_linear_value(player, bonus_card) / layout._BONUS_VALUE_SCALE
        )
    return np.concatenate([held, count, stepped, linear])


def _opp_bonus_count(opp: state.Player) -> np.ndarray:
    """Opponent bonus-card count as a single scalar. The *identities* of the
    opponent's bonus cards are hidden information, so only how many they hold is
    observable."""
    return np.array(
        [len(opp.bonus_cards) / layout._BONUS_COUNT_SCALE], dtype=np.float32
    )


def _bird_attr_vector(bird: cards.Bird) -> np.ndarray:
    """Dense, normalized view of a bird's immutable card attributes — the rich
    ``N`` half of each board/tray slot's ``(identity one-hot, attributes)``
    encoding.

    Layout (``layout._BIRD_ATTR_DIM`` dims): victory points; food cost 6-vector
    (5 specific foods then wild); nest 4-one-hot (STAR wildcard = all-ones, NONE
    = all-zeros); habitat multi-hot; flocking and predator flags; wingspan; egg
    limit; power-color one-hot (NONE => all-zero); plays-another-bird flag;
    caches-food flag (1 when any power effect caches food); 7-wide multi-hot of
    curated bonus categories; 13-dim power-exchange vector (what the bird's
    ability does in resource terms — same slot semantics as the choice exchange
    stripe, normalized by ``layout._EXCHANGE_SCALE``); or-cost flag (1 when the
    bird's food cost is an OR choice — pay 1 accepted or 2 non-accepted)."""
    vec = np.zeros(layout._BIRD_ATTR_DIM, dtype=np.float32)

    # Victory points and the printed food cost (6-vector: 5 specific, then wild).
    vec[layout._OFF_ATTR_POINTS] = bird.points / layout._POINTS_SCALE
    for i in range(layout._FOOD_COST_VEC_DIM):
        vec[layout._OFF_ATTR_FOOD_COST + i] = (
            bird.food_cost.counts[i] / layout._PER_FOOD_COST_SCALE
        )

    # Nest: 4-way one-hot over the concrete nests; STAR is a wildcard (all ones)
    # and a missing nest leaves the block zero.
    if bird.nest == cards.NestType.STAR:
        vec[
            layout._OFF_ATTR_NEST : layout._OFF_ATTR_NEST + len(layout._NEST_BASE_TYPES)
        ] = 1.0
    else:
        for i, nest in enumerate(layout._NEST_BASE_TYPES):
            if bird.nest == nest:
                vec[layout._OFF_ATTR_NEST + i] = 1.0
                break

    # Habitat multi-hot, the two behavioural flags, and the remaining scalars.
    for i, habitat in enumerate(cards.ALL_HABITATS):
        if habitat in bird.habitats:
            vec[layout._OFF_ATTR_HAB + i] = 1.0
    vec[layout._OFF_ATTR_FLOCK] = 1.0 if bird.flocking else 0.0
    vec[layout._OFF_ATTR_PRED] = 1.0 if bird.predator else 0.0
    vec[layout._OFF_ATTR_WINGSPAN] = bird.wingspan_cm / layout._WINGSPAN_SCALE
    vec[layout._OFF_ATTR_EGG_LIMIT] = bird.egg_limit / layout._EGG_LIMIT_SCALE

    # Power color (one-hot; NONE leaves all zero), plays-another-bird, caches-food.
    for i, color in enumerate(layout._COLORS):
        if bird.color == color:
            vec[layout._OFF_ATTR_COLOR + i] = 1.0
            break
    vec[layout._OFF_ATTR_PLAYS_BIRD] = 1.0 if bird.plays_another_bird else 0.0
    vec[layout._OFF_ATTR_CACHES_FOOD] = 1.0 if _is_caching_bird(bird) else 0.0

    # Curated bonus categories: 7-wide multi-hot over the intrinsic-property
    # subset. Unknown names (dropped categories) return None and are no-ops.
    for category in bird.bonus_categories:
        idx = layout._BONUS_NAME_TO_INDEX.get(category)
        if idx is not None:
            vec[layout._OFF_ATTR_BONUS_CATS + idx] = 1.0

    # Power exchange: what the bird's ability does in resource terms.
    vec[
        layout._OFF_ATTR_POWER_EX : layout._OFF_ATTR_POWER_EX + layout._EXCHANGE_DIM
    ] = _bird_power_exchange_vector(bird)

    # OR-cost flag: 1.0 when the bird's food cost is an OR choice (pay 1
    # accepted food OR 2 non-accepted), 0.0 for standard AND costs.
    vec[layout._OFF_ATTR_OR_COST] = 1.0 if bird.food_cost.is_or_cost else 0.0

    return vec


def _board_slots_continuous(player: state.Player) -> np.ndarray:
    """The continuous per-slot board stripe for ``player``: one fixed slot per
    board position (``N_HABITATS x ROW_SLOTS``), each carrying only the slot's
    mutable state (eggs, egg-capacity remaining, cached food per type, tucked
    cards, activations). The bird's *identity* — and, through the shared card
    table, its static attributes — is emitted separately in the card-index block.
    Empty slots stay zero. Slot order is positional (NOT sorted) so a slot's
    mutable state — and its matching card-index entry — stays bound to the specific
    bird occupying it."""
    vec = np.zeros(layout._BOARD_CONT_STRIPE_DIM, dtype=np.float32)
    for hab_idx, habitat in enumerate(cards.ALL_HABITATS):
        for slot, pb in enumerate(player.board[habitat]):
            if slot >= state.ROW_SLOTS:
                break
            _write_slot_continuous(
                vec, (hab_idx * state.ROW_SLOTS + slot) * layout._SLOT_CONT_DIM, pb
            )
    return vec


def _write_slot_continuous(vec: np.ndarray, base: int, pb: state.PlayedBird) -> None:
    """Write one occupied board slot's mutable features into
    ``vec[base : base + layout._SLOT_CONT_DIM]`` (no identity, no attributes — the
    identity rides the card-index block and its attributes the shared card table)."""
    mut = base + layout._OFF_SLOT_MUT
    vec[mut + layout._SLOT_MUT_EGGS] = pb.eggs / layout._EGG_COUNT_SCALE
    vec[mut + layout._SLOT_MUT_EGG_CAP] = (
        max(pb.bird.egg_limit - pb.eggs, 0) / layout._EGG_COUNT_SCALE
    )
    for i, food in enumerate(cards.ALL_FOODS):
        vec[mut + layout._SLOT_MUT_CACHED + i] = (
            pb.cached_food[food] / layout._CACHED_FOOD_SCALE
        )
    vec[mut + layout._SLOT_MUT_TUCKED] = pb.tucked_cards / layout._TUCKED_SCALE
    vec[mut + layout._SLOT_MUT_ACTIVATIONS] = pb.activations / layout._ACTIVATIONS_SCALE


def card_feature_matrix() -> np.ndarray:
    """The constant ``[HAND_MULTIHOT_DIM + 1, CARD_FEATURE_DIM]`` feature table the
    model's card encoder consumes.

    Row 0 is all zeros — the padding / empty-slot row (``cards.bird_index + 1`` with
    0 meaning "no card"). Row ``bird_index + 1`` is that bird's static attribute
    vector concatenated with its identity one-hot: ``[_bird_attr_vector(bird) (44) ⊕
    one_hot(bird_index) (180)]``. The encoder maps this fixed matrix to the shared
    ``[181, card_embed_dim]`` card table every board / tray / hand / choice slot
    looks up, so a card has exactly one representation, derived from both its
    attributes and a learned per-card component."""
    rows = layout.HAND_MULTIHOT_DIM + 1
    matrix = np.zeros((rows, layout.CARD_FEATURE_DIM), dtype=np.float32)
    for bird in cards.load_all()[0]:
        idx = cards.bird_index(bird)
        matrix[idx + 1, : layout._BIRD_ATTR_DIM] = _bird_attr_vector(bird)
        matrix[idx + 1, layout._BIRD_ATTR_DIM + idx] = 1.0
    return matrix


def card_summary_matrix() -> np.ndarray:
    """The constant ``[HAND_MULTIHOT_DIM + 1, HAND_SUMMARY_DIM]`` per-card summary
    table for deriving a card *set*'s 10-dim summary in-model.

    Row 0 is all zeros — the padding / empty-slot row (``cards.bird_index + 1``
    with 0 meaning "no card"), so an empty slot adds nothing to the set summary.
    Row ``bird_index + 1`` is that bird's :func:`_hand_summary_row`. Reducing the
    selected rows — sum over the leading ``layout.HAND_SUMMARY_SUM_DIMS`` dims,
    max (OR) over the food flags — reproduces ``_summary_hand`` for the same set
    of cards, which is what lets the model feed the shared hand encoder a tray /
    kept-set summary it derives from index columns or a multi-hot alone."""
    rows = layout.HAND_MULTIHOT_DIM + 1
    matrix = np.zeros((rows, layout.HAND_SUMMARY_DIM), dtype=np.float32)
    for bird in cards.load_all()[0]:
        matrix[cards.bird_index(bird) + 1] = _hand_summary_row(bird)
    return matrix


def _card_index_block(
    me: state.Player, opp: state.Player, game_state: state.GameState
) -> np.ndarray:
    """Contiguous integer card indices the model looks up in its shared card
    table: both boards' positional slots (POV then opponent) followed by the
    ``TRAY_SIZE`` public tray slots. Each entry is ``bird_index + 1`` for an
    occupied slot and 0 for an empty one (the card table's zeroed padding row).
    Board and tray slots are both positional: tray slot 0 is the left card,
    slot 2 is the right card."""
    vec = np.zeros(layout.N_CARD_INDEX_SLOTS, dtype=np.float32)
    offset = 0
    for player in (me, opp):
        for hab_idx, habitat in enumerate(cards.ALL_HABITATS):
            for slot, pb in enumerate(player.board[habitat]):
                if slot >= state.ROW_SLOTS:
                    break
                vec[offset + hab_idx * state.ROW_SLOTS + slot] = (
                    cards.bird_index(pb.bird) + 1
                )
        offset += layout._SLOTS_PER_BOARD
    for slot, bird in enumerate(game_state.tray):
        vec[offset + slot] = cards.bird_index(bird) + 1 if bird is not None else 0
    return vec


def _round_goals_all_rounds(
    game_state: state.GameState, me: state.Player
) -> np.ndarray:
    """All four round-goal slots from ``me``'s POV: each = the goal's category
    one-hot plus ``me``'s count, the opponent's count, and the placement VP ``me``
    would earn if that round scored now. Encoding every round (not just the
    current one) lets the model plan toward later-round goals it is already
    accumulating progress on. Already-scored rounds read the frozen
    at-scoring standings (``GameState.scored_goals``) — their stripes never
    move again, however the boards evolve."""
    from wingspan.engine import scoring

    vec = np.zeros(layout._ROUND_GOALS_STRIPE_DIM, dtype=np.float32)
    for round_idx in range(layout._NUM_ROUNDS):
        if round_idx >= len(game_state.round_goals):
            break
        base = round_idx * layout._ROUND_GOAL_SLOT_DIM
        goal = game_state.round_goals[round_idx]
        if goal.category in layout._GOAL_CATEGORIES:
            vec[base + layout._GOAL_CATEGORIES.index(goal.category)] = 1.0
        standing = scoring.round_goal_standing_for_round(game_state, me, round_idx)
        vec[base + layout._ROUND_GOAL_MY_COUNT] = (
            standing.count / layout._GOAL_COUNT_SCALE
        )
        vec[base + layout._ROUND_GOAL_OPP_COUNT] = (
            standing.opp_count / layout._GOAL_COUNT_SCALE
        )
        vec[base + layout._ROUND_GOAL_VP] = (
            standing.vp / layout._ROUND_GOAL_POINTS_SCALE
        )
    return vec


def _summary_birdfeeder(state: state.GameState) -> np.ndarray:
    return np.array(
        [
            *(
                state.birdfeeder.counts[food] / layout._BIRDFEEDER_COUNT_SCALE
                for food in cards.ALL_FOODS
            ),
            state.birdfeeder.choice_dice / layout._BIRDFEEDER_COUNT_SCALE,
            # The optional pre-gain reset (Rule 2) is on offer: every die shows
            # the same face. Derivable from the counts above, but surfaced as an
            # explicit flag so the model reads it directly.
            1.0 if state.birdfeeder.reset_available() else 0.0,
        ],
        dtype=np.float32,
    )


def _summary_turn_state(game_state: state.GameState, me: state.Player) -> np.ndarray:
    """27 dims: 26-dim player-turn one-hot + is_first_player flag.

    The turn one-hot marks which of the player's 26 personal turns (across
    all 4 rounds) they are currently on, counting from 0. It is all-zeros
    during setup (``turn_counter == 0``). The trailing flag is 1.0 when
    ``me`` goes first in the current round, 0.0 when second."""
    out = np.zeros(layout.N_PLAYER_TURNS + 1, dtype=np.float32)

    # 26-dim turn one-hot (all-zeros during setup)
    if game_state.turn_counter != 0:
        player_turn = (
            layout._ROUND_CUBE_OFFSETS[game_state.round_idx]
            + state.ROUND_CUBES[game_state.round_idx]
            - me.action_cubes_left
        )
        out[player_turn] = 1.0

    # Is-first-player flag: 1.0 when me goes first this round.
    out[layout.N_PLAYER_TURNS] = float(
        me.id == (game_state.start_player + game_state.round_idx) % 2
    )
    return out


def _summary_misc_scalars(
    game_state: state.GameState, me: state.Player, opp: state.Player
) -> np.ndarray:
    """4 dims: my round-goal VP, opponent round-goal VP, tray size, deck size."""
    return np.array(
        [
            me.round_goal_points / layout._ROUND_GOAL_POINTS_SCALE,
            opp.round_goal_points / layout._ROUND_GOAL_POINTS_SCALE,
            sum(1 for bird in game_state.tray if bird is not None)
            / layout._TRAY_SIZE_SCALE,
            len(game_state.bird_deck) / layout._DECK_SIZE_SCALE,
        ],
        dtype=np.float32,
    )


def _encode_decision_type(
    decision: layout._AnyDecision | None, spec: layout.EncodingSpec
) -> np.ndarray:
    out = np.zeros(layout.decision_type_dim(spec), dtype=np.float32)
    if decision is not None:
        out[layout._DECISION_TYPE_INDEX[type(decision)]] = 1.0
    return out


#### Power-feature helpers ####

# EffectKinds whose presence means the bird caches food as part of its power.
_CACHE_EFFECT_KINDS: frozenset[cards.EffectKind] = frozenset(
    [
        cards.EffectKind.CACHE_FOOD,
        cards.EffectKind.GAIN_FOOD_FEEDER_MAY_CACHE,
        cards.EffectKind.ROLL_NOT_IN_FEEDER_CACHE,
        cards.EffectKind.PINK_GAIN_FOOD_CACHE,
    ]
)


def _is_caching_bird(bird: cards.Bird) -> bool:
    return any(effect.kind in _CACHE_EFFECT_KINDS for effect in bird.power.effects)


def _bird_power_exchange_vector(bird: cards.Bird) -> np.ndarray:
    """Build the 13-dim power-exchange vector for ``bird``.

    Accumulates each effect's resource contribution into the same slot layout
    as the choice-row exchange stripe, then normalizes by ``_EXCHANGE_SCALE``.
    UNIMPLEMENTED and unknown effects contribute zero (correct default)."""
    vec = np.zeros(layout._EXCHANGE_DIM, dtype=np.float32)
    for effect in bird.power.effects:
        _accumulate_effect_exchange(vec, effect)
    return vec / layout._EXCHANGE_SCALE


def _accumulate_effect_exchange(vec: np.ndarray, effect: cards.Effect) -> None:
    """Add one effect's resource exchange to ``vec`` (unnormalized).

    Each EffectKind maps to zero or more exchange slots; compound kinds (e.g.
    TUCK_FROM_HAND_THEN_DRAW) set both cost and gain slots. Pink reactive kinds
    use the self-gain slots (the reacting player's perspective). Conditional
    or partial-probability effects (PREDATOR_HUNT, FEWEST_*, ROLL_NOT_IN_FEEDER_*)
    are mapped by their nominal exchange — the model learns to discount uncertain
    outcomes via training, not by zeroing the signal here."""
    amount = effect.amount
    kind = effect.kind

    # Food gains (from supply, feeder, die, or compound tuck-then-gain)
    if kind in (
        cards.EffectKind.GAIN_FOOD_SUPPLY,
        cards.EffectKind.GAIN_FOOD_BIRDFEEDER,
        cards.EffectKind.GAIN_FOOD_FROM_FEEDER_CHOICE,
        cards.EffectKind.GAIN_DIE_ANY,
        cards.EffectKind.GAIN_ALL_FOOD_FEEDER,
        cards.EffectKind.FEWEST_FOREST_GAINS_DIE,
        cards.EffectKind.FEWEST_WETLAND_DRAWS_CARD,
    ):
        vec[layout._EXCHANGE_FOOD_TO_GAIN] += amount

    elif kind in (
        cards.EffectKind.TUCK_FROM_HAND_THEN_GAIN_FOOD_SUPPLY,
        cards.EffectKind.TUCK_FROM_HAND_THEN_GAIN_FOOD_CHOICE,
    ):
        vec[layout._EXCHANGE_CARDS_TO_DISCARD] += 1
        vec[layout._EXCHANGE_FOOD_TO_GAIN] += amount

    # Cache gains (food cached on the bird itself)
    elif kind in (
        cards.EffectKind.CACHE_FOOD,
        cards.EffectKind.ROLL_NOT_IN_FEEDER_CACHE,
        cards.EffectKind.GAIN_FOOD_FEEDER_MAY_CACHE,
        cards.EffectKind.PINK_GAIN_FOOD_CACHE,
    ):
        vec[layout._EXCHANGE_CACHE_TO_GAIN] += amount

    # Egg gains
    elif kind in (
        cards.EffectKind.LAY_EGG_ON_THIS,
        cards.EffectKind.LAY_EGG_ANY,
    ):
        vec[layout._EXCHANGE_EGGS_TO_GAIN] += amount

    elif kind in (
        cards.EffectKind.TUCK_FROM_HAND_THEN_LAY_ON_THIS,
        cards.EffectKind.TUCK_FROM_HAND_THEN_LAY_ANY,
    ):
        vec[layout._EXCHANGE_CARDS_TO_DISCARD] += 1
        vec[layout._EXCHANGE_EGGS_TO_GAIN] += amount

    # Card draws
    elif kind in (
        cards.EffectKind.DRAW_CARDS,
        cards.EffectKind.DRAW_FROM_TRAY_ALL,
        cards.EffectKind.DRAW_N_PLUS_ONE_DRAFT,
        cards.EffectKind.DRAW_CARDS_THEN_DISCARD_EOT,
    ):
        vec[layout._EXCHANGE_CARDS_TO_DRAW] += amount

    elif kind == cards.EffectKind.TUCK_FROM_HAND_THEN_DRAW:
        vec[layout._EXCHANGE_CARDS_TO_DISCARD] += 1
        vec[layout._EXCHANGE_CARDS_TO_DRAW] += amount

    # Cards tucked (from deck onto a bird)
    elif kind in (
        cards.EffectKind.TUCK_FROM_DECK,
        cards.EffectKind.TUCK_FROM_DECK_PAID,
    ):
        if kind == cards.EffectKind.TUCK_FROM_DECK_PAID:
            vec[layout._EXCHANGE_EGGS_TO_PAY] += 1
        vec[layout._EXCHANGE_CARDS_TO_TUCK] += amount

    # Cards discarded from hand (tuck-from-hand as a cost with no secondary)
    elif kind == cards.EffectKind.TUCK_FROM_HAND:
        vec[layout._EXCHANGE_CARDS_TO_DISCARD] += 1

    # Egg-cost exchanges
    elif kind == cards.EffectKind.DISCARD_EGG_FOR_CARDS:
        vec[layout._EXCHANGE_EGGS_TO_PAY] += 1
        vec[layout._EXCHANGE_CARDS_TO_DRAW] += amount

    elif kind == cards.EffectKind.DISCARD_EGG_FOR_WILD:
        vec[layout._EXCHANGE_EGGS_TO_PAY] += 1
        vec[layout._EXCHANGE_FOOD_TO_GAIN] += 1

    # Wild food trade (net zero food but signals a conversion power)
    elif kind == cards.EffectKind.TRADE_WILD_FOOD:
        vec[layout._EXCHANGE_FOOD_TO_PAY] += 1
        vec[layout._EXCHANGE_FOOD_TO_GAIN] += 1

    # Extra bird plays
    elif kind in (
        cards.EffectKind.PLAY_ADDITIONAL_BIRD,
        cards.EffectKind.PLAY_ADDITIONAL_BIRD_HERE,
    ):
        vec[layout._EXCHANGE_PLAYS_TO_GAIN] += 1

    # All-players effects: self-gain + opponent-gain
    elif kind == cards.EffectKind.ALL_PLAYERS_GAIN_FOOD:
        vec[layout._EXCHANGE_FOOD_TO_GAIN] += amount
        vec[layout._EXCHANGE_OPP_FOOD_TO_GAIN] += amount

    elif kind == cards.EffectKind.EACH_PLAYER_GAINS_DIE_CHOOSE_ORDER:
        vec[layout._EXCHANGE_FOOD_TO_GAIN] += 1
        vec[layout._EXCHANGE_OPP_FOOD_TO_GAIN] += 1

    elif kind == cards.EffectKind.ALL_PLAYERS_DRAW:
        vec[layout._EXCHANGE_CARDS_TO_DRAW] += amount
        vec[layout._EXCHANGE_OPP_CARDS_TO_DRAW] += amount

    elif kind == cards.EffectKind.ALL_PLAYERS_LAY_EGG_ON_NEST:
        vec[layout._EXCHANGE_EGGS_TO_GAIN] += amount
        vec[layout._EXCHANGE_OPP_EGGS_TO_GAIN] += amount

    elif kind == cards.EffectKind.LAY_EGG_ALL_NEST:
        vec[layout._EXCHANGE_EGGS_TO_GAIN] += amount
        vec[layout._EXCHANGE_OPP_EGGS_TO_GAIN] += amount

    # Pink reactive effects (reacting player's gain)
    elif kind == cards.EffectKind.PINK_PLAY_BIRD_GAIN:
        vec[layout._EXCHANGE_FOOD_TO_GAIN] += amount

    elif kind == cards.EffectKind.PINK_PLAY_BIRD_TUCK:
        vec[layout._EXCHANGE_CARDS_TO_TUCK] += amount

    elif kind == cards.EffectKind.PINK_LAY_EGG_ON_NEST:
        vec[layout._EXCHANGE_EGGS_TO_GAIN] += amount
