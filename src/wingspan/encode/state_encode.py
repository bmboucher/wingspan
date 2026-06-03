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
        _summary_birdfeeder(state),  # 6 (5 food faces + choice dice)
        _summary_misc_scalars(state, me, opp),  # 7
        _round_goals_all_rounds(state, me, opp),  # layout._ROUND_GOALS_STRIPE_DIM
        _card_index_block(me, opp, state),  # layout.N_CARD_INDEX_SLOTS — board+tray ids
        _hand_identity(me),  # layout.HAND_MULTIHOT_DIM — multi-hot of my hand
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
    held ride the separate hand identity multi-hot (``_hand_identity``)."""
    vec = np.zeros(10, dtype=np.float32)
    vec[0] = len(player.hand) / layout._HAND_SIZE_SCALE
    for bird in player.hand:
        for i, habitat in enumerate(cards.ALL_HABITATS):
            if habitat in bird.habitats:
                vec[1 + i] += 1.0 / layout._HAND_SIZE_SCALE
        for food_idx in range(layout._FOOD_COST_VEC_DIM):  # 5 foods + wild
            if bird.food_cost.counts[food_idx] > 0:
                vec[4 + food_idx] = 1.0
    return vec


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

    Layout (``layout._BIRD_ATTR_DIM`` dims): victory points; food cost 6-vector (5
    specific foods then wild); nest 4-one-hot (a STAR nest is a wildcard encoded
    all-ones, NONE all-zeros); habitat multi-hot; flocking and predator flags;
    wingspan; egg limit; power-color one-hot (NONE => all-zero); swift-start
    flag; and a 26-wide multi-hot of the bonus-card categories the bird
    statically qualifies for (the "test" predicates such as 'named after a
    person' or a wingspan threshold), keyed to the same ``bonus_index`` space as
    the bonus-progress stripes."""
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

    # Power color (one-hot; NONE leaves all zero) and the swift-start flag.
    for i, color in enumerate(layout._COLORS):
        if bird.color == color:
            vec[layout._OFF_ATTR_COLOR + i] = 1.0
            break
    vec[layout._OFF_ATTR_SWIFT] = 1.0 if bird.is_swift_start else 0.0

    # "Test" predicates: the bonus cards this bird statically qualifies for.
    for category in bird.bonus_categories:
        idx = layout._BONUS_NAME_TO_INDEX.get(category)
        if idx is not None:
            vec[layout._OFF_ATTR_BONUS_CATS + idx] = 1.0

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
    vector concatenated with its identity one-hot: ``[_bird_attr_vector(bird) (49) ⊕
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


def _card_index_block(
    me: state.Player, opp: state.Player, game_state: state.GameState
) -> np.ndarray:
    """Contiguous integer card indices the model looks up in its shared card
    table: both boards' positional slots (POV then opponent) followed by the
    up-to-``TRAY_SIZE`` public tray. Each entry is ``bird_index + 1`` for an
    occupied slot and 0 for an empty one (the card table's zeroed padding row).
    Board slots are positional (matching ``_board_slots_continuous``); tray birds
    are sorted by ``bird_index`` so the tray encoding is order-invariant."""
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
    for slot, bird in enumerate(sorted(game_state.tray, key=cards.bird_index)):
        if slot >= state.TRAY_SIZE:
            break
        vec[offset + slot] = cards.bird_index(bird) + 1
    return vec


def _round_goals_all_rounds(
    game_state: state.GameState, me: state.Player, opp: state.Player
) -> np.ndarray:
    """All four round-goal slots from ``me``'s POV: each = the goal's category
    one-hot plus ``me``'s count, the opponent's count, and the placement VP ``me``
    would earn if that round scored now. Encoding every round (not just the
    current one) lets the model plan toward later-round goals it is already
    accumulating progress on."""
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
            scoring.eval_goal(opp, goal) / layout._GOAL_COUNT_SCALE
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
        ],
        dtype=np.float32,
    )


def _summary_misc_scalars(
    state: state.GameState, me: state.Player, opp: state.Player
) -> np.ndarray:
    return np.array(
        [
            state.round_idx / 3.0,
            me.action_cubes_left / layout._ACTION_CUBES_SCALE,
            opp.action_cubes_left / layout._ACTION_CUBES_SCALE,
            me.round_goal_points / layout._ROUND_GOAL_POINTS_SCALE,
            opp.round_goal_points / layout._ROUND_GOAL_POINTS_SCALE,
            len(state.tray) / layout._TRAY_SIZE_SCALE,
            len(state.bird_deck) / layout._DECK_SIZE_SCALE,
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
