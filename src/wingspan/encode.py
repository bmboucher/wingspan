"""State and per-choice encoders for RL.

Two responsibilities live here, and they meet in the model:

* ``encode_state`` produces a fixed-size dense feature vector summarizing the
  game from the perspective of the player who is about to decide (not
  necessarily ``state.current_player`` — opponent prompts during a "each
  player chooses" power must encode from that player's POV). The state vector
  is concatenated with a one-hot ``DecisionType`` so the trunk knows which
  decision is being asked.

* ``encode_choices`` produces a ``(n_choices, CHOICE_FEATURE_DIM)`` matrix
  describing each legal choice with structured features. Unlike the old
  positional-slot encoding, slot N here means "the N-th candidate at this
  decision" *with its own attribute vector* — the network scores each
  candidate as ``(state, choice_features[i])`` and the action space becomes
  implicitly variable.

The featurizer dispatches on ``DecisionType`` so each branch can pull the
right structure out of ``Choice.payload``: a ``cards.Bird`` exposes its costs and
power color, a board-target ``(habitat, slot)`` is looked up on the asking
player's board for current egg/cache state, a payment dict exposes its food
composition, and so on. Unused slots stay zero.
"""

from __future__ import annotations

import logging
import typing

import numpy as np

from wingspan import cards, decisions, state

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public constants — sanity bounds + normalization scales

# Choice-count safety bounds. The new encoder no longer truncates: every
# choice gets a feature row. ``SOFT_CHOICE_WARN_THRESHOLD`` produces a one-off
# log entry per call site if a decision balloons unexpectedly large; the
# hard cap is a defensive assert that should never trip in real play.
SOFT_CHOICE_WARN_THRESHOLD = 20
# Hard cap on the per-decision choice count. The setup decision
# (``SETUP_CHOOSE_HAND_FOOD_BONUS``) intentionally enumerates all 504
# combinations for the standard 5-card / 2-bonus deal, so the cap sits a
# little above that to leave room for future deal variants while still
# catching runaway choice generation.
MAX_CHOICES_HARD = 600

# Goal-category one-hot length (mirrors the round-goal stripe).
MAX_GOAL_CATEGORIES = 18

# Normalization scales for raw card / board values. Picked so most values
# land in roughly [0, 1.5]; the network can rescale internally if needed.
_POINTS_SCALE = 9.0
_FOOD_COST_SCALE = 7.0
_EGG_LIMIT_SCALE = 6.0
_WINGSPAN_SCALE = 200.0
_PER_FOOD_COST_SCALE = 3.0
_ROW_SLOTS_SCALE = 5.0
_EGG_COUNT_SCALE = 6.0
_CACHED_FOOD_SCALE = 6.0
_TUCKED_SCALE = 6.0
_ACTION_CUBES_SCALE = 8.0
_ROUND_GOAL_POINTS_SCALE = 10.0
_PAYMENT_COUNT_SCALE = 4.0
_DECK_SIZE_SCALE = 100.0
_TRAY_SIZE_SCALE = 3.0
_HAND_SIZE_SCALE = 10.0
_BIRDFEEDER_COUNT_SCALE = 5.0
_FOOD_INVENTORY_SCALE = 6.0
_PLAYER_ID_SCALE = 4.0  # MainAction encoded index normalizer

# ---------------------------------------------------------------------------
# Choice feature layout
#
# A single uniform feature vector with type-specific stripes. Each branch in
# ``_featurize_choice`` fills only the stripes relevant to that decision
# type; the rest stay zero.

_KIND_DIM = 6  # bird, food, habitat, payment, board_target, special
_BIRD_DIM = 21  # numeric attributes + color/nest one-hots + per-food cost
_FOOD_DIM = 5  # food one-hot
_HABITAT_DIM = 3  # habitat one-hot
_PAYMENT_DIM = 5  # count per food
_BOARD_TARGET_DIM = 8  # habitat (3), slot, eggs, capacity_remaining, cached, tucked
_SPECIAL_DIM = 3  # is_skip, encoded_slot/4, setup_is_keep

CHOICE_FEATURE_DIM = (
    _KIND_DIM
    + _BIRD_DIM
    + _FOOD_DIM
    + _HABITAT_DIM
    + _PAYMENT_DIM
    + _BOARD_TARGET_DIM
    + _SPECIAL_DIM
)

# Stripe offsets (cumulative)
_OFF_KIND = 0
_OFF_BIRD = _OFF_KIND + _KIND_DIM
_OFF_FOOD = _OFF_BIRD + _BIRD_DIM
_OFF_HAB = _OFF_FOOD + _FOOD_DIM
_OFF_PAY = _OFF_HAB + _HABITAT_DIM
_OFF_BOARD = _OFF_PAY + _PAYMENT_DIM
_OFF_SPECIAL = _OFF_BOARD + _BOARD_TARGET_DIM

# Within-KIND indices
_KIND_BIRD = 0
_KIND_FOOD = 1
_KIND_HABITAT = 2
_KIND_PAYMENT = 3
_KIND_BOARD_TARGET = 4
_KIND_SPECIAL = 5

# Within-SPECIAL indices
_SPECIAL_IS_SKIP = 0
_SPECIAL_ENCODED_SLOT = 1
_SPECIAL_IS_KEEP = 2


# ---------------------------------------------------------------------------
# Decision-type one-hot. Indexed by Decision subclass so adding a new
# decision is a single registration in ``ALL_DECISION_CLASSES``.

DECISION_TYPE_DIM = len(decisions.ALL_DECISION_CLASSES)
_DECISION_TYPE_INDEX: dict[type[decisions.Decision[typing.Any]], int] = {
    cls: i for i, cls in enumerate(decisions.ALL_DECISION_CLASSES)
}

_AnyDecision = decisions.Decision[typing.Any]
_ChoiceFeaturizer = typing.Callable[..., None]


# ---------------------------------------------------------------------------
# Stable global ordering of goal categories

_GOAL_CATEGORIES = [
    "birds_forest",
    "birds_grassland",
    "birds_wetland",
    "eggs_forest",
    "eggs_grassland",
    "eggs_wetland",
    "eggs_bowl",
    "eggs_cavity",
    "eggs_ground",
    "eggs_platform",
    "bowl_birds_with_eggs",
    "cavity_birds_with_eggs",
    "ground_birds_with_eggs",
    "platform_birds_with_eggs",
    "tucked_cards",
    "wingspan_under_30",
    "wingspan_over_65",
]


# ---------------------------------------------------------------------------
# Public API


def encode_state(
    state: state.GameState, decision: _AnyDecision | None = None
) -> np.ndarray:
    """Encode the game from the perspective of ``decision.player_id``.

    If ``decision`` is ``None`` we fall back to ``state.current_player`` and
    leave the decision-type stripe zero — useful for value-only inference or
    tests. Returns a float32 array of length ``state_size()``.
    """
    pov = decision.player_id if decision is not None else state.current_player
    me = state.players[pov]
    opp = state.players[1 - pov] if len(state.players) > 1 else me

    parts: list[np.ndarray] = [
        _summary_food(me),  # 5
        _summary_food(opp),  # 5
        _summary_board(me),  # 18
        _summary_board(opp),  # 18
        _summary_hand(me),  # 8
        np.array([len(opp.hand) / _HAND_SIZE_SCALE], dtype=np.float32),
        _summary_birdfeeder(state),  # 5
        _summary_misc_scalars(state, me, opp),  # 7
        _summary_round_goal(state),  # MAX_GOAL_CATEGORIES
        _encode_decision_type(decision),  # DECISION_TYPE_DIM
    ]
    return np.concatenate(parts).astype(np.float32)


def state_size() -> int:
    """Total length of the vector returned by ``encode_state``."""
    return 5 + 5 + 18 + 18 + 8 + 1 + 5 + 7 + MAX_GOAL_CATEGORIES + DECISION_TYPE_DIM


def encode_choices(decision: _AnyDecision, state: state.GameState) -> np.ndarray:
    """Featurize every choice in ``decision``.

    Returns a float32 array of shape ``(n_choices, CHOICE_FEATURE_DIM)``.
    All returned rows correspond to legal choices — there is no padding or
    truncation, and the action space is implicitly variable across decisions.
    The caller (training loop) handles batched padding + masking.
    """
    n = len(decision.choices)
    decision_name = type(decision).__name__
    assert n > 0, f"empty Decision: {decision_name}"
    assert n <= MAX_CHOICES_HARD, (
        f"decision {decision_name} produced {n} choices, "
        f"exceeds MAX_CHOICES_HARD={MAX_CHOICES_HARD}"
    )
    if n > SOFT_CHOICE_WARN_THRESHOLD:
        logger.warning(
            "Decision %s exposes %d choices (> %d soft threshold) for player %d",
            decision_name,
            n,
            SOFT_CHOICE_WARN_THRESHOLD,
            decision.player_id,
        )
    feats = np.zeros((n, CHOICE_FEATURE_DIM), dtype=np.float32)
    for i, choice in enumerate(decision.choices):
        feats[i] = _featurize_choice(decision, choice, state)
    return feats


# Back-compat shim: legacy callers expect ``encode_decision``. The semantics
# changed (per-choice features instead of a global mask), so the return is
# different — callers that consumed the old (mask, action_ids) tuple need to
# be updated.
def encode_decision(decision: _AnyDecision, state: state.GameState) -> np.ndarray:
    """Alias for :func:`encode_choices`."""
    return encode_choices(decision, state)


###### PRIVATE #######

#### State summary helpers ####


def _summary_food(player: state.Player) -> np.ndarray:
    return np.array(
        [player.food[f] / _FOOD_INVENTORY_SCALE for f in cards.ALL_FOODS],
        dtype=np.float32,
    )


def _summary_board(player: state.Player) -> np.ndarray:
    parts: list[np.ndarray] = []
    for h in cards.ALL_HABITATS:
        row = player.board[h]
        parts.append(
            np.array(
                [
                    len(row) / _ROW_SLOTS_SCALE,
                    sum(pb.eggs for pb in row) / _EGG_COUNT_SCALE,
                    sum(pb.bird.points for pb in row)
                    / (_POINTS_SCALE * _ROW_SLOTS_SCALE),
                    sum(pb.tucked_cards for pb in row) / _TUCKED_SCALE,
                    sum(pb.cached_food for pb in row) / _CACHED_FOOD_SCALE,
                    sum(1 for pb in row if pb.bird.color == cards.PowerColor.BROWN)
                    / _ROW_SLOTS_SCALE,
                ],
                dtype=np.float32,
            )
        )
    return np.concatenate(parts)


def _summary_hand(player: state.Player) -> np.ndarray:
    if not player.hand:
        return np.zeros(8, dtype=np.float32)
    pts = [b.points for b in player.hand]
    costs = [b.food_cost.total for b in player.hand]
    eggs = [b.egg_limit for b in player.hand]
    return np.array(
        [
            len(player.hand) / _HAND_SIZE_SCALE,
            float(np.mean(pts)) / _POINTS_SCALE,
            float(np.max(pts)) / _POINTS_SCALE,
            float(np.mean(costs)) / _FOOD_COST_SCALE,
            float(np.min(costs)) / _FOOD_COST_SCALE,
            float(np.mean(eggs)) / _EGG_LIMIT_SCALE,
            sum(1 for b in player.hand if cards.Habitat.FOREST in b.habitats)
            / _HAND_SIZE_SCALE,
            sum(1 for b in player.hand if cards.Habitat.WETLAND in b.habitats)
            / _HAND_SIZE_SCALE,
        ],
        dtype=np.float32,
    )


def _summary_birdfeeder(state: state.GameState) -> np.ndarray:
    return np.array(
        [state.birdfeeder.counts[f] / _BIRDFEEDER_COUNT_SCALE for f in cards.ALL_FOODS],
        dtype=np.float32,
    )


def _summary_misc_scalars(
    state: state.GameState, me: state.Player, opp: state.Player
) -> np.ndarray:
    return np.array(
        [
            state.round_idx / 3.0,
            me.action_cubes_left / _ACTION_CUBES_SCALE,
            opp.action_cubes_left / _ACTION_CUBES_SCALE,
            me.round_goal_points / _ROUND_GOAL_POINTS_SCALE,
            opp.round_goal_points / _ROUND_GOAL_POINTS_SCALE,
            len(state.tray) / _TRAY_SIZE_SCALE,
            len(state.bird_deck) / _DECK_SIZE_SCALE,
        ],
        dtype=np.float32,
    )


def _summary_round_goal(state: state.GameState) -> np.ndarray:
    rg = np.zeros(MAX_GOAL_CATEGORIES, dtype=np.float32)
    if 0 <= state.round_idx < len(state.round_goals):
        cat = state.round_goals[state.round_idx].category
        if cat in _GOAL_CATEGORIES:
            rg[_GOAL_CATEGORIES.index(cat)] = 1.0
    return rg


def _encode_decision_type(decision: _AnyDecision | None) -> np.ndarray:
    out = np.zeros(DECISION_TYPE_DIM, dtype=np.float32)
    if decision is not None:
        out[_DECISION_TYPE_INDEX[type(decision)]] = 1.0
    return out


#### Per-choice featurization ####
#
# Dispatch is by the concrete ``Choice`` subclass. Each handler reads typed
# fields directly off the choice rather than unpacking an opaque payload.
# A few decisions need the surrounding Decision for context (the setup
# decision exposes ``dealt_cards``; ``DrawSourceChoice`` looks up tray
# contents from game state); that's passed in as an extra argument.


def _featurize_choice(
    decision: _AnyDecision,
    choice: decisions.Choice,
    state: state.GameState,
) -> np.ndarray:
    """Fill a CHOICE_FEATURE_DIM vector for one (decision, choice) pair."""
    feat = np.zeros(CHOICE_FEATURE_DIM, dtype=np.float32)
    handler = _CHOICE_FEATURIZERS.get(type(choice), _featurize_default)
    handler(feat, decision, choice, state)
    return feat


def _featurize_default(
    feat: np.ndarray,
    decision: _AnyDecision,
    choice: decisions.Choice,
    state: state.GameState,
) -> None:
    feat[_OFF_KIND + _KIND_SPECIAL] = 1.0


def _featurize_skip(
    feat: np.ndarray,
    decision: _AnyDecision,
    choice: decisions.SkipChoice,
    state: state.GameState,
) -> None:
    feat[_OFF_KIND + _KIND_SPECIAL] = 1.0
    feat[_OFF_SPECIAL + _SPECIAL_IS_SKIP] = 1.0


def _featurize_pay_cost(
    feat: np.ndarray,
    decision: _AnyDecision,
    choice: decisions.PayCostChoice,
    state: state.GameState,
) -> None:
    # The 'accept the offered cost' branch is distinct from skip — the
    # network can learn to prefer or avoid it independently.
    feat[_OFF_KIND + _KIND_SPECIAL] = 1.0


def _featurize_main_action(
    feat: np.ndarray,
    decision: _AnyDecision,
    choice: decisions.MainActionChoice,
    state: state.GameState,
) -> None:
    feat[_OFF_KIND + _KIND_SPECIAL] = 1.0
    feat[_OFF_SPECIAL + _SPECIAL_ENCODED_SLOT] = (
        _MAIN_ACTION_INDEX[choice.action] / _PLAYER_ID_SCALE
    )


def _featurize_bird(
    feat: np.ndarray,
    decision: _AnyDecision,
    choice: decisions.BirdChoice,
    state: state.GameState,
) -> None:
    feat[_OFF_KIND + _KIND_BIRD] = 1.0
    _fill_bird(feat, choice.bird)


def _featurize_played_bird(
    feat: np.ndarray,
    decision: _AnyDecision,
    choice: decisions.PlayedBirdChoice,
    state: state.GameState,
) -> None:
    pb = choice.played_bird
    feat[_OFF_KIND + _KIND_BIRD] = 1.0
    _fill_bird(feat, pb.bird)
    # Surface board-target dynamic state too (eggs/cache/tucked) even
    # though we don't know its row index here.
    feat[_OFF_BOARD + 4] = pb.eggs / _EGG_COUNT_SCALE
    cap = max(pb.bird.egg_limit - pb.eggs, 0)
    feat[_OFF_BOARD + 5] = cap / _EGG_COUNT_SCALE
    feat[_OFF_BOARD + 6] = pb.cached_food / _CACHED_FOOD_SCALE
    feat[_OFF_BOARD + 7] = pb.tucked_cards / _TUCKED_SCALE


def _featurize_habitat(
    feat: np.ndarray,
    decision: _AnyDecision,
    choice: decisions.HabitatChoice,
    state: state.GameState,
) -> None:
    feat[_OFF_KIND + _KIND_HABITAT] = 1.0
    _fill_habitat(feat, choice.habitat)


def _featurize_food(
    feat: np.ndarray,
    decision: _AnyDecision,
    choice: decisions.FoodChoice,
    state: state.GameState,
) -> None:
    feat[_OFF_KIND + _KIND_FOOD] = 1.0
    _fill_food(feat, choice.food)


def _featurize_food_payment(
    feat: np.ndarray,
    decision: _AnyDecision,
    choice: decisions.FoodPaymentChoice,
    state: state.GameState,
) -> None:
    feat[_OFF_KIND + _KIND_PAYMENT] = 1.0
    _fill_payment(feat, choice.payment)


def _featurize_board_target(
    feat: np.ndarray,
    decision: _AnyDecision,
    choice: decisions.BoardTargetChoice,
    state: state.GameState,
) -> None:
    feat[_OFF_KIND + _KIND_BOARD_TARGET] = 1.0
    _fill_board_target(feat, choice.habitat, choice.slot, state, decision.player_id)


def _featurize_bonus_card(
    feat: np.ndarray,
    decision: _AnyDecision,
    choice: decisions.BonusCardChoice,
    state: state.GameState,
) -> None:
    # No bonus-card embedding yet; encode identity via id hash so distinct
    # bonus options aren't collapsed.
    feat[_OFF_KIND + _KIND_SPECIAL] = 1.0
    feat[_OFF_SPECIAL + _SPECIAL_IS_KEEP] = (choice.bonus_card.id % 16) / 16.0


def _featurize_draw_source(
    feat: np.ndarray,
    decision: _AnyDecision,
    choice: decisions.DrawSourceChoice,
    state: state.GameState,
) -> None:
    if (
        choice.source == "tray"
        and choice.tray_index is not None
        and 0 <= choice.tray_index < len(state.tray)
    ):
        feat[_OFF_KIND + _KIND_BIRD] = 1.0
        _fill_bird(feat, state.tray[choice.tray_index])
    else:
        feat[_OFF_KIND + _KIND_SPECIAL] = 1.0


def _featurize_player_id(
    feat: np.ndarray,
    decision: _AnyDecision,
    choice: decisions.PlayerIdChoice,
    state: state.GameState,
) -> None:
    # Flag whether the choice means "me" so the network can learn
    # self-vs-opponent preference cheaply.
    feat[_OFF_KIND + _KIND_SPECIAL] = 1.0
    feat[_OFF_SPECIAL + _SPECIAL_IS_KEEP] = (
        1.0 if choice.player_id == decision.player_id else 0.0
    )


def _featurize_setup(
    feat: np.ndarray,
    decision: _AnyDecision,
    choice: decisions.SetupChoice,
    state: state.GameState,
) -> None:
    """Featurize a single combined setup pick.

    The 504 candidates share a state vector, so the network has to read the
    choice features to distinguish them. We surface (a) aggregate stats of
    the kept-card subset, (b) a multi-hot of foods spent in the PAYMENT
    stripe, and (c) a normalized bonus-card id in the SPECIAL stripe so the
    bonus-pick dimension isn't collapsed.
    """
    feat[_OFF_KIND + _KIND_SPECIAL] = 1.0
    # PAY stripe encodes the foods *spent* (complement of kept_foods), so the
    # network sees the same payment signal as every other paying decision.
    for i, food in enumerate(cards.ALL_FOODS):
        if food not in choice.kept_foods:
            feat[_OFF_PAY + i] = 1.0 / _PAYMENT_COUNT_SCALE
    kept = choice.kept_cards
    if kept:
        feat[_OFF_BIRD + 0] = sum(b.points for b in kept) / (
            _POINTS_SCALE * _ROW_SLOTS_SCALE
        )
        feat[_OFF_BIRD + 1] = sum(b.food_cost.total for b in kept) / (
            _FOOD_COST_SCALE * _ROW_SLOTS_SCALE
        )
        feat[_OFF_BIRD + 3] = sum(b.egg_limit for b in kept) / (
            _EGG_LIMIT_SCALE * _ROW_SLOTS_SCALE
        )
    feat[_OFF_BIRD + 4] = len(kept) / _ROW_SLOTS_SCALE
    if choice.bonus_card is not None:
        feat[_OFF_SPECIAL + _SPECIAL_IS_KEEP] = (choice.bonus_card.id % 16) / 16.0


_MAIN_ACTION_INDEX: dict[decisions.MainAction, int] = {
    decisions.MainAction.PLAY_BIRD: 0,
    decisions.MainAction.GAIN_FOOD: 1,
    decisions.MainAction.LAY_EGGS: 2,
    decisions.MainAction.DRAW_CARDS: 3,
}


_CHOICE_FEATURIZERS: dict[type[decisions.Choice], _ChoiceFeaturizer] = {
    decisions.SkipChoice: _featurize_skip,
    decisions.PayCostChoice: _featurize_pay_cost,
    decisions.MainActionChoice: _featurize_main_action,
    decisions.BirdChoice: _featurize_bird,
    decisions.PlayedBirdChoice: _featurize_played_bird,
    decisions.HabitatChoice: _featurize_habitat,
    decisions.FoodChoice: _featurize_food,
    decisions.FoodPaymentChoice: _featurize_food_payment,
    decisions.BoardTargetChoice: _featurize_board_target,
    decisions.BonusCardChoice: _featurize_bonus_card,
    decisions.DrawSourceChoice: _featurize_draw_source,
    decisions.PlayerIdChoice: _featurize_player_id,
    decisions.SetupChoice: _featurize_setup,
}


#### Stripe fillers ####

_COLORS = [
    cards.PowerColor.BROWN,
    cards.PowerColor.WHITE,
    cards.PowerColor.PINK,
    cards.PowerColor.YELLOW,
]
_NESTS = [
    cards.NestType.BOWL,
    cards.NestType.CAVITY,
    cards.NestType.GROUND,
    cards.NestType.PLATFORM,
    cards.NestType.STAR,
]


def _fill_bird(feat: np.ndarray, bird: cards.Bird) -> None:
    o = _OFF_BIRD
    feat[o + 0] = bird.points / _POINTS_SCALE
    feat[o + 1] = bird.food_cost.total / _FOOD_COST_SCALE
    feat[o + 2] = bird.food_cost.wild / _FOOD_COST_SCALE
    feat[o + 3] = bird.egg_limit / _EGG_LIMIT_SCALE
    feat[o + 4] = bird.wingspan_cm / _WINGSPAN_SCALE
    feat[o + 5] = 1.0 if bird.predator else 0.0
    feat[o + 6] = 1.0 if bird.flocking else 0.0
    for i, col in enumerate(_COLORS):
        if bird.color == col:
            feat[o + 7 + i] = 1.0
            break
    for i, nst in enumerate(_NESTS):
        if bird.nest == nst:
            feat[o + 11 + i] = 1.0
            break
    for i in range(cards.N_FOODS):
        feat[o + 16 + i] = bird.food_cost.counts[i] / _PER_FOOD_COST_SCALE


def _fill_food(feat: np.ndarray, food: cards.Food) -> None:
    for i, f in enumerate(cards.ALL_FOODS):
        if f == food:
            feat[_OFF_FOOD + i] = 1.0
            break


def _fill_habitat(feat: np.ndarray, habitat: cards.Habitat) -> None:
    for i, h in enumerate(cards.ALL_HABITATS):
        if h == habitat:
            feat[_OFF_HAB + i] = 1.0
            break


def _fill_payment(feat: np.ndarray, payment: state.FoodPool) -> None:
    for i in range(cards.N_FOODS):
        feat[_OFF_PAY + i] = payment.counts[i] / _PAYMENT_COUNT_SCALE


def _fill_board_target(
    feat: np.ndarray,
    habitat: cards.Habitat,
    slot: int,
    state: state.GameState,
    player_id: int,
) -> None:
    for i, h in enumerate(cards.ALL_HABITATS):
        if h == habitat:
            feat[_OFF_BOARD + i] = 1.0
            break
    feat[_OFF_BOARD + 3] = slot / _ROW_SLOTS_SCALE
    player = state.players[player_id]
    row = player.board[habitat]
    if 0 <= slot < len(row):
        pb = row[slot]
        feat[_OFF_BOARD + 4] = pb.eggs / _EGG_COUNT_SCALE
        cap = max(pb.bird.egg_limit - pb.eggs, 0)
        feat[_OFF_BOARD + 5] = cap / _EGG_COUNT_SCALE
        feat[_OFF_BOARD + 6] = pb.cached_food / _CACHED_FOOD_SCALE
        feat[_OFF_BOARD + 7] = pb.tucked_cards / _TUCKED_SCALE
