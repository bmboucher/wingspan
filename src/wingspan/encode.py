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

The featurizer dispatches on the concrete ``Choice`` subclass so each branch
reads the typed fields it needs: a ``cards.Bird`` exposes its costs, power
color, and identity; a board-target ``(habitat, slot)`` is looked up on the
asking player's board for current egg/cache state; a payment exposes its food
composition; and so on. Unused stripes stay zero.
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
# combinations for the standard 5-card / 2-bonus deal, and a food-rich
# late-game ``PlayBirdDecision`` enumerates one candidate per
# ``(bird, habitat, payment)`` combination — which has been observed past 600
# (e.g. 637), a trajectory-dependent spike that would otherwise abort an
# unattended training run hours in (TRAINING.md §4.3). The cap therefore sits
# well above any legitimate width while still catching genuinely runaway
# choice generation (a sign of a bug, not normal play).
MAX_CHOICES_HARD = 2000

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
_EXCHANGE_SCALE = 3.0  # accept-exchange paid/gained quantity normalizer

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
_EXCHANGE_DIM = 3  # accept-exchange terms: eggs paid, cards gained, tucks gained
#                    (the food paid, if any, reuses the FOOD stripe)
# Card-identity stripes: a one-hot over every core-set bird / bonus card, so a
# specific card — or, for the setup pick and the hand, a *set* of cards as a
# multi-hot — is encoded by identity alongside its attribute stripe. The first
# linear layer over this stripe is a learned per-card embedding, exactly the
# per-card value signal the card-power analysis wants. Sized from the loaded
# catalog (180 birds / 26 bonus cards in the core set).
_BIRD_ID_DIM = cards.n_birds()
_BONUS_ID_DIM = cards.n_bonus_cards()

CHOICE_FEATURE_DIM = (
    _KIND_DIM
    + _BIRD_DIM
    + _FOOD_DIM
    + _HABITAT_DIM
    + _PAYMENT_DIM
    + _BOARD_TARGET_DIM
    + _SPECIAL_DIM
    + _EXCHANGE_DIM
    + _BIRD_ID_DIM
    + _BONUS_ID_DIM
)

# Stripe offsets (cumulative)
_OFF_KIND = 0
_OFF_BIRD = _OFF_KIND + _KIND_DIM
_OFF_FOOD = _OFF_BIRD + _BIRD_DIM
_OFF_HAB = _OFF_FOOD + _FOOD_DIM
_OFF_PAY = _OFF_HAB + _HABITAT_DIM
_OFF_BOARD = _OFF_PAY + _PAYMENT_DIM
_OFF_SPECIAL = _OFF_BOARD + _BOARD_TARGET_DIM
_OFF_EXCHANGE = _OFF_SPECIAL + _SPECIAL_DIM
_OFF_BIRD_ID = _OFF_EXCHANGE + _EXCHANGE_DIM
_OFF_BONUS_ID = _OFF_BIRD_ID + _BIRD_ID_DIM

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

# Within-EXCHANGE indices (an AcceptExchange PayCostChoice's trade terms)
_EXCHANGE_PAID_EGGS = 0
_EXCHANGE_GAINED_CARDS = 1
_EXCHANGE_GAINED_TUCKS = 2


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
        _hand_identity(me),  # _BIRD_ID_DIM — multi-hot of my hand
        np.array([len(opp.hand) / _HAND_SIZE_SCALE], dtype=np.float32),
        _summary_birdfeeder(state),  # 5
        _summary_misc_scalars(state, me, opp),  # 7
        _summary_round_goal(state),  # MAX_GOAL_CATEGORIES
        _encode_decision_type(decision),  # DECISION_TYPE_DIM
    ]
    return np.concatenate(parts).astype(np.float32)


def state_size() -> int:
    """Total length of the vector returned by ``encode_state``."""
    return (
        5
        + 5
        + 18
        + 18
        + 8
        + _BIRD_ID_DIM
        + 1
        + 5
        + 7
        + MAX_GOAL_CATEGORIES
        + DECISION_TYPE_DIM
    )


def encode_choices(decision: _AnyDecision, state: state.GameState) -> np.ndarray:
    """Featurize every choice in ``decision``.

    Returns a float32 array of shape ``(n_choices, CHOICE_FEATURE_DIM)``.
    All returned rows correspond to legal choices — there is no padding or
    truncation, and the action space is implicitly variable across decisions.
    The caller (training loop) handles batched padding + masking.
    """
    n_choices = len(decision.choices)
    decision_name = type(decision).__name__
    assert n_choices > 0, f"empty Decision: {decision_name}"
    assert n_choices <= MAX_CHOICES_HARD, (
        f"decision {decision_name} produced {n_choices} choices, "
        f"exceeds MAX_CHOICES_HARD={MAX_CHOICES_HARD}"
    )
    if n_choices > SOFT_CHOICE_WARN_THRESHOLD:
        logger.warning(
            "Decision %s exposes %d choices (> %d soft threshold) for player %d",
            decision_name,
            n_choices,
            SOFT_CHOICE_WARN_THRESHOLD,
            decision.player_id,
        )
    feats = np.zeros((n_choices, CHOICE_FEATURE_DIM), dtype=np.float32)
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
        [player.food[food] / _FOOD_INVENTORY_SCALE for food in cards.ALL_FOODS],
        dtype=np.float32,
    )


def _summary_board(player: state.Player) -> np.ndarray:
    parts: list[np.ndarray] = []
    for habitat in cards.ALL_HABITATS:
        row = player.board[habitat]
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
    pts = [bird.points for bird in player.hand]
    costs = [bird.food_cost.total for bird in player.hand]
    eggs = [bird.egg_limit for bird in player.hand]
    return np.array(
        [
            len(player.hand) / _HAND_SIZE_SCALE,
            float(np.mean(pts)) / _POINTS_SCALE,
            float(np.max(pts)) / _POINTS_SCALE,
            float(np.mean(costs)) / _FOOD_COST_SCALE,
            float(np.min(costs)) / _FOOD_COST_SCALE,
            float(np.mean(eggs)) / _EGG_LIMIT_SCALE,
            sum(1 for bird in player.hand if cards.Habitat.FOREST in bird.habitats)
            / _HAND_SIZE_SCALE,
            sum(1 for bird in player.hand if cards.Habitat.WETLAND in bird.habitats)
            / _HAND_SIZE_SCALE,
        ],
        dtype=np.float32,
    )


def _hand_identity(player: state.Player) -> np.ndarray:
    """Multi-hot over all core-set birds marking which are in ``player``'s hand.

    Pairs with ``_summary_hand``'s aggregate stats (identity + attributes) so
    every scoring head and the value head can read the *specific* cards held,
    not just their summary. Opponent hands are hidden information, so only the
    POV player's hand is encoded by identity (the opponent contributes its
    size only)."""
    vec = np.zeros(_BIRD_ID_DIM, dtype=np.float32)
    for bird in player.hand:
        vec[cards.bird_index(bird)] = 1.0
    return vec


def _summary_birdfeeder(state: state.GameState) -> np.ndarray:
    return np.array(
        [
            state.birdfeeder.counts[food] / _BIRDFEEDER_COUNT_SCALE
            for food in cards.ALL_FOODS
        ],
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
    # The 'accept the offered exchange' branch is distinct from skip — the
    # network can learn to prefer or avoid it independently. KIND_SPECIAL marks
    # it a commit token; the trade terms live in the FOOD stripe (the food paid,
    # if any) and the EXCHANGE stripe (eggs paid, cards / tucks gained) so the
    # commit-to-cost head weighs what is gained against what is paid.
    feat[_OFF_KIND + _KIND_SPECIAL] = 1.0
    if choice.paid_food is not None:
        _fill_food(feat, choice.paid_food)
    feat[_OFF_EXCHANGE + _EXCHANGE_PAID_EGGS] = choice.paid_egg_count / _EXCHANGE_SCALE
    feat[_OFF_EXCHANGE + _EXCHANGE_GAINED_CARDS] = (
        choice.gained_card_count / _EXCHANGE_SCALE
    )
    feat[_OFF_EXCHANGE + _EXCHANGE_GAINED_TUCKS] = (
        choice.gained_tuck_count / _EXCHANGE_SCALE
    )


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


def _featurize_play_bird(
    feat: np.ndarray,
    decision: _AnyDecision,
    choice: decisions.PlayBirdChoice,
    state: state.GameState,
) -> None:
    # A play candidate from ``PlayBirdDecision``: the bird stripe (identity +
    # attributes) carries the card, and the habitat + payment stripes carry the
    # bundled habitat / food-payment picks. KIND stays BIRD — it is fundamentally
    # a bird play — while the extra stripes distinguish the (habitat, payment)
    # variants of the same bird.
    feat[_OFF_KIND + _KIND_BIRD] = 1.0
    _fill_bird(feat, choice.bird)
    _fill_habitat(feat, choice.habitat)
    _fill_payment(feat, choice.payment)


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
    # Identity via the bonus one-hot stripe (a learned per-bonus embedding),
    # replacing the old id-hash so distinct bonus cards are fully distinguished.
    feat[_OFF_KIND + _KIND_SPECIAL] = 1.0
    _fill_bonus_identity(feat, choice.bonus_card)


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

    The 504 candidates share a state vector, so the network reads the choice
    features to tell them apart. We surface (a) a multi-hot of the *specific*
    kept birds in the bird-identity stripe — so the setup head can finally learn
    card-specific opening synergies (DECISIONS.md §3.1) — alongside (b) aggregate
    stats of the kept-card subset, (c) a multi-hot of foods spent in the PAYMENT
    stripe, and (d) the kept bonus card's identity one-hot.
    """
    feat[_OFF_KIND + _KIND_SPECIAL] = 1.0
    # PAY stripe encodes the foods *spent* (complement of kept_foods), so the
    # network sees the same payment signal as every other paying decision.
    for i, food in enumerate(cards.ALL_FOODS):
        if food not in choice.kept_foods:
            feat[_OFF_PAY + i] = 1.0 / _PAYMENT_COUNT_SCALE
    kept = choice.kept_cards
    # Identity multi-hot of the kept birds (the headline §3.1 fix) plus the
    # aggregate stats that summarise the subset.
    for bird in kept:
        _fill_bird_identity(feat, bird)
    if kept:
        feat[_OFF_BIRD + 0] = sum(bird.points for bird in kept) / (
            _POINTS_SCALE * _ROW_SLOTS_SCALE
        )
        feat[_OFF_BIRD + 1] = sum(bird.food_cost.total for bird in kept) / (
            _FOOD_COST_SCALE * _ROW_SLOTS_SCALE
        )
        feat[_OFF_BIRD + 3] = sum(bird.egg_limit for bird in kept) / (
            _EGG_LIMIT_SCALE * _ROW_SLOTS_SCALE
        )
    feat[_OFF_BIRD + 4] = len(kept) / _ROW_SLOTS_SCALE
    if choice.bonus_card is not None:
        _fill_bonus_identity(feat, choice.bonus_card)


# Index per main-action type, spread across the SPECIAL stripe so the options
# are distinguishable. ``MainActionDecision`` now scores only the action *type*
# (including ``PLAY_BIRD``), so all four are featureless type tokens here; the
# rich bird / habitat / payment features live on the follow-up
# ``PlayBirdDecision``'s ``PlayBirdChoice`` candidates instead.
_MAIN_ACTION_INDEX: dict[decisions.MainAction, int] = {
    decisions.MainAction.GAIN_FOOD: 0,
    decisions.MainAction.LAY_EGGS: 1,
    decisions.MainAction.DRAW_CARDS: 2,
    decisions.MainAction.PLAY_BIRD: 3,
}


_CHOICE_FEATURIZERS: dict[type[decisions.Choice], _ChoiceFeaturizer] = {
    decisions.SkipChoice: _featurize_skip,
    decisions.PayCostChoice: _featurize_pay_cost,
    decisions.MainActionChoice: _featurize_main_action,
    decisions.BirdChoice: _featurize_bird,
    decisions.PlayBirdChoice: _featurize_play_bird,
    decisions.PlayedBirdChoice: _featurize_played_bird,
    decisions.HabitatChoice: _featurize_habitat,
    decisions.FoodChoice: _featurize_food,
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
    off = _OFF_BIRD
    feat[off + 0] = bird.points / _POINTS_SCALE
    feat[off + 1] = bird.food_cost.total / _FOOD_COST_SCALE
    feat[off + 2] = bird.food_cost.wild / _FOOD_COST_SCALE
    feat[off + 3] = bird.egg_limit / _EGG_LIMIT_SCALE
    feat[off + 4] = bird.wingspan_cm / _WINGSPAN_SCALE
    feat[off + 5] = 1.0 if bird.predator else 0.0
    feat[off + 6] = 1.0 if bird.flocking else 0.0
    for i, col in enumerate(_COLORS):
        if bird.color == col:
            feat[off + 7 + i] = 1.0
            break
    for i, nst in enumerate(_NESTS):
        if bird.nest == nst:
            feat[off + 11 + i] = 1.0
            break
    for i in range(cards.N_FOODS):
        feat[off + 16 + i] = bird.food_cost.counts[i] / _PER_FOOD_COST_SCALE
    _fill_bird_identity(feat, bird)


def _fill_bird_identity(feat: np.ndarray, bird: cards.Bird) -> None:
    """Set the bird-identity one-hot bit for ``bird``. Called for every
    bird-carrying choice, and once per card to build a kept-set / hand multi-hot.
    The first linear layer over this stripe is a learned per-card embedding."""
    feat[_OFF_BIRD_ID + cards.bird_index(bird)] = 1.0


def _fill_bonus_identity(feat: np.ndarray, bonus_card: cards.BonusCard) -> None:
    """Set the bonus-card identity one-hot bit for ``bonus_card``."""
    feat[_OFF_BONUS_ID + cards.bonus_index(bonus_card)] = 1.0


def _fill_food(feat: np.ndarray, food: cards.Food) -> None:
    for i, candidate in enumerate(cards.ALL_FOODS):
        if candidate == food:
            feat[_OFF_FOOD + i] = 1.0
            break


def _fill_habitat(feat: np.ndarray, habitat: cards.Habitat) -> None:
    for i, candidate in enumerate(cards.ALL_HABITATS):
        if candidate == habitat:
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
    for i, candidate in enumerate(cards.ALL_HABITATS):
        if candidate == habitat:
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
