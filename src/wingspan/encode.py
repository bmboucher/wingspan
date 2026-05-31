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

# Decision class names already warned about for crossing each choice-count
# threshold, so each notice fires once per class per process rather than on every
# wide decision (the setup deal alone would otherwise log it twice per game).
# One set per threshold so the soft and runaway notices are independent.
_WARNED_WIDE: set[str] = set()
_WARNED_RUNAWAY: set[str] = set()


# ---------------------------------------------------------------------------
# Public constants — sanity bounds + normalization scales

# Choice-count safety bounds. The new encoder no longer truncates: every choice
# gets a feature row, and an over-wide decision is never fatal — both thresholds
# below only drive (deduped) log notices. ``SOFT_CHOICE_WARN_THRESHOLD`` flags a
# decision merely wider than typical; ``RUNAWAY_CHOICE_THRESHOLD`` flags one so
# wide it almost certainly signals a bug rather than real play.
SOFT_CHOICE_WARN_THRESHOLD = 20
# Width past which a decision is treated as a likely runaway-generation bug and
# gets a loud (but non-fatal) warning. The setup decision
# (``SETUP_CHOOSE_HAND_FOOD_BONUS``) intentionally enumerates all 504
# combinations for the standard 5-card / 2-bonus deal, and a food-rich late-game
# ``PlayBirdDecision`` enumerates one candidate per ``(bird, habitat, payment)``
# combination — which has been observed in the low thousands (TRAINING.md §4.3).
# The threshold therefore sits well above any legitimate width; exceeding it
# warns once per decision class and proceeds rather than aborting an unattended
# training run hours in.
RUNAWAY_CHOICE_THRESHOLD = 10000

# Goal-category one-hot length (mirrors the round-goal stripe). Sized with a
# little headroom above the number of distinct core goal categories.
MAX_GOAL_CATEGORIES = 20

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
_BONUS_VALUE_SCALE = 7.0  # max single-card bonus VP (Bird Feeder 8+: 7 VP)
_ACTIVATIONS_SCALE = 4.0  # per-bird activations within a round rarely exceed this
_BONUS_COUNT_SCALE = 5.0  # bonus qualifying-bird count / opponent bonus-card count
_GOAL_COUNT_SCALE = 5.0  # round-goal category counts

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
# State-stripe layout: the per-card identity + attribute encoding shared by the
# board- and tray-slot stripes, plus the board/tray/round-goal stripe sizes.

# Power colors and nests, used both by the per-choice ``_fill_bird`` stripe and
# by the rich state-stripe ``_bird_attr_vector``.
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
# The four concrete nests; a STAR nest is a wildcard encoded as all-ones over
# these, a missing nest (NONE) as all-zeros.
_NEST_BASE_TYPES = [
    cards.NestType.BOWL,
    cards.NestType.CAVITY,
    cards.NestType.GROUND,
    cards.NestType.PLATFORM,
]

# Bonus-card index keyed by printed name, so a bird's ``bonus_categories`` (the
# cards it statically qualifies for) can be encoded as a multi-hot aligned to the
# same ``cards.bonus_index`` space the bonus-progress stripes use. Built once
# from the canonical (lru-cached) bonus list.
_BONUS_NAME_TO_INDEX: dict[str, int] = {
    bonus_card.name: cards.bonus_index(bonus_card) for bonus_card in cards.load_all()[1]
}

# Rich per-card attribute vector (the ``N`` half of each slot's identity+attrs
# encoding). Offsets are cumulative; see ``_bird_attr_vector`` for the meaning.
_FOOD_COST_VEC_DIM = cards.N_FOODS + 1  # 5 specific foods + wild
_OFF_ATTR_POINTS = 0
_OFF_ATTR_FOOD_COST = _OFF_ATTR_POINTS + 1
_OFF_ATTR_NEST = _OFF_ATTR_FOOD_COST + _FOOD_COST_VEC_DIM
_OFF_ATTR_HAB = _OFF_ATTR_NEST + len(_NEST_BASE_TYPES)
_OFF_ATTR_FLOCK = _OFF_ATTR_HAB + len(cards.ALL_HABITATS)
_OFF_ATTR_PRED = _OFF_ATTR_FLOCK + 1
_OFF_ATTR_WINGSPAN = _OFF_ATTR_PRED + 1
_OFF_ATTR_EGG_LIMIT = _OFF_ATTR_WINGSPAN + 1
_OFF_ATTR_COLOR = _OFF_ATTR_EGG_LIMIT + 1
_OFF_ATTR_SWIFT = _OFF_ATTR_COLOR + len(_COLORS)
_OFF_ATTR_BONUS_CATS = _OFF_ATTR_SWIFT + 1
_BIRD_ATTR_DIM = _OFF_ATTR_BONUS_CATS + _BONUS_ID_DIM  # 49

# Per-board-slot continuous block: attribute vector then mutable state, with NO
# identity one-hot. The bird's identity is emitted separately as an integer index
# in the card-index block and looked up by the model's shared card embedding.
_OFF_SLOT_ATTR = 0
_OFF_SLOT_MUT = _OFF_SLOT_ATTR + _BIRD_ATTR_DIM
# Mutable: eggs, egg-capacity-remaining, cached food per type, tucked, activations.
_SLOT_MUT_EGGS = 0
_SLOT_MUT_EGG_CAP = 1
_SLOT_MUT_CACHED = 2  # start of the N_FOODS cached-by-type block
_SLOT_MUT_TUCKED = _SLOT_MUT_CACHED + cards.N_FOODS
_SLOT_MUT_ACTIVATIONS = _SLOT_MUT_TUCKED + 1
_SLOT_MUT_DIM = _SLOT_MUT_ACTIVATIONS + 1
_SLOT_CONT_DIM = _BIRD_ATTR_DIM + _SLOT_MUT_DIM
_SLOTS_PER_BOARD = state.N_HABITATS * state.ROW_SLOTS
_BOARD_CONT_STRIPE_DIM = _SLOTS_PER_BOARD * _SLOT_CONT_DIM

# Per-tray-slot continuous block: attribute vector only (no mutable state, no
# identity one-hot — the identity rides the card-index block). Order-invariant.
_TRAY_CONT_SLOT_DIM = _BIRD_ATTR_DIM
_TRAY_CONT_STRIPE_DIM = state.TRAY_SIZE * _TRAY_CONT_SLOT_DIM

# Round-goal state stripe: all four rounds, each = category one-hot
# (MAX_GOAL_CATEGORIES) + my count + opponent count + current placement VP.
_NUM_ROUNDS = len(state.ROUND_GOAL_PAYOUTS_2P)
_ROUND_GOAL_MY_COUNT = MAX_GOAL_CATEGORIES
_ROUND_GOAL_OPP_COUNT = MAX_GOAL_CATEGORIES + 1
_ROUND_GOAL_VP = MAX_GOAL_CATEGORIES + 2
_ROUND_GOAL_SLOT_DIM = MAX_GOAL_CATEGORIES + 3
_ROUND_GOALS_STRIPE_DIM = _NUM_ROUNDS * _ROUND_GOAL_SLOT_DIM


# ---------------------------------------------------------------------------
# Model-facing flat-vector layout for the shared card embedding.
#
# encode_state groups every per-slot card *identity* into one contiguous block of
# integer indices (board me 15, board opp 15, tray 3), each ``bird_index + 1``
# with 0 meaning "empty slot". The model gathers this block, looks the indices up
# in a single shared ``nn.Embedding`` (padding_idx 0), and concatenates the result
# with the continuous features. The hand is carried as a multi-hot the model
# mean-pools through the same embedding weight. These offsets are the contract the
# model splits on; the decision-type one-hot stays the final stripe.

N_BOARD_INDEX_SLOTS = 2 * _SLOTS_PER_BOARD  # POV board + opponent board
N_CARD_INDEX_SLOTS = N_BOARD_INDEX_SLOTS + state.TRAY_SIZE
HAND_MULTIHOT_DIM = _BIRD_ID_DIM

# Continuous prefix preceding the card-index block, summed over the encode_state
# parts in order (everything except the index block, hand multi-hot, and the
# trailing decision-type stripe).
_CONT_PREFIX_DIM = (
    5  # my food
    + 5  # opponent food
    + 2 * _BOARD_CONT_STRIPE_DIM  # board continuous (me, opp)
    + _TRAY_CONT_STRIPE_DIM  # tray continuous
    + 18  # my board summary
    + 18  # opponent board summary
    + 8  # my hand summary
    + 4 * _BONUS_ID_DIM  # bonus progress (held + count + stepped + linear)
    + 1  # opponent bonus-card count
    + 1  # opponent hand size
    + 5  # birdfeeder
    + 7  # misc scalars
    + _ROUND_GOALS_STRIPE_DIM  # all four round goals
)
OFF_CARD_INDEX = _CONT_PREFIX_DIM
OFF_HAND_MULTIHOT = OFF_CARD_INDEX + N_CARD_INDEX_SLOTS
OFF_DECISION_TYPE = OFF_HAND_MULTIHOT + HAND_MULTIHOT_DIM

# Choice-vector card-identity stripe. The model embeds it through the same shared
# table (a single-card candidate's one-hot maps to that card's embedding; the
# setup pick's kept-set multi-hot rides the same matmul as a sum).
CHOICE_BIRD_ID_OFFSET = _OFF_BIRD_ID
CHOICE_BIRD_ID_DIM = _BIRD_ID_DIM
CHOICE_BONUS_ID_OFFSET = _OFF_BONUS_ID


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
    "total_birds",
    "egg_sets_3habitats",
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
        _board_slots_continuous(me),  # _BOARD_CONT_STRIPE_DIM — per-slot attrs+mut
        _board_slots_continuous(opp),  # _BOARD_CONT_STRIPE_DIM — opponent (public)
        _tray_slots_continuous(state),  # _TRAY_CONT_STRIPE_DIM — public tray attrs
        _summary_board(me),  # 18 — kept aggregate
        _summary_board(opp),  # 18 — kept aggregate
        _summary_hand(me),  # 8
        _bonus_progress(me),  # 4 * _BONUS_ID_DIM — held + count + stepped + linear
        _opp_bonus_count(opp),  # 1 — opponent bonus-card count (hidden identity)
        np.array([len(opp.hand) / _HAND_SIZE_SCALE], dtype=np.float32),
        _summary_birdfeeder(state),  # 6 (5 food faces + choice dice)
        _summary_misc_scalars(state, me, opp),  # 7
        _round_goals_all_rounds(state, me, opp),  # _ROUND_GOALS_STRIPE_DIM
        _card_index_block(me, opp, state),  # N_CARD_INDEX_SLOTS — board+tray ids
        _hand_identity(me),  # HAND_MULTIHOT_DIM — multi-hot of my hand
        _encode_decision_type(decision),  # DECISION_TYPE_DIM (stays last)
    ]
    return np.concatenate(parts).astype(np.float32)


def state_size() -> int:
    """Total length of the vector returned by ``encode_state``."""
    return (
        5
        + 5
        + _BOARD_CONT_STRIPE_DIM  # my board (per-slot attrs + mutable)
        + _BOARD_CONT_STRIPE_DIM  # opponent board (per-slot attrs + mutable)
        + _TRAY_CONT_STRIPE_DIM  # public tray (per-slot attrs)
        + 18
        + 18
        + 8
        + 4 * _BONUS_ID_DIM  # bonus: held + count + stepped + linear
        + 1  # opponent bonus-card count
        + 1  # opponent hand size
        + 6  # birdfeeder: 5 single-food faces + choice-die count
        + 7
        + _ROUND_GOALS_STRIPE_DIM  # all four round goals
        + N_CARD_INDEX_SLOTS  # board + tray card-identity indices
        + HAND_MULTIHOT_DIM  # my hand multi-hot
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
    # A decision wider than the runaway threshold almost certainly signals a bug
    # in choice generation, but truncating or aborting would silently drop legal
    # moves / kill an unattended run — so record it (once per class) and proceed,
    # featurizing every choice as usual. Kept at WARNING (not ERROR) so it lands
    # in the run log without being loud enough for a default console handler to
    # surface it onto the live dashboard (which corrupts the rich.Live canvas).
    if n_choices > RUNAWAY_CHOICE_THRESHOLD and decision_name not in _WARNED_RUNAWAY:
        _WARNED_RUNAWAY.add(decision_name)
        logger.warning(
            "Decision %s produced %d choices (> %d runaway threshold) for "
            "player %d — featurizing all of them, but this likely signals a "
            "choice-generation bug",
            decision_name,
            n_choices,
            RUNAWAY_CHOICE_THRESHOLD,
            decision.player_id,
        )
    # The soft-threshold notice is a one-off-per-decision-class signal that a
    # decision ballooned wider than typical. SetupDecision (504) and a food-rich
    # PlayBirdDecision routinely and legitimately exceed it, so logging on every
    # such decision floods the log and adds per-call overhead in the hot path —
    # dedupe by class name so it fires once per class per process. Logged at INFO
    # (it is informational, not a fault): it still reaches the dashboard's file
    # log but never the console, so it can't flicker the live "FLYWAY CONTROL"
    # display the way a WARNING surfaced by a stray stderr handler would.
    if n_choices > SOFT_CHOICE_WARN_THRESHOLD and decision_name not in _WARNED_WIDE:
        _WARNED_WIDE.add(decision_name)
        logger.info(
            "Decision %s exposes %d choices (> %d soft threshold) for player %d",
            decision_name,
            n_choices,
            SOFT_CHOICE_WARN_THRESHOLD,
            decision.player_id,
        )
    # Featurize straight into each row view rather than building a throwaway
    # CHOICE_FEATURE_DIM array per candidate and copying it in — the rows start
    # zeroed, and the handlers only ever index-assign their own stripes.
    feats = np.zeros((n_choices, CHOICE_FEATURE_DIM), dtype=np.float32)
    for i, choice in enumerate(decision.choices):
        _featurize_choice(feats[i], decision, choice, state)
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
                    sum(pb.cached_food.total() for pb in row) / _CACHED_FOOD_SCALE,
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

    held = np.zeros(_BONUS_ID_DIM, dtype=np.float32)
    count = np.zeros(_BONUS_ID_DIM, dtype=np.float32)
    stepped = np.zeros(_BONUS_ID_DIM, dtype=np.float32)
    linear = np.zeros(_BONUS_ID_DIM, dtype=np.float32)
    for bonus_card in player.bonus_cards:
        idx = cards.bonus_index(bonus_card)
        held[idx] = 1.0
        count[idx] = (
            scoring.bonus_qualifying_count(player, bonus_card) / _BONUS_COUNT_SCALE
        )
        stepped[idx] = scoring.bonus_score(player, bonus_card) / _BONUS_VALUE_SCALE
        linear[idx] = (
            scoring.bonus_linear_value(player, bonus_card) / _BONUS_VALUE_SCALE
        )
    return np.concatenate([held, count, stepped, linear])


def _opp_bonus_count(opp: state.Player) -> np.ndarray:
    """Opponent bonus-card count as a single scalar. The *identities* of the
    opponent's bonus cards are hidden information, so only how many they hold is
    observable."""
    return np.array([len(opp.bonus_cards) / _BONUS_COUNT_SCALE], dtype=np.float32)


def _bird_attr_vector(bird: cards.Bird) -> np.ndarray:
    """Dense, normalized view of a bird's immutable card attributes — the rich
    ``N`` half of each board/tray slot's ``(identity one-hot, attributes)``
    encoding.

    Layout (``_BIRD_ATTR_DIM`` dims): victory points; food cost 6-vector (5
    specific foods then wild); nest 4-one-hot (a STAR nest is a wildcard encoded
    all-ones, NONE all-zeros); habitat multi-hot; flocking and predator flags;
    wingspan; egg limit; power-color one-hot (NONE => all-zero); swift-start
    flag; and a 26-wide multi-hot of the bonus-card categories the bird
    statically qualifies for (the "test" predicates such as 'named after a
    person' or a wingspan threshold), keyed to the same ``bonus_index`` space as
    the bonus-progress stripes."""
    vec = np.zeros(_BIRD_ATTR_DIM, dtype=np.float32)

    # Victory points and the printed food cost (6-vector: 5 specific, then wild).
    vec[_OFF_ATTR_POINTS] = bird.points / _POINTS_SCALE
    for i in range(_FOOD_COST_VEC_DIM):
        vec[_OFF_ATTR_FOOD_COST + i] = bird.food_cost.counts[i] / _PER_FOOD_COST_SCALE

    # Nest: 4-way one-hot over the concrete nests; STAR is a wildcard (all ones)
    # and a missing nest leaves the block zero.
    if bird.nest == cards.NestType.STAR:
        vec[_OFF_ATTR_NEST : _OFF_ATTR_NEST + len(_NEST_BASE_TYPES)] = 1.0
    else:
        for i, nest in enumerate(_NEST_BASE_TYPES):
            if bird.nest == nest:
                vec[_OFF_ATTR_NEST + i] = 1.0
                break

    # Habitat multi-hot, the two behavioural flags, and the remaining scalars.
    for i, habitat in enumerate(cards.ALL_HABITATS):
        if habitat in bird.habitats:
            vec[_OFF_ATTR_HAB + i] = 1.0
    vec[_OFF_ATTR_FLOCK] = 1.0 if bird.flocking else 0.0
    vec[_OFF_ATTR_PRED] = 1.0 if bird.predator else 0.0
    vec[_OFF_ATTR_WINGSPAN] = bird.wingspan_cm / _WINGSPAN_SCALE
    vec[_OFF_ATTR_EGG_LIMIT] = bird.egg_limit / _EGG_LIMIT_SCALE

    # Power color (one-hot; NONE leaves all zero) and the swift-start flag.
    for i, color in enumerate(_COLORS):
        if bird.color == color:
            vec[_OFF_ATTR_COLOR + i] = 1.0
            break
    vec[_OFF_ATTR_SWIFT] = 1.0 if bird.is_swift_start else 0.0

    # "Test" predicates: the bonus cards this bird statically qualifies for.
    for category in bird.bonus_categories:
        idx = _BONUS_NAME_TO_INDEX.get(category)
        if idx is not None:
            vec[_OFF_ATTR_BONUS_CATS + idx] = 1.0

    return vec


def _board_slots_continuous(player: state.Player) -> np.ndarray:
    """The continuous per-slot board stripe for ``player``: one fixed slot per
    board position (``N_HABITATS x ROW_SLOTS``), each carrying the bird's
    attribute vector and per-slot mutable state (eggs, egg-capacity remaining,
    cached food per type, tucked cards, activations). The bird's *identity* is
    emitted separately in the card-index block. Empty slots stay zero. Slot order
    is positional (NOT sorted) so a slot's mutable state — and its matching
    card-index entry — stays bound to the specific bird occupying it."""
    vec = np.zeros(_BOARD_CONT_STRIPE_DIM, dtype=np.float32)
    for hab_idx, habitat in enumerate(cards.ALL_HABITATS):
        for slot, pb in enumerate(player.board[habitat]):
            if slot >= state.ROW_SLOTS:
                break
            _write_slot_continuous(
                vec, (hab_idx * state.ROW_SLOTS + slot) * _SLOT_CONT_DIM, pb
            )
    return vec


def _write_slot_continuous(vec: np.ndarray, base: int, pb: state.PlayedBird) -> None:
    """Write one occupied board slot's continuous features into
    ``vec[base : base + _SLOT_CONT_DIM]``: attribute vector then per-slot mutable
    state (no identity — that rides the card-index block)."""
    attr_at = base + _OFF_SLOT_ATTR
    vec[attr_at : attr_at + _BIRD_ATTR_DIM] = _bird_attr_vector(pb.bird)

    mut = base + _OFF_SLOT_MUT
    vec[mut + _SLOT_MUT_EGGS] = pb.eggs / _EGG_COUNT_SCALE
    vec[mut + _SLOT_MUT_EGG_CAP] = (
        max(pb.bird.egg_limit - pb.eggs, 0) / _EGG_COUNT_SCALE
    )
    for i, food in enumerate(cards.ALL_FOODS):
        vec[mut + _SLOT_MUT_CACHED + i] = pb.cached_food[food] / _CACHED_FOOD_SCALE
    vec[mut + _SLOT_MUT_TUCKED] = pb.tucked_cards / _TUCKED_SCALE
    vec[mut + _SLOT_MUT_ACTIVATIONS] = pb.activations / _ACTIVATIONS_SCALE


def _tray_slots_continuous(game_state: state.GameState) -> np.ndarray:
    """The public face-up bird tray's continuous features: up to ``TRAY_SIZE``
    slots, each the bird's attribute vector (no mutable state, no identity — the
    identity rides the card-index block). The tray slots are interchangeable, so
    birds are sorted by ``cards.bird_index`` to make the encoding order-invariant;
    trailing slots stay zero when the tray is short."""
    vec = np.zeros(_TRAY_CONT_STRIPE_DIM, dtype=np.float32)
    for slot, bird in enumerate(sorted(game_state.tray, key=cards.bird_index)):
        if slot >= state.TRAY_SIZE:
            break
        base = slot * _TRAY_CONT_SLOT_DIM
        vec[base : base + _BIRD_ATTR_DIM] = _bird_attr_vector(bird)
    return vec


def _card_index_block(
    me: state.Player, opp: state.Player, game_state: state.GameState
) -> np.ndarray:
    """Contiguous integer card indices the model looks up in its shared card
    embedding: both boards' positional slots (POV then opponent) followed by the
    up-to-``TRAY_SIZE`` public tray. Each entry is ``bird_index + 1`` for an
    occupied slot and 0 for an empty one (the embedding's padding index). Board
    slots are positional (matching ``_board_slots_continuous``); tray birds are
    sorted by ``bird_index`` (matching ``_tray_slots_continuous``) so each index
    lines up with its continuous attribute slot."""
    vec = np.zeros(N_CARD_INDEX_SLOTS, dtype=np.float32)
    offset = 0
    for player in (me, opp):
        for hab_idx, habitat in enumerate(cards.ALL_HABITATS):
            for slot, pb in enumerate(player.board[habitat]):
                if slot >= state.ROW_SLOTS:
                    break
                vec[offset + hab_idx * state.ROW_SLOTS + slot] = (
                    cards.bird_index(pb.bird) + 1
                )
        offset += _SLOTS_PER_BOARD
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

    vec = np.zeros(_ROUND_GOALS_STRIPE_DIM, dtype=np.float32)
    for round_idx in range(_NUM_ROUNDS):
        if round_idx >= len(game_state.round_goals):
            break
        base = round_idx * _ROUND_GOAL_SLOT_DIM
        goal = game_state.round_goals[round_idx]
        if goal.category in _GOAL_CATEGORIES:
            vec[base + _GOAL_CATEGORIES.index(goal.category)] = 1.0
        standing = scoring.round_goal_standing_for_round(game_state, me, round_idx)
        vec[base + _ROUND_GOAL_MY_COUNT] = standing.count / _GOAL_COUNT_SCALE
        vec[base + _ROUND_GOAL_OPP_COUNT] = (
            scoring.eval_goal(opp, goal) / _GOAL_COUNT_SCALE
        )
        vec[base + _ROUND_GOAL_VP] = standing.vp / _ROUND_GOAL_POINTS_SCALE
    return vec


def _summary_birdfeeder(state: state.GameState) -> np.ndarray:
    return np.array(
        [
            *(
                state.birdfeeder.counts[food] / _BIRDFEEDER_COUNT_SCALE
                for food in cards.ALL_FOODS
            ),
            state.birdfeeder.choice_dice / _BIRDFEEDER_COUNT_SCALE,
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
    feat: np.ndarray,
    decision: _AnyDecision,
    choice: decisions.Choice,
    state: state.GameState,
) -> None:
    """Fill the pre-zeroed CHOICE_FEATURE_DIM row ``feat`` for one
    (decision, choice) pair, dispatching on the concrete Choice subclass.

    Writes into the caller's row view rather than allocating a fresh vector, so
    ``encode_choices`` builds its ``(n_choices, DIM)`` matrix with no per-row
    throwaway. The typed ``choice`` parameter keeps ``type(choice)`` a known
    ``type[Choice]`` for the dispatch lookup."""
    _CHOICE_FEATURIZERS.get(type(choice), _featurize_default)(
        feat, decision, choice, state
    )


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


def _featurize_reset_birdfeeder(
    feat: np.ndarray,
    decision: _AnyDecision,
    choice: decisions.ResetBirdfeederChoice,
    state: state.GameState,
) -> None:
    # The "yes, reroll" affirmative. Carries no data, so only the special-kind
    # bit is set; the decision-type stripe identifies the reset decision and the
    # absent is-skip bit distinguishes it from the paired ``SkipChoice``.
    feat[_OFF_KIND + _KIND_SPECIAL] = 1.0


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
    feat[_OFF_BOARD + 6] = pb.cached_food.total() / _CACHED_FOOD_SCALE
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
    # aggregate stats that summarise the subset. One pass sets each identity bit
    # and accumulates all three sums (the setup deal featurizes 504 candidates,
    # so folding three generator passes into one matters).
    if kept:
        points = 0.0
        cost = 0.0
        eggs = 0.0
        for bird in kept:
            feat[_OFF_BIRD_ID + cards.bird_index(bird)] = 1.0
            points += bird.points
            cost += bird.food_cost.total
            eggs += bird.egg_limit
        feat[_OFF_BIRD + 0] = points / (_POINTS_SCALE * _ROW_SLOTS_SCALE)
        feat[_OFF_BIRD + 1] = cost / (_FOOD_COST_SCALE * _ROW_SLOTS_SCALE)
        feat[_OFF_BIRD + 3] = eggs / (_EGG_LIMIT_SCALE * _ROW_SLOTS_SCALE)
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
    decisions.ResetBirdfeederChoice: _featurize_reset_birdfeeder,
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
        feat[_OFF_BOARD + 6] = pb.cached_food.total() / _CACHED_FOOD_SCALE
        feat[_OFF_BOARD + 7] = pb.tucked_cards / _TUCKED_SCALE
