"""The encoder's fixed layout: feature dimensions, stripe offsets, and the
normalization scales, plus the decision-type / goal-category orderings.

Every constant the state and choice encoders share lives here in one place and
in dependency order — the ``_OFF_*`` chain is evaluated top-to-bottom, and the
``encode_state`` stripe order and these offsets are what trained checkpoints are
aligned to, so nothing here may be reordered or renumbered.
"""

from __future__ import annotations

import typing

from wingspan import cards, decisions, state

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


def trunk_input_dim(state_dim: int, card_embed_dim: int) -> int:
    """The state trunk's first-``Linear`` input width: the flat ``state_dim`` with
    the card-index block and the hand multi-hot replaced by their shared-embedding
    lookups — one ``card_embed_dim`` vector per index slot, plus one mean-pooled
    hand embedding. The model splits the flat state on exactly this basis, so this
    is the single source of truth for the post-embedding width (used both by
    ``model.PolicyValueNet`` and by the configurator's parameter accounting)."""
    return (
        state_dim
        - N_CARD_INDEX_SLOTS  # index columns -> per-slot embeddings
        - HAND_MULTIHOT_DIM  # hand multi-hot -> one pooled embedding
        + N_CARD_INDEX_SLOTS * card_embed_dim
        + card_embed_dim
    )


def choice_input_dim(choice_dim: int, card_embed_dim: int) -> int:
    """The per-choice encoder's first-``Linear`` input width: the flat
    ``choice_dim`` with the candidate's bird-identity one-hot replaced by its
    shared-embedding lookup (one ``card_embed_dim`` vector)."""
    return choice_dim - CHOICE_BIRD_ID_DIM + card_embed_dim


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

# Public alias of the goal-category ordering, re-exported from the package so
# the (separately-encoded) setup model can build its own round-goal one-hots
# against the same stable category order the state encoder uses.
GOAL_CATEGORIES = _GOAL_CATEGORIES
