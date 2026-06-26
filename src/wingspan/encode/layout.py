"""The encoder's fixed layout: feature dimensions, stripe offsets, and the
normalization scales, plus the decision-type / goal-category orderings.

Most of the layout is fixed, but the *shape* of the main model is config-driven
on a single axis — whether the opening (``SetupDecision``) is scored by the main
model or delegated to the separate setup model. :class:`EncodingSpec` captures
that axis (``include_setup``); the setup-only pieces are kept LAST in every
stable order (the choice ``setup_agg`` and ``kept_multihot`` stripes, the
decision-type one-hot's setup column, the ``SETUP`` scoring head), so excluding
them is a clean truncation of trailing dimensions that leaves every other
offset and index unchanged. The spec-dependent totals are therefore exposed as
functions (``state_feature_dim`` / ``choice_feature_dim`` /
``decision_type_dim`` / ``num_families``); the module constants
``CHOICE_FEATURE_DIM`` / ``DECISION_TYPE_DIM`` are the default-spec
(setup-excluded) values.

Stripe offsets are derived by sequential accumulation from ordered
:class:`~stripes.descriptors.StripeSpec` lists, eliminating the risk of
arithmetic errors in a hand-written cumulative-sum chain.  The public layout
objects (:data:`CHOICE_BASE_LAYOUT`, :data:`CHOICE_FULL_LAYOUT`,
:data:`CARD_ATTR_LAYOUT`, :data:`STATE_CONT_LAYOUT`) expose the canonical
stripe geometry; the ``_OFF_*`` aliases are derived from them so that every
downstream consumer reads consistent, auto-computed offsets.  Changing the
order or sizes of stripes is a FRESH (checkpoint-invalidating) change.
See CLAUDE.md "Checkpoint compatibility policy".
"""

from __future__ import annotations

import typing

import pydantic

from wingspan import cards, decisions, state

# ---------------------------------------------------------------------------
# Encoding spec — the one config-driven axis of the encoders' shape.


class EncodingSpec(pydantic.BaseModel):
    """The config-driven shape of the state/choice encoders.

    ``include_setup`` selects whether the main model carries the opening:
    ``True`` keeps the ``SetupDecision`` decision-type column, the ``setup_agg``
    choice stripe, and the ``SETUP`` scoring head (the ``use_setup_model=False``
    fallback that scores the opening with the main net); ``False`` drops all
    three, because the opening is handled by the separate setup model.

    The field defaults to ``False`` so a bare spec matches the default training
    config (``TrainConfig.use_setup_model=True`` ⇒ the main net excludes setup):
    bare ``PolicyValueNet()`` / ``encode_state()`` calls and a configured default
    run then agree on the shape. A run derives its spec from
    ``spec_for(use_setup_model)``. Frozen so it is hashable (it keys the cached
    size functions below) and serializable into ``model_config.json``."""

    model_config = pydantic.ConfigDict(frozen=True)

    include_setup: bool = False


# The default spec matches a default run (``use_setup_model=True`` ⇒ setup
# excluded from the main net); bare encoder / model calls use it.
DEFAULT_SPEC = EncodingSpec()


def spec_for(use_setup_model: bool) -> EncodingSpec:
    """The encoding spec implied by a run's ``use_setup_model`` flag: the main
    model includes setup exactly when the separate setup model is *off*."""
    return EncodingSpec(include_setup=not use_setup_model)


# Deferred until after DEFAULT_SPEC is defined: stripes.choice accesses
# layout.DEFAULT_SPEC as a function-default argument (evaluated at import
# time), so this import must come AFTER DEFAULT_SPEC to avoid a circular
# AttributeError when Python re-enters this partially-initialized module.
from wingspan.encode.stripes import descriptors as _stripe_descriptors  # noqa: E402

# ---------------------------------------------------------------------------
# Public constants — sanity bounds + normalization scales

# Choice-count safety bounds. The encoder never truncates: every choice gets a
# feature row, and an over-wide decision is never fatal — both thresholds below
# only drive (deduped) log notices. ``SOFT_CHOICE_WARN_THRESHOLD`` flags a
# decision merely wider than typical; ``RUNAWAY_CHOICE_THRESHOLD`` flags one so
# wide it almost certainly signals a bug rather than real play.
SOFT_CHOICE_WARN_THRESHOLD = 20
RUNAWAY_CHOICE_THRESHOLD = 10000

# Goal-category one-hot length (mirrors the round-goal stripe). Exactly fits
# the 20 core goal categories in _GOAL_CATEGORIES — growing it is a FRESH
# (checkpoint-invalidating) change.
MAX_GOAL_CATEGORIES = 20

# One-hot dimensions for round number and action-cube counts.
# N_ROUNDS: one hot position per round (0..3 → 4 dims).
# MAX_ACTION_CUBES: one hot position per cube count (0..8 → 9 dims) — equals
# state.ROUND_CUBES[0], the maximum cubes at the start of round 1.
# Both are retained for the v0.2 and v0.3 compat shims that reference them.
N_ROUNDS: int = 4
MAX_ACTION_CUBES: int = 8

# Total personal turns across all 4 rounds (8+7+6+5 = 26). Used by the
# turn_state stripe to encode which of the player's 26 turns they are on.
N_PLAYER_TURNS: int = sum(state.ROUND_CUBES)

# Cumulative cube-count offsets per round: ROUND_CUBES[0..i-1] summed.
# Entry i is the turn index at which round i begins.  [0, 8, 15, 21].
_ROUND_CUBE_OFFSETS: list[int] = [
    sum(state.ROUND_CUBES[:i]) for i in range(len(state.ROUND_CUBES))
]

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
_EXCHANGE_SCALE = 3.0  # accept-exchange paid/gained quantity normalizer
_BONUS_VALUE_SCALE = 7.0  # max single-card bonus VP (Bird Feeder 8+: 7 VP)
_ACTIVATIONS_SCALE = 4.0  # per-bird activations within a round rarely exceed this
_BONUS_COUNT_SCALE = 5.0  # bonus qualifying-bird count / opponent bonus-card count
_GOAL_COUNT_SCALE = 5.0  # round-goal category counts

# Dimensions of the hand-summary stripe: hand size + per-habitat bird counts +
# food-cost multi-hot (see ``state_encode._summary_hand``). Used both as a named
# constant in ``_CONT_PREFIX_DIM`` and as the second input block to the distinct
# hand encoder when ``use_distinct_hand_model`` is active.
HAND_SUMMARY_DIM = 10

# How a *set* of cards combines its members' per-card summary rows
# (``state_encode.card_summary_matrix``): the leading dims (the set-size term and
# the per-habitat counts) sum over the set; the remaining food-cost flags combine
# by max (= OR, every entry is >= 0). Shared by the numpy encoder and the
# torch-side derivation (``wingspan.model.hand_model``) so the two cannot disagree on
# the split.
HAND_SUMMARY_SUM_DIMS = 4

# One board's fixed slot count (3 habitats x 5 columns). Defined here because
# both the choice board_target stripe and the state board stripes size from it.
_SLOTS_PER_BOARD = state.N_HABITATS * state.ROW_SLOTS  # 15

# ---------------------------------------------------------------------------
# Choice feature layout
#
# A single uniform feature vector with type-specific stripes. Each branch in
# ``_featurize_choice`` fills only the stripes relevant to that decision type;
# the rest stay zero. Every offset below is fixed; only the trailing
# ``setup_agg`` / ``kept_multihot`` stripes are conditional (present iff
# ``EncodingSpec.include_setup``).

_KIND_DIM = 6  # bird, food, habitat, payment, board_target, special
_GAIN_FOOD_DIM = 7  # 5 foods + take-choice-die-as-invertebrate + ...-as-seed
_PAY_FOOD_DIM = 5  # food payment: count per food
_MAIN_ACTION_DIM = 4  # one-hot over the four main actions
_SPECIAL_DIM = 2  # is_skip, is_self
_EXCHANGE_DIM = (
    13  # symmetric pay->gain terms: 8 self + 4 opponent-gain + 1 cache (below)
)
_BONUS_DELTA_DIM = 3  # candidate bird's contribution to held bonus cards (below)
_GOAL_DELTA_SLOT_DIM = 2  # count_delta + vp_delta per round-goal slot
_GOAL_DELTA_DIM = 4 * _GOAL_DELTA_SLOT_DIM  # 8 (4 round goals × 2 scalars)
_GOAL_DELTA_COUNT = 0  # within-slot: count change (÷ _GOAL_COUNT_SCALE)
_GOAL_DELTA_VP = 1  # within-slot: VP delta (÷ _ROUND_GOAL_POINTS_SCALE)
_BONUS_VALUE_DIM = 5  # candidate bonus card's value to the deciding player (below)
_SETUP_DIM = 4  # setup kept-subset aggregates (only when include_setup)

# The board_target stripe is a per-board-slot block: 4 scalars repeated over
# every board slot. Per slot: lay_eggs, pay_eggs, cached_total (all food types
# summed), tucked. The targeted slot's occupant rides the ``bird_id`` column;
# its location is marked by ``board_hab`` (habitat one-hot) + ``board_col``
# (column one-hot). The full board state already rides the state vector.
_BT_SLOT_SCALARS = 4
_BT_LAY_EGGS = 0
_BT_PAY_EGGS = 1
_BT_CACHED_TOTAL = 2  # summed cached food (all types combined)
_BT_TUCKED = 3
_BOARD_TARGET_DIM = _SLOTS_PER_BOARD * _BT_SLOT_SCALARS  # 15 * 4 = 60

# Location one-hots for the single board slot relevant to each choice: the
# habitat (3 dims) and column (5 dims) of the landing slot, the targeted slot,
# or the current slot of the relevant bird. Pass-through (not embedded).
_BOARD_HAB_DIM = state.N_HABITATS  # 3
_BOARD_COL_DIM = state.ROW_SLOTS  # 5

# Card-identity stripes. The candidate bird is a single integer index column
# (``bird_index + 1``, 0 = no bird) the model looks up in the shared card table
# (the same ``[181, D]`` table the state board / tray slots use). On
# board-target rows it also carries the targeted slot's occupant. The bonus
# card stays a one-hot. A setup pick's kept *set* of cards rides the trailing
# ``kept_multihot`` stripe (a multi-hot the model sums through the card table),
# present iff ``include_setup``. ``_BIRD_ID_DIM`` is the catalog size (180
# core-set birds) and also feeds the state-side hand multi-hot / card-feature
# constants.
_BIRD_ID_DIM = cards.n_birds()
_BONUS_ID_DIM = cards.n_bonus_cards()
_CHOICE_BIRD_ID_DIM = 1  # the candidate bird's single index column
_KEPT_MULTIHOT_DIM = _BIRD_ID_DIM  # setup kept-set multi-hot (only when include_setup)

# Stripe offsets auto-accumulated from ordered StripeSpec lists. The bird-index
# column (the card region the model embeds) sits just before bonus_id; the
# conditional setup_agg and kept_multihot stripes trail everything — so the
# card-region offset the model slices on stays invariant to ``include_setup``
# (the trailing kept_multihot region is by construction the row's final columns).
_CHOICE_STRIPE_SPECS: list[_stripe_descriptors.StripeSpec] = [
    _stripe_descriptors.StripeSpec(name="kind", size=_KIND_DIM),
    _stripe_descriptors.StripeSpec(name="gain_food", size=_GAIN_FOOD_DIM),
    _stripe_descriptors.StripeSpec(name="pay", size=_PAY_FOOD_DIM),
    _stripe_descriptors.StripeSpec(name="board", size=_BOARD_TARGET_DIM),
    _stripe_descriptors.StripeSpec(name="main_action", size=_MAIN_ACTION_DIM),
    _stripe_descriptors.StripeSpec(name="special", size=_SPECIAL_DIM),
    _stripe_descriptors.StripeSpec(name="exchange", size=_EXCHANGE_DIM),
    _stripe_descriptors.StripeSpec(name="board_hab", size=_BOARD_HAB_DIM),
    _stripe_descriptors.StripeSpec(name="board_col", size=_BOARD_COL_DIM),
    _stripe_descriptors.StripeSpec(name="bird_id", size=_CHOICE_BIRD_ID_DIM),
    _stripe_descriptors.StripeSpec(name="bonus_id", size=_BONUS_ID_DIM),
    _stripe_descriptors.StripeSpec(name="bonus_delta", size=_BONUS_DELTA_DIM),
    _stripe_descriptors.StripeSpec(name="goal_delta", size=_GOAL_DELTA_DIM),
    _stripe_descriptors.StripeSpec(name="bonus_value", size=_BONUS_VALUE_DIM),
    _stripe_descriptors.StripeSpec(name="becomes_playable", size=_BIRD_ID_DIM),
]
_CHOICE_SETUP_STRIPE_SPECS: list[_stripe_descriptors.StripeSpec] = [
    _stripe_descriptors.StripeSpec(name="setup_agg", size=_SETUP_DIM),
    _stripe_descriptors.StripeSpec(name="kept_multihot", size=_KEPT_MULTIHOT_DIM),
]
CHOICE_BASE_LAYOUT = _stripe_descriptors.VectorLayout.from_stripe_specs(
    _CHOICE_STRIPE_SPECS
)
CHOICE_FULL_LAYOUT = _stripe_descriptors.VectorLayout.from_stripe_specs(
    _CHOICE_STRIPE_SPECS + _CHOICE_SETUP_STRIPE_SPECS
)
_OFF_KIND = CHOICE_BASE_LAYOUT.offset_of("kind")
_OFF_GAIN_FOOD = CHOICE_BASE_LAYOUT.offset_of("gain_food")
_OFF_PAY = CHOICE_BASE_LAYOUT.offset_of("pay")
_OFF_BOARD = CHOICE_BASE_LAYOUT.offset_of("board")
_OFF_MAIN_ACTION = CHOICE_BASE_LAYOUT.offset_of("main_action")
_OFF_SPECIAL = CHOICE_BASE_LAYOUT.offset_of("special")
_OFF_EXCHANGE = CHOICE_BASE_LAYOUT.offset_of("exchange")
_OFF_BOARD_HAB = CHOICE_BASE_LAYOUT.offset_of("board_hab")
_OFF_BOARD_COL = CHOICE_BASE_LAYOUT.offset_of("board_col")
_OFF_BIRD_ID = CHOICE_BASE_LAYOUT.offset_of("bird_id")
_OFF_BONUS_ID = CHOICE_BASE_LAYOUT.offset_of("bonus_id")
_OFF_BONUS_DELTA = CHOICE_BASE_LAYOUT.offset_of("bonus_delta")
_OFF_GOAL_DELTA = CHOICE_BASE_LAYOUT.offset_of("goal_delta")
_OFF_BONUS_VALUE = CHOICE_BASE_LAYOUT.offset_of("bonus_value")
_CHOICE_BASE_DIM = CHOICE_BASE_LAYOUT.total_size
_OFF_SETUP = CHOICE_FULL_LAYOUT.offset_of(
    "setup_agg"
)  # trailing; present iff include_setup
_OFF_KEPT_MULTIHOT = CHOICE_FULL_LAYOUT.offset_of(
    "kept_multihot"
)  # trailing; present iff include_setup

# Within-KIND indices
_KIND_BIRD = 0
_KIND_FOOD = 1
_KIND_HABITAT = 2
_KIND_PAYMENT = 3
_KIND_BOARD_TARGET = 4
_KIND_SPECIAL = 5

# Within-SPECIAL indices
_SPECIAL_IS_SKIP = 0
_SPECIAL_IS_SELF = 1

# Within-GAIN_FOOD: 0..N_FOODS-1 are the plain ALL_FOODS dice; the final two are
# the invertebrate/seed choice die taken as invertebrate / as seed.
_GAIN_FOOD_CHOICE_INV = cards.N_FOODS  # 5
_GAIN_FOOD_CHOICE_SEED = cards.N_FOODS + 1  # 6

# Within-MAIN_ACTION one-hot: the stable order of the four main actions.
_MAIN_ACTION_ORDER = [
    decisions.MainAction.GAIN_FOOD,
    decisions.MainAction.LAY_EGGS,
    decisions.MainAction.DRAW_CARDS,
    decisions.MainAction.PLAY_BIRD,
]

# Within-EXCHANGE indices: a PayCostChoice's symmetric pay->gain terms, as
# counts/magnitudes normalized by _EXCHANGE_SCALE. A self block (what the deciding
# player gives up / receives) then an opponent-gain block (what a shared-benefit
# power also grants the opponent — reserved until such an *optional* trade is
# modelled; the mandatory each-player powers keep their own decisions). The food
# *type* paid still rides the PAY_FOOD stripe; here food is a magnitude.
_EXCHANGE_CARDS_TO_DISCARD = 0
_EXCHANGE_FOOD_TO_PAY = 1
_EXCHANGE_EGGS_TO_PAY = 2
_EXCHANGE_FOOD_TO_GAIN = 3
_EXCHANGE_EGGS_TO_GAIN = 4
_EXCHANGE_CARDS_TO_DRAW = 5
_EXCHANGE_CARDS_TO_TUCK = 6
_EXCHANGE_OPP_FOOD_TO_GAIN = 7
_EXCHANGE_OPP_EGGS_TO_GAIN = 8
_EXCHANGE_OPP_CARDS_TO_DRAW = 9
_EXCHANGE_OPP_CARDS_TO_TUCK = 10
# Appended after the opponent-gain block (offsets above are checkpoint-aligned):
# extra bird plays the accept unlocks (the extra-play commit's gained_play_count).
_EXCHANGE_PLAYS_TO_GAIN = 11
# FRESH (slot 12): food cached on the bird (the cache-vs-keep accept's gained_cache_count).
# Adding this slot widens CHOICE_FEATURE_DIM by 1, which architecture_key detects via
# choice_dim — old checkpoints are refused cleanly with no architecture.py edits.
_EXCHANGE_CACHE_TO_GAIN = 12

# Within-BONUS_DELTA indices: a candidate bird's contribution to the deciding
# player's HELD bonus cards (filled for play / keep-bird / tray draw-source
# candidates — see ``choice_encode._fill_bonus_delta``). The marginal values
# price the +1 qualifying bird this candidate would add to the board.
_BONUS_DELTA_QUAL = 0  # held bonus cards this bird qualifies for (÷ _BONUS_COUNT_SCALE)
_BONUS_DELTA_STEPPED = 1  # summed stepped-VP delta at count+1 (÷ _BONUS_VALUE_SCALE)
_BONUS_DELTA_LINEAR = 2  # summed linear-VP delta at count+1 (÷ _BONUS_VALUE_SCALE)

# Within-BONUS_VALUE indices: the candidate bonus CARD's value to the deciding
# player — it prices the offered card itself, not a bird (filled for
# BonusCardChoice rows and a setup pick's kept bonus — see
# ``choice_encode._fill_bonus_value``). Where bonus_delta is marginal (the +1
# bird against held cards), these are standing values: what the card pays on the
# current board, plus how many hand/kept and tray birds could still qualify it.
_BONUS_VALUE_QUAL = 0  # board birds qualifying for this bonus (÷ _BONUS_COUNT_SCALE)
_BONUS_VALUE_STEPPED = 1  # stepped VP at that count (÷ _BONUS_VALUE_SCALE)
_BONUS_VALUE_LINEAR = 2  # linear VP at that count (÷ _BONUS_VALUE_SCALE)
_BONUS_VALUE_HAND = 3  # hand/kept birds qualifying (÷ _BONUS_COUNT_SCALE)
_BONUS_VALUE_TRAY = 4  # tray birds qualifying (÷ _BONUS_COUNT_SCALE)

# Within-SETUP indices (a setup pick's kept-subset aggregate stats — summaries the
# shared card table cannot reconstruct from the kept-bird identity multi-hot alone).
_SETUP_AGG_POINTS = 0
_SETUP_AGG_COST = 1
_SETUP_AGG_EGGS = 2
_SETUP_KEPT_COUNT = 3


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

# Curated subset of bonus cards whose qualifying condition is intrinsic to the
# card (not state-dependent at game-end, not already captured by another stripe).
# Ordered stably; the dense 0..N-1 index here is independent of cards.bonus_index().
#
# Dropped:
#   state-dependent  — Breeding Manager, Ecologist, Oologist, Visionary Leader
#   covered by food_cost stripe — Bird Feeder, Fishery Manager, Food Web Expert,
#       Omnivore Specialist, Rodentologist, Viticulturalist
#   covered by nest stripe — Enclosure Builder, Nest Box Builder, Platform Builder,
#       Wildlife Gardener
#   covered by habitats stripe — Forester, Prairie Manager, Wetland Scientist,
#       Bird Bander (multi-habitat = sum of habitat bits > 1)
#   covered by flocking/predator flags — Bird Counter, Falconer
#   fan-made — Caprimulgiform Specialist
_KEPT_BONUS_NAMES: tuple[str, ...] = (
    "Anatomist",
    "Backyard Birder",
    "Cartographer",
    "Historian",
    "Large Bird Specialist",
    "Passerine Specialist",
    "Photographer",
)
_BONUS_NAME_TO_INDEX: dict[str, int] = {
    name: index for index, name in enumerate(_KEPT_BONUS_NAMES)
}

# Rich per-card attribute vector (the ``N`` half of each slot's identity+attrs
# encoding). Offsets auto-accumulated from CARD_ATTR_LAYOUT; see
# ``_bird_attr_vector`` for the meaning of each stripe.
_FOOD_COST_VEC_DIM = cards.N_FOODS + 1  # 5 specific foods + wild
_BONUS_CATS_DIM = len(_KEPT_BONUS_NAMES)  # 7 curated categories (was 26)
_OR_COST_FLAG_DIM = 1  # 1 if the bird's food cost is an OR choice, 0 for AND
_ATTR_STRIPE_SPECS: list[_stripe_descriptors.StripeSpec] = [
    _stripe_descriptors.StripeSpec(name="points", size=1),
    _stripe_descriptors.StripeSpec(name="food_cost", size=_FOOD_COST_VEC_DIM),
    _stripe_descriptors.StripeSpec(name="nest", size=len(_NEST_BASE_TYPES)),
    _stripe_descriptors.StripeSpec(name="habitats", size=len(cards.ALL_HABITATS)),
    _stripe_descriptors.StripeSpec(name="flock", size=1),
    _stripe_descriptors.StripeSpec(name="pred", size=1),
    _stripe_descriptors.StripeSpec(name="wingspan", size=1),
    _stripe_descriptors.StripeSpec(name="egg_limit", size=1),
    _stripe_descriptors.StripeSpec(name="color", size=len(_COLORS)),
    _stripe_descriptors.StripeSpec(name="plays_bird", size=1),
    _stripe_descriptors.StripeSpec(name="caches_food", size=1),
    _stripe_descriptors.StripeSpec(name="bonus_cats", size=_BONUS_CATS_DIM),
    _stripe_descriptors.StripeSpec(name="power_ex", size=_EXCHANGE_DIM),
    _stripe_descriptors.StripeSpec(name="or_cost", size=_OR_COST_FLAG_DIM),
]
CARD_ATTR_LAYOUT = _stripe_descriptors.VectorLayout.from_stripe_specs(
    _ATTR_STRIPE_SPECS
)
_OFF_ATTR_POINTS = CARD_ATTR_LAYOUT.offset_of("points")
_OFF_ATTR_FOOD_COST = CARD_ATTR_LAYOUT.offset_of("food_cost")
_OFF_ATTR_NEST = CARD_ATTR_LAYOUT.offset_of("nest")
_OFF_ATTR_HAB = CARD_ATTR_LAYOUT.offset_of("habitats")
_OFF_ATTR_FLOCK = CARD_ATTR_LAYOUT.offset_of("flock")
_OFF_ATTR_PRED = CARD_ATTR_LAYOUT.offset_of("pred")
_OFF_ATTR_WINGSPAN = CARD_ATTR_LAYOUT.offset_of("wingspan")
_OFF_ATTR_EGG_LIMIT = CARD_ATTR_LAYOUT.offset_of("egg_limit")
_OFF_ATTR_COLOR = CARD_ATTR_LAYOUT.offset_of("color")
_OFF_ATTR_PLAYS_BIRD = CARD_ATTR_LAYOUT.offset_of("plays_bird")
_OFF_ATTR_CACHES_FOOD = CARD_ATTR_LAYOUT.offset_of("caches_food")
_OFF_ATTR_BONUS_CATS = CARD_ATTR_LAYOUT.offset_of("bonus_cats")
_OFF_ATTR_POWER_EX = CARD_ATTR_LAYOUT.offset_of("power_ex")
_OFF_ATTR_OR_COST = CARD_ATTR_LAYOUT.offset_of("or_cost")
_BIRD_ATTR_DIM = CARD_ATTR_LAYOUT.total_size  # 45 (was 44)

# The model's card encoder consumes, per card, this attribute vector concatenated
# with the card's identity one-hot, and outputs the shared ``[181, D]`` card table
# that every board / tray / hand / choice slot looks up. Defined here beside the
# attribute layout it builds on, so the encoder-input builder
# (``state_encode.card_feature_matrix``) and the model read one constant.
CARD_FEATURE_DIM = _BIRD_ATTR_DIM + _BIRD_ID_DIM  # 45 + 180 = 225 (was 224)

# Per-board-slot continuous block: mutable per-slot state only, with NO identity
# and NO attribute vector. The bird's identity is emitted separately as an integer
# index in the card-index block and looked up by the model's shared card table,
# which already carries the static attributes; only the slot's mutable state lives
# here.
_OFF_SLOT_MUT = 0
# Mutable: eggs, egg-capacity-remaining, cached food per type, tucked, activations.
_SLOT_MUT_EGGS = 0
_SLOT_MUT_EGG_CAP = 1
_SLOT_MUT_CACHED = 2  # start of the N_FOODS cached-by-type block
_SLOT_MUT_TUCKED = _SLOT_MUT_CACHED + cards.N_FOODS
_SLOT_MUT_ACTIVATIONS = _SLOT_MUT_TUCKED + 1
_SLOT_MUT_DIM = _SLOT_MUT_ACTIVATIONS + 1
_SLOT_CONT_DIM = _SLOT_MUT_DIM
_BOARD_CONT_STRIPE_DIM = _SLOTS_PER_BOARD * _SLOT_CONT_DIM

# Public aliases consumed by the board self-attention path (model/core.py). These
# are live-encoding constants only — a future FRESH change that shifts board-stripe
# offsets must freeze them in the era seam (StateEmbedOffsets), per the
# 2026-06-14 lesson in VERSIONING.md.
SLOTS_PER_BOARD: int = _SLOTS_PER_BOARD  # 15 — slots on one player's board
SLOT_SCALAR_DIM: int = _SLOT_MUT_DIM  # 9 — mutable scalars per board slot
BOARD_CONT_STRIPE_DIM: int = _BOARD_CONT_STRIPE_DIM  # 135 — 15 × 9

# The public face-up tray carries no continuous block at all: a tray bird has no
# mutable per-slot state, and its static attributes ride the shared card table via
# the card-index block. The stripe is therefore empty.
_TRAY_CONT_STRIPE_DIM = 0

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
# in a single shared ``nn.Embedding`` (padding row 0), and concatenates the result
# with the continuous features. The hand is carried as a multi-hot the model
# mean-pools through the same embedding weight. These offsets are the contract the
# model splits on; the decision-type one-hot stays the final state stripe.

N_BOARD_INDEX_SLOTS = 2 * _SLOTS_PER_BOARD  # POV board + opponent board
N_CARD_INDEX_SLOTS = N_BOARD_INDEX_SLOTS + state.TRAY_SIZE
HAND_MULTIHOT_DIM = _BIRD_ID_DIM
N_HAND_PLAYABLE_MULTIHOTS = 2
"""Number of extra hand-playability multi-hot stripes appended after ``hand_multihot``
in the state vector: ``hand_playable_me`` (playable right now) and
``hand_playable_eggs_me`` (egg-blocked but food/slot ready). Pre-0.6 artifacts
have 0 extra stripes; the compat shim freezes the offsets at the old values."""

# Per-habitat fields retained in the v0.9 compacted board-summary (row_length +
# total_eggs). Pre-0.9 shims used 6 fields; this constant keeps the live width
# explicit and avoids a magic 2 in the stripe spec below.
_BOARD_SUMMARY_FIELDS_PER_HABITAT = 2

_STATE_CONT_STRIPE_SPECS: list[_stripe_descriptors.StripeSpec] = [
    _stripe_descriptors.StripeSpec(
        name="turn_state",
        size=N_PLAYER_TURNS + 1,  # 26-dim player-turn one-hot + is_first_player flag
    ),
    _stripe_descriptors.StripeSpec(name="food_me", size=cards.N_FOODS),
    _stripe_descriptors.StripeSpec(name="food_opp", size=cards.N_FOODS),
    _stripe_descriptors.StripeSpec(name="board_me", size=_BOARD_CONT_STRIPE_DIM),
    _stripe_descriptors.StripeSpec(name="board_opp", size=_BOARD_CONT_STRIPE_DIM),
    _stripe_descriptors.StripeSpec(
        name="board_summary_me",
        size=state.N_HABITATS * _BOARD_SUMMARY_FIELDS_PER_HABITAT,
    ),
    _stripe_descriptors.StripeSpec(
        name="board_summary_opp",
        size=state.N_HABITATS * _BOARD_SUMMARY_FIELDS_PER_HABITAT,
    ),
    # hand_summary_me removed at the 0.9 compaction (carried into the 1.0
    # baseline): derived in-model from hand_multihot via set_summary_from_multihot
    # (same mechanism as the playability + tray-set stripes). No pre-1.0 shim
    # restores it — every loadable artifact already omits the inline stripe.
    _stripe_descriptors.StripeSpec(name="bonus_progress", size=4 * _BONUS_ID_DIM),
    _stripe_descriptors.StripeSpec(name="opp_bonus_count", size=1),
    _stripe_descriptors.StripeSpec(name="opp_hand_size", size=1),
    _stripe_descriptors.StripeSpec(name="birdfeeder", size=7),
    _stripe_descriptors.StripeSpec(
        name="misc_scalars",
        size=2,  # tray size + deck size (goal pts removed in v0.9)
    ),
    _stripe_descriptors.StripeSpec(name="round_goals", size=_ROUND_GOALS_STRIPE_DIM),
    _stripe_descriptors.StripeSpec(name="card_idx_block", size=N_CARD_INDEX_SLOTS),
    _stripe_descriptors.StripeSpec(name="hand_multihot", size=HAND_MULTIHOT_DIM),
    _stripe_descriptors.StripeSpec(name="hand_playable_me", size=HAND_MULTIHOT_DIM),
    _stripe_descriptors.StripeSpec(
        name="hand_playable_eggs_me", size=HAND_MULTIHOT_DIM
    ),
]
STATE_CONT_LAYOUT = _stripe_descriptors.VectorLayout.from_stripe_specs(
    _STATE_CONT_STRIPE_SPECS
)

# Offsets of the two 135-dim per-board continuous stripes within the state vector.
# Used by the board self-attention path (model/core.py) to slice the mutable
# scalars for each player's 15 slots. Live-encoding only — same era-seam caveat
# as SLOTS_PER_BOARD / SLOT_SCALAR_DIM above.
OFF_BOARD_ME: int = STATE_CONT_LAYOUT.offset_of("board_me")
OFF_BOARD_OPP: int = STATE_CONT_LAYOUT.offset_of("board_opp")

# Frozen: the byte offset of the hand-summary stripe in the *pre-0.9* state vector
# (= 343 = sum of all stripes before it in the v0.8 layout).  The stripe was
# removed from the live layout in v0.9 (the model derives it in-model from the
# hand multi-hot); pre-0.9 compat shims still need this constant to slice the
# frozen 1155-dim vector at the correct column.
HAND_SUMMARY_OFFSET: int = 343

# The raw input width of the hand encoder: multi-hot identity + summary stats.
HAND_ENCODER_INPUT_DIM = HAND_MULTIHOT_DIM + HAND_SUMMARY_DIM

# Continuous prefix preceding the card-index block; ``card_idx_block`` and
# ``hand_multihot`` are included in STATE_CONT_LAYOUT for offset arithmetic,
# but the prefix boundary is the start of ``card_idx_block``.
_CONT_PREFIX_DIM = STATE_CONT_LAYOUT.offset_of("card_idx_block")
OFF_CARD_INDEX = _CONT_PREFIX_DIM
OFF_HAND_MULTIHOT: int = STATE_CONT_LAYOUT.offset_of("hand_multihot")
OFF_DECISION_TYPE: int = STATE_CONT_LAYOUT.total_size

# Choice-vector card region the model embeds through the shared card table. The
# bird-index column sits just before bonus_id; the board_hab / board_col one-hots
# immediately precede it as pass-through features. These offsets are invariant
# to ``include_setup`` and the model slices on plain constants. The model embeds
# the candidate's index column to that card's vector (masked to zero when no
# bird); when ``include_setup``, the trailing kept_multihot stripe is summed
# through the card table into one more vector.
CHOICE_BOARD_HAB_OFFSET: int = _OFF_BOARD_HAB
CHOICE_BOARD_HAB_DIM: int = _BOARD_HAB_DIM
CHOICE_BOARD_COL_OFFSET: int = _OFF_BOARD_COL
CHOICE_BOARD_COL_DIM: int = _BOARD_COL_DIM
CHOICE_BIRD_ID_OFFSET = _OFF_BIRD_ID
CHOICE_BIRD_ID_DIM = _CHOICE_BIRD_ID_DIM
CHOICE_BONUS_ID_OFFSET = _OFF_BONUS_ID
CHOICE_SETUP_OFFSET = _OFF_SETUP
CHOICE_KEPT_MULTIHOT_OFFSET = _OFF_KEPT_MULTIHOT
CHOICE_KEPT_MULTIHOT_DIM = _KEPT_MULTIHOT_DIM
CHOICE_BECOMES_PLAYABLE_OFFSET: int = CHOICE_BASE_LAYOUT.offset_of("becomes_playable")
CHOICE_BECOMES_PLAYABLE_DIM: int = _BIRD_ID_DIM


def trunk_input_dim(
    state_dim: int,
    card_embed_dim: int,
    *,
    use_distinct_hand_model: bool = False,
    hand_summary_in_state: bool = False,
    hand_embed_dim: int | None = None,
    pooled_hand_width: int | None = None,
    tray_set_embedding: bool = False,
    n_playable_multihots: int = 0,
) -> int:
    """The state trunk's first-``Linear`` input width: the flat ``state_dim`` with
    the card-index block and the hand multi-hot replaced by their shared-embedding
    lookups — one ``card_embed_dim`` vector per index slot, plus one hand embedding.

    When ``use_distinct_hand_model`` is ``False`` (default) the hand multi-hot is
    pooled over the shared card vectors. The output width is ``pooled_hand_width``
    (from ``architecture.ModelArchitecture.pooled_hand_width``); when ``None``,
    defaults to ``card_embed_dim`` (MEAN mode / legacy back-compat).

    When ``True`` a dedicated hand encoder produces the hand embedding. In the live
    v0.9+ encoding the hand-summary stripe is *not* present in the state vector
    (it is derived in-model); in pre-0.9 frozen vectors it is present and must be
    excised from the continuous block. Set ``hand_summary_in_state=True`` when the
    state vector carries the 10-dim stripe so that this function correctly subtracts
    it from the continuous feed. ``PolicyValueNet._build_trunk`` derives this flag
    from ``StateEmbedOffsets.hand_summary_end > hand_summary`` so it is always
    era-correct without requiring an explicit override.

    ``tray_set_embedding`` (which requires the distinct hand encoder) appends one
    more ``hand_embed_dim``-wide vector: the tray *set* embedded through the same
    hand encoder, derived in-model from the three tray index columns.

    ``n_playable_multihots`` counts the extra hand-playability multi-hot blocks that
    follow the hand multi-hot in the state vector (``N_HAND_PLAYABLE_MULTIHOTS`` in
    live encoding, 0 for pre-0.6 compat shims). Each is removed from the flat
    state and re-embedded through the same path, adding one set-embedding-wide
    vector per block.

    This is the single source of truth for the post-embedding width (used by both
    ``model.PolicyValueNet`` and the configurator's parameter accounting)."""
    if use_distinct_hand_model:
        hand_width = hand_embed_dim if hand_embed_dim is not None else card_embed_dim
    else:
        hand_width = (
            pooled_hand_width if pooled_hand_width is not None else card_embed_dim
        )
    base = (
        state_dim
        - N_CARD_INDEX_SLOTS  # index columns -> per-slot embeddings
        - HAND_MULTIHOT_DIM  # hand multi-hot -> one hand embedding
        - n_playable_multihots
        * HAND_MULTIHOT_DIM  # extra playability multi-hots removed
        + N_CARD_INDEX_SLOTS * card_embed_dim
        + hand_width
        + n_playable_multihots * hand_width  # each embedded as a card set
    )
    if use_distinct_hand_model and hand_summary_in_state:
        # Stripe is present in the (pre-0.9) state vector and redirected into the
        # hand encoder's input; subtract it from the continuous feed.
        base -= HAND_SUMMARY_DIM
    if tray_set_embedding:
        base += hand_width
    return base


def choice_input_dim(
    choice_dim: int,
    card_embed_dim: int,
    *,
    include_setup: bool = False,
    has_becomes_playable: bool = True,
) -> int:
    """The per-choice encoder's first-``Linear`` input width: the flat
    ``choice_dim`` with the candidate's bird-index column replaced by its
    shared-embedding lookup — one ``card_embed_dim`` vector for the candidate.
    The ``board_hab`` / ``board_col`` one-hots pass through unchanged (they are
    not embedded). When ``include_setup``, the trailing kept_multihot stripe
    likewise collapses to one summed embedding; ``choice_dim`` alone cannot
    reveal whether the trailing setup stripes are present, so the flag is
    explicit (default matches ``DEFAULT_SPEC``).

    When ``has_becomes_playable`` is True (the live 0.9+ encoding), the
    ``becomes_playable`` 180-dim multi-hot is replaced by one summed embedding;
    set to False for pre-0.6 compat shims whose choice vector lacks the stripe."""
    base = (
        choice_dim
        - CHOICE_BIRD_ID_DIM  # candidate index column -> one embedding
        + card_embed_dim
    )
    if has_becomes_playable:
        base += (
            card_embed_dim - CHOICE_BECOMES_PLAYABLE_DIM
        )  # multi-hot -> one embedding
    if include_setup:
        base += card_embed_dim - CHOICE_KEPT_MULTIHOT_DIM  # multi-hot -> one embedding
    return base


def choice_passthrough_dim(
    choice_dim: int, *, include_setup: bool = False, has_becomes_playable: bool = True
) -> int:
    """The choice columns that pass straight through to the encoder — the flat
    ``choice_dim`` minus every card-region stripe the model replaces with a
    shared-embedding lookup (the candidate index column, the ``becomes_playable``
    multi-hot when present, and — when ``include_setup`` — the kept-set multi-hot).
    The ``board_hab`` / ``board_col`` one-hots pass through unchanged and are
    counted in the total. The architecture diagram's "additional inputs" count."""
    extra = choice_dim - CHOICE_BIRD_ID_DIM
    if has_becomes_playable:
        extra -= CHOICE_BECOMES_PLAYABLE_DIM
    if include_setup:
        extra -= CHOICE_KEPT_MULTIHOT_DIM
    return extra


# ---------------------------------------------------------------------------
# Spec-dependent totals + the decision-type one-hot. Indexed by Decision
# subclass so adding a new decision is a single registration in
# ``ALL_DECISION_CLASSES``. ``SetupDecision`` is last there, so excluding it
# (``include_setup=False``) drops only the trailing one-hot column.

_DECISION_TYPE_INDEX: dict[type[decisions.Decision[typing.Any]], int] = {
    cls: i for i, cls in enumerate(decisions.ALL_DECISION_CLASSES)
}


def decision_type_dim(spec: EncodingSpec = DEFAULT_SPEC) -> int:
    """Width of the decision-type one-hot stripe for ``spec`` — the number of
    decision classes the main model covers (one fewer when setup is excluded)."""
    return len(decisions.active_decision_classes(spec.include_setup))


def num_families(spec: EncodingSpec = DEFAULT_SPEC) -> int:
    """Number of scoring heads (judgment families) the main model trains for
    ``spec`` — one fewer when the ``SETUP`` head is excluded."""
    return len(decisions.active_decision_families(spec.include_setup))


def choice_feature_dim(spec: EncodingSpec = DEFAULT_SPEC) -> int:
    """Width of one choice feature row for ``spec``: the fixed base plus the
    trailing ``setup_agg`` + ``kept_multihot`` stripes when ``include_setup``."""
    return _CHOICE_BASE_DIM + (
        _SETUP_DIM + _KEPT_MULTIHOT_DIM if spec.include_setup else 0
    )


def state_feature_dim(spec: EncodingSpec = DEFAULT_SPEC) -> int:
    """Length of the state vector for ``spec``: the fixed prefix through the
    card / hand blocks plus the (spec-sized) trailing decision-type one-hot."""
    return OFF_DECISION_TYPE + decision_type_dim(spec)


def choice_layout(
    spec: EncodingSpec = DEFAULT_SPEC,
) -> _stripe_descriptors.VectorLayout:
    """The auto-computed choice stripe layout for ``spec``."""
    if spec.include_setup:
        return CHOICE_FULL_LAYOUT
    return CHOICE_BASE_LAYOUT


def state_cont_layout() -> _stripe_descriptors.VectorLayout:
    """The auto-computed state continuous-prefix layout (decision-type excluded)."""
    return STATE_CONT_LAYOUT


def card_attr_layout() -> _stripe_descriptors.VectorLayout:
    """The auto-computed per-card attribute vector layout."""
    return CARD_ATTR_LAYOUT


# Default-spec (setup-included) public sizes. A configured run derives its own
# sizes via the functions above from ``spec_for(use_setup_model)``.
DECISION_TYPE_DIM = decision_type_dim(DEFAULT_SPEC)
CHOICE_FEATURE_DIM = choice_feature_dim(DEFAULT_SPEC)

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
    "birds_no_eggs",
]
# Append-only: trained weights are aligned to these one-hot indices, and
# adding a 20th entry consumed the last slot of MAX_GOAL_CATEGORIES headroom —
# a 21st category must grow MAX_GOAL_CATEGORIES (a FRESH, checkpoint-
# invalidating change).

# Public alias of the goal-category ordering, re-exported from the package so
# the (separately-encoded) setup model can build its own round-goal one-hots
# against the same stable category order the state encoder uses.
GOAL_CATEGORIES = _GOAL_CATEGORIES
