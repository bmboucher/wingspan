"""The encoder's fixed layout: feature dimensions, stripe offsets, and the
normalization scales, plus the decision-type / goal-category orderings.

Most of the layout is fixed, but the *shape* of the main model is config-driven
on a single axis — whether the opening (``SetupDecision``) is scored by the main
model or delegated to the separate setup model. :class:`EncodingSpec` captures
that axis (``include_setup``); the setup-only pieces are kept LAST in every
stable order (the choice ``setup_agg`` stripe, the decision-type one-hot's setup
column, the ``SETUP`` scoring head), so excluding them is a clean truncation of
trailing dimensions that leaves every other offset and index unchanged. The
spec-dependent totals are therefore exposed as functions
(``state_feature_dim`` / ``choice_feature_dim`` / ``decision_type_dim`` /
``num_families``); the module constants ``CHOICE_FEATURE_DIM`` /
``DECISION_TYPE_DIM`` are the default-spec (setup-included) values.

The ``_OFF_*`` chain is evaluated top-to-bottom; the ``encode_state`` stripe
order and these offsets are part of the checkpoint format — reordering or
renumbering anything here is a FRESH (checkpoint-invalidating) change. See
CLAUDE.md "Checkpoint compatibility policy".
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


# ---------------------------------------------------------------------------
# Public constants — sanity bounds + normalization scales

# Choice-count safety bounds. The encoder never truncates: every choice gets a
# feature row, and an over-wide decision is never fatal — both thresholds below
# only drive (deduped) log notices. ``SOFT_CHOICE_WARN_THRESHOLD`` flags a
# decision merely wider than typical; ``RUNAWAY_CHOICE_THRESHOLD`` flags one so
# wide it almost certainly signals a bug rather than real play.
SOFT_CHOICE_WARN_THRESHOLD = 20
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
# torch-side derivation (``wingspan.hand_model``) so the two cannot disagree on
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
# ``setup_agg`` stripe is conditional (present iff ``EncodingSpec.include_setup``).

_KIND_DIM = 6  # bird, food, habitat, payment, board_target, special
_GAIN_FOOD_DIM = 7  # 5 foods + take-choice-die-as-invertebrate + ...-as-seed
_HABITAT_DIM = 3  # habitat one-hot
_PAY_FOOD_DIM = 5  # food payment: count per food
_MAIN_ACTION_DIM = 4  # one-hot over the four main actions
_SPECIAL_DIM = 2  # is_skip, is_self
_EXCHANGE_DIM = 12  # symmetric pay->gain terms: 8 self + 4 opponent-gain (below)
_BONUS_DELTA_DIM = 3  # candidate bird's contribution to held bonus cards (below)
_GOAL_DELTA_SLOT_DIM = 2  # count_delta + vp_delta per round-goal slot
_GOAL_DELTA_DIM = 4 * _GOAL_DELTA_SLOT_DIM  # 8 (4 round goals × 2 scalars)
_GOAL_DELTA_COUNT = 0  # within-slot: count change (÷ _GOAL_COUNT_SCALE)
_GOAL_DELTA_VP = 1  # within-slot: VP delta (÷ _ROUND_GOAL_POINTS_SCALE)
_SETUP_DIM = 4  # setup kept-subset aggregates (only when include_setup)

# The board_target stripe is a per-board-slot block: 8 scalars repeated over
# every board slot, paired with a parallel integer card-index block the model
# embeds through the shared card table (one card vector per slot). Per slot:
# lay_eggs, pay_eggs, cached food x5 (ALL_FOODS order), tucked.
_BT_SLOT_SCALARS = 8
_BT_LAY_EGGS = 0
_BT_PAY_EGGS = 1
_BT_CACHED = 2  # start of the N_FOODS cached-by-type block
_BT_TUCKED = _BT_CACHED + cards.N_FOODS  # 7
_BOARD_TARGET_DIM = _SLOTS_PER_BOARD * _BT_SLOT_SCALARS  # 15 * 8 = 120
_BOARD_IDX_SLOTS = _SLOTS_PER_BOARD  # 15 integer card indices, embedded by the model

# Card-identity stripes: a one-hot over every core-set bird / bonus card, so a
# specific card — or, for the setup pick, a *set* of cards as a multi-hot — is
# encoded by identity. The model maps the bird stripe through the shared card
# encoder (the same ``[181, D]`` table the state board / tray slots use). Sized
# from the loaded catalog (180 birds / 26 bonus cards in the core set).
_BIRD_ID_DIM = cards.n_birds()
_BONUS_ID_DIM = cards.n_bonus_cards()

# Stripe offsets (cumulative). The board-index block and bird-identity one-hot
# (the two card regions the model embeds) sit together just before bonus_id;
# bonus_id is followed by bonus_delta then goal_delta (per-candidate contribution
# scalars), and the conditional setup_agg stripe trails everything — so the
# card-region offsets the model slices on stay invariant to ``include_setup``.
_OFF_KIND = 0
_OFF_GAIN_FOOD = _OFF_KIND + _KIND_DIM
_OFF_HAB = _OFF_GAIN_FOOD + _GAIN_FOOD_DIM
_OFF_PAY = _OFF_HAB + _HABITAT_DIM
_OFF_BOARD = _OFF_PAY + _PAY_FOOD_DIM
_OFF_MAIN_ACTION = _OFF_BOARD + _BOARD_TARGET_DIM
_OFF_SPECIAL = _OFF_MAIN_ACTION + _MAIN_ACTION_DIM
_OFF_EXCHANGE = _OFF_SPECIAL + _SPECIAL_DIM
_OFF_BOARD_IDX = _OFF_EXCHANGE + _EXCHANGE_DIM
_OFF_BIRD_ID = _OFF_BOARD_IDX + _BOARD_IDX_SLOTS
_OFF_BONUS_ID = _OFF_BIRD_ID + _BIRD_ID_DIM
_OFF_BONUS_DELTA = _OFF_BONUS_ID + _BONUS_ID_DIM
_OFF_GOAL_DELTA = _OFF_BONUS_DELTA + _BONUS_DELTA_DIM
_OFF_SETUP = _OFF_GOAL_DELTA + _GOAL_DELTA_DIM  # trailing; present iff include_setup
_CHOICE_BASE_DIM = _OFF_SETUP  # row width without setup_agg (include_setup=False)

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

# Within-BONUS_DELTA indices: a candidate bird's contribution to the deciding
# player's HELD bonus cards (filled for play / keep-bird / tray draw-source
# candidates — see ``choice_encode._fill_bonus_delta``). The marginal values
# price the +1 qualifying bird this candidate would add to the board.
_BONUS_DELTA_QUAL = 0  # held bonus cards this bird qualifies for (÷ _BONUS_COUNT_SCALE)
_BONUS_DELTA_STEPPED = 1  # summed stepped-VP delta at count+1 (÷ _BONUS_VALUE_SCALE)
_BONUS_DELTA_LINEAR = 2  # summed linear-VP delta at count+1 (÷ _BONUS_VALUE_SCALE)

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

# The model's card encoder consumes, per card, this attribute vector concatenated
# with the card's identity one-hot, and outputs the shared ``[181, D]`` card table
# that every board / tray / hand / choice slot looks up. Defined here beside the
# attribute layout it builds on, so the encoder-input builder
# (``state_encode.card_feature_matrix``) and the model read one constant.
CARD_FEATURE_DIM = _BIRD_ATTR_DIM + _BIRD_ID_DIM  # 49 + 180 = 229

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

# Offset of the 10-dim hand-summary stripe within the state vector (= within the
# continuous prefix, since it precedes the card-index block). Used by the model's
# ``_embed_state`` when ``use_distinct_hand_model`` is active to split the 10 dims
# away from the continuous trunk feed and redirect them into the hand encoder.
HAND_SUMMARY_OFFSET: int = (
    5
    + 5  # food inventories (me + opp)
    + 2 * _BOARD_CONT_STRIPE_DIM  # board continuous (me, opp)
    + _TRAY_CONT_STRIPE_DIM  # tray continuous
    + 18
    + 18  # board summaries (me, opp)
)

# The raw input width of the hand encoder: multi-hot identity + summary stats.
HAND_ENCODER_INPUT_DIM = HAND_MULTIHOT_DIM + HAND_SUMMARY_DIM

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
    + HAND_SUMMARY_DIM  # my hand summary
    + 4 * _BONUS_ID_DIM  # bonus progress (held + count + stepped + linear)
    + 1  # opponent bonus-card count
    + 1  # opponent hand size
    + 6  # birdfeeder (5 single-food faces + choice-die count)
    + 7  # misc scalars
    + _ROUND_GOALS_STRIPE_DIM  # all four round goals
)
OFF_CARD_INDEX = _CONT_PREFIX_DIM
OFF_HAND_MULTIHOT = OFF_CARD_INDEX + N_CARD_INDEX_SLOTS
OFF_DECISION_TYPE = OFF_HAND_MULTIHOT + HAND_MULTIHOT_DIM

# Choice-vector card regions the model embeds through the shared card table. The
# board-index block sits immediately before the candidate bird one-hot; both
# precede bonus_id and the trailing (conditional) setup_agg stripe, so these
# offsets are invariant to ``include_setup`` and the model slices on plain
# constants. The model embeds a single-card candidate's one-hot to that card's
# vector and each of the 15 board-slot indices to its card vector.
CHOICE_BOARD_IDX_OFFSET = _OFF_BOARD_IDX
CHOICE_BOARD_IDX_SLOTS = _BOARD_IDX_SLOTS
CHOICE_BIRD_ID_OFFSET = _OFF_BIRD_ID
CHOICE_BIRD_ID_DIM = _BIRD_ID_DIM
CHOICE_BONUS_ID_OFFSET = _OFF_BONUS_ID
CHOICE_SETUP_OFFSET = _OFF_SETUP


def trunk_input_dim(
    state_dim: int,
    card_embed_dim: int,
    *,
    use_distinct_hand_model: bool = False,
    hand_embed_dim: int | None = None,
    tray_set_embedding: bool = False,
) -> int:
    """The state trunk's first-``Linear`` input width: the flat ``state_dim`` with
    the card-index block and the hand multi-hot replaced by their shared-embedding
    lookups — one ``card_embed_dim`` vector per index slot, plus one hand embedding.

    When ``use_distinct_hand_model`` is ``False`` (default) the hand embedding is a
    mean-pool of the held cards' shared card vectors (``card_embed_dim`` wide), and
    the 10-dim hand-summary stripe in the continuous prefix reaches the trunk as-is.

    When ``True`` a dedicated hand encoder produces the hand embedding from the
    multi-hot concatenated with the hand-summary; the 10-dim hand-summary stripe is
    redirected into that encoder instead of passing through to the trunk, so the
    trunk's continuous input is ``HAND_SUMMARY_DIM`` narrower. The encoder's output
    width is ``hand_embed_dim`` (``None`` = match ``card_embed_dim``, mirroring
    ``architecture.ModelArchitecture.hand_embed_width``).

    ``tray_set_embedding`` (which requires the distinct hand encoder) appends one
    more ``hand_embed_dim``-wide vector: the tray *set* embedded through the same
    hand encoder, derived in-model from the three tray index columns — the tray's
    per-slot card-table lookups are unchanged, giving 3·M + N tray dims in total.

    This is the single source of truth for the post-embedding width (used by both
    ``model.PolicyValueNet`` and the configurator's parameter accounting)."""
    hand_width = (
        (hand_embed_dim if hand_embed_dim is not None else card_embed_dim)
        if use_distinct_hand_model
        else card_embed_dim
    )
    base = (
        state_dim
        - N_CARD_INDEX_SLOTS  # index columns -> per-slot embeddings
        - HAND_MULTIHOT_DIM  # hand multi-hot -> one hand embedding
        + N_CARD_INDEX_SLOTS * card_embed_dim
        + hand_width
    )
    if use_distinct_hand_model:
        base -= HAND_SUMMARY_DIM
    if tray_set_embedding:
        base += hand_width
    return base


def choice_input_dim(choice_dim: int, card_embed_dim: int) -> int:
    """The per-choice encoder's first-``Linear`` input width: the flat
    ``choice_dim`` with the candidate's bird-identity one-hot AND the 15-slot
    board-index block replaced by their shared-embedding lookups — one
    ``card_embed_dim`` vector per board slot plus one for the candidate."""
    return (
        choice_dim
        - CHOICE_BIRD_ID_DIM  # candidate one-hot -> one embedding
        - CHOICE_BOARD_IDX_SLOTS  # board index columns -> per-slot embeddings
        + card_embed_dim
        + CHOICE_BOARD_IDX_SLOTS * card_embed_dim
    )


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
    trailing ``setup_agg`` stripe when ``include_setup``."""
    return _CHOICE_BASE_DIM + (_SETUP_DIM if spec.include_setup else 0)


def state_feature_dim(spec: EncodingSpec = DEFAULT_SPEC) -> int:
    """Length of the state vector for ``spec``: the fixed prefix through the
    card / hand blocks plus the (spec-sized) trailing decision-type one-hot."""
    return OFF_DECISION_TYPE + decision_type_dim(spec)


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
]

# Public alias of the goal-category ordering, re-exported from the package so
# the (separately-encoded) setup model can build its own round-goal one-hots
# against the same stable category order the state encoder uses.
GOAL_CATEGORIES = _GOAL_CATEGORIES
