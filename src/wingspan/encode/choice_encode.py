# pyright: reportPrivateUsage=false
# (this encoder reads the shared, package-private layout constants in
# ``layout`` -- a deliberate intra-package coupling, not a privacy break)
"""The choice encoder: ``encode_choices`` featurizes every legal choice in a
decision into a ``(n_choices, choice_feature_dim(spec))`` matrix, dispatching on
the concrete ``Choice`` subclass through ``_CHOICE_FEATURIZERS``. The ``_fill_*``
helpers write the shared per-card / per-food / per-habitat / per-board stripes.
"""

from __future__ import annotations

import logging

import numpy as np

from wingspan import cards, decisions, state
from wingspan.encode import layout

logger = logging.getLogger("wingspan.encode")

# Decision class names already warned about for crossing each choice-count
# threshold, so each notice fires once per class per process rather than on every
# wide decision (the setup deal alone would otherwise log it twice per game).
# One set per threshold so the soft and runaway notices are independent.
_WARNED_WIDE: set[str] = set()
_WARNED_RUNAWAY: set[str] = set()


def encode_choices(
    decision: layout._AnyDecision,
    state: state.GameState,
    spec: layout.EncodingSpec = layout.DEFAULT_SPEC,
) -> np.ndarray:
    """Featurize every choice in ``decision``.

    Returns a float32 array of shape ``(n_choices, layout.choice_feature_dim(spec))``.
    All returned rows correspond to legal choices — there is no padding or
    truncation, and the action space is implicitly variable across decisions.
    The caller (training loop) handles batched padding + masking. ``spec`` selects
    the config-driven row width (only the trailing ``setup_agg`` stripe varies).
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
    if (
        n_choices > layout.RUNAWAY_CHOICE_THRESHOLD
        and decision_name not in _WARNED_RUNAWAY
    ):
        _WARNED_RUNAWAY.add(decision_name)
        logger.warning(
            "Decision %s produced %d choices (> %d runaway threshold) for "
            "player %d — featurizing all of them, but this likely signals a "
            "choice-generation bug",
            decision_name,
            n_choices,
            layout.RUNAWAY_CHOICE_THRESHOLD,
            decision.player_id,
        )
    # The soft-threshold notice is a one-off-per-decision-class signal that a
    # decision ballooned wider than typical. SetupDecision (504) and a food-rich
    # PayBirdFoodDecision routinely and legitimately exceed it, so logging on every
    # such decision floods the log and adds per-call overhead in the hot path —
    # dedupe by class name so it fires once per class per process. Logged at INFO
    # (it is informational, not a fault): it still reaches the dashboard's file
    # log but never the console, so it can't flicker the live FLIGHT PLAN
    # display the way a WARNING surfaced by a stray stderr handler would.
    if (
        n_choices > layout.SOFT_CHOICE_WARN_THRESHOLD
        and decision_name not in _WARNED_WIDE
    ):
        _WARNED_WIDE.add(decision_name)
        logger.info(
            "Decision %s exposes %d choices (> %d soft threshold) for player %d",
            decision_name,
            n_choices,
            layout.SOFT_CHOICE_WARN_THRESHOLD,
            decision.player_id,
        )
    # Featurize straight into each row view rather than building a throwaway
    # array per candidate and copying it in — the rows start zeroed, and the
    # handlers only ever index-assign their own stripes.
    feats = np.zeros((n_choices, layout.choice_feature_dim(spec)), dtype=np.float32)
    for i, choice in enumerate(decision.choices):
        _featurize_choice(feats[i], decision, choice, state)
    return feats


###### PRIVATE #######

#### Per-choice featurization ####


def _featurize_choice(
    feat: np.ndarray,
    decision: layout._AnyDecision,
    choice: decisions.Choice,
    state: state.GameState,
) -> None:
    """Fill the pre-zeroed choice-feature row ``feat`` for one (decision, choice)
    pair, dispatching on the concrete Choice subclass.

    Writes into the caller's row view rather than allocating a fresh vector, so
    ``encode_choices`` builds its ``(n_choices, DIM)`` matrix with no per-row
    throwaway. The typed ``choice`` parameter keeps ``type(choice)`` a known
    ``type[Choice]`` for the dispatch lookup."""
    _CHOICE_FEATURIZERS.get(type(choice), _featurize_default)(
        feat, decision, choice, state
    )


def _featurize_default(
    feat: np.ndarray,
    decision: layout._AnyDecision,
    choice: decisions.Choice,
    state: state.GameState,
) -> None:
    feat[layout._OFF_KIND + layout._KIND_SPECIAL] = 1.0


def _featurize_skip(
    feat: np.ndarray,
    decision: layout._AnyDecision,
    choice: decisions.SkipChoice,
    state: state.GameState,
) -> None:
    feat[layout._OFF_KIND + layout._KIND_SPECIAL] = 1.0
    feat[layout._OFF_SPECIAL + layout._SPECIAL_IS_SKIP] = 1.0


def _featurize_reset_birdfeeder(
    feat: np.ndarray,
    decision: layout._AnyDecision,
    choice: decisions.ResetBirdfeederChoice,
    state: state.GameState,
) -> None:
    # The "yes, reroll" affirmative. Carries no data, so only the special-kind
    # bit is set; the decision-type stripe identifies the reset decision and the
    # absent is-skip bit distinguishes it from the paired ``SkipChoice``.
    feat[layout._OFF_KIND + layout._KIND_SPECIAL] = 1.0


def _featurize_pay_cost(
    feat: np.ndarray,
    decision: layout._AnyDecision,
    choice: decisions.PayCostChoice,
    state: state.GameState,
) -> None:
    # The 'accept the offered exchange' branch is distinct from skip — the network
    # can learn to prefer or avoid it independently. KIND_SPECIAL marks it a commit
    # token; the trade's resource ledger lives in the EXCHANGE stripe (a symmetric
    # pay->gain block, self then opponent-gain) so the skip-optional head weighs
    # what is gained against what is paid. The food *type* paid, if any, also rides
    # the PAY_FOOD stripe; in the EXCHANGE stripe food is a magnitude.
    feat[layout._OFF_KIND + layout._KIND_SPECIAL] = 1.0
    if choice.paid_food is not None:
        _add_pay_food(feat, choice.paid_food)
    exchange_terms = {
        layout._EXCHANGE_CARDS_TO_DISCARD: choice.paid_card_count,
        layout._EXCHANGE_FOOD_TO_PAY: (
            choice.paid_food_count if choice.paid_food is None else 1
        ),
        layout._EXCHANGE_EGGS_TO_PAY: choice.paid_egg_count,
        layout._EXCHANGE_FOOD_TO_GAIN: choice.gained_food_count,
        layout._EXCHANGE_EGGS_TO_GAIN: choice.gained_egg_count,
        layout._EXCHANGE_CARDS_TO_DRAW: choice.gained_card_count,
        layout._EXCHANGE_CARDS_TO_TUCK: choice.gained_tuck_count,
        layout._EXCHANGE_PLAYS_TO_GAIN: choice.gained_play_count,
        layout._EXCHANGE_OPP_FOOD_TO_GAIN: choice.opp_gained_food_count,
        layout._EXCHANGE_OPP_EGGS_TO_GAIN: choice.opp_gained_egg_count,
        layout._EXCHANGE_OPP_CARDS_TO_DRAW: choice.opp_gained_card_count,
        layout._EXCHANGE_OPP_CARDS_TO_TUCK: choice.opp_gained_tuck_count,
    }
    for index, count in exchange_terms.items():
        feat[layout._OFF_EXCHANGE + index] = count / layout._EXCHANGE_SCALE


def _featurize_main_action(
    feat: np.ndarray,
    decision: layout._AnyDecision,
    choice: decisions.MainActionChoice,
    state: state.GameState,
) -> None:
    # A one-hot over the four main actions — never an index-as-scalar (the four
    # actions have no ordinal relationship). KIND stays SPECIAL; the dedicated
    # main_action stripe distinguishes the four options.
    feat[layout._OFF_KIND + layout._KIND_SPECIAL] = 1.0
    for i, action in enumerate(layout._MAIN_ACTION_ORDER):
        if choice.action == action:
            feat[layout._OFF_MAIN_ACTION + i] = 1.0
            break


def _featurize_bird(
    feat: np.ndarray,
    decision: layout._AnyDecision,
    choice: decisions.BirdChoice,
    state: state.GameState,
) -> None:
    # A candidate bird from a hand / drawn pile (keep, tuck, or discard picks).
    # The bonus_delta stripe prices what acquiring — or for tuck/discard, giving
    # up — this bird means for the held bonus cards; the decision-type stripe in
    # the state vector tells the net which direction applies.
    feat[layout._OFF_KIND + layout._KIND_BIRD] = 1.0
    _fill_bird_identity(feat, choice.bird)
    _fill_bonus_delta(feat, state.players[decision.player_id], choice.bird)
    _fill_goal_delta(feat, decision.player_id, choice.bird, state)


def _featurize_play_bird(
    feat: np.ndarray,
    decision: layout._AnyDecision,
    choice: decisions.PlayBirdChoice,
    state: state.GameState,
) -> None:
    # A play candidate from ``PlayBirdDecision``: the bird-identity stripe carries
    # the card (its attributes ride the shared card table), and the habitat stripe
    # carries the bundled habitat pick. KIND stays BIRD — it is fundamentally a
    # bird play — while the habitat stripe distinguishes the per-habitat variants
    # of the same bird. The costs are follow-up decisions (RemoveEggDecision /
    # PayBirdFoodDecision), so no payment stripe is filled here; the bonus_delta
    # and goal_delta stripes price the play's contribution to held bonus cards and
    # round-goal standings.
    feat[layout._OFF_KIND + layout._KIND_BIRD] = 1.0
    _fill_bird_identity(feat, choice.bird)
    _fill_habitat(feat, choice.habitat)
    _fill_bonus_delta(feat, state.players[decision.player_id], choice.bird)
    _fill_goal_delta(feat, decision.player_id, choice.bird, state)


def _featurize_food_payment(
    feat: np.ndarray,
    decision: layout._AnyDecision,
    choice: decisions.FoodPaymentChoice,
    state: state.GameState,
) -> None:
    # A complete payment multiset for a committed bird play (PayBirdFoodDecision).
    # KIND_PAYMENT marks the row a whole-payment pick; the PAY stripe carries the
    # candidate's per-food counts, and the committed play rides along as context —
    # bird identity (embedded through the shared card table) plus habitat — so the
    # spend-food head sees *what* the payment is for, not just the tokens leaving.
    feat[layout._OFF_KIND + layout._KIND_PAYMENT] = 1.0
    _fill_payment(feat, choice.payment)
    if isinstance(decision, decisions.PayBirdFoodDecision):
        _fill_bird_identity(feat, decision.bird)
        _fill_habitat(feat, decision.habitat)


def _featurize_played_bird(
    feat: np.ndarray,
    decision: layout._AnyDecision,
    choice: decisions.PlayedBirdChoice,
    state: state.GameState,
) -> None:
    # A bird already in play, by reference (move-bird / repeat-power, MISC_RARE).
    # The candidate is identified by its bird-identity stripe (embedded); the
    # board block is filled for context with no add/take flag (this is not an egg
    # decision). The bird is on the deciding player's own board in the core set.
    feat[layout._OFF_KIND + layout._KIND_BIRD] = 1.0
    _fill_bird_identity(feat, choice.played_bird.bird)
    _fill_board_slots(feat, state.players[decision.player_id], None, None, False, False)


def _featurize_habitat(
    feat: np.ndarray,
    decision: layout._AnyDecision,
    choice: decisions.HabitatChoice,
    state: state.GameState,
) -> None:
    feat[layout._OFF_KIND + layout._KIND_HABITAT] = 1.0
    _fill_habitat(feat, choice.habitat)


def _featurize_food(
    feat: np.ndarray,
    decision: layout._AnyDecision,
    choice: decisions.FoodChoice,
    state: state.GameState,
) -> None:
    feat[layout._OFF_KIND + layout._KIND_FOOD] = 1.0
    _fill_gain_food(feat, choice.food, choice.from_choice_die)


def _featurize_board_target(
    feat: np.ndarray,
    decision: layout._AnyDecision,
    choice: decisions.BoardTargetChoice,
    state: state.GameState,
) -> None:
    # A board-slot target: fill the whole 15-slot board block from the deciding
    # player's board and flag the targeted slot as laying an egg (lay-egg decision)
    # or paying an egg (remove-egg decision). The occupying birds ride the parallel
    # card-index block the model embeds.
    feat[layout._OFF_KIND + layout._KIND_BOARD_TARGET] = 1.0
    is_lay = isinstance(decision, decisions.LayEggDecision)
    is_pay = isinstance(decision, decisions.RemoveEggDecision)
    _fill_board_slots(
        feat,
        state.players[decision.player_id],
        choice.habitat,
        choice.slot,
        is_lay,
        is_pay,
    )


def _featurize_bonus_card(
    feat: np.ndarray,
    decision: layout._AnyDecision,
    choice: decisions.BonusCardChoice,
    state: state.GameState,
) -> None:
    # Identity via the bonus one-hot stripe (a learned per-bonus embedding),
    # replacing the old id-hash so distinct bonus cards are fully distinguished.
    feat[layout._OFF_KIND + layout._KIND_SPECIAL] = 1.0
    _fill_bonus_identity(feat, choice.bonus_card)


def _featurize_draw_source(
    feat: np.ndarray,
    decision: layout._AnyDecision,
    choice: decisions.DrawSourceChoice,
    state: state.GameState,
) -> None:
    tray_bird: cards.Bird | None = None
    if choice.source == "tray" and choice.tray_index is not None:
        if 0 <= choice.tray_index < len(state.tray):
            tray_bird = state.tray[choice.tray_index]
    if tray_bird is not None:
        feat[layout._OFF_KIND + layout._KIND_BIRD] = 1.0
        _fill_bird_identity(feat, tray_bird)
        _fill_bonus_delta(feat, state.players[decision.player_id], tray_bird)
        _fill_goal_delta(feat, decision.player_id, tray_bird, state)
    else:
        feat[layout._OFF_KIND + layout._KIND_SPECIAL] = 1.0


def _featurize_player_id(
    feat: np.ndarray,
    decision: layout._AnyDecision,
    choice: decisions.PlayerIdChoice,
    state: state.GameState,
) -> None:
    # Flag whether this player option is the deciding player ("self") so the
    # network can learn self-vs-opponent preference cheaply (e.g. the Hummingbird
    # food-gain order pick — going first is usually best).
    feat[layout._OFF_KIND + layout._KIND_SPECIAL] = 1.0
    feat[layout._OFF_SPECIAL + layout._SPECIAL_IS_SELF] = (
        1.0 if choice.player_id == decision.player_id else 0.0
    )


def _featurize_setup(
    feat: np.ndarray,
    decision: layout._AnyDecision,
    choice: decisions.SetupChoice,
    state: state.GameState,
) -> None:
    """Featurize a single combined setup pick.

    Only reached when the main model carries setup (``include_setup``), so the
    row is wide enough to hold the trailing ``setup_agg`` stripe. The 504
    candidates share a state vector, so the network reads the choice features to
    tell them apart: (a) a multi-hot of the *specific* kept birds in the
    bird-identity stripe, (b) aggregate stats of the kept-card subset in the
    setup_agg stripe, (c) a multi-hot of foods spent in the PAY_FOOD stripe, and
    (d) the kept bonus card's identity one-hot.
    """
    feat[layout._OFF_KIND + layout._KIND_SPECIAL] = 1.0
    # PAY_FOOD stripe encodes the foods *spent* (complement of kept_foods), so the
    # network sees the same payment signal as every other paying decision.
    for i, food in enumerate(cards.ALL_FOODS):
        if food not in choice.kept_foods:
            feat[layout._OFF_PAY + i] = 1.0 / layout._PAYMENT_COUNT_SCALE
    kept = choice.kept_cards
    # Identity multi-hot of the kept birds plus the setup_agg aggregate stats that
    # summarise the subset. One pass sets each identity bit and accumulates all
    # three sums (the setup deal featurizes 504 candidates, so folding three
    # generator passes into one matters). The aggregates live in the dedicated
    # SETUP stripe because they are kept-*subset* summaries the shared card table
    # cannot reconstruct from the identity multi-hot.
    if kept:
        points = 0.0
        cost = 0.0
        eggs = 0.0
        for bird in kept:
            feat[layout._OFF_BIRD_ID + cards.bird_index(bird)] = 1.0
            points += bird.points
            cost += bird.food_cost.total
            eggs += bird.egg_limit
        feat[layout._OFF_SETUP + layout._SETUP_AGG_POINTS] = points / (
            layout._POINTS_SCALE * layout._ROW_SLOTS_SCALE
        )
        feat[layout._OFF_SETUP + layout._SETUP_AGG_COST] = cost / (
            layout._FOOD_COST_SCALE * layout._ROW_SLOTS_SCALE
        )
        feat[layout._OFF_SETUP + layout._SETUP_AGG_EGGS] = eggs / (
            layout._EGG_LIMIT_SCALE * layout._ROW_SLOTS_SCALE
        )
    feat[layout._OFF_SETUP + layout._SETUP_KEPT_COUNT] = (
        len(kept) / layout._ROW_SLOTS_SCALE
    )
    if choice.bonus_card is not None:
        _fill_bonus_identity(feat, choice.bonus_card)


_CHOICE_FEATURIZERS: dict[type[decisions.Choice], layout._ChoiceFeaturizer] = {
    decisions.SkipChoice: _featurize_skip,
    decisions.ResetBirdfeederChoice: _featurize_reset_birdfeeder,
    decisions.PayCostChoice: _featurize_pay_cost,
    decisions.MainActionChoice: _featurize_main_action,
    decisions.BirdChoice: _featurize_bird,
    decisions.PlayBirdChoice: _featurize_play_bird,
    decisions.FoodPaymentChoice: _featurize_food_payment,
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


def _fill_bird_identity(feat: np.ndarray, bird: cards.Bird) -> None:
    """Set the bird-identity one-hot bit for ``bird``. Called for every
    bird-carrying choice, and once per card to build a kept-set multi-hot. The
    model maps this stripe through the shared card encoder, so a candidate's
    static attributes and its learned per-card vector arrive together."""
    feat[layout._OFF_BIRD_ID + cards.bird_index(bird)] = 1.0


def _fill_bonus_identity(feat: np.ndarray, bonus_card: cards.BonusCard) -> None:
    """Set the bonus-card identity one-hot bit for ``bonus_card``."""
    feat[layout._OFF_BONUS_ID + cards.bonus_index(bonus_card)] = 1.0


def _fill_bonus_delta(feat: np.ndarray, player: state.Player, bird: cards.Bird) -> None:
    """Fill the bonus_delta stripe: how much ``bird`` reaching ``player``'s board
    would advance the bonus cards ``player`` currently holds. Three scalars — the
    count of held cards the bird qualifies for, and the summed stepped / linear
    VP gain from the +1 qualifying bird — so the net reads a candidate's bonus
    contribution directly instead of inferring it from the bonus-progress and
    card-attribute stripes. All zero when the bird qualifies for no held card."""
    from wingspan.engine import scoring  # local: keeps encode engine-free at import

    qual = 0
    stepped = 0.0
    linear = 0.0
    for bonus_card in player.bonus_cards:
        if bonus_card.name not in bird.bonus_categories:
            continue
        qual += 1
        count = scoring.bonus_qualifying_count(player, bonus_card)
        stepped += scoring.bonus_score_for_count(
            bonus_card, count + 1
        ) - scoring.bonus_score_for_count(bonus_card, count)
        linear += scoring.bonus_linear_value_for_count(
            bonus_card, count + 1
        ) - scoring.bonus_linear_value_for_count(bonus_card, count)
    if qual == 0:
        return
    base = layout._OFF_BONUS_DELTA
    feat[base + layout._BONUS_DELTA_QUAL] = qual / layout._BONUS_COUNT_SCALE
    feat[base + layout._BONUS_DELTA_STEPPED] = stepped / layout._BONUS_VALUE_SCALE
    feat[base + layout._BONUS_DELTA_LINEAR] = linear / layout._BONUS_VALUE_SCALE


def _fill_goal_delta(
    feat: np.ndarray,
    player_id: int,
    bird: cards.Bird,
    game_state: state.GameState,
) -> None:
    """Fill the goal_delta stripe: for each of the 4 round goals, how much
    playing ``bird`` would change the deciding player's category count and
    placement VP. count_delta is always 0 or 1 (freshly played birds start
    with no eggs or tucks); vp_delta depends on current standings. Both stay
    zero for goals where the bird has no immediate static effect."""
    from wingspan.engine import scoring  # local: keeps encode engine-free at import

    player = game_state.players[player_id]
    opp = game_state.players[1 - player_id]

    for goal_idx, goal in enumerate(game_state.round_goals):
        payout = state.ROUND_GOAL_PAYOUTS_2P[goal_idx]
        count_delta, vp_delta = scoring.goal_vp_delta_for_bird(
            player, opp, goal, bird, payout
        )
        if count_delta == 0:
            continue
        base = layout._OFF_GOAL_DELTA + goal_idx * layout._GOAL_DELTA_SLOT_DIM
        feat[base + layout._GOAL_DELTA_COUNT] = count_delta / layout._GOAL_COUNT_SCALE
        feat[base + layout._GOAL_DELTA_VP] = vp_delta / layout._ROUND_GOAL_POINTS_SCALE


def _fill_gain_food(feat: np.ndarray, food: cards.Food, from_choice_die: bool) -> None:
    """Set the gain_food stripe for a food selection. The first ``N_FOODS`` slots
    are the plain single-food dice; the final two are the invertebrate/seed choice
    die taken as invertebrate or as seed. ``from_choice_die`` (only ever true for
    invertebrate/seed at a feeder gain) selects the choice-die slots, so the model
    scores burning a flexible choice die apart from spending a rigid single face.
    Also serves spend decisions (which never set ``from_choice_die``)."""
    if from_choice_die:
        if food == cards.Food.INVERTEBRATE:
            feat[layout._OFF_GAIN_FOOD + layout._GAIN_FOOD_CHOICE_INV] = 1.0
        elif food == cards.Food.SEED:
            feat[layout._OFF_GAIN_FOOD + layout._GAIN_FOOD_CHOICE_SEED] = 1.0
        return
    for i, candidate in enumerate(cards.ALL_FOODS):
        if candidate == food:
            feat[layout._OFF_GAIN_FOOD + i] = 1.0
            break


def _add_pay_food(feat: np.ndarray, food: cards.Food) -> None:
    """Add one unit of ``food`` to the pay_food count stripe (a single-food
    payment, e.g. a PayCostChoice's ``paid_food``)."""
    for i, candidate in enumerate(cards.ALL_FOODS):
        if candidate == food:
            feat[layout._OFF_PAY + i] += 1.0 / layout._PAYMENT_COUNT_SCALE
            break


def _fill_habitat(feat: np.ndarray, habitat: cards.Habitat) -> None:
    for i, candidate in enumerate(cards.ALL_HABITATS):
        if candidate == habitat:
            feat[layout._OFF_HAB + i] = 1.0
            break


def _fill_payment(feat: np.ndarray, payment: state.FoodPool) -> None:
    for i in range(cards.N_FOODS):
        feat[layout._OFF_PAY + i] = payment.counts[i] / layout._PAYMENT_COUNT_SCALE


def _fill_board_slots(
    feat: np.ndarray,
    player: state.Player,
    target_habitat: cards.Habitat | None,
    target_slot: int | None,
    is_lay: bool,
    is_pay: bool,
) -> None:
    """Fill the board_target stripe: per board slot, 8 scalars (lay_eggs,
    pay_eggs, cached food x5 in ALL_FOODS order, tucked) plus a parallel
    integer card index the model embeds. Slot order is positional
    (ALL_HABITATS x ROW_SLOTS), matching the state board stripe and card-index
    block. The targeted slot (``target_habitat``/``target_slot``) is flagged
    lay or pay; empty slots and untargeted slots leave their flags zero."""
    for hab_idx, habitat in enumerate(cards.ALL_HABITATS):
        row = player.board[habitat]
        for slot in range(state.ROW_SLOTS):
            slot_index = hab_idx * state.ROW_SLOTS + slot
            scalar_base = layout._OFF_BOARD + slot_index * layout._BT_SLOT_SCALARS
            if (
                target_habitat is not None
                and habitat == target_habitat
                and slot == target_slot
            ):
                if is_lay:
                    feat[scalar_base + layout._BT_LAY_EGGS] = 1.0
                if is_pay:
                    feat[scalar_base + layout._BT_PAY_EGGS] = 1.0
            if slot >= len(row):
                continue
            pb = row[slot]
            for i, food in enumerate(cards.ALL_FOODS):
                feat[scalar_base + layout._BT_CACHED + i] = (
                    pb.cached_food[food] / layout._CACHED_FOOD_SCALE
                )
            feat[scalar_base + layout._BT_TUCKED] = (
                pb.tucked_cards / layout._TUCKED_SCALE
            )
            feat[layout._OFF_BOARD_IDX + slot_index] = cards.bird_index(pb.bird) + 1
