# pyright: reportPrivateUsage=false
# (this encoder reads the shared, package-private layout constants in
# ``layout`` -- a deliberate intra-package coupling, not a privacy break)
"""The choice encoder: ``encode_choices`` featurizes every legal choice in a
decision into a ``(n_choices, CHOICE_FEATURE_DIM)`` matrix, dispatching on the
concrete ``Choice`` subclass through ``_CHOICE_FEATURIZERS``. The ``_fill_*``
helpers write the shared per-card / per-food / per-habitat stripes.
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


def encode_choices(decision: layout._AnyDecision, state: state.GameState) -> np.ndarray:
    """Featurize every choice in ``decision``.

    Returns a float32 array of shape ``(n_choices, layout.CHOICE_FEATURE_DIM)``.
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
    # PlayBirdDecision routinely and legitimately exceed it, so logging on every
    # such decision floods the log and adds per-call overhead in the hot path —
    # dedupe by class name so it fires once per class per process. Logged at INFO
    # (it is informational, not a fault): it still reaches the dashboard's file
    # log but never the console, so it can't flicker the live "FLYWAY CONTROL"
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
    # layout.CHOICE_FEATURE_DIM array per candidate and copying it in — the rows start
    # zeroed, and the handlers only ever index-assign their own stripes.
    feats = np.zeros((n_choices, layout.CHOICE_FEATURE_DIM), dtype=np.float32)
    for i, choice in enumerate(decision.choices):
        _featurize_choice(feats[i], decision, choice, state)
    return feats


# Back-compat shim: legacy callers expect ``encode_decision``. The semantics
# changed (per-choice features instead of a global mask), so the return is
# different — callers that consumed the old (mask, action_ids) tuple need to
# be updated.
def encode_decision(
    decision: layout._AnyDecision, state: state.GameState
) -> np.ndarray:
    """Alias for :func:`encode_choices`."""
    return encode_choices(decision, state)


###### PRIVATE #######

#### Per-choice featurization ####


def _featurize_choice(
    feat: np.ndarray,
    decision: layout._AnyDecision,
    choice: decisions.Choice,
    state: state.GameState,
) -> None:
    """Fill the pre-zeroed layout.CHOICE_FEATURE_DIM row ``feat`` for one
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
    # The 'accept the offered exchange' branch is distinct from skip — the
    # network can learn to prefer or avoid it independently. KIND_SPECIAL marks
    # it a commit token; the trade terms live in the FOOD stripe (the food paid,
    # if any) and the EXCHANGE stripe (eggs paid, cards / tucks gained) so the
    # commit-to-cost head weighs what is gained against what is paid.
    feat[layout._OFF_KIND + layout._KIND_SPECIAL] = 1.0
    if choice.paid_food is not None:
        _fill_food(feat, choice.paid_food)
    feat[layout._OFF_EXCHANGE + layout._EXCHANGE_PAID_EGGS] = (
        choice.paid_egg_count / layout._EXCHANGE_SCALE
    )
    feat[layout._OFF_EXCHANGE + layout._EXCHANGE_GAINED_CARDS] = (
        choice.gained_card_count / layout._EXCHANGE_SCALE
    )
    feat[layout._OFF_EXCHANGE + layout._EXCHANGE_GAINED_TUCKS] = (
        choice.gained_tuck_count / layout._EXCHANGE_SCALE
    )


def _featurize_main_action(
    feat: np.ndarray,
    decision: layout._AnyDecision,
    choice: decisions.MainActionChoice,
    state: state.GameState,
) -> None:
    feat[layout._OFF_KIND + layout._KIND_SPECIAL] = 1.0
    feat[layout._OFF_SPECIAL + layout._SPECIAL_ENCODED_SLOT] = (
        _MAIN_ACTION_INDEX[choice.action] / layout._PLAYER_ID_SCALE
    )


def _featurize_bird(
    feat: np.ndarray,
    decision: layout._AnyDecision,
    choice: decisions.BirdChoice,
    state: state.GameState,
) -> None:
    feat[layout._OFF_KIND + layout._KIND_BIRD] = 1.0
    _fill_bird_identity(feat, choice.bird)


def _featurize_play_bird(
    feat: np.ndarray,
    decision: layout._AnyDecision,
    choice: decisions.PlayBirdChoice,
    state: state.GameState,
) -> None:
    # A play candidate from ``PlayBirdDecision``: the bird-identity stripe carries
    # the card (its attributes ride the shared card table), and the habitat +
    # payment stripes carry the bundled habitat / food-payment picks. KIND stays
    # BIRD — it is fundamentally a bird play — while the extra stripes distinguish
    # the (habitat, payment) variants of the same bird.
    feat[layout._OFF_KIND + layout._KIND_BIRD] = 1.0
    _fill_bird_identity(feat, choice.bird)
    _fill_habitat(feat, choice.habitat)
    _fill_payment(feat, choice.payment)


def _featurize_played_bird(
    feat: np.ndarray,
    decision: layout._AnyDecision,
    choice: decisions.PlayedBirdChoice,
    state: state.GameState,
) -> None:
    pb = choice.played_bird
    feat[layout._OFF_KIND + layout._KIND_BIRD] = 1.0
    _fill_bird_identity(feat, pb.bird)
    # Surface board-target dynamic state too (eggs/cache/tucked) even
    # though we don't know its row index here.
    feat[layout._OFF_BOARD + 4] = pb.eggs / layout._EGG_COUNT_SCALE
    cap = max(pb.bird.egg_limit - pb.eggs, 0)
    feat[layout._OFF_BOARD + 5] = cap / layout._EGG_COUNT_SCALE
    feat[layout._OFF_BOARD + 6] = pb.cached_food.total() / layout._CACHED_FOOD_SCALE
    feat[layout._OFF_BOARD + 7] = pb.tucked_cards / layout._TUCKED_SCALE


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
    _fill_food(feat, choice.food)


def _featurize_board_target(
    feat: np.ndarray,
    decision: layout._AnyDecision,
    choice: decisions.BoardTargetChoice,
    state: state.GameState,
) -> None:
    feat[layout._OFF_KIND + layout._KIND_BOARD_TARGET] = 1.0
    _fill_board_target(feat, choice.habitat, choice.slot, state, decision.player_id)


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
    if (
        choice.source == "tray"
        and choice.tray_index is not None
        and 0 <= choice.tray_index < len(state.tray)
    ):
        feat[layout._OFF_KIND + layout._KIND_BIRD] = 1.0
        _fill_bird_identity(feat, state.tray[choice.tray_index])
    else:
        feat[layout._OFF_KIND + layout._KIND_SPECIAL] = 1.0


def _featurize_player_id(
    feat: np.ndarray,
    decision: layout._AnyDecision,
    choice: decisions.PlayerIdChoice,
    state: state.GameState,
) -> None:
    # Flag whether the choice means "me" so the network can learn
    # self-vs-opponent preference cheaply.
    feat[layout._OFF_KIND + layout._KIND_SPECIAL] = 1.0
    feat[layout._OFF_SPECIAL + layout._SPECIAL_IS_KEEP] = (
        1.0 if choice.player_id == decision.player_id else 0.0
    )


def _featurize_setup(
    feat: np.ndarray,
    decision: layout._AnyDecision,
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
    feat[layout._OFF_KIND + layout._KIND_SPECIAL] = 1.0
    # PAY stripe encodes the foods *spent* (complement of kept_foods), so the
    # network sees the same payment signal as every other paying decision.
    for i, food in enumerate(cards.ALL_FOODS):
        if food not in choice.kept_foods:
            feat[layout._OFF_PAY + i] = 1.0 / layout._PAYMENT_COUNT_SCALE
    kept = choice.kept_cards
    # Identity multi-hot of the kept birds (the headline §3.1 fix) plus the
    # SETUP-stripe aggregate stats that summarise the subset. One pass sets each
    # identity bit and accumulates all three sums (the setup deal featurizes 504
    # candidates, so folding three generator passes into one matters). The
    # aggregates live in the dedicated SETUP stripe because they are kept-*subset*
    # summaries the shared card table cannot reconstruct from the identity multi-hot.
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


_CHOICE_FEATURIZERS: dict[type[decisions.Choice], layout._ChoiceFeaturizer] = {
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


def _fill_bird_identity(feat: np.ndarray, bird: cards.Bird) -> None:
    """Set the bird-identity one-hot bit for ``bird``. Called for every
    bird-carrying choice, and once per card to build a kept-set / hand multi-hot.
    The model maps this stripe through the shared card encoder, so a candidate's
    static attributes and its learned per-card vector arrive together."""
    feat[layout._OFF_BIRD_ID + cards.bird_index(bird)] = 1.0


def _fill_bonus_identity(feat: np.ndarray, bonus_card: cards.BonusCard) -> None:
    """Set the bonus-card identity one-hot bit for ``bonus_card``."""
    feat[layout._OFF_BONUS_ID + cards.bonus_index(bonus_card)] = 1.0


def _fill_food(feat: np.ndarray, food: cards.Food) -> None:
    for i, candidate in enumerate(cards.ALL_FOODS):
        if candidate == food:
            feat[layout._OFF_FOOD + i] = 1.0
            break


def _fill_habitat(feat: np.ndarray, habitat: cards.Habitat) -> None:
    for i, candidate in enumerate(cards.ALL_HABITATS):
        if candidate == habitat:
            feat[layout._OFF_HAB + i] = 1.0
            break


def _fill_payment(feat: np.ndarray, payment: state.FoodPool) -> None:
    for i in range(cards.N_FOODS):
        feat[layout._OFF_PAY + i] = payment.counts[i] / layout._PAYMENT_COUNT_SCALE


def _fill_board_target(
    feat: np.ndarray,
    habitat: cards.Habitat,
    slot: int,
    state: state.GameState,
    player_id: int,
) -> None:
    for i, candidate in enumerate(cards.ALL_HABITATS):
        if candidate == habitat:
            feat[layout._OFF_BOARD + i] = 1.0
            break
    feat[layout._OFF_BOARD + 3] = slot / layout._ROW_SLOTS_SCALE
    player = state.players[player_id]
    row = player.board[habitat]
    if 0 <= slot < len(row):
        pb = row[slot]
        feat[layout._OFF_BOARD + 4] = pb.eggs / layout._EGG_COUNT_SCALE
        cap = max(pb.bird.egg_limit - pb.eggs, 0)
        feat[layout._OFF_BOARD + 5] = cap / layout._EGG_COUNT_SCALE
        feat[layout._OFF_BOARD + 6] = pb.cached_food.total() / layout._CACHED_FOOD_SCALE
        feat[layout._OFF_BOARD + 7] = pb.tucked_cards / layout._TUCKED_SCALE
