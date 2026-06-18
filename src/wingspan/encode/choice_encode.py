# pyright: reportPrivateUsage=false
# (this encoder reads the shared, package-private layout constants in
# ``layout`` -- a deliberate intra-package coupling, not a privacy break)
"""The choice encoder: ``encode_choices`` featurizes every legal choice in a
decision into a ``(n_choices, choice_feature_dim(spec))`` matrix, dispatching on
the concrete ``Choice`` subclass through ``_CHOICE_FEATURIZERS``. The ``_fill_*``
helpers write the shared per-card / per-food / per-board stripes.
"""

from __future__ import annotations

import logging
import typing

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
    *,
    has_becomes_playable: bool = True,
) -> np.ndarray:
    """Featurize every choice in ``decision``.

    Returns a float32 array of shape ``(n_choices, layout.choice_feature_dim(spec))``.
    All returned rows correspond to legal choices — there is no padding or
    truncation, and the action space is implicitly variable across decisions.
    The caller (training loop) handles batched padding + masking. ``spec`` selects
    the config-driven row width (only the trailing ``setup_agg`` stripe varies).

    ``has_becomes_playable`` selects whether the ``becomes_playable`` 180-dim
    stripe is included in each row. Set to ``False`` for pre-0.6 compat shims
    whose choice vectors lack the stripe; defaults to ``True`` for live encoding.
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
    # Compute the deciding player's already-playable set once; used by the
    # becomes_playable fillers so each row tests only not-yet-playable birds.
    # Skip the playability computation when has_becomes_playable is False (compat
    # shim path) to avoid the engine import and the classification cost entirely.
    if has_becomes_playable:
        from wingspan.engine import playability as _playability

        player = state.players[decision.player_id]
        playable_now, _ = _playability.classify_hand_playability(player)
    else:
        playable_now = []

    # Row width depends on whether the becomes_playable stripe is included.
    row_dim = (
        layout.choice_feature_dim(spec)
        if has_becomes_playable
        else layout.choice_feature_dim(spec) - layout.CHOICE_BECOMES_PLAYABLE_DIM
    )
    feats = np.zeros((n_choices, row_dim), dtype=np.float32)
    for i, choice in enumerate(decision.choices):
        _featurize_choice(
            feats[i], decision, choice, state, playable_now, has_becomes_playable
        )
    return feats


###### PRIVATE #######

#### Per-choice featurization ####


def _featurize_choice(
    feat: np.ndarray,
    decision: layout._AnyDecision,
    choice: decisions.Choice,
    state: state.GameState,
    playable_now: list[cards.Bird],
    has_becomes_playable: bool = True,
) -> None:
    """Fill the pre-zeroed choice-feature row ``feat`` for one (decision, choice)
    pair, dispatching on the concrete Choice subclass.

    Writes into the caller's row view rather than allocating a fresh vector, so
    ``encode_choices`` builds its ``(n_choices, DIM)`` matrix with no per-row
    throwaway. The typed ``choice`` parameter keeps ``type(choice)`` a known
    ``type[Choice]`` for the dispatch lookup. ``has_becomes_playable`` is
    threaded to featurizers that fill the ``becomes_playable`` stripe."""
    _CHOICE_FEATURIZERS.get(type(choice), _featurize_default)(
        feat, decision, choice, state, playable_now, has_becomes_playable
    )


def _featurize_default(
    feat: np.ndarray,
    decision: layout._AnyDecision,
    choice: decisions.Choice,
    state: state.GameState,
    playable_now: list[cards.Bird],
    has_becomes_playable: bool = True,
) -> None:
    feat[layout._OFF_KIND + layout._KIND_SPECIAL] = 1.0


def _featurize_skip(
    feat: np.ndarray,
    decision: layout._AnyDecision,
    choice: decisions.SkipChoice,
    state: state.GameState,
    playable_now: list[cards.Bird],
    has_becomes_playable: bool = True,
) -> None:
    feat[layout._OFF_KIND + layout._KIND_SPECIAL] = 1.0
    feat[layout._OFF_SPECIAL + layout._SPECIAL_IS_SKIP] = 1.0


def _featurize_reset_birdfeeder(
    feat: np.ndarray,
    decision: layout._AnyDecision,
    choice: decisions.ResetBirdfeederChoice,
    state: state.GameState,
    playable_now: list[cards.Bird],
    has_becomes_playable: bool = True,
) -> None:
    # The "yes, reroll" affirmative. Carries no data, so only the special-kind
    # bit is set; the decision-type stripe identifies the reset decision and the
    # absent is-skip bit distinguishes it from the paired ``SkipChoice``.
    feat[layout._OFF_KIND + layout._KIND_SPECIAL] = 1.0


def _featurize_tuck_activate(
    feat: np.ndarray,
    decision: layout._AnyDecision,
    choice: decisions.TuckActivateChoice,
    state: state.GameState,
    playable_now: list[cards.Bird],
    has_becomes_playable: bool = True,
) -> None:
    # The "yes, tuck" commit token — analogous to PayCostChoice but simpler:
    # KIND_SPECIAL marks it as a non-bird commit; the EXCHANGE stripe's
    # cards_to_tuck slot carries how many cards the player is committing to
    # tuck so the SKIP_OPTIONAL head can weigh the tuck's value.
    feat[layout._OFF_KIND + layout._KIND_SPECIAL] = 1.0
    feat[layout._OFF_EXCHANGE + layout._EXCHANGE_CARDS_TO_TUCK] = (
        choice.cards_to_tuck / layout._EXCHANGE_SCALE
    )


def _featurize_pay_cost(
    feat: np.ndarray,
    decision: layout._AnyDecision,
    choice: decisions.PayCostChoice,
    state: state.GameState,
    playable_now: list[cards.Bird],
    has_becomes_playable: bool = True,
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
        layout._EXCHANGE_CACHE_TO_GAIN: choice.gained_cache_count,
    }
    for index, count in exchange_terms.items():
        feat[layout._OFF_EXCHANGE + index] = count / layout._EXCHANGE_SCALE
    # Consequence pricing for the committed terms. Net hand-card flow prices
    # the hand-counting bonus card (a draw grows the end-game hand, a discard
    # shrinks it); committed egg terms price an optimistic round-goal bound —
    # the target picks are follow-up decisions, so the accept row advertises
    # the best the player could realize (exact deltas land on those rows).
    player = state.players[decision.player_id]
    delta_cards = choice.gained_card_count - choice.paid_card_count
    if delta_cards != 0:
        _fill_bonus_delta_for_hand(feat, player, delta_cards)
    if choice.gained_egg_count > 0:
        _fill_goal_delta_best_case(
            feat, decision.player_id, choice.gained_egg_count, state
        )
    elif choice.paid_egg_count > 0:
        _fill_goal_delta_best_case(
            feat, decision.player_id, -choice.paid_egg_count, state
        )
    # Becomes-playable: food or egg gains that unlock new hand birds.
    if has_becomes_playable and (
        choice.gained_food_count > 0 or choice.gained_egg_count > 0
    ):
        from wingspan.engine import playability as _playability

        all_newly: list[cards.Bird] = []
        seen_ids: set[int] = set()
        if choice.gained_food_count > 0:
            feeder_newly = _playability.newly_playable_after_feeder_food(
                player, state.birdfeeder, already_playable=playable_now
            )
            for bird in feeder_newly:
                if id(bird) not in seen_ids:
                    all_newly.append(bird)
                    seen_ids.add(id(bird))
        if choice.gained_egg_count > 0:
            egg_newly = _playability.newly_playable_after_egg(
                player, choice.gained_egg_count, already_playable=playable_now
            )
            for bird in egg_newly:
                if id(bird) not in seen_ids:
                    all_newly.append(bird)
                    seen_ids.add(id(bird))
        if all_newly:
            _fill_becomes_playable(feat, all_newly)


def _featurize_main_action(
    feat: np.ndarray,
    decision: layout._AnyDecision,
    choice: decisions.MainActionChoice,
    state: state.GameState,
    playable_now: list[cards.Bird],
    has_becomes_playable: bool = True,
) -> None:
    # A one-hot over the four main actions — never an index-as-scalar (the four
    # actions have no ordinal relationship). KIND stays SPECIAL; the dedicated
    # main_action stripe distinguishes the four options.
    feat[layout._OFF_KIND + layout._KIND_SPECIAL] = 1.0
    for i, action in enumerate(layout._MAIN_ACTION_ORDER):
        if choice.action == action:
            feat[layout._OFF_MAIN_ACTION + i] = 1.0
            break
    # Consequence pricing on the *whether* row (the targets, if any, are
    # follow-up decisions): DRAW_CARDS grows the hand by the wetland track
    # count, which is what the hand-counting bonus card pays on; LAY_EGGS
    # advertises the capacity-capped best case the grassland track's eggs
    # could realize against each unscored goal. GAIN_FOOD and PLAY_BIRD touch
    # no goal or bonus count directly and stay featureless.
    player = state.players[decision.player_id]
    if choice.action == decisions.MainAction.DRAW_CARDS:
        _fill_bonus_delta_for_hand(feat, player, player.board.draw_cards_count())
    elif choice.action == decisions.MainAction.LAY_EGGS:
        _fill_goal_delta_best_case(
            feat, decision.player_id, player.board.lay_eggs_count(), state
        )
    # Becomes-playable: forecast which hand birds would unlock after this action.
    if has_becomes_playable:
        if choice.action == decisions.MainAction.GAIN_FOOD:
            from wingspan.engine import playability as _playability

            newly = _playability.newly_playable_after_feeder_food(
                player, state.birdfeeder, already_playable=playable_now
            )
            _fill_becomes_playable(feat, newly)
        elif choice.action == decisions.MainAction.LAY_EGGS:
            from wingspan.engine import playability as _playability

            n_eggs = player.board.lay_eggs_count()
            newly = _playability.newly_playable_after_egg(
                player, n_eggs, already_playable=playable_now
            )
            _fill_becomes_playable(feat, newly)


def _featurize_bird(
    feat: np.ndarray,
    decision: layout._AnyDecision,
    choice: decisions.BirdChoice,
    state: state.GameState,
    playable_now: list[cards.Bird],
    has_becomes_playable: bool = True,
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
    playable_now: list[cards.Bird],
    has_becomes_playable: bool = True,
) -> None:
    # A play candidate from ``PlayBirdDecision``: the bird-index column carries
    # the card (its attributes ride the shared card table), and the landing-slot
    # marker in the board-index block carries the bundled habitat pick as the
    # exact slot the bird would occupy — the model reads the resulting location
    # directly instead of inferring it from a habitat flag. KIND stays BIRD — it
    # is fundamentally a bird play — while the landing slot distinguishes the
    # per-habitat variants of the same bird. The costs are follow-up decisions
    # (RemoveEggDecision / PayBirdFoodDecision), so no payment stripe is filled
    # here; the bonus_delta and goal_delta stripes price the play's contribution
    # to held bonus cards and round-goal standings.
    player = state.players[decision.player_id]
    feat[layout._OFF_KIND + layout._KIND_BIRD] = 1.0
    _fill_bird_identity(feat, choice.bird)
    _fill_landing_slot(feat, player, choice.bird, choice.habitat)
    _fill_bonus_delta(feat, player, choice.bird, play_habitat=choice.habitat)
    _fill_goal_delta(feat, decision.player_id, choice.bird, state)


def _featurize_food_payment(
    feat: np.ndarray,
    decision: layout._AnyDecision,
    choice: decisions.FoodPaymentChoice,
    state: state.GameState,
    playable_now: list[cards.Bird],
    has_becomes_playable: bool = True,
) -> None:
    # A complete payment multiset for a committed bird play (PayBirdFoodDecision).
    # KIND_PAYMENT marks the row a whole-payment pick; the PAY stripe carries the
    # candidate's per-food counts, and the committed play rides along as context —
    # bird identity (embedded through the shared card table) plus its landing
    # slot in the board-index block (the payment is asked before the bird is
    # placed, so the row's next free slot is where it will land) — so the
    # spend-food head sees *what* the payment is for, not just the tokens leaving.
    feat[layout._OFF_KIND + layout._KIND_PAYMENT] = 1.0
    _fill_payment(feat, choice.payment)
    if isinstance(decision, decisions.PayBirdFoodDecision):
        _fill_bird_identity(feat, decision.bird)
        _fill_landing_slot(
            feat, state.players[decision.player_id], decision.bird, decision.habitat
        )


def _featurize_played_bird(
    feat: np.ndarray,
    decision: layout._AnyDecision,
    choice: decisions.PlayedBirdChoice,
    state: state.GameState,
    playable_now: list[cards.Bird],
    has_becomes_playable: bool = True,
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
    playable_now: list[cards.Bird],
    has_becomes_playable: bool = True,
) -> None:
    # A habitat pick is always a move-bird destination
    # (``BirdPowerPickHabitatDecision`` is the only decision that offers
    # ``HabitatChoice``). Each row prices relocating the moving bird: its
    # identity (through the shared card table), its landing slot in the
    # board-index block — the exact slot it would occupy, so the model reads
    # the resulting location instead of inferring it from a habitat flag — and
    # the round-goal / bonus consequences of the move (habitat bird counts, the
    # egg block riding along, the habitat-spread bonus card). The "stay" row
    # marks the bird's *current* slot (it is the rightmost of its row — the
    # power only fires then) and its deltas are naturally all-zero.
    assert isinstance(decision, decisions.BirdPowerPickHabitatDecision)
    feat[layout._OFF_KIND + layout._KIND_HABITAT] = 1.0
    player = state.players[decision.player_id]
    moving_bird = decision.moving_bird
    if choice.habitat == decision.from_habitat:
        current_slot = len(player.board[choice.habitat]) - 1
        _fill_board_index(feat, moving_bird.bird, choice.habitat, current_slot)
    else:
        _fill_landing_slot(feat, player, moving_bird.bird, choice.habitat)
    _fill_bird_identity(feat, moving_bird.bird)
    _fill_goal_delta_for_move(
        feat,
        decision.player_id,
        decision.from_habitat,
        choice.habitat,
        moving_bird,
        state,
    )
    _fill_bonus_delta_for_move(feat, player, decision.from_habitat, choice.habitat)


def _featurize_food(
    feat: np.ndarray,
    decision: layout._AnyDecision,
    choice: decisions.FoodChoice,
    state: state.GameState,
    playable_now: list[cards.Bird],
    has_becomes_playable: bool = True,
) -> None:
    feat[layout._OFF_KIND + layout._KIND_FOOD] = 1.0
    _fill_gain_food(feat, choice.food, choice.from_choice_die)
    if has_becomes_playable and isinstance(decision, decisions.GainFoodDecision):
        from wingspan.engine import playability as _playability

        player = state.players[decision.player_id]
        newly = _playability.newly_playable_after_food(
            player, choice.food, already_playable=playable_now
        )
        _fill_becomes_playable(feat, newly)


def _featurize_board_target(
    feat: np.ndarray,
    decision: layout._AnyDecision,
    choice: decisions.BoardTargetChoice,
    state: state.GameState,
    playable_now: list[cards.Bird],
    has_becomes_playable: bool = True,
) -> None:
    # A board-slot target: fill the whole 15-slot board block from the deciding
    # player's board and flag the targeted slot as laying an egg (lay-egg decision)
    # or paying an egg (remove-egg decision). The occupying birds ride the parallel
    # card-index block the model embeds.
    feat[layout._OFF_KIND + layout._KIND_BOARD_TARGET] = 1.0
    is_lay = isinstance(decision, decisions.LayEggDecision)
    is_pay = isinstance(decision, decisions.RemoveEggDecision)
    player = state.players[decision.player_id]
    _fill_board_slots(
        feat,
        player,
        choice.habitat,
        choice.slot,
        is_lay,
        is_pay,
    )
    # The egg event's consequences for this specific target: every egg
    # lay / removal in the engine lands on the deciding player's own board, so
    # the targeted slot fully determines the round-goal and bonus-card deltas
    # (habitat totals, nest totals, has-eggs crossings, the egg-set minimum,
    # and the egg-counting dynamic bonus thresholds).
    if is_lay or is_pay:
        row = player.board[choice.habitat]
        if 0 <= choice.slot < len(row):  # the engine only offers occupied slots
            played_bird = row[choice.slot]
            delta_eggs = 1 if is_lay else -1
            _fill_goal_delta_for_egg(
                feat, decision.player_id, choice.habitat, played_bird, delta_eggs, state
            )
            _fill_bonus_delta_for_egg(feat, player, played_bird, delta_eggs)


def _featurize_bonus_card(
    feat: np.ndarray,
    decision: layout._AnyDecision,
    choice: decisions.BonusCardChoice,
    state: state.GameState,
    playable_now: list[cards.Bird],
    has_becomes_playable: bool = True,
) -> None:
    # Identity via the bonus one-hot stripe (a learned per-bonus embedding),
    # replacing the old id-hash so distinct bonus cards are fully distinguished.
    # The bonus_value stripe prices the candidate card itself — its standing VP on
    # the current board plus the hand/tray birds that could still qualify it — so
    # the net reads what the offered card is worth instead of inferring it from
    # identity alone (the candidate is not yet held, so the state-side
    # bonus-progress stripes carry nothing for it).
    feat[layout._OFF_KIND + layout._KIND_SPECIAL] = 1.0
    _fill_bonus_identity(feat, choice.bonus_card)
    player = state.players[decision.player_id]
    _fill_bonus_value(feat, player, choice.bonus_card, player.hand, state.tray)


def _featurize_draw_source(
    feat: np.ndarray,
    decision: layout._AnyDecision,
    choice: decisions.DrawSourceChoice,
    state: state.GameState,
    playable_now: list[cards.Bird],
    has_becomes_playable: bool = True,
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
        # The deck row stays identity-free (a blind draw is the value of
        # information), but a draw from any source grows the hand by one — so
        # the hand-counting bonus term filled on the tray rows (via
        # ``_fill_bonus_delta``) must ride the deck row too, keeping the
        # within-decision comparison neutral.
        feat[layout._OFF_KIND + layout._KIND_SPECIAL] = 1.0
        _fill_bonus_delta_for_hand(feat, state.players[decision.player_id], 1)


def _featurize_player_id(
    feat: np.ndarray,
    decision: layout._AnyDecision,
    choice: decisions.PlayerIdChoice,
    state: state.GameState,
    playable_now: list[cards.Bird],
    has_becomes_playable: bool = True,
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
    playable_now: list[cards.Bird],
    has_becomes_playable: bool = True,
) -> None:
    """Featurize a single combined setup pick.

    Only reached when the main model carries setup (``include_setup``), so the
    row is wide enough to hold the trailing ``setup_agg`` and ``kept_multihot``
    stripes. The 504 candidates share a state vector, so the network reads the
    choice features to tell them apart: (a) a multi-hot of the *specific* kept
    birds in the dedicated kept_multihot stripe (the single-candidate bird-index
    column stays zero — a setup pick is a set, not one bird), (b) aggregate
    stats of the kept-card subset in the setup_agg stripe, (c) a multi-hot of
    foods spent in the PAY_FOOD stripe, and (d) the kept bonus card's identity
    one-hot.
    """
    feat[layout._OFF_KIND + layout._KIND_SPECIAL] = 1.0
    # PAY_FOOD stripe encodes the foods *spent* (complement of kept_foods), so the
    # network sees the same payment signal as every other paying decision.
    for i, food in enumerate(cards.ALL_FOODS):
        if food not in choice.kept_foods:
            feat[layout._OFF_PAY + i] = 1.0 / layout._PAYMENT_COUNT_SCALE
    kept = choice.kept_cards
    # Identity multi-hot of the kept birds (in the trailing kept_multihot
    # stripe, which the model sums through the shared card table) plus the
    # setup_agg aggregate stats that summarise the subset. One pass sets each
    # identity bit and accumulates all three sums (the setup deal featurizes
    # 504 candidates, so folding three generator passes into one matters). The
    # aggregates live in the dedicated SETUP stripe because they are
    # kept-*subset* summaries the shared card table cannot reconstruct from the
    # identity multi-hot.
    if kept:
        points = 0.0
        cost = 0.0
        eggs = 0.0
        for bird in kept:
            feat[layout._OFF_KEPT_MULTIHOT + cards.bird_index(bird)] = 1.0
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
    # The kept bonus rides identity + the bonus_value stripe. The hand source is
    # the kept subset, NOT ``player.hand`` — at the setup ask the hand still holds
    # all dealt cards, so counting it would credit birds this pick discards.
    if choice.bonus_card is not None:
        _fill_bonus_identity(feat, choice.bonus_card)
        _fill_bonus_value(
            feat,
            state.players[decision.player_id],
            choice.bonus_card,
            kept,
            state.tray,
        )


# Maps each Choice subclass to its featurizer; drives encode_choices and the stripe fillers below.
_CHOICE_FEATURIZERS: dict[type[decisions.Choice], layout._ChoiceFeaturizer] = {
    decisions.SkipChoice: _featurize_skip,
    decisions.TuckActivateChoice: _featurize_tuck_activate,
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
    """Write the candidate bird's index column (``bird_index + 1``; the zeroed
    default means "no bird"). Called for every bird-carrying choice. The model
    looks the index up in the shared card table, so a candidate's static
    attributes and its learned per-card vector arrive together."""
    feat[layout._OFF_BIRD_ID] = cards.bird_index(bird) + 1


def _fill_becomes_playable(feat: np.ndarray, birds: list[cards.Bird]) -> None:
    """Set the ``becomes_playable`` stripe bit for each bird in ``birds``."""
    for bird in birds:
        feat[layout.CHOICE_BECOMES_PLAYABLE_OFFSET + cards.bird_index(bird)] = 1.0


def _fill_board_index(
    feat: np.ndarray, bird: cards.Bird, habitat: cards.Habitat, slot: int
) -> None:
    """Write ``bird``'s card index at one positional board slot of the
    board-index block (``hab_idx * ROW_SLOTS + slot``, the same indexing as
    ``_fill_board_slots``). The single marked slot is how placement rows carry
    their resulting location; every other board-index column stays zero (the
    full current board already rides the state vector)."""
    for hab_idx, candidate in enumerate(cards.ALL_HABITATS):
        if candidate == habitat:
            slot_index = hab_idx * state.ROW_SLOTS + slot
            feat[layout._OFF_BOARD_IDX + slot_index] = cards.bird_index(bird) + 1
            break


def _fill_landing_slot(
    feat: np.ndarray, player: state.Player, bird: cards.Bird, habitat: cards.Habitat
) -> None:
    """Mark where ``bird`` would land if placed in ``habitat`` now: its card
    index at the row's next free slot (rows fill left to right, and legality /
    the engine's placement order guarantee the row isn't full)."""
    _fill_board_index(feat, bird, habitat, len(player.board[habitat]))


def _fill_bonus_identity(feat: np.ndarray, bonus_card: cards.BonusCard) -> None:
    """Set the bonus-card identity one-hot bit for ``bonus_card``."""
    feat[layout._OFF_BONUS_ID + cards.bonus_index(bonus_card)] = 1.0


def _fill_bonus_delta(
    feat: np.ndarray,
    player: state.Player,
    bird: cards.Bird,
    play_habitat: cards.Habitat | None = None,
) -> None:
    """Fill the bonus_delta stripe: how much taking ``bird`` would advance the
    bonus cards ``player`` currently holds. Three scalars — the count of held
    cards the bird moves, and the summed stepped / linear VP gain — so the net
    reads a candidate's bonus contribution directly instead of inferring it
    from the bonus-progress and card-attribute stripes.

    Static cards price the bird's eventual +1 board qualifier (its printed
    categories). Dynamic cards price by row direction: an acquire row
    (``play_habitat is None`` — keep / tray-draw picks) grows the hand by one,
    which the hand-counting card pays on; a play row (``play_habitat`` set)
    can grow the smallest habitat row, which the habitat-spread card pays on.
    All zero when no held card is moved."""
    from wingspan.engine import scoring  # local: keeps encode engine-free at import

    qual = 0
    stepped = 0.0
    linear = 0.0
    for bonus_card in player.bonus_cards:
        if bonus_card.name in bird.bonus_categories:
            count_delta = 1
        elif play_habitat is None:
            count_delta = scoring.bonus_count_delta_for_hand(bonus_card, 1)
        else:
            count_delta = scoring.bonus_count_delta_for_play_habitat(
                bonus_card, player, play_habitat
            )
        if count_delta == 0:
            continue
        qual += 1
        count = scoring.bonus_qualifying_count(player, bonus_card)
        stepped_delta, linear_delta = scoring.bonus_vp_deltas_for_count_change(
            bonus_card, count, count + count_delta
        )
        stepped += stepped_delta
        linear += linear_delta
    _write_bonus_delta(feat, qual, stepped, linear)


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
    zero for goals where the bird has no immediate static effect, and for
    goals whose round has already been scored (a scored goal's payout is
    frozen — no choice can change it)."""
    from wingspan.engine import scoring  # local: keeps encode engine-free at import

    player = game_state.players[player_id]
    opp = game_state.players[1 - player_id]

    for goal_idx, goal in enumerate(game_state.round_goals):
        if goal_idx < len(game_state.scored_goals):
            continue
        payout = state.ROUND_GOAL_PAYOUTS_2P[goal_idx]
        count_delta, vp_delta = scoring.goal_vp_delta_for_bird(
            player, opp, goal, bird, payout
        )
        _write_goal_delta(feat, goal_idx, count_delta, vp_delta)


def _fill_goal_delta_for_egg(
    feat: np.ndarray,
    player_id: int,
    habitat: cards.Habitat,
    played_bird: state.PlayedBird,
    delta_eggs: int,
    game_state: state.GameState,
) -> None:
    """Fill the goal_delta stripe for an egg event: per unscored round goal,
    how laying (``delta_eggs > 0``) or removing (``< 0``) that many eggs on
    ``played_bird`` at ``habitat`` would move the deciding player's category
    count and placement VP. The 12 egg-driven core goal categories all price
    through :func:`scoring.goal_count_delta_for_egg`."""
    from wingspan.engine import scoring  # local: keeps encode engine-free at import

    player = game_state.players[player_id]
    opp = game_state.players[1 - player_id]
    for goal_idx, goal in enumerate(game_state.round_goals):
        if goal_idx < len(game_state.scored_goals):
            continue
        payout = state.ROUND_GOAL_PAYOUTS_2P[goal_idx]
        count_delta, vp_delta = scoring.goal_vp_delta_for_egg(
            player, opp, goal, habitat, played_bird, payout, delta_eggs
        )
        _write_goal_delta(feat, goal_idx, count_delta, vp_delta)


def _fill_goal_delta_for_move(
    feat: np.ndarray,
    player_id: int,
    from_habitat: cards.Habitat,
    to_habitat: cards.Habitat,
    played_bird: state.PlayedBird,
    game_state: state.GameState,
) -> None:
    """Fill the goal_delta stripe for relocating ``played_bird`` (with its
    eggs) between habitat rows: per unscored round goal, the count / placement
    VP swing of the move. A stay (``from == to``) writes nothing."""
    from wingspan.engine import scoring  # local: keeps encode engine-free at import

    player = game_state.players[player_id]
    opp = game_state.players[1 - player_id]
    for goal_idx, goal in enumerate(game_state.round_goals):
        if goal_idx < len(game_state.scored_goals):
            continue
        payout = state.ROUND_GOAL_PAYOUTS_2P[goal_idx]
        count_delta, vp_delta = scoring.goal_vp_delta_for_move(
            player, opp, goal, from_habitat, to_habitat, played_bird, payout
        )
        _write_goal_delta(feat, goal_idx, count_delta, vp_delta)


def _fill_goal_delta_best_case(
    feat: np.ndarray,
    player_id: int,
    n_eggs: int,
    game_state: state.GameState,
) -> None:
    """Fill the goal_delta stripe for a *commitment* to lay (``n_eggs > 0``)
    or remove (``< 0``) eggs whose targets are picked later: per unscored
    round goal, the capacity-capped optimistic bound from
    :func:`scoring.goal_best_case_for_eggs`. The follow-up target rows carry
    the exact per-target deltas."""
    from wingspan.engine import scoring  # local: keeps encode engine-free at import

    player = game_state.players[player_id]
    opp = game_state.players[1 - player_id]
    for goal_idx, goal in enumerate(game_state.round_goals):
        if goal_idx < len(game_state.scored_goals):
            continue
        payout = state.ROUND_GOAL_PAYOUTS_2P[goal_idx]
        count_delta, vp_delta = scoring.goal_best_case_for_eggs(
            player, opp, goal, payout, n_eggs
        )
        _write_goal_delta(feat, goal_idx, count_delta, vp_delta)


def _fill_bonus_delta_for_egg(
    feat: np.ndarray,
    player: state.Player,
    played_bird: state.PlayedBird,
    delta_eggs: int,
) -> None:
    """Fill the bonus_delta stripe for an egg event on ``played_bird``: the
    held egg-counting dynamic bonus cards whose qualifying threshold the event
    crosses, and the summed stepped / linear VP swing."""
    from wingspan.engine import scoring  # local: keeps encode engine-free at import

    qual = 0
    stepped = 0.0
    linear = 0.0
    for bonus_card in player.bonus_cards:
        count_delta = scoring.bonus_count_delta_for_egg(
            bonus_card, played_bird, delta_eggs
        )
        if count_delta == 0:
            continue
        qual += 1
        count = scoring.bonus_qualifying_count(player, bonus_card)
        stepped_delta, linear_delta = scoring.bonus_vp_deltas_for_count_change(
            bonus_card, count, count + count_delta
        )
        stepped += stepped_delta
        linear += linear_delta
    _write_bonus_delta(feat, qual, stepped, linear)


def _fill_bonus_delta_for_hand(
    feat: np.ndarray, player: state.Player, delta_cards: int
) -> None:
    """Fill the bonus_delta stripe for the hand growing or shrinking by
    ``delta_cards`` — what a committed draw / discard means for the held
    hand-counting bonus card."""
    from wingspan.engine import scoring  # local: keeps encode engine-free at import

    qual = 0
    stepped = 0.0
    linear = 0.0
    for bonus_card in player.bonus_cards:
        count_delta = scoring.bonus_count_delta_for_hand(bonus_card, delta_cards)
        if count_delta == 0:
            continue
        qual += 1
        count = scoring.bonus_qualifying_count(player, bonus_card)
        stepped_delta, linear_delta = scoring.bonus_vp_deltas_for_count_change(
            bonus_card, count, count + count_delta
        )
        stepped += stepped_delta
        linear += linear_delta
    _write_bonus_delta(feat, qual, stepped, linear)


def _fill_bonus_delta_for_move(
    feat: np.ndarray,
    player: state.Player,
    from_habitat: cards.Habitat,
    to_habitat: cards.Habitat,
) -> None:
    """Fill the bonus_delta stripe for moving one bird between habitat rows —
    nonzero only when the held habitat-spread bonus card's minimum row count
    shifts."""
    from wingspan.engine import scoring  # local: keeps encode engine-free at import

    qual = 0
    stepped = 0.0
    linear = 0.0
    for bonus_card in player.bonus_cards:
        count_delta = scoring.bonus_count_delta_for_move(
            bonus_card, player, from_habitat, to_habitat
        )
        if count_delta == 0:
            continue
        qual += 1
        count = scoring.bonus_qualifying_count(player, bonus_card)
        stepped_delta, linear_delta = scoring.bonus_vp_deltas_for_count_change(
            bonus_card, count, count + count_delta
        )
        stepped += stepped_delta
        linear += linear_delta
    _write_bonus_delta(feat, qual, stepped, linear)


def _write_goal_delta(
    feat: np.ndarray, goal_idx: int, count_delta: int, vp_delta: int
) -> None:
    """Write one goal slot of the goal_delta stripe (normalized); a zero
    count delta writes nothing (the slot stays zero, the no-effect signal)."""
    if count_delta == 0:
        return
    base = layout._OFF_GOAL_DELTA + goal_idx * layout._GOAL_DELTA_SLOT_DIM
    feat[base + layout._GOAL_DELTA_COUNT] = count_delta / layout._GOAL_COUNT_SCALE
    feat[base + layout._GOAL_DELTA_VP] = vp_delta / layout._ROUND_GOAL_POINTS_SCALE


def _write_bonus_delta(
    feat: np.ndarray, qual: int, stepped: float, linear: float
) -> None:
    """Write the three bonus_delta scalars (normalized); a zero affected-card
    count writes nothing (the stripe stays zero, the no-effect signal)."""
    if qual == 0:
        return
    base = layout._OFF_BONUS_DELTA
    feat[base + layout._BONUS_DELTA_QUAL] = qual / layout._BONUS_COUNT_SCALE
    feat[base + layout._BONUS_DELTA_STEPPED] = stepped / layout._BONUS_VALUE_SCALE
    feat[base + layout._BONUS_DELTA_LINEAR] = linear / layout._BONUS_VALUE_SCALE


def _fill_bonus_value(
    feat: np.ndarray,
    player: state.Player,
    bonus_card: cards.BonusCard,
    hand_source: typing.Iterable[cards.Bird],
    tray: typing.Sequence[cards.Bird | None],
) -> None:
    """Fill the bonus_value stripe: the value of the candidate ``bonus_card``
    itself to ``player`` — the board birds qualifying now, the stepped / linear
    VP that count pays, and how many hand and tray birds could still qualify it.
    Where ``_fill_bonus_delta`` prices a bird against the held bonuses, this
    prices an offered bonus card against the player's position. ``hand_source``
    is the bird set counted for hand potential: ``player.hand`` for an in-game
    pick, the kept-card subset for a setup pick (where the kept birds are not
    yet in hand). All five scalars are always written — at zero board qualifiers
    the trio is simply 0, which IS the candidate's standing value."""
    from wingspan.engine import scoring  # local: keeps encode engine-free at import

    # The board trio: qualifiers in play and the VP the card pays at that count.
    count = scoring.bonus_qualifying_count(player, bonus_card)
    base = layout._OFF_BONUS_VALUE
    feat[base + layout._BONUS_VALUE_QUAL] = count / layout._BONUS_COUNT_SCALE
    feat[base + layout._BONUS_VALUE_STEPPED] = (
        scoring.bonus_score_for_count(bonus_card, count) / layout._BONUS_VALUE_SCALE
    )
    feat[base + layout._BONUS_VALUE_LINEAR] = (
        scoring.bonus_linear_value_for_count(bonus_card, count)
        / layout._BONUS_VALUE_SCALE
    )

    # Potential: birds not yet in play that pass the card's static test. The
    # hand-counting dynamic card is the exception — every card in the hand
    # source counts toward it, whatever its printed categories.
    if scoring.bonus_count_delta_for_hand(bonus_card, 1) > 0:
        hand_qual = sum(1 for _bird in hand_source)
    else:
        hand_qual = sum(
            1 for bird in hand_source if bonus_card.name in bird.bonus_categories
        )
    feat[base + layout._BONUS_VALUE_HAND] = hand_qual / layout._BONUS_COUNT_SCALE
    tray_qual = sum(
        1
        for tray_bird in tray
        if tray_bird is not None and bonus_card.name in tray_bird.bonus_categories
    )
    feat[base + layout._BONUS_VALUE_TRAY] = tray_qual / layout._BONUS_COUNT_SCALE


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
