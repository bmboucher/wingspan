"""Optimistic exchange forecasts for MAIN_ACTION commitment rows.

The main-action menu (``MainActionDecision``) commits to a habitat row before
its follow-up decisions (which bird, which food, which egg) resolve. This
module prices that commitment: a greedy, per-action forecast of the resources
the row could plausibly deliver, mirroring the engine's real activation
order — base track gain, then the row's one-shot trade-arrow conversion, then
every already-in-play brown bird's power, right-to-left — under running
coarse feasibility budgets (food, eggs, hand size, spare egg room). The
forecast is optimistic, not exact: conditional and partial-probability
effects are priced at nominal face value (see :func:`effect_exchange_ledger`),
and each phase's gate uses the budgets left over from the phases before it,
not a full re-simulation of the engine.

Pure free functions over ``(state.Player, state.GameState)`` — no
``engine.core.Engine`` import. Lives in ``engine`` because it speaks engine
vocabulary (tracks, conversion gates, activation order) and returns
``decisions.ExchangeLedger``; encoders import it locally
(``from wingspan.engine import forecast``) to keep ``wingspan.encode``
engine-free at import time — the same pattern ``choice_encode`` uses for
``scoring`` and ``playability``.
"""

from __future__ import annotations

import pydantic

from wingspan import cards, decisions, state

# ---------------------------------------------------------------------------
# The habitat each habitat-row main action forecasts. PLAY_BIRD has no row —
# its exchange stripe stays zero at the choice-encode layer by design.
_ACTION_HABITATS: dict[decisions.MainAction, cards.Habitat] = {
    decisions.MainAction.GAIN_FOOD: cards.Habitat.FOREST,
    decisions.MainAction.LAY_EGGS: cards.Habitat.GRASSLAND,
    decisions.MainAction.DRAW_CARDS: cards.Habitat.WETLAND,
}


def effect_exchange_ledger(
    effect: cards.Effect, *, include_eot_discard: bool
) -> decisions.ExchangeLedger:
    """Map one bird-power ``effect`` onto an :class:`decisions.ExchangeLedger`.

    The single source of truth for how an ``EffectKind`` translates into
    ledger fields: every kind maps to zero or more fields; compound kinds
    (e.g. ``TUCK_FROM_HAND_THEN_DRAW``) set both a cost and a gain field. Pink
    reactive kinds use the self-gain fields (the reacting player's
    perspective). Conditional or partial-probability effects
    (``PREDATOR_HUNT``, ``FEWEST_*``, ``ROLL_NOT_IN_FEEDER_*``) are mapped by
    their nominal exchange — the model learns to discount uncertain outcomes
    via training, not by zeroing the signal here. Unknown/``UNIMPLEMENTED``
    kinds contribute nothing (the correct default).

    ``include_eot_discard`` is the era seam for the mapper's one known gap:
    ``DRAW_CARDS_THEN_DISCARD_EOT`` draws now but discards *later*, at end of
    turn, after every other row power resolves — a real cost the original
    state-encoder mapper omitted. ``True`` adds that ``paid_card_count += 1``
    side (used by the live optimistic forecast,
    :func:`habitat_action_exchange_forecast`, and by ``state_encode`` from the
    v1.5 era onward); ``False`` reproduces the old byte-for-byte behavior
    (used by frozen era-<=1.4 artifacts — see ``wingspan.compat.v1_4``)."""
    ledger: dict[str, int] = {}
    amount = effect.amount
    kind = effect.kind
    handled = _map_gain_and_cache(ledger, kind, amount)
    if not handled:
        handled = _map_egg_and_draw(
            ledger, kind, amount, include_eot_discard=include_eot_discard
        )
    if not handled:
        _map_conversion_and_shared(ledger, kind, amount)
    return _build_ledger(ledger)


def habitat_action_exchange_forecast(
    player: state.Player, game_state: state.GameState, action: decisions.MainAction
) -> decisions.ExchangeLedger:
    """Optimistic exchange forecast for committing to ``action``'s habitat row.

    GAIN_FOOD / LAY_EGGS / DRAW_CARDS only — raises ``ValueError`` for
    ``PLAY_BIRD``, which has no row to forecast. Runs three phases in the
    engine's real activation order, threading a single set of running
    :class:`_Budgets` through all of them so later phases only advertise what
    is left after earlier ones commit:

    1. :func:`_apply_base_gain` — the row's base printed track gain.
    2. :func:`_apply_conversion` — the row's one-shot trade-arrow conversion,
       if the row lands on a trade space and the engine's exact gates are met.
    3. :func:`_apply_row_powers` — every already-in-play brown bird's power,
       right-to-left (the engine's real column-activation order).
    """
    if action not in _ACTION_HABITATS:
        raise ValueError(f"no habitat-action exchange forecast for {action}")
    habitat = _ACTION_HABITATS[action]

    ledger: dict[str, int] = {}
    budgets = _Budgets(
        food=player.food.total(),
        eggs=player.total_eggs,
        hand=len(player.hand),
        egg_room=sum(
            max(played_bird.bird.egg_limit - played_bird.eggs, 0)
            for row in player.board.values()
            for played_bird in row
        ),
    )
    _apply_base_gain(ledger, budgets, player, game_state, action)
    _apply_conversion(ledger, budgets, player, game_state, action, habitat)
    _apply_row_powers(ledger, budgets, player, habitat)
    return _build_ledger(ledger)


###### PRIVATE #######

#### Ledger construction ####


def _build_ledger(counts: dict[str, int]) -> decisions.ExchangeLedger:
    """Build an :class:`decisions.ExchangeLedger` from a sparse ``{field:
    count}`` accumulator (only nonzero fields are typically present; every
    other field takes its 0 default). Spelled out field-by-field rather than
    ``ExchangeLedger(**counts)`` because ``counts`` is a plain
    ``dict[str, int]`` — a dict-splat can't type-check against
    ``ExchangeLedger``'s one non-int field (``paid_food: cards.Food | None``)
    under strict pyright."""
    return decisions.ExchangeLedger(
        paid_card_count=counts.get("paid_card_count", 0),
        paid_food_count=counts.get("paid_food_count", 0),
        paid_egg_count=counts.get("paid_egg_count", 0),
        gained_food_count=counts.get("gained_food_count", 0),
        gained_egg_count=counts.get("gained_egg_count", 0),
        gained_card_count=counts.get("gained_card_count", 0),
        gained_tuck_count=counts.get("gained_tuck_count", 0),
        gained_play_count=counts.get("gained_play_count", 0),
        gained_cache_count=counts.get("gained_cache_count", 0),
        opp_gained_food_count=counts.get("opp_gained_food_count", 0),
        opp_gained_egg_count=counts.get("opp_gained_egg_count", 0),
        opp_gained_card_count=counts.get("opp_gained_card_count", 0),
        opp_gained_tuck_count=counts.get("opp_gained_tuck_count", 0),
    )


#### Feasibility budgets ####


class _Budgets(pydantic.BaseModel):
    """Running coarse feasibility budgets tracked while forecasting a habitat
    action's exchange — decremented by costs and incremented by gains as each
    phase (base gain, one-shot conversion, brown row powers) applies, so later
    phases only see the resources actually left over from earlier ones."""

    food: int
    eggs: int
    hand: int
    egg_room: int


def _cards_available(game_state: state.GameState) -> bool:
    """Whether a card could still be drawn from the tray, deck, or discard
    pile. Reimplements ``engine.actions._cards_available_to_draw`` against
    ``GameState`` directly, since this module takes no ``Engine``."""
    return bool(
        any(bird is not None for bird in game_state.tray)
        or game_state.bird_deck
        or game_state.bird_discard
    )


#### Phase 1: base track gain ####


def _apply_base_gain(
    ledger: dict[str, int],
    budgets: _Budgets,
    player: state.Player,
    game_state: state.GameState,
    action: decisions.MainAction,
) -> None:
    """The row's base printed track gain, updating ``budgets`` in place."""
    if action == decisions.MainAction.GAIN_FOOD:
        gained = player.board.gain_food_count()
        _bump(ledger, "gained_food_count", gained)
        budgets.food += gained
    elif action == decisions.MainAction.LAY_EGGS:
        laid = min(player.board.lay_eggs_count(), budgets.egg_room)
        _bump(ledger, "gained_egg_count", laid)
        budgets.eggs += laid
        budgets.egg_room -= laid
    elif action == decisions.MainAction.DRAW_CARDS and _cards_available(game_state):
        drawn = player.board.draw_cards_count()
        _bump(ledger, "gained_card_count", drawn)
        budgets.hand += drawn


#### Phase 2: one-shot trade-arrow conversion ####


def _apply_conversion(
    ledger: dict[str, int],
    budgets: _Budgets,
    player: state.Player,
    game_state: state.GameState,
    action: decisions.MainAction,
    habitat: cards.Habitat,
) -> None:
    """The row's one-shot trade-arrow conversion, gated exactly as the engine
    gates it (``engine.actions._convert_gain_food`` / ``_convert_lay_eggs`` /
    ``_convert_draw_cards``), updating ``budgets`` in place."""
    if not player.board.action_offers_convert(habitat):
        return
    if action == decisions.MainAction.GAIN_FOOD:
        if budgets.hand < 1:
            return
        _bump(ledger, "paid_card_count", 1)
        _bump(ledger, "gained_food_count", 1)
        budgets.hand -= 1
        budgets.food += 1
    elif action == decisions.MainAction.LAY_EGGS:
        if budgets.food < 1 or budgets.egg_room < 1:
            return
        _bump(ledger, "paid_food_count", 1)
        _bump(ledger, "gained_egg_count", 1)
        budgets.food -= 1
        budgets.eggs += 1
        budgets.egg_room -= 1
    elif action == decisions.MainAction.DRAW_CARDS:
        if budgets.eggs < 1 or not _cards_available(game_state):
            return
        _bump(ledger, "paid_egg_count", 1)
        _bump(ledger, "gained_card_count", 1)
        budgets.eggs -= 1
        budgets.hand += 1


#### Phase 3: brown row powers, right-to-left ####


def _apply_row_powers(
    ledger: dict[str, int],
    budgets: _Budgets,
    player: state.Player,
    habitat: cards.Habitat,
) -> None:
    """Every brown bird already in ``habitat``'s row, right-to-left (the
    engine's real column-activation order), each effect gated by the budgets
    left over from Phase 1 + 2."""
    for played_bird in reversed(player.board[habitat]):
        if played_bird.bird.color != cards.PowerColor.BROWN:
            continue
        for effect in played_bird.bird.power.effects:
            _apply_one_row_power_effect(ledger, budgets, effect)


def _apply_one_row_power_effect(
    ledger: dict[str, int], budgets: _Budgets, effect: cards.Effect
) -> None:
    """Apply one brown-power effect's optimistic delta to ``ledger`` and
    ``budgets``, or skip it entirely if any cost side exceeds its budget.

    The self egg gain is capped at the room actually remaining. The
    end-of-turn discard side of ``DRAW_CARDS_THEN_DISCARD_EOT`` is charged to
    the ledger but is neither a precondition nor a hand-budget decrement — it
    resolves at end of turn, after every remaining row power has already drawn
    against the same hand (the ``paid_card_count`` this kind contributes *is*
    the EOT side), so the power fires even from an empty hand."""
    delta = effect_exchange_ledger(effect, include_eot_discard=True)
    is_eot_discard = effect.kind == cards.EffectKind.DRAW_CARDS_THEN_DISCARD_EOT
    cards_needed_now = 0 if is_eot_discard else delta.paid_card_count
    if (
        cards_needed_now > budgets.hand
        or delta.paid_food_count > budgets.food
        or delta.paid_egg_count > budgets.eggs
    ):
        return
    eggs_gain = min(delta.gained_egg_count, budgets.egg_room)

    if not is_eot_discard:
        budgets.hand -= delta.paid_card_count
    budgets.food -= delta.paid_food_count
    budgets.eggs -= delta.paid_egg_count
    budgets.hand += delta.gained_card_count
    budgets.food += delta.gained_food_count
    budgets.eggs += eggs_gain
    budgets.egg_room -= eggs_gain

    _bump(ledger, "paid_card_count", delta.paid_card_count)
    _bump(ledger, "paid_food_count", delta.paid_food_count)
    _bump(ledger, "paid_egg_count", delta.paid_egg_count)
    _bump(ledger, "gained_food_count", delta.gained_food_count)
    _bump(ledger, "gained_egg_count", eggs_gain)
    _bump(ledger, "gained_card_count", delta.gained_card_count)
    _bump(ledger, "gained_tuck_count", delta.gained_tuck_count)
    _bump(ledger, "gained_play_count", delta.gained_play_count)
    _bump(ledger, "gained_cache_count", delta.gained_cache_count)
    _bump(ledger, "opp_gained_food_count", delta.opp_gained_food_count)
    _bump(ledger, "opp_gained_egg_count", delta.opp_gained_egg_count)
    _bump(ledger, "opp_gained_card_count", delta.opp_gained_card_count)
    _bump(ledger, "opp_gained_tuck_count", delta.opp_gained_tuck_count)


#### Effect -> ledger-field branch table (ported from state_encode) ####


def _map_gain_and_cache(
    vec: dict[str, int], kind: cards.EffectKind, amount: int
) -> bool:
    """Food gains (from supply, feeder, die, or compound tuck-then-gain) and
    cache gains (food cached on the bird itself). Returns whether ``kind`` was
    handled by this group."""
    # Food gains
    if kind in (
        cards.EffectKind.GAIN_FOOD_SUPPLY,
        cards.EffectKind.GAIN_FOOD_BIRDFEEDER,
        cards.EffectKind.GAIN_FOOD_FROM_FEEDER_CHOICE,
        cards.EffectKind.GAIN_DIE_ANY,
        cards.EffectKind.GAIN_ALL_FOOD_FEEDER,
        cards.EffectKind.FEWEST_FOREST_GAINS_DIE,
        cards.EffectKind.FEWEST_WETLAND_DRAWS_CARD,
    ):
        _bump(vec, "gained_food_count", amount)
    elif kind in (
        cards.EffectKind.TUCK_FROM_HAND_THEN_GAIN_FOOD_SUPPLY,
        cards.EffectKind.TUCK_FROM_HAND_THEN_GAIN_FOOD_CHOICE,
    ):
        _bump(vec, "paid_card_count", 1)
        _bump(vec, "gained_food_count", amount)

    # Cache gains (food cached on the bird itself)
    elif kind in (
        cards.EffectKind.CACHE_FOOD,
        cards.EffectKind.ROLL_NOT_IN_FEEDER_CACHE,
        cards.EffectKind.GAIN_FOOD_FEEDER_MAY_CACHE,
        cards.EffectKind.PINK_GAIN_FOOD_CACHE,
    ):
        _bump(vec, "gained_cache_count", amount)
    else:
        return False
    return True


def _map_egg_and_draw(
    vec: dict[str, int],
    kind: cards.EffectKind,
    amount: int,
    *,
    include_eot_discard: bool,
) -> bool:
    """Egg gains, card draws (and their tuck-from-hand-then-X compounds), and
    tucks/discards that cost a card or an egg. Returns whether ``kind`` was
    handled by this group."""
    # Egg gains
    if kind in (cards.EffectKind.LAY_EGG_ON_THIS, cards.EffectKind.LAY_EGG_ANY):
        _bump(vec, "gained_egg_count", amount)
    elif kind in (
        cards.EffectKind.TUCK_FROM_HAND_THEN_LAY_ON_THIS,
        cards.EffectKind.TUCK_FROM_HAND_THEN_LAY_ANY,
    ):
        _bump(vec, "paid_card_count", 1)
        _bump(vec, "gained_egg_count", amount)

    # Card draws
    elif kind in (
        cards.EffectKind.DRAW_CARDS,
        cards.EffectKind.DRAW_FROM_TRAY_ALL,
        cards.EffectKind.DRAW_N_PLUS_ONE_DRAFT,
        cards.EffectKind.DRAW_CARDS_THEN_DISCARD_EOT,
    ):
        _bump(vec, "gained_card_count", amount)
        if kind == cards.EffectKind.DRAW_CARDS_THEN_DISCARD_EOT and include_eot_discard:
            _bump(vec, "paid_card_count", 1)
    elif kind == cards.EffectKind.TUCK_FROM_HAND_THEN_DRAW:
        _bump(vec, "paid_card_count", 1)
        _bump(vec, "gained_card_count", amount)

    # Cards tucked (from deck onto a bird) / discarded (tuck from hand as a cost)
    elif kind in (
        cards.EffectKind.TUCK_FROM_DECK,
        cards.EffectKind.TUCK_FROM_DECK_PAID,
    ):
        if kind == cards.EffectKind.TUCK_FROM_DECK_PAID:
            _bump(vec, "paid_egg_count", 1)
        _bump(vec, "gained_tuck_count", amount)
    elif kind == cards.EffectKind.TUCK_FROM_HAND:
        _bump(vec, "paid_card_count", 1)

    # Egg-cost exchanges
    elif kind == cards.EffectKind.DISCARD_EGG_FOR_CARDS:
        _bump(vec, "paid_egg_count", 1)
        _bump(vec, "gained_card_count", amount)
    elif kind == cards.EffectKind.DISCARD_EGG_FOR_WILD:
        _bump(vec, "paid_egg_count", 1)
        _bump(vec, "gained_food_count", 1)
    else:
        return False
    return True


def _map_conversion_and_shared(
    vec: dict[str, int], kind: cards.EffectKind, amount: int
) -> bool:
    """Wild food trades, extra bird plays, all-players shared effects, and
    pink reactive gains (the reacting player's own gain). Returns whether
    ``kind`` was handled by this group; ``UNIMPLEMENTED`` and unknown kinds
    fall through and contribute nothing (the correct default)."""
    # Wild food trade (net zero food but signals a conversion power)
    if kind == cards.EffectKind.TRADE_WILD_FOOD:
        _bump(vec, "paid_food_count", 1)
        _bump(vec, "gained_food_count", 1)

    # Extra bird plays
    elif kind in (
        cards.EffectKind.PLAY_ADDITIONAL_BIRD,
        cards.EffectKind.PLAY_ADDITIONAL_BIRD_HERE,
    ):
        _bump(vec, "gained_play_count", 1)

    # All-players effects: self-gain + opponent-gain
    elif kind == cards.EffectKind.ALL_PLAYERS_GAIN_FOOD:
        _bump(vec, "gained_food_count", amount)
        _bump(vec, "opp_gained_food_count", amount)
    elif kind == cards.EffectKind.EACH_PLAYER_GAINS_DIE_CHOOSE_ORDER:
        _bump(vec, "gained_food_count", 1)
        _bump(vec, "opp_gained_food_count", 1)
    elif kind == cards.EffectKind.ALL_PLAYERS_DRAW:
        _bump(vec, "gained_card_count", amount)
        _bump(vec, "opp_gained_card_count", amount)
    elif kind in (
        cards.EffectKind.ALL_PLAYERS_LAY_EGG_ON_NEST,
        cards.EffectKind.LAY_EGG_ALL_NEST,
    ):
        _bump(vec, "gained_egg_count", amount)
        _bump(vec, "opp_gained_egg_count", amount)

    # Pink reactive effects (reacting player's gain)
    elif kind == cards.EffectKind.PINK_PLAY_BIRD_GAIN:
        _bump(vec, "gained_food_count", amount)
    elif kind == cards.EffectKind.PINK_PLAY_BIRD_TUCK:
        _bump(vec, "gained_tuck_count", amount)
    elif kind == cards.EffectKind.PINK_LAY_EGG_ON_NEST:
        _bump(vec, "gained_egg_count", amount)
    else:
        return False
    return True


def _bump(ledger: dict[str, int], field: str, amount: int) -> None:
    ledger[field] = ledger.get(field, 0) + amount
