"""Setup-phase resolution helpers, split out of ``engine.core.Engine``.

These are the mutation and follow-up-decision steps that turn a resolved
``decisions.SetupChoice`` into board/state changes plus (under the deferred
regimes) the in-game bonus-card and food decisions that complete the opening
setup. They are free functions whose first argument is the live ``Engine``,
matching the sibling-module convention used by ``actions.py`` and the other
engine modules — so a forthcoming aid preview can call them directly without
routing through ``Engine`` methods.

``core._resolve_setup_choice`` (the ask-the-agent path) and
``core._setup_phase_fixed`` (the fixed-setup / setup-model path) are the two
call sites; both apply the chosen keep with :func:`apply_setup_choice`, then
resolve any deferral with :func:`resolve_deferred_setup_bonus` and
:func:`resolve_deferred_setup_food`.
"""

from __future__ import annotations

import typing

from wingspan import cards, decisions, state
from wingspan.engine import actions, ledger
from wingspan.gamelog import models as gamelog_models

if typing.TYPE_CHECKING:
    from wingspan.engine import core


def apply_setup_choice(
    engine: "core.Engine",
    player: state.Player,
    dealt_cards: list[cards.Bird],
    dealt_bonus: list[cards.BonusCard],
    sc: decisions.SetupChoice,
    *,
    defer_food: bool = False,
) -> None:
    """Mutate ``player`` / ``engine.state`` to reflect the chosen setup combination.

    When ``defer_food`` is True (the ``split_setup_food`` regime), the food
    update is skipped here — food is instead resolved by a subsequent call to
    :func:`resolve_deferred_setup_food`."""
    kept = list(sc.kept_cards)
    for card in dealt_cards:
        if card not in kept:
            ledger.discard_from_hand(
                engine, player, card, purpose=gamelog_models.EffectPurpose.SETUP
            )
    if not defer_food:
        for food in cards.ALL_FOODS:
            if food not in sc.kept_foods:
                ledger.spend_food(
                    engine,
                    player,
                    food,
                    purpose=gamelog_models.EffectPurpose.SETUP,
                )
    if sc.bonus_card is not None:
        player.bonus_cards.append(sc.bonus_card)
        for bonus in dealt_bonus:
            if bonus is not sc.bonus_card:
                engine.state.bonus_discard.append(bonus)


def resolve_deferred_setup_bonus(
    engine: "core.Engine",
    player: state.Player,
    dealt_bonus: list[cards.BonusCard],
    sc: decisions.SetupChoice,
) -> cards.BonusCard | None:
    """Pick ``player``'s bonus card via the in-game ``CHOOSE_BONUS`` head when
    the setup keep deferred it (the ``split_setup_bonus`` regime).

    A keep with ``bonus_card is None`` while bonus cards were dealt is the
    deferral signal — it never arises in the combined-keep regime, where the
    enumerator always assigns one of the dealt bonuses. The pick is asked over a
    minimal "start of round 1" snapshot (round 0, full action cubes; the tray /
    birdfeeder / round goals were set by ``new_game``) so the in-game head scores
    the bonus over a faithful opening, and it routes through ``Engine.ask`` so a
    collecting agent records it like any other in-game decision.

    Returns the bonus card ``player`` ends up keeping: ``sc.bonus_card`` on the
    non-deferred path (``None`` when no bonus was dealt at all), or the agent's
    pick on the deferred path. Engine call sites ignore the return value; it
    exists so other callers (e.g. an aid preview) can learn the resolved bonus
    without re-deriving it."""
    if not dealt_bonus or sc.bonus_card is not None:
        # Bonus was included in the combined setup choice (non-deferred path).
        # Fire setup_applied now so all listeners receive it; the HTML log
        # handler checks choice.bonus_card to decide whether to open a phase.
        engine.instrumentation.setup_applied(engine=engine, player=player, choice=sc)
        return sc.bonus_card
    engine.log_section(f"=== SETUP: {player.name} CHOOSING BONUS CARD ===")
    # Fire setup_applied at the start of this segment so the HTML log's
    # zip(phases, segments) pairs this phase with the bonus-choosing narration.
    engine.instrumentation.setup_applied(engine=engine, player=player, choice=sc)
    # ``_play_round`` resets the cubes again before real play, so pre-loading
    # them here only shapes the encoded snapshot the bonus pick is scored over.
    for seat in engine.state.players:
        seat.action_cubes_left = state.ROUND_CUBES[0]
    engine.state.current_player = player.id
    decision = decisions.BirdPowerPickBonusCardDecision(
        player_id=player.id,
        prompt=f"[{player.name}] keep a bonus card",
        choices=[
            decisions.BonusCardChoice(label=bonus.name, bonus_card=bonus)
            for bonus in dealt_bonus
        ],
    )
    chosen = engine.ask(engine.agent_for(player), decision)
    player.bonus_cards.append(chosen.bonus_card)
    for bonus in dealt_bonus:
        if bonus is not chosen.bonus_card:
            engine.state.bonus_discard.append(bonus)
    engine.log(f"[{player.name}] keeps bonus: {chosen.bonus_card.name}")
    return chosen.bonus_card


def resolve_deferred_setup_food(
    engine: "core.Engine",
    player: state.Player,
    agent: "core.Agent",
    n_kept: int,
    *,
    defer_food: bool,
) -> None:
    """Resolve the opening food pick via in-game decisions when the setup keep
    deferred it (the ``split_setup_food`` regime).

    Food decisions depend on how many birds were kept, splitting evenly around
    the "pay-for-cards" midpoint so both spend and gain sides of food valuation
    get training signal:

      0 kept → keep all 5 food (no decision needed)
      1 kept → 5 food dealt, 1 × SpendFoodDecision (discard 1)
      2 kept → 5 food dealt, 2 × SpendFoodDecision (no repeat)
      3 kept → zero food,    2 × GainFoodDecision  (no repeat)
      4 kept → zero food,    1 × GainFoodDecision
      5 kept → keep no food (no decision needed)

    All decisions are mandatory (no SkipChoice offered) and route through
    ``Engine.ask`` so collecting agents record them like any in-game decision.
    The encoder's action-cubes snapshot is pre-loaded to round-1 values (same
    as the bonus-deferral path) so the heads score food over a faithful opening.

    Under the ``combine_gain_food`` regime the gain/discard split is replaced
    by a single "gain ``5 - n_kept`` foods" pick over the distinct-foods
    subsets (one die of each food on offer), from a zeroed food pool.
    """
    if not defer_food:
        return

    # Pre-load round-1 cubes so the in-game encoder sees a faithful opening.
    # ``_play_round`` resets them again before real play.
    for seat in engine.state.players:
        seat.action_cubes_left = state.ROUND_CUBES[0]
    engine.state.current_player = player.id

    if engine.combine_gain_food:
        # One combined "gain N foods" pick over N-subsets of the five distinct
        # foods (one die of each on offer), replacing the gain/discard split.
        # N = 5 - n_kept (the foods kept after paying for birds); start at 0.
        clear_setup_food(engine, player)
        n_keep = len(cards.ALL_FOODS) - n_kept
        actions.combined_supply_gain(
            engine,
            agent,
            player,
            n_keep,
            per_food_capacity=1,
            prompt=f"[{player.name}] setup: choose {n_keep} food to keep",
        )
        return

    if n_kept >= 3:
        # High-keep: player would have no food left after paying for birds.
        # Zero out the post-deal food pool and grant food via gain decisions.
        clear_setup_food(engine, player)
        n_gains = min(5 - n_kept, 2)  # 3 kept → 2, 4 kept → 1, 5 kept → 0
        gained: set[cards.Food] = set()
        for gain_num in range(n_gains):
            available = [food for food in cards.ALL_FOODS if food not in gained]
            decision = decisions.GainFoodDecision(
                player_id=player.id,
                prompt=(
                    f"[{player.name}] setup: choose food to gain "
                    f"({gain_num + 1}/{n_gains})"
                ),
                choices=[
                    decisions.FoodChoice(label=food.value, food=food)
                    for food in available
                ],
            )
            chosen = engine.ask(agent, decision)
            assert isinstance(chosen, decisions.FoodChoice)
            ledger.gain_food(
                engine,
                player,
                chosen.food,
                source=gamelog_models.FoodSource.SUPPLY,
            )
            gained.add(chosen.food)
    else:
        # Low-keep: player has 5 food from the deal and discards down.
        n_spends = n_kept  # 0 kept → 0, 1 kept → 1, 2 kept → 2
        for spend_num in range(n_spends):
            held = [food for food in cards.ALL_FOODS if player.food[food] > 0]
            decision = decisions.SpendFoodDecision(
                player_id=player.id,
                prompt=(
                    f"[{player.name}] setup: choose food to discard "
                    f"({spend_num + 1}/{n_spends})"
                ),
                choices=[
                    decisions.FoodChoice(label=food.value, food=food) for food in held
                ],
            )
            chosen = engine.ask(agent, decision)
            assert isinstance(chosen, decisions.FoodChoice)
            ledger.spend_food(
                engine,
                player,
                chosen.food,
                purpose=gamelog_models.EffectPurpose.SETUP,
            )


def clear_setup_food(engine: "core.Engine", player: state.Player) -> None:
    """Zero ``player``'s post-deal food pool before the deferred-food regimes
    grant it back through in-game decisions.

    Recorded as one :class:`~wingspan.gamelog.models.SpendFoodEffect` per
    token rather than a bulk assignment, so the ledger stays a complete
    account of every token that entered or left a supply."""
    for food in cards.ALL_FOODS:
        if player.food[food]:
            ledger.spend_food(
                engine,
                player,
                food,
                player.food[food],
                purpose=gamelog_models.EffectPurpose.SETUP,
            )
