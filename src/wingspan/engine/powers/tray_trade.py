# pyright: reportUnusedFunction=false
# (every function here is a power handler registered via @registry.handles;
# none is called by name, so pyright's unused-function check is a false positive)
"""Tray-draw, wild-food trade, and fewest-habitat-birds handlers.

Each handler registers itself with ``@registry.handles`` and is imported by the
``powers`` package ``__init__`` so the dispatch table is populated on load.
"""

from __future__ import annotations

import typing

from wingspan import cards, decisions, state
from wingspan.engine.powers import dispatch, registry

if typing.TYPE_CHECKING:
    from wingspan.engine import core


@registry.handles(cards.EffectKind.DRAW_FROM_TRAY_ALL)
def _h_draw_from_tray_all(
    engine: "core.Engine",
    agent: "core.Agent",
    player: state.Player,
    pb: state.PlayedBird,
    habitat: cards.Habitat,
    eff: cards.Effect,
    trigger: str,
) -> None:
    # Brant (generic for N): take every face-up card in the tray.
    # The end-of-turn refill (core._take_turn) handles refilling; no mid-turn refill.
    st = engine.state
    bird = pb.bird
    taken = [card for card in st.tray if card is not None]
    st.tray = [None] * state.TRAY_SIZE
    player.hand.extend(taken)
    engine.log(
        f"  {bird.name}: drew {len(taken)} card(s) from tray: "
        f"{[card.name for card in taken]}"
    )


@registry.handles(cards.EffectKind.TRADE_WILD_FOOD)
def _h_trade_wild_food(
    engine: "core.Engine",
    agent: "core.Agent",
    player: state.Player,
    pb: state.PlayedBird,
    habitat: cards.Habitat,
    eff: cards.Effect,
    trigger: str,
) -> None:
    # Green Heron: discard 1 food, then gain 1 food from the supply.
    # The trade is forced — no activation gate (gap #18). Silent skip on no food.
    # Step 1: SPEND_FOOD — which food to discard (mandatory if any food held).
    # Step 2: GAIN_FOOD — which food to gain from supply (mandatory; the just-returned
    #   food is back in supply, so trading a food for itself is legal but wasteful).
    bird = pb.bird

    # Pre-flight: need a food to give up.
    if player.total_food() <= 0:
        engine.log(f"  {bird.name}: no food to trade; power skipped")
        return

    # Step 1 — mandatory discard.
    lose_food = _trade_discard_step(engine, agent, player, bird)

    # Step 2 — mandatory gain from supply.
    gain_food = _trade_gain_step(engine, agent, player, bird)

    engine.log(
        f"  {bird.name}: discarded 1 {lose_food.value}, gained 1 {gain_food.value}"
    )


def _trade_discard_step(
    engine: "core.Engine",
    agent: "core.Agent",
    player: state.Player,
    bird: cards.Bird,
) -> cards.Food:
    """The 'lose 1 food back to the supply' half of a wild-food trade; returns
    the discarded food."""
    lose_ch = engine.ask(
        agent,
        decisions.SpendFoodDecision(
            player_id=player.id,
            prompt=f"[{player.name}] discard 1 food back to the supply (from {bird.name})",
            choices=[
                decisions.FoodChoice(label=food.value, food=food)
                for food in cards.ALL_FOODS
                if player.food.get(food, 0) > 0
            ],
        ),
    )
    assert isinstance(lose_ch, decisions.FoodChoice)
    lose_food = lose_ch.food
    player.food[lose_food] -= 1
    return lose_food


def _trade_gain_step(
    engine: "core.Engine",
    agent: "core.Agent",
    player: state.Player,
    bird: cards.Bird,
) -> cards.Food:
    """The 'gain 1 food from the supply' half of a wild-food trade; returns
    the gained food. Supply is infinite so all 5 food types are always available."""
    gain_ch = engine.ask(
        agent,
        decisions.GainFoodDecision(
            player_id=player.id,
            prompt=f"[{player.name}] gain 1 food from the supply ({bird.name})",
            choices=[
                decisions.FoodChoice(label=food.value, food=food)
                for food in cards.ALL_FOODS
            ],
        ),
    )
    assert isinstance(gain_ch, decisions.FoodChoice)
    gain_food = gain_ch.food
    player.food[gain_food] += 1
    return gain_food


def _fewest_habitat_gate(
    engine: "core.Engine",
    agent: "core.Agent",
    player: state.Player,
    bird: cards.Bird,
    habitat: cards.Habitat,
    make_accept_choice: typing.Callable[[int], decisions.PayCostChoice],
) -> tuple[list[int], int] | None:
    """Common pre-check for FEWEST_HABITAT effects.

    Handles auto-skip (active player doesn't have fewest habitat birds) and the
    tied-case veto gate (all tied players share the benefit). Returns
    ``(counts, fewest)`` if the effect should proceed, or ``None`` if it was
    auto-skipped or vetoed.

    ``make_accept_choice(n_opponents_tied)`` is called only when a veto is
    needed; it must return the fully-populated ``PayCostChoice`` for that tied
    scenario.
    """
    st = engine.state
    counts = [len(other.board[habitat]) for other in st.players]
    fewest = min(counts)

    # Auto-skip: only opponent benefits if the active player has more than fewest.
    if len(player.board[habitat]) != fewest:
        engine.log(
            f"  {bird.name}: [{player.name}] has more {habitat.value} birds"
            f" than opponent; power auto-skipped"
        )
        return None

    # Tie check: all tied players share the benefit, so offer a veto gate.
    n_tied = sum(1 for count in counts if count == fewest)
    if n_tied > 1:
        n_opponents_tied = n_tied - 1
        accepted = dispatch.offer_activation_veto(
            engine,
            agent,
            player,
            f"[{player.name}] activate {bird.name}?"
            f" (tied fewest {habitat.value} birds; all tied players benefit)",
            make_accept_choice(n_opponents_tied),
        )
        if not accepted:
            engine.log(f"  {bird.name}: [{player.name}] skipped activation")
            return None

    return counts, fewest


@registry.handles(cards.EffectKind.FEWEST_FOREST_GAINS_DIE)
def _h_fewest_forest_gains_die(
    engine: "core.Engine",
    agent: "core.Agent",
    player: state.Player,
    pb: state.PlayedBird,
    habitat: cards.Habitat,
    eff: cards.Effect,
    trigger: str,
) -> None:
    from wingspan.engine import actions

    bird = pb.bird
    result = _fewest_habitat_gate(
        engine,
        agent,
        player,
        bird,
        cards.Habitat.FOREST,
        lambda n_opp: decisions.PayCostChoice(
            label="activate",
            gained_food_count=eff.amount,
            opp_gained_food_count=n_opp * eff.amount,
        ),
    )
    if result is None:
        return

    counts, fewest = result
    for other_player, forest_count in zip(engine.state.players, counts):
        if forest_count != fewest:
            continue
        responder = engine.agent_for(other_player)
        gained = actions.take_one_from_feeder(
            engine,
            responder,
            other_player,
            prompt=f"[{other_player.name}] take 1 die from birdfeeder (from {bird.name})",
        )
        assert gained is not None  # unrestricted menu, post-reset
        engine.log(
            f"  {bird.name}: [{other_player.name}] +1 {gained.value} from birdfeeder"
        )


@registry.handles(cards.EffectKind.FEWEST_WETLAND_DRAWS_CARD)
def _h_fewest_wetland_draws_card(
    engine: "core.Engine",
    agent: "core.Agent",
    player: state.Player,
    pb: state.PlayedBird,
    habitat: cards.Habitat,
    eff: cards.Effect,
    trigger: str,
) -> None:
    # "Player(s) with the fewest birds in their [wetland] draw 1 [card]."
    # American Bittern, Common Loon. Same gate logic as _h_fewest_forest_gains_die
    # via _fewest_habitat_gate; the tied case offers a veto because the opponent
    # also draws.
    from wingspan.engine import actions

    bird = pb.bird
    result = _fewest_habitat_gate(
        engine,
        agent,
        player,
        bird,
        cards.Habitat.WETLAND,
        lambda n_opp: decisions.PayCostChoice(
            label="activate",
            gained_card_count=1,
            opp_gained_card_count=n_opp,
        ),
    )
    if result is None:
        return

    counts, fewest = result
    for other_player, wetland_count in zip(engine.state.players, counts):
        if wetland_count != fewest:
            continue
        responder = engine.agent_for(other_player)
        actions.draw_one_card(engine, responder, other_player)
        engine.log(f"  {bird.name}: [{other_player.name}] drew 1 card")
