# pyright: reportUnusedFunction=false
# (every function here is a power handler registered via @registry.handles;
# none is called by name, so pyright's unused-function check is a false positive)
"""Tray-draw, wild-food trade, and fewest-forest-birds handlers.

Each handler registers itself with ``@registry.handles`` and is imported by the
``powers`` package ``__init__`` so the dispatch table is populated on load.
"""

from __future__ import annotations

import typing

from wingspan import cards, decisions, state
from wingspan.engine.powers import registry

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
    # Brant (generic for N): take every face-up card in the tray, then refill.
    st = engine.state
    bird = pb.bird
    taken = [card for card in st.tray if card is not None]
    st.tray = [None] * state.TRAY_SIZE
    player.hand.extend(taken)
    st.refill_tray()
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
    # Green Heron: swap one food for another, modelled as two atomic decisions —
    # "gain 1 food from the supply" (GAIN_FOOD head; declining cancels the whole
    # trade) then "lose 1 food back to the supply" (SPEND_FOOD head). The player
    # must hold a food to give up for the trade to be live; the lose step is
    # unconstrained (a rational agent gives up a *different* food, achieving the
    # swap, but giving up the just-gained food is legal and a harmless no-op).
    st = engine.state
    bird = pb.bird
    if player.total_food() <= 0:
        engine.log(f"  {bird.name}: no food to trade; power skipped")
        return
    gain_choices: list[decisions.FoodChoice | decisions.SkipChoice] = [
        decisions.FoodChoice(label=food.value, food=food)
        for food in cards.ALL_FOODS
        if st.food_supply.get(food, 0) > 0
    ]
    if not gain_choices:
        engine.log(f"  {bird.name}: supply empty; power skipped")
        return
    gain_choices.append(decisions.SkipChoice(label="skip"))
    gain_ch = engine.ask(
        agent,
        decisions.GainFoodDecision(
            player_id=player.id,
            prompt=(
                f"[{player.name}] gain 1 food from the supply to trade "
                f"(or skip) from {bird.name}"
            ),
            choices=gain_choices,
        ),
    )
    if isinstance(gain_ch, decisions.SkipChoice):
        engine.log(f"  {bird.name}: declined to trade")
        return
    gain_food = gain_ch.food
    st.food_supply[gain_food] -= 1
    player.food[gain_food] += 1

    lose_food = _trade_discard_step(engine, agent, player, bird)
    engine.log(
        f"  {bird.name}: gained 1 {gain_food.value}, discarded 1 {lose_food.value}"
    )


def _trade_discard_step(
    engine: "core.Engine",
    agent: "core.Agent",
    player: state.Player,
    bird: cards.Bird,
) -> cards.Food:
    """The 'lose 1 food back to the supply' half of a wild-food trade; returns
    the discarded food."""
    st = engine.state
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
    st.food_supply[lose_food] = st.food_supply.get(lose_food, 0) + 1
    return lose_food


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

    st = engine.state
    bird = pb.bird
    counts = [len(other.board[cards.Habitat.FOREST]) for other in st.players]
    fewest = min(counts)

    # Auto-skip: activating would only benefit the opponent(s) who have fewer
    # forest birds, not the active player — a rational player never does this.
    if len(player.board[cards.Habitat.FOREST]) != fewest:
        engine.log(
            f"  {bird.name}: [{player.name}] has more forest birds than opponent;"
            f" power auto-skipped"
        )
        return

    for other_player, forest_count in zip(st.players, counts):
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
