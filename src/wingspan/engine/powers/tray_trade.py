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
    # Step 1: SKIP_OPTIONAL activation gate — do you want to trade?
    # Step 2: SPEND_FOOD — which food to discard (mandatory after activation).
    # Step 3: GAIN_FOOD — which food to gain from supply (mandatory; post-discard
    #   supply includes the just-returned food, so trading a food for itself is
    #   legal but wasteful — the model learns to avoid it).
    bird = pb.bird

    # Pre-flight: need a food to give up.
    if player.total_food() <= 0:
        engine.log(f"  {bird.name}: no food to trade; power skipped")
        return

    # Step 1 — activation gate.
    commit_ch = engine.ask(
        agent,
        decisions.AcceptExchangeDecision(
            player_id=player.id,
            prompt=f"[{player.name}] trade 1 food for another from the supply ({bird.name})?",
            choices=[
                decisions.PayCostChoice(
                    label="trade 1 food -> 1 food (supply)",
                    paid_food_count=1,
                    gained_food_count=1,
                ),
                decisions.SkipChoice(label="skip"),
            ],
        ),
    )
    if isinstance(commit_ch, decisions.SkipChoice):
        engine.log(f"  {bird.name}: declined to trade")
        return

    # Step 2 — mandatory discard.
    lose_food = _trade_discard_step(engine, agent, player, bird)

    # Step 3 — mandatory gain from supply.
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
    # American Bittern, Common Loon. Mirrors _h_fewest_forest_gains_die but draws
    # a card instead of taking a die. Three-way logic:
    #   strictly fewer wetland: only active player benefits → mandatory activation
    #   tied fewest: both players draw → rational to activate (mandatory)
    #   strictly more wetland: only opponent benefits → auto-skip
    from wingspan.engine import actions

    st = engine.state
    bird = pb.bird
    counts = [len(other.board[cards.Habitat.WETLAND]) for other in st.players]
    fewest = min(counts)

    # Auto-skip: activating only benefits the opponent, never the active player.
    if len(player.board[cards.Habitat.WETLAND]) != fewest:
        engine.log(
            f"  {bird.name}: [{player.name}] has more wetland birds than opponent;"
            f" power auto-skipped"
        )
        return

    for other_player, wetland_count in zip(st.players, counts):
        if wetland_count != fewest:
            continue
        responder = engine.agent_for(other_player)
        actions.draw_one_card(engine, responder, other_player)
        engine.log(f"  {bird.name}: [{other_player.name}] drew 1 card")
