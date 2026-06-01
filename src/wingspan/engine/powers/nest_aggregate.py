# pyright: reportUnusedFunction=false
# (every function here is a power handler registered via @registry.handles;
# none is called by name, so pyright's unused-function check is a false positive)
"""Nest-targeted egg and aggregate food / tuck handlers.

Each handler registers itself with ``@registry.handles`` and is imported by the
``powers`` package ``__init__`` so the dispatch table is populated on load.
"""

from __future__ import annotations

import typing

from wingspan import cards, decisions, state
from wingspan.engine.powers import registry

if typing.TYPE_CHECKING:
    from wingspan.engine import core


@registry.handles(cards.EffectKind.LAY_EGG_ALL_NEST)
def _h_lay_egg_all_nest(
    engine: "core.Engine",
    agent: "core.Agent",
    player: state.Player,
    pb: state.PlayedBird,
    habitat: cards.Habitat,
    eff: cards.Effect,
    trigger: str,
) -> None:
    bird = pb.bird
    assert eff.nest is not None
    nest = eff.nest
    count = 0
    for row in player.board.values():
        for pb_t in row:
            if pb_t.bird.nest != nest:
                continue
            cap = pb_t.bird.egg_limit - pb_t.eggs
            add = min(eff.amount, cap)
            if add > 0:
                pb_t.eggs += add
                count += add
    engine.log(f"  {bird.name}: laid {count} egg(s) on all [{nest.value}] birds")


@registry.handles(cards.EffectKind.GAIN_ALL_FOOD_FEEDER)
def _h_gain_all_food_feeder(
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
    assert eff.food is not None
    food = eff.food
    actions.offer_birdfeeder_reset(engine, agent, player)
    count = st.birdfeeder.gainable_count(food)
    if count > 0:
        for _ in range(count):
            actions.gain_feeder_die(engine, player, food)
        engine.log(f"  {bird.name}: gained all {count} {food.value} from birdfeeder")
    else:
        engine.log(f"  {bird.name}: no {food.value} in birdfeeder; skipped")


@registry.handles(cards.EffectKind.TUCK_FROM_DECK_PAID)
def _h_tuck_from_deck_paid(
    engine: "core.Engine",
    agent: "core.Agent",
    player: state.Player,
    pb: state.PlayedBird,
    habitat: cards.Habitat,
    eff: cards.Effect,
    trigger: str,
) -> None:
    st = engine.state
    bird = pb.bird
    assert eff.food is not None
    if player.food.get(eff.food, 0) <= 0:
        engine.log(f"  {bird.name}: no {eff.food.value} to spend; power skipped")
        return
    ch = engine.ask(
        agent,
        decisions.AcceptExchangeDecision(
            player_id=player.id,
            prompt=(
                f"[{player.name}] discard 1 {eff.food.value} to tuck {eff.amount} "
                f"cards behind {bird.name}? (or skip)"
            ),
            choices=[
                decisions.PayCostChoice(
                    label=f"pay 1 {eff.food.value}",
                    paid_food=eff.food,
                    gained_tuck_count=eff.amount,
                ),
                decisions.SkipChoice(label="skip"),
            ],
        ),
    )
    if isinstance(ch, decisions.SkipChoice):
        engine.log(f"  {bird.name}: declined to spend {eff.food.value}")
        return
    player.food[eff.food] -= 1
    tucked = 0
    for _ in range(eff.amount):
        drawn_card = st.draw_bird()
        if drawn_card is None:
            break
        tucked += 1  # tucked card leaves the deck for good
    pb.tucked_cards += tucked
    engine.log(
        f"  {bird.name}: paid 1 {eff.food.value}, tucked {tucked} card(s) from deck"
    )
