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
from wingspan.engine.powers import dispatch, registry

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

    # Preview the total eggs that would be laid so the gate ledger is accurate (gap #15).
    potential_count = sum(
        min(eff.amount, pb_t.bird.egg_limit - pb_t.eggs)
        for row in player.board.values()
        for pb_t in row
        if cards.nest_matches(pb_t.bird.nest, nest) and pb_t.eggs < pb_t.bird.egg_limit
    )
    if potential_count == 0:
        engine.log(f"  {bird.name}: no [{nest.value}] bird with room; skipped")
        return

    # When the anti-egg-goal is active, gate before laying (gap #15).
    anti_egg_goal = (
        engine.state.round_goals[engine.state.round_idx].category == "birds_no_eggs"
    )
    if anti_egg_goal:
        accepted = dispatch.offer_activation_veto(
            engine,
            agent,
            player,
            f"[{player.name}] lay eggs on all [{nest.value}] birds ({bird.name})? (or skip)",
            decisions.PayCostChoice(label="lay eggs", gained_egg_count=potential_count),
        )
        if not accepted:
            engine.log(f"  {bird.name}: [{player.name}] skipped")
            return

    count = 0
    for row in player.board.values():
        for pb_t in row:
            if not cards.nest_matches(pb_t.bird.nest, nest):
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

    bird = pb.bird
    assert eff.food is not None
    food = eff.food
    count = actions.take_all_of_food(engine, agent, player, food)
    if count > 0:
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
