# pyright: reportUnusedFunction=false
# (every function here is a power handler registered via @registry.handles;
# none is called by name, so pyright's unused-function check is a false positive)
"""The discard-an-egg-for-a-wild-food trade handler.

Each handler registers itself with ``@registry.handles`` and is imported by the
``powers`` package ``__init__`` so the dispatch table is populated on load.
"""

from __future__ import annotations

import typing

from wingspan import cards, decisions, state
from wingspan.engine.powers import registry

if typing.TYPE_CHECKING:
    from wingspan.engine import core


@registry.handles(cards.EffectKind.DISCARD_EGG_FOR_WILD)
def _h_discard_egg_for_wild(
    engine: "core.Engine",
    agent: "core.Agent",
    player: state.Player,
    pb: state.PlayedBird,
    habitat: cards.Habitat,
    eff: cards.Effect,
    trigger: str,
) -> None:
    # "Discard 1 [egg] from any of your other birds to gain N [wild] from the
    # supply." Optional: skip if there are no eligible eggs or the player
    # would rather not spend one.
    bird = pb.bird
    egg_choices: list[decisions.BoardTargetChoice | decisions.SkipChoice] = [
        decisions.BoardTargetChoice(
            label=f"{pb_other.bird.name}@{egg_habitat.value}[{slot}]",
            habitat=egg_habitat,
            slot=slot,
        )
        for egg_habitat, row in player.board.items()
        for slot, pb_other in enumerate(row)
        if pb_other is not pb and pb_other.eggs > 0
    ]
    if not egg_choices:
        engine.log(f"  {bird.name}: no other bird has an egg; power skipped")
        return
    egg_choices.append(decisions.SkipChoice(label="skip"))
    ch = engine.ask(
        agent,
        decisions.RemoveEggDecision(
            player_id=player.id,
            prompt=(
                f"[{player.name}] discard an egg from another bird to gain "
                f"{eff.amount} [wild] (or skip)"
            ),
            choices=egg_choices,
        ),
    )
    if isinstance(ch, decisions.SkipChoice):
        engine.log(f"  {bird.name}: declined to discard an egg")
        return
    source = player.board[ch.habitat][ch.slot]
    source.eggs -= 1
    engine.log(f"  {bird.name}: discarded 1 egg from {source.bird.name}")
    _gain_wild_from_supply(engine, agent, player, bird, eff.amount)


def _gain_wild_from_supply(
    engine: "core.Engine",
    agent: "core.Agent",
    player: state.Player,
    bird: cards.Bird,
    amount: int,
) -> None:
    """Let ``player`` pick ``amount`` wild foods one at a time from the supply
    (stopping early if the supply empties), crediting each to their tray."""
    st = engine.state
    for _ in range(amount):
        available = [
            food for food in cards.ALL_FOODS if st.food_supply.get(food, 0) > 0
        ]
        if not available:
            break
        food_ch = engine.ask(
            agent,
            decisions.GainFoodDecision(
                player_id=player.id,
                prompt=f"[{player.name}] pick 1 [wild] from supply (from {bird.name})",
                choices=[
                    decisions.FoodChoice(label=food.value, food=food)
                    for food in available
                ],
            ),
        )
        assert isinstance(food_ch, decisions.FoodChoice)
        chosen_food = food_ch.food
        st.food_supply[chosen_food] -= 1
        player.food[chosen_food] += 1
        engine.log(f"  {bird.name}: +1 {chosen_food.value} from supply")
