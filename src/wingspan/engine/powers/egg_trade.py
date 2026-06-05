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
    # supply." Two-step: first an AcceptExchangeDecision (is the trade worth
    # it?), then a mandatory RemoveEggDecision (which egg to give up?). The
    # skip lives entirely in the first step; by the time the second step runs
    # the commitment is settled.
    bird = pb.bird
    egg_targets: list[decisions.BoardTargetChoice | decisions.SkipChoice] = [
        decisions.BoardTargetChoice(
            label=f"{pb_other.bird.name}@{egg_habitat.value}[{slot}]",
            habitat=egg_habitat,
            slot=slot,
        )
        for egg_habitat, row in player.board.items()
        for slot, pb_other in enumerate(row)
        if pb_other is not pb and pb_other.eggs > 0
    ]
    if not egg_targets:
        engine.log(f"  {bird.name}: no other bird has an egg; power skipped")
        return

    # Step 1: should I make this trade at all?
    commit_ch = engine.ask(
        agent,
        decisions.AcceptExchangeDecision(
            player_id=player.id,
            prompt=(
                f"[{player.name}] discard 1 egg from another bird to gain "
                f"{eff.amount} [wild]? (or skip)"
            ),
            choices=[
                decisions.PayCostChoice(
                    label=f"discard 1 egg for {eff.amount} [wild]",
                    paid_egg_count=1,
                    gained_food_count=eff.amount,
                ),
                decisions.SkipChoice(label="skip"),
            ],
        ),
    )
    if isinstance(commit_ch, decisions.SkipChoice):
        engine.log(f"  {bird.name}: declined to discard an egg")
        return

    # Step 2: which egg to give up? (mandatory — commitment settled above)
    egg_ch = engine.ask(
        agent,
        decisions.RemoveEggDecision(
            player_id=player.id,
            prompt=f"[{player.name}] discard an egg from another bird ({bird.name})",
            choices=egg_targets,
        ),
    )
    assert isinstance(egg_ch, decisions.BoardTargetChoice)
    source = player.board[egg_ch.habitat][egg_ch.slot]
    source.eggs -= 1
    engine.log(f"  {bird.name}: discarded 1 egg from {source.bird.name}")
    _gain_wild_from_supply(engine, agent, player, bird, eff.amount)


@registry.handles(cards.EffectKind.DISCARD_EGG_FOR_CARDS)
def _h_discard_egg_for_cards(
    engine: "core.Engine",
    agent: "core.Agent",
    player: state.Player,
    pb: state.PlayedBird,
    habitat: cards.Habitat,
    eff: cards.Effect,
    trigger: str,
) -> None:
    # "Discard 1 [egg] to draw N [card]." — Franklin's Gull, Killdeer.
    # Step 1: AcceptExchangeDecision gate (skip if no eggs on board).
    # Step 2: mandatory RemoveEggDecision (which egg to give up).
    # Step 3: draw N cards.
    from wingspan.engine import actions

    bird = pb.bird
    egg_targets: list[decisions.BoardTargetChoice | decisions.SkipChoice] = [
        decisions.BoardTargetChoice(
            label=f"{pb_other.bird.name}@{egg_hab.value}[{slot}]",
            habitat=egg_hab,
            slot=slot,
        )
        for egg_hab, row in player.board.items()
        for slot, pb_other in enumerate(row)
        if pb_other.eggs > 0
    ]
    if not egg_targets:
        engine.log(f"  {bird.name}: no eggs on board; power skipped")
        return

    # Step 1: is the trade worth it?
    commit_ch = engine.ask(
        agent,
        decisions.AcceptExchangeDecision(
            player_id=player.id,
            prompt=(
                f"[{player.name}] discard 1 egg to draw {eff.amount} card(s)"
                f" ({bird.name})? (or skip)"
            ),
            choices=[
                decisions.PayCostChoice(
                    label=f"discard 1 egg for {eff.amount} card(s)",
                    paid_egg_count=1,
                    gained_card_count=eff.amount,
                ),
                decisions.SkipChoice(label="skip"),
            ],
        ),
    )
    if isinstance(commit_ch, decisions.SkipChoice):
        engine.log(f"  {bird.name}: declined to discard an egg")
        return

    # Step 2: which egg to discard?
    egg_ch = engine.ask(
        agent,
        decisions.RemoveEggDecision(
            player_id=player.id,
            prompt=f"[{player.name}] discard 1 egg ({bird.name})",
            choices=egg_targets,
        ),
    )
    assert isinstance(egg_ch, decisions.BoardTargetChoice)
    source = player.board[egg_ch.habitat][egg_ch.slot]
    source.eggs -= 1
    engine.log(f"  {bird.name}: discarded 1 egg from {source.bird.name}")

    # Step 3: draw the cards.
    for _ in range(eff.amount):
        actions.draw_one_card(engine, agent, player)


def _gain_wild_from_supply(
    engine: "core.Engine",
    agent: "core.Agent",
    player: state.Player,
    bird: cards.Bird,
    amount: int,
) -> None:
    """Let ``player`` pick ``amount`` wild foods one at a time from the infinite
    supply, crediting each to their tray."""
    for _ in range(amount):
        food_ch = engine.ask(
            agent,
            decisions.GainFoodDecision(
                player_id=player.id,
                prompt=f"[{player.name}] pick 1 [wild] from supply (from {bird.name})",
                choices=[
                    decisions.FoodChoice(label=food.value, food=food)
                    for food in cards.ALL_FOODS
                ],
            ),
        )
        assert isinstance(food_ch, decisions.FoodChoice)
        chosen_food = food_ch.food
        player.food[chosen_food] += 1
        engine.log(f"  {bird.name}: +1 {chosen_food.value} from supply")
