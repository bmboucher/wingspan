# pyright: reportUnusedFunction=false
# (every function here is a power handler registered via @registry.handles;
# none is called by name, so pyright's unused-function check is a false positive)
"""Handlers that prompt every player (each-player die draft, all lay egg).

Each handler registers itself with ``@registry.handles`` and is imported by the
``powers`` package ``__init__`` so the dispatch table is populated on load.
"""

from __future__ import annotations

import typing

from wingspan import cards, decisions, state
from wingspan.engine.powers import dispatch, registry

if typing.TYPE_CHECKING:
    from wingspan.engine import core


@registry.handles(cards.EffectKind.EACH_PLAYER_GAINS_DIE_CHOOSE_ORDER)
def _h_each_player_gains_die_choose_order(
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
    n_players = len(st.players)
    start_ch = engine.ask(
        agent,
        decisions.BirdPowerPickStartingPlayerDecision(
            player_id=player.id,
            prompt=f"[{player.name}] pick the starting player for {bird.name}",
            choices=[
                decisions.PlayerIdChoice(
                    label=f"{candidate.name} (P{candidate.id})", player_id=candidate.id
                )
                for candidate in st.players
            ],
        ),
    )
    start_idx = start_ch.player_id
    engine.log(
        f"  {bird.name}: each player gains {eff.amount} [die] from feeder, "
        f"starting with P{start_idx}"
    )
    for offset in range(n_players):
        current_idx = (start_idx + offset) % n_players
        _player_gains_dice(engine, st.players[current_idx], eff.amount, bird)


def _player_gains_dice(
    engine: "core.Engine",
    target_player: state.Player,
    amount: int,
    bird: cards.Bird,
) -> None:
    """``target_player`` takes ``amount`` dice from the birdfeeder one at a time
    (resetting the feeder first if offered), crediting each to their supply."""
    from wingspan.engine import actions

    st = engine.state
    responder = engine.agent_for(target_player)
    for _ in range(amount):
        actions.offer_birdfeeder_reset(engine, responder, target_player)
        food_ch = engine.ask(
            responder,
            decisions.GainFoodDecision(
                player_id=target_player.id,
                prompt=f"[{target_player.name}] take 1 die from birdfeeder ({bird.name})",
                choices=[
                    decisions.FoodChoice(
                        label=st.birdfeeder.gain_option_label(food, combo),
                        food=food,
                        from_choice_die=combo,
                    )
                    for food, combo in st.birdfeeder.gain_options()
                ],
            ),
        )
        assert isinstance(food_ch, decisions.FoodChoice)
        actions.gain_feeder_die(
            engine, target_player, food_ch.food, from_choice_die=food_ch.from_choice_die
        )
        engine.log(f"  [{target_player.name}] +1 {food_ch.food.value} from birdfeeder")


@registry.handles(cards.EffectKind.ALL_PLAYERS_LAY_EGG_ON_NEST)
def _h_all_players_lay_egg_on_nest(
    engine: "core.Engine",
    agent: "core.Agent",
    player: state.Player,
    pb: state.PlayedBird,
    habitat: cards.Habitat,
    eff: cards.Effect,
    trigger: str,
) -> None:
    # "All players lay 1 [egg] on any 1 [<nest>] bird." Wingspan's
    # "All players" effects resolve starting with the active player and
    # proceeding clockwise. The optional second sentence ("You may lay 1
    # [egg] on 1 additional [<nest>] bird.") is encoded as ``eff.amount``
    # extra optional layings the active player gets after every other
    # player has resolved.
    st = engine.state
    bird = pb.bird
    assert eff.nest is not None, "ALL_PLAYERS_LAY_EGG_ON_NEST requires nest"
    nest = eff.nest
    extra_for_self = eff.amount
    engine.log(
        f"  {bird.name}: all players lay 1 egg on a [{nest.value}] bird"
        + (
            f" (active player may lay {extra_for_self} additional)"
            if extra_for_self
            else ""
        )
    )
    n_players = len(st.players)
    active_idx = player.id
    for offset in range(n_players):
        other_player = st.players[(active_idx + offset) % n_players]
        dispatch.lay_one_egg_on_nest(engine, other_player, nest, label=bird.name)
    for _ in range(extra_for_self):
        dispatch.lay_one_egg_on_nest(
            engine, player, nest, label=bird.name, optional=True
        )
