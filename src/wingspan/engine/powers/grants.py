# pyright: reportUnusedFunction=false
# (every function here is a power handler registered via @registry.handles;
# none is called by name, so pyright's unused-function check is a false positive)
"""Direct food / egg / card grant handlers (the simplest effects).

Each handler registers itself with ``@registry.handles`` and is imported by the
``powers`` package ``__init__`` so the dispatch table is populated on load.
"""

from __future__ import annotations

import typing

from wingspan import cards, decisions, state
from wingspan.engine.powers import registry

if typing.TYPE_CHECKING:
    from wingspan.engine import core


@registry.handles(cards.EffectKind.GAIN_FOOD_SUPPLY)
def _h_gain_food_supply(
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
    if eff.food and st.food_supply.get(eff.food, 0) >= eff.amount:
        st.food_supply[eff.food] -= eff.amount
        player.food[eff.food] += eff.amount
        engine.log(f"  {bird.name}: +{eff.amount} {eff.food.value} from supply")


@registry.handles(cards.EffectKind.GAIN_FOOD_BIRDFEEDER)
def _h_gain_food_birdfeeder(
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
    if eff.food:
        actions.offer_birdfeeder_reset(engine, agent, player)
        take = min(eff.amount, st.birdfeeder.gainable_count(eff.food))
        for _ in range(take):
            actions.gain_feeder_die(engine, player, eff.food)
        if take:
            engine.log(f"  {bird.name}: +{take} {eff.food.value} from birdfeeder")


@registry.handles(cards.EffectKind.GAIN_FOOD_FROM_FEEDER_CHOICE)
def _h_gain_food_from_feeder_choice(
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
    food_a, food_b = eff.food_a, eff.food_b
    assert food_a is not None and food_b is not None
    actions.offer_birdfeeder_reset(engine, agent, player)
    gainable = st.birdfeeder.gainable_foods()
    avail = [food for food in (food_a, food_b) if food in gainable]
    if not avail:
        engine.log(
            f"  {bird.name}: neither {food_a.value} nor {food_b.value}"
            f" in birdfeeder; skipped"
        )
        return
    actions.take_one_from_feeder(
        engine,
        agent,
        player,
        pb,
        avail,
        reason="gain_food_from_feeder_choice",
    )


@registry.handles(cards.EffectKind.GAIN_DIE_ANY)
def _h_gain_die_any(
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
    actions.offer_birdfeeder_reset(engine, agent, player)
    avail = st.birdfeeder.gainable_foods()
    if not avail:
        engine.log(f"  {bird.name}: birdfeeder empty; skipped")
        return
    actions.take_one_from_feeder(
        engine, agent, player, pb, avail, reason="gain_die_any"
    )


@registry.handles(cards.EffectKind.LAY_EGG_ON_THIS)
def _h_lay_egg_on_this(
    engine: "core.Engine",
    agent: "core.Agent",
    player: state.Player,
    pb: state.PlayedBird,
    habitat: cards.Habitat,
    eff: cards.Effect,
    trigger: str,
) -> None:
    cap = pb.bird.egg_limit - pb.eggs
    to_lay = min(eff.amount, cap)
    pb.eggs += to_lay
    if to_lay:
        engine.log(f"  {pb.bird.name}: +{to_lay} egg on itself")


@registry.handles(cards.EffectKind.LAY_EGG_ANY)
def _h_lay_egg_any(
    engine: "core.Engine",
    agent: "core.Agent",
    player: state.Player,
    pb: state.PlayedBird,
    habitat: cards.Habitat,
    eff: cards.Effect,
    trigger: str,
) -> None:
    from wingspan.engine import actions

    for _ in range(eff.amount):
        actions.lay_one_egg(engine, agent, player)


@registry.handles(cards.EffectKind.DRAW_CARDS)
def _h_draw_cards(
    engine: "core.Engine",
    agent: "core.Agent",
    player: state.Player,
    pb: state.PlayedBird,
    habitat: cards.Habitat,
    eff: cards.Effect,
    trigger: str,
) -> None:
    from wingspan.engine import actions

    for _ in range(eff.amount):
        actions.draw_one_card(engine, agent, player)


@registry.handles(cards.EffectKind.CACHE_FOOD)
def _h_cache_food(
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
    if eff.food and st.food_supply.get(eff.food, 0) >= eff.amount:
        st.food_supply[eff.food] -= eff.amount
        pb.cached_food[eff.food] += eff.amount
        engine.log(f"  {bird.name}: cached {eff.amount} {eff.food.value}")


@registry.handles(cards.EffectKind.TUCK_FROM_HAND)
def _h_tuck_from_hand(
    engine: "core.Engine",
    agent: "core.Agent",
    player: state.Player,
    pb: state.PlayedBird,
    habitat: cards.Habitat,
    eff: cards.Effect,
    trigger: str,
) -> None:
    bird = pb.bird
    for _ in range(eff.amount):
        if not player.hand:
            break
        choices: list[decisions.BirdChoice | decisions.SkipChoice] = [
            decisions.BirdChoice(label=card.name, bird=card) for card in player.hand
        ]
        choices.append(decisions.SkipChoice(label="skip"))
        ch = engine.ask(
            agent,
            decisions.BirdPowerTuckFromHandDecision(
                player_id=player.id,
                prompt=f"[{player.name}] tuck 1 card behind {bird.name} (or skip)",
                choices=choices,
            ),
        )
        if isinstance(ch, decisions.SkipChoice):
            break
        player.hand.remove(ch.bird)
        pb.tucked_cards += 1
        engine.log(f"  {bird.name}: tucked {ch.bird.name}")


@registry.handles(cards.EffectKind.PLAY_ADDITIONAL_BIRD)
def _h_play_additional_bird(
    engine: "core.Engine",
    agent: "core.Agent",
    player: state.Player,
    pb: state.PlayedBird,
    habitat: cards.Habitat,
    eff: cards.Effect,
    trigger: str,
) -> None:
    bird = pb.bird
    if not eff.habitat or eff.habitat == habitat:
        engine.state.turn_extra_plays += 1
        engine.log(f"  {bird.name}: granted +1 extra play")


@registry.handles(cards.EffectKind.ALL_PLAYERS_GAIN_FOOD)
def _h_all_players_gain_food(
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
    if not eff.food:
        return
    for other_player in st.players:
        if st.food_supply.get(eff.food, 0) >= eff.amount:
            st.food_supply[eff.food] -= eff.amount
            other_player.food[eff.food] += eff.amount
    engine.log(f"  {bird.name}: all players +{eff.amount} {eff.food.value}")


@registry.handles(cards.EffectKind.ALL_PLAYERS_DRAW)
def _h_all_players_draw(
    engine: "core.Engine",
    agent: "core.Agent",
    player: state.Player,
    pb: state.PlayedBird,
    habitat: cards.Habitat,
    eff: cards.Effect,
    trigger: str,
) -> None:
    from wingspan.engine import actions

    for other_player in engine.state.players:
        for _ in range(eff.amount):
            actions.draw_one_card(engine, agent, other_player)


@registry.handles(cards.EffectKind.DRAW_BONUS)
def _h_draw_bonus(
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
    drawn: list[cards.BonusCard] = []
    for _ in range(eff.amount):
        if st.bonus_deck:
            drawn.append(st.bonus_deck.pop())
    player.bonus_cards.extend(drawn)
    engine.log(f"  {bird.name}: drew {len(drawn)} bonus card(s)")
