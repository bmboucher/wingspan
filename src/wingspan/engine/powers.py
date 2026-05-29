"""Bird power dispatch.

``dispatch_power`` iterates a played bird's parsed ``Power`` effects and
forwards each to ``apply_effect``. ``apply_effect`` is one big switch over
``EffectKind`` — each branch is short, so the module reads as a catalog of
what every effect type does to game state.
"""

from __future__ import annotations

import typing

from wingspan import cards, decisions, state
from wingspan.engine import reactors

if typing.TYPE_CHECKING:
    from wingspan.engine import core


_EffectHandler = typing.Callable[
    [
        "core.Engine",
        "core.Agent",
        state.Player,
        state.PlayedBird,
        cards.Habitat,
        cards.Effect,
        str,
    ],
    None,
]


def dispatch_power(
    engine: "core.Engine",
    agent: "core.Agent",
    player: state.Player,
    pb: state.PlayedBird,
    habitat: cards.Habitat,
    trigger: str,
) -> None:
    """Iterate every parsed effect on ``pb`` and apply each."""
    for eff in pb.bird.power.effects:
        apply_effect(engine, agent, player, pb, habitat, eff, trigger)


def apply_effect(
    engine: "core.Engine",
    agent: "core.Agent",
    player: state.Player,
    pb: state.PlayedBird,
    habitat: cards.Habitat,
    eff: cards.Effect,
    trigger: str,
) -> None:
    """Apply a single ``Effect`` to the game state. Effects that don't match
    any known pattern (``UNIMPLEMENTED``) and pink reactor effects fire from
    elsewhere; both are no-ops here."""
    handler = _HANDLERS.get(eff.kind)
    if handler is None:
        # Pink reactor effects are not dispatched from here — they fire from
        # the engine's reactor hooks after the triggering action.
        if eff.kind in (
            cards.EffectKind.PINK_LAY_EGG_ON_NEST,
            cards.EffectKind.PINK_PREDATOR_FEEDER,
        ):
            return
        if eff.kind == cards.EffectKind.UNIMPLEMENTED:
            engine.log(
                f"  (power on {pb.bird.name} not modeled: "
                f"{pb.bird.raw_power_text!r}; skipped)"
            )
        return
    handler(engine, agent, player, pb, habitat, eff, trigger)


def lay_one_egg_on_nest(
    engine: "core.Engine",
    target_player: state.Player,
    nest: cards.NestType,
    label: str,
    optional: bool = False,
) -> None:
    """Ask ``target_player`` to pick one of their birds whose nest matches
    ``nest`` and whose ``eggs < egg_limit`` and add 1 egg there. No-op if none
    match. If ``optional`` is True, the player may also choose to skip."""
    eligible: list[decisions.BoardTargetChoice | decisions.SkipChoice] = [
        decisions.BoardTargetChoice(
            label=f"{pb.bird.name}@{habitat.value}[{slot}]({pb.eggs}/{pb.bird.egg_limit})",
            habitat=habitat,
            slot=slot,
        )
        for habitat, row in target_player.board.items()
        for slot, pb in enumerate(row)
        if pb.bird.nest == nest and pb.eggs < pb.bird.egg_limit
    ]
    if not eligible:
        engine.log(
            f"  {label}: [{target_player.name}] has no [{nest.value}] bird with room; skipped"
        )
        return
    if optional:
        eligible.append(decisions.SkipChoice(label="skip"))
    prompt = f"[{target_player.name}] lay 1 egg on a [{nest.value}] bird ({label})" + (
        " (or skip)" if optional else ""
    )
    ch = engine.ask(
        engine.agent_for(target_player),
        decisions.LayEggPickBirdDecision(
            player_id=target_player.id,
            prompt=prompt,
            choices=eligible,
        ),
    )
    if isinstance(ch, decisions.SkipChoice):
        engine.log(f"  {label}: [{target_player.name}] skipped optional extra egg")
        return
    target_player.board[ch.habitat][ch.slot].eggs += 1
    engine.log(
        f"  {label}: [{target_player.name}] laid 1 egg on "
        f"{target_player.board[ch.habitat][ch.slot].bird.name}@{ch.habitat.value}[{ch.slot}]"
    )


###### PRIVATE #######

#### Simple food/egg/card grants ####


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


def _h_gain_food_birdfeeder(
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
    if eff.food and st.birdfeeder.counts.get(eff.food, 0) > 0:
        take = min(eff.amount, st.birdfeeder.counts[eff.food])
        st.birdfeeder.counts[eff.food] -= take
        player.food[eff.food] += take
        engine.log(f"  {bird.name}: +{take} {eff.food.value} from birdfeeder")


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
    avail = [food for food in (food_a, food_b) if st.birdfeeder.counts.get(food, 0) > 0]
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
    avail = [food for food, count in st.birdfeeder.counts.items() if count > 0]
    if not avail:
        engine.log(f"  {bird.name}: birdfeeder empty; skipped")
        return
    actions.take_one_from_feeder(
        engine, agent, player, pb, avail, reason="gain_die_any"
    )


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
        pb.cached_food += eff.amount
        engine.log(f"  {bird.name}: cached {eff.amount} {eff.food.value}")


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


#### Egg-for-wild trade ####


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
    st = engine.state
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
        decisions.PlayBirdPickEggToPayDecision(
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
    for _ in range(eff.amount):
        available = [
            food for food in cards.ALL_FOODS if st.food_supply.get(food, 0) > 0
        ]
        if not available:
            break
        food_ch = engine.ask(
            agent,
            decisions.BirdPowerPickFoodDecision(
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


#### Each-player and all-players multi-actor effects ####


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
    stop_outer = False
    for offset in range(n_players):
        if stop_outer:
            break
        current_idx = (start_idx + offset) % n_players
        current_player = st.players[current_idx]
        responder = engine.agent_for(current_player)
        for _ in range(eff.amount):
            avail = [
                (food, count)
                for food, count in st.birdfeeder.counts.items()
                if count > 0
            ]
            if not avail:
                if st.birdfeeder.total() > 0:
                    st.birdfeeder.reroll(st.rng)
                    engine.log(
                        f"  {bird.name}: birdfeeder rerolled to "
                        f"{st.birdfeeder.counts.format()}"
                    )
                    avail = [
                        (food, count)
                        for food, count in st.birdfeeder.counts.items()
                        if count > 0
                    ]
                if not avail:
                    engine.log(f"  {bird.name}: birdfeeder empty; stopping power early")
                    stop_outer = True
                    break
            food_ch = engine.ask(
                responder,
                decisions.GainFoodPickDieDecision(
                    player_id=current_player.id,
                    prompt=f"[{current_player.name}] take 1 die from birdfeeder ({bird.name})",
                    choices=[
                        decisions.FoodChoice(label=f"{food.value}({count})", food=food)
                        for food, count in avail
                    ],
                ),
            )
            chosen_food = food_ch.food
            st.birdfeeder.counts[chosen_food] -= 1
            current_player.food[chosen_food] += 1
            engine.log(
                f"  [{current_player.name}] +1 {chosen_food.value} from birdfeeder"
            )


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
        lay_one_egg_on_nest(engine, other_player, nest, label=bird.name)
    for _ in range(extra_for_self):
        lay_one_egg_on_nest(engine, player, nest, label=bird.name, optional=True)


#### Tray, trade, fewest-birds effects ####


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
    taken = list(st.tray)
    st.tray.clear()
    player.hand.extend(taken)
    st.refill_tray()
    engine.log(
        f"  {bird.name}: drew {len(taken)} card(s) from tray: "
        f"{[card.name for card in taken]}"
    )


def _h_trade_wild_food(
    engine: "core.Engine",
    agent: "core.Agent",
    player: state.Player,
    pb: state.PlayedBird,
    habitat: cards.Habitat,
    eff: cards.Effect,
    trigger: str,
) -> None:
    # Green Heron: trade 1 food back to supply for any other food type.
    st = engine.state
    bird = pb.bird
    if player.total_food() <= 0:
        engine.log(f"  {bird.name}: no food to trade; power skipped")
        return
    food_choices: list[
        decisions.FoodChoice | decisions.SkipChoice | decisions.PayCostChoice
    ] = [
        decisions.FoodChoice(label=food.value, food=food)
        for food in cards.ALL_FOODS
        if player.food.get(food, 0) > 0
    ]
    food_choices.append(decisions.SkipChoice(label="skip"))
    ch = engine.ask(
        agent,
        decisions.BirdPowerPickFoodDecision(
            player_id=player.id,
            prompt=f"[{player.name}] discard 1 food to trade (or skip) from {bird.name}",
            choices=food_choices,
        ),
    )
    if isinstance(ch, decisions.SkipChoice):
        engine.log(f"  {bird.name}: declined to trade")
        return
    assert isinstance(ch, decisions.FoodChoice)
    discard_food = ch.food
    gain_choices: list[
        decisions.FoodChoice | decisions.SkipChoice | decisions.PayCostChoice
    ] = [
        decisions.FoodChoice(label=food.value, food=food)
        for food in cards.ALL_FOODS
        if food != discard_food and st.food_supply.get(food, 0) > 0
    ]
    if not gain_choices:
        engine.log(f"  {bird.name}: no other food type available in supply; skipped")
        return
    player.food[discard_food] -= 1
    st.food_supply[discard_food] = st.food_supply.get(discard_food, 0) + 1
    ch = engine.ask(
        agent,
        decisions.BirdPowerPickFoodDecision(
            player_id=player.id,
            prompt=f"[{player.name}] pick a different food from supply (from {bird.name})",
            choices=gain_choices,
        ),
    )
    assert isinstance(ch, decisions.FoodChoice)
    gain_food = ch.food
    st.food_supply[gain_food] -= 1
    player.food[gain_food] += 1
    engine.log(f"  {bird.name}: traded 1 {discard_food.value} -> 1 {gain_food.value}")


def _h_fewest_forest_gains_die(
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
    if st.birdfeeder.total() <= 0:
        engine.log(f"  {bird.name}: birdfeeder empty; power skipped")
        return
    counts = [len(other.board[cards.Habitat.FOREST]) for other in st.players]
    fewest = min(counts)
    for other_player, forest_count in zip(st.players, counts):
        if forest_count != fewest:
            continue
        avail = [
            (food, count) for food, count in st.birdfeeder.counts.items() if count > 0
        ]
        if not avail:
            break
        ch = engine.ask(
            engine.agent_for(other_player),
            decisions.BirdPowerPickFoodDecision(
                player_id=other_player.id,
                prompt=f"[{other_player.name}] take 1 die from birdfeeder (from {bird.name})",
                choices=[
                    decisions.FoodChoice(label=f"{food.value}({count})", food=food)
                    for food, count in avail
                ],
            ),
        )
        assert isinstance(ch, decisions.FoodChoice)
        chosen_food = ch.food
        st.birdfeeder.counts[chosen_food] -= 1
        other_player.food[chosen_food] += 1
        engine.log(
            f"  {bird.name}: [{other_player.name}] +1 {chosen_food.value} from birdfeeder"
        )


#### Additional play / drafting effects ####


def _h_play_additional_bird_here(
    engine: "core.Engine",
    agent: "core.Agent",
    player: state.Player,
    pb: state.PlayedBird,
    habitat: cards.Habitat,
    eff: cards.Effect,
    trigger: str,
) -> None:
    # House Wren: grants +1 extra play in this bird's habitat. The habitat
    # restriction is tracked on ``state.turn_extra_play_habitat`` and enforced
    # when offering legal cards in the extra-plays loop.
    bird = pb.bird
    engine.state.turn_extra_plays += 1
    engine.state.turn_extra_play_habitat = habitat
    engine.log(
        f"  {bird.name}: granted +1 extra play (restricted to [{habitat.value}])"
    )


def _h_draw_n_plus_one_draft(
    engine: "core.Engine",
    agent: "core.Agent",
    player: state.Player,
    pb: state.PlayedBird,
    habitat: cards.Habitat,
    eff: cards.Effect,
    trigger: str,
) -> None:
    # American Oystercatcher: draw (#players+1) cards. Each non-active player
    # (clockwise from active+1) picks one card; the active player keeps what
    # remains. Works for any N >= 2.
    st = engine.state
    bird = pb.bird
    n_players = len(st.players)
    n_draw = n_players + 1
    drawn: list[cards.Bird] = []
    for _ in range(n_draw):
        drawn_card = st.draw_bird()
        if drawn_card is None:
            break
        drawn.append(drawn_card)
    if not drawn:
        engine.log(f"  {bird.name}: deck empty; power skipped")
        return
    for offset in range(1, n_players):
        if not drawn:
            break
        picker = st.players[(player.id + offset) % n_players]
        ch = engine.ask(
            engine.agent_for(picker),
            decisions.BirdPowerPickBirdFromHandDecision(
                player_id=picker.id,
                prompt=f"[{picker.name}] pick a card to keep (from {bird.name})",
                choices=[
                    decisions.BirdChoice(label=candidate.name, bird=candidate)
                    for candidate in drawn
                ],
            ),
        )
        kept_card = ch.bird
        drawn.remove(kept_card)
        picker.hand.append(kept_card)
        engine.log(f"  {bird.name}: [{picker.name}] kept {kept_card.name}")
    for leftover in drawn:
        player.hand.append(leftover)
        engine.log(f"  {bird.name}: [{player.name}] keeps leftover {leftover.name}")


def _h_draw_bonus_keep(
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
    n_draw = max(eff.amount, 1)
    keep = max(eff.keep_count or 1, 1)
    drawn: list[cards.BonusCard] = []
    for _ in range(n_draw):
        if not st.bonus_deck:
            break
        drawn.append(st.bonus_deck.pop())
    if not drawn:
        engine.log(f"  {bird.name}: bonus deck empty; power skipped")
        return
    keep = min(keep, len(drawn))
    for _ in range(keep):
        ch = engine.ask(
            agent,
            decisions.BirdPowerPickBonusCardDecision(
                player_id=player.id,
                prompt=f"[{player.name}] keep a bonus card (from {bird.name})",
                choices=[
                    decisions.BonusCardChoice(label=card.name, bonus_card=card)
                    for card in drawn
                ],
            ),
        )
        kept = ch.bonus_card
        drawn.remove(kept)
        player.bonus_cards.append(kept)
        engine.log(f"  {bird.name}: kept bonus '{kept.name}'")
    for leftover in drawn:
        st.bonus_discard.append(leftover)


#### Nest-targeted and aggregate effects ####


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


def _h_gain_all_food_feeder(
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
    food = eff.food
    count = st.birdfeeder.counts.get(food, 0)
    if count > 0:
        st.birdfeeder.counts[food] = 0
        player.food[food] += count
        engine.log(f"  {bird.name}: gained all {count} {food.value} from birdfeeder")
    else:
        engine.log(f"  {bird.name}: no {food.value} in birdfeeder; skipped")


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
        decisions.BirdPowerPickFoodDecision(
            player_id=player.id,
            prompt=(
                f"[{player.name}] discard 1 {eff.food.value} to tuck {eff.amount} "
                f"cards behind {bird.name}? (or skip)"
            ),
            choices=[
                decisions.PayCostChoice(label=f"pay 1 {eff.food.value}"),
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


#### Predator / movement / repeat effects ####


def _h_predator_hunt(
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
    cap = eff.max_wingspan_cm
    assert cap is not None
    prey = st.draw_bird()
    if prey is None:
        engine.log(f"  {bird.name}: deck empty; predator hunt skipped")
        return
    if prey.wingspan_cm and prey.wingspan_cm < cap:
        pb.tucked_cards += 1
        engine.log(
            f"  {bird.name}: hunted {prey.name} ({prey.wingspan_cm}cm < {cap}cm) — tucked"
        )
        reactors.trigger_pink_predator_success(engine, player)
    else:
        st.bird_discard.append(prey)
        engine.log(
            f"  {bird.name}: hunt missed ({prey.name}, {prey.wingspan_cm}cm) — discarded"
        )


def _h_move_bird_if_rightmost(
    engine: "core.Engine",
    agent: "core.Agent",
    player: state.Player,
    pb: state.PlayedBird,
    habitat: cards.Habitat,
    eff: cards.Effect,
    trigger: str,
) -> None:
    bird = pb.bird
    row = player.board[habitat]
    if not row or row[-1] is not pb:
        engine.log(f"  {bird.name}: not rightmost in [{habitat.value}]; power skipped")
        return
    targets = [
        candidate
        for candidate in cards.ALL_HABITATS
        if candidate != habitat and player.can_play_in(candidate)
    ]
    if not targets:
        engine.log(f"  {bird.name}: no other habitat with space; power skipped")
        return
    if len(targets) == 1:
        target = targets[0]
    else:
        ch = engine.ask(
            agent,
            decisions.BirdPowerPickHabitatDecision(
                player_id=player.id,
                prompt=f"[{player.name}] move {bird.name} to which habitat?",
                choices=[
                    decisions.HabitatChoice(label=candidate.value, habitat=candidate)
                    for candidate in targets
                ],
            ),
        )
        target = ch.habitat
    row.pop()
    player.board[target].append(pb)
    engine.log(f"  {bird.name}: moved from [{habitat.value}] to [{target.value}]")


def _h_repeat_brown_power(
    engine: "core.Engine",
    agent: "core.Agent",
    player: state.Player,
    pb: state.PlayedBird,
    habitat: cards.Habitat,
    eff: cards.Effect,
    trigger: str,
) -> None:
    bird = pb.bird
    others = [
        other
        for other in player.board[habitat]
        if other is not pb
        and other.bird.color == cards.PowerColor.BROWN
        and any(
            effect.kind
            not in (
                cards.EffectKind.UNIMPLEMENTED,
                cards.EffectKind.REPEAT_BROWN_POWER,
                cards.EffectKind.REPEAT_PREDATOR_POWER,
            )
            for effect in other.bird.power.effects
        )
    ]
    if not others:
        engine.log(f"  {bird.name}: no other brown bird here to repeat; skipped")
        return
    if len(others) == 1:
        target_pb = others[0]
    else:
        ch = engine.ask(
            agent,
            decisions.BirdPowerPickPlayedBirdDecision(
                player_id=player.id,
                prompt=f"[{player.name}] repeat which bird's brown power?",
                choices=[
                    decisions.PlayedBirdChoice(label=other.bird.name, played_bird=other)
                    for other in others
                ],
            ),
        )
        target_pb = ch.played_bird
    engine.log(f"  {bird.name}: repeats {target_pb.bird.name}'s power")
    for sub in target_pb.bird.power.effects:
        if sub.kind in (
            cards.EffectKind.REPEAT_BROWN_POWER,
            cards.EffectKind.REPEAT_PREDATOR_POWER,
        ):
            continue
        apply_effect(engine, agent, player, target_pb, habitat, sub, trigger="repeat")


def _h_repeat_predator_power(
    engine: "core.Engine",
    agent: "core.Agent",
    player: state.Player,
    pb: state.PlayedBird,
    habitat: cards.Habitat,
    eff: cards.Effect,
    trigger: str,
) -> None:
    bird = pb.bird
    others = [
        other
        for other in player.board[habitat]
        if other is not pb
        and other.bird.predator
        and any(
            effect.kind == cards.EffectKind.PREDATOR_HUNT
            for effect in other.bird.power.effects
        )
    ]
    if not others:
        engine.log(f"  {bird.name}: no other predator here to repeat; skipped")
        return
    if len(others) == 1:
        target_pb = others[0]
    else:
        ch = engine.ask(
            agent,
            decisions.BirdPowerPickPlayedBirdDecision(
                player_id=player.id,
                prompt=f"[{player.name}] repeat which predator's power?",
                choices=[
                    decisions.PlayedBirdChoice(label=other.bird.name, played_bird=other)
                    for other in others
                ],
            ),
        )
        target_pb = ch.played_bird
    engine.log(f"  {bird.name}: repeats {target_pb.bird.name}'s predator power")
    for sub in target_pb.bird.power.effects:
        if sub.kind == cards.EffectKind.PREDATOR_HUNT:
            apply_effect(
                engine, agent, player, target_pb, habitat, sub, trigger="repeat"
            )


#### Dispatch table ####

_HANDLERS: dict[cards.EffectKind, _EffectHandler] = {
    cards.EffectKind.GAIN_FOOD_SUPPLY: _h_gain_food_supply,
    cards.EffectKind.GAIN_FOOD_BIRDFEEDER: _h_gain_food_birdfeeder,
    cards.EffectKind.GAIN_FOOD_FROM_FEEDER_CHOICE: _h_gain_food_from_feeder_choice,
    cards.EffectKind.GAIN_DIE_ANY: _h_gain_die_any,
    cards.EffectKind.LAY_EGG_ON_THIS: _h_lay_egg_on_this,
    cards.EffectKind.LAY_EGG_ANY: _h_lay_egg_any,
    cards.EffectKind.DRAW_CARDS: _h_draw_cards,
    cards.EffectKind.CACHE_FOOD: _h_cache_food,
    cards.EffectKind.TUCK_FROM_HAND: _h_tuck_from_hand,
    cards.EffectKind.PLAY_ADDITIONAL_BIRD: _h_play_additional_bird,
    cards.EffectKind.ALL_PLAYERS_GAIN_FOOD: _h_all_players_gain_food,
    cards.EffectKind.ALL_PLAYERS_DRAW: _h_all_players_draw,
    cards.EffectKind.DRAW_BONUS: _h_draw_bonus,
    cards.EffectKind.DISCARD_EGG_FOR_WILD: _h_discard_egg_for_wild,
    cards.EffectKind.EACH_PLAYER_GAINS_DIE_CHOOSE_ORDER: _h_each_player_gains_die_choose_order,
    cards.EffectKind.ALL_PLAYERS_LAY_EGG_ON_NEST: _h_all_players_lay_egg_on_nest,
    cards.EffectKind.DRAW_FROM_TRAY_ALL: _h_draw_from_tray_all,
    cards.EffectKind.TRADE_WILD_FOOD: _h_trade_wild_food,
    cards.EffectKind.FEWEST_FOREST_GAINS_DIE: _h_fewest_forest_gains_die,
    cards.EffectKind.PLAY_ADDITIONAL_BIRD_HERE: _h_play_additional_bird_here,
    cards.EffectKind.DRAW_N_PLUS_ONE_DRAFT: _h_draw_n_plus_one_draft,
    cards.EffectKind.DRAW_BONUS_KEEP: _h_draw_bonus_keep,
    cards.EffectKind.LAY_EGG_ALL_NEST: _h_lay_egg_all_nest,
    cards.EffectKind.GAIN_ALL_FOOD_FEEDER: _h_gain_all_food_feeder,
    cards.EffectKind.TUCK_FROM_DECK_PAID: _h_tuck_from_deck_paid,
    cards.EffectKind.PREDATOR_HUNT: _h_predator_hunt,
    cards.EffectKind.MOVE_BIRD_IF_RIGHTMOST: _h_move_bird_if_rightmost,
    cards.EffectKind.REPEAT_BROWN_POWER: _h_repeat_brown_power,
    cards.EffectKind.REPEAT_PREDATOR_POWER: _h_repeat_predator_power,
}
