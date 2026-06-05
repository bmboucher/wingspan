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

    bird = pb.bird
    if eff.food:
        take = actions.take_all_of_food(
            engine, agent, player, eff.food, limit=eff.amount
        )
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

    bird = pb.bird
    food_a, food_b = eff.food_a, eff.food_b
    assert food_a is not None and food_b is not None
    gained = actions.take_one_from_feeder(
        engine,
        agent,
        player,
        prompt=f"[{player.name}] pick 1 from birdfeeder for {bird.name}",
        allowed=[food_a, food_b],
    )
    if gained is None:
        engine.log(
            f"  {bird.name}: neither {food_a.value} nor {food_b.value}"
            f" in birdfeeder; skipped"
        )
        return
    engine.log(f"  {bird.name}: +1 {gained.value} from birdfeeder")


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

    bird = pb.bird
    gained = actions.take_one_from_feeder(
        engine,
        agent,
        player,
        prompt=f"[{player.name}] pick 1 from birdfeeder for {bird.name}",
    )
    assert gained is not None  # unrestricted menu, post-reset
    engine.log(f"  {bird.name}: +1 {gained.value} from birdfeeder")


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

    # When the active round goal rewards birds-without-eggs, laying is no longer
    # automatically beneficial — offer an AcceptExchangeDecision before each egg
    # so the SKIP_OPTIONAL head can decide. Outside that goal, mandatory.
    st = engine.state
    anti_egg_goal = st.round_goals[st.round_idx].category == "birds_no_eggs"

    for _ in range(eff.amount):
        if anti_egg_goal:
            commit_ch = engine.ask(
                agent,
                decisions.AcceptExchangeDecision(
                    player_id=player.id,
                    prompt=f"[{player.name}] lay 1 egg on any bird ({pb.bird.name})? (or skip)",
                    choices=[
                        decisions.PayCostChoice(label="lay 1 egg", gained_egg_count=1),
                        decisions.SkipChoice(label="skip"),
                    ],
                ),
            )
            if isinstance(commit_ch, decisions.SkipChoice):
                engine.log(f"  {pb.bird.name}: [{player.name}] skipped optional egg")
                continue
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


@registry.handles(cards.EffectKind.ROLL_NOT_IN_FEEDER_CACHE)
def _h_roll_not_in_feeder_cache(
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
    dice_out = state.BIRDFEEDER_DICE - st.birdfeeder.total()

    if dice_out <= 0:
        engine.log(f"  {bird.name}: no dice outside feeder; skipped")
        return

    # Roll the outside dice using the same 6-face distribution as the feeder.
    roll_counts = state.FoodPool()
    choice_rolled = 0
    for _ in range(dice_out):
        face = st.rng.randint(0, cards.N_FOODS)
        if face < cards.N_FOODS:
            roll_counts[cards.ALL_FOODS[face]] += 1
        else:
            choice_rolled += 1

    # Format the result in the same style as Birdfeeder.format().
    roll_str = roll_counts.format()
    if choice_rolled:
        choice_part = f"{choice_rolled}choice"
        roll_str = choice_part if roll_str == "(empty)" else f"{roll_str}+{choice_part}"
    die_word = "die" if dice_out == 1 else "dice"
    engine.log(f"  {bird.name}: rolled {dice_out} {die_word}: {roll_str}")

    assert eff.food is not None
    if roll_counts[eff.food] > 0 and st.food_supply.get(eff.food, 0) >= eff.amount:
        st.food_supply[eff.food] -= eff.amount
        pb.cached_food[eff.food] += eff.amount
        engine.log(f"  {bird.name}: cached {eff.amount} {eff.food.value}")
    elif roll_counts[eff.food] == 0:
        engine.log(f"  {bird.name}: no {eff.food.value} rolled; nothing cached")


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
            engine.log_skipped_decision(player.id, "no choices")
            break
        gate_ch = engine.ask(
            agent,
            decisions.ActivateTuckDecision(
                player_id=player.id,
                prompt=f"[{player.name}] tuck 1 card behind {bird.name}? (or skip)",
                choices=[
                    decisions.TuckActivateChoice(label="tuck 1 card", cards_to_tuck=1),
                    decisions.SkipChoice(label="skip"),
                ],
            ),
        )
        if isinstance(gate_ch, decisions.SkipChoice):
            break
        choices = [
            decisions.BirdChoice(label=card.name, bird=card) for card in player.hand
        ]
        ch = engine.ask(
            agent,
            decisions.BirdPowerTuckFromHandDecision(
                player_id=player.id,
                prompt=f"[{player.name}] tuck 1 card behind {bird.name}",
                choices=choices,
            ),
        )
        player.hand.remove(ch.bird)
        pb.tucked_cards += 1
        engine.log(f"  {bird.name}: tucked {ch.bird.name}")


@registry.handles(cards.EffectKind.TUCK_FROM_HAND_THEN_DRAW)
def _h_tuck_from_hand_then_draw(
    engine: "core.Engine",
    agent: "core.Agent",
    player: state.Player,
    pb: state.PlayedBird,
    habitat: cards.Habitat,
    eff: cards.Effect,
    trigger: str,
) -> None:
    """Tuck 1 from hand (optional); if accepted, draw 1 card."""
    from wingspan.engine import actions

    bird = pb.bird
    if not player.hand:
        engine.log_skipped_decision(player.id, "no choices")
        return
    gate_ch = engine.ask(
        agent,
        decisions.ActivateTuckDecision(
            player_id=player.id,
            prompt=f"[{player.name}] tuck 1 card behind {bird.name}? (or skip)",
            choices=[
                decisions.TuckActivateChoice(label="tuck 1 card", cards_to_tuck=1),
                decisions.SkipChoice(label="skip"),
            ],
        ),
    )
    if isinstance(gate_ch, decisions.SkipChoice):
        return
    choices = [decisions.BirdChoice(label=card.name, bird=card) for card in player.hand]
    ch = engine.ask(
        agent,
        decisions.BirdPowerTuckFromHandDecision(
            player_id=player.id,
            prompt=f"[{player.name}] tuck 1 card behind {bird.name}",
            choices=choices,
        ),
    )
    player.hand.remove(ch.bird)
    pb.tucked_cards += 1
    engine.log(f"  {bird.name}: tucked {ch.bird.name}")
    for _ in range(eff.amount):
        actions.draw_one_card(engine, agent, player)


@registry.handles(cards.EffectKind.TUCK_FROM_HAND_THEN_LAY_ON_THIS)
def _h_tuck_from_hand_then_lay_on_this(
    engine: "core.Engine",
    agent: "core.Agent",
    player: state.Player,
    pb: state.PlayedBird,
    habitat: cards.Habitat,
    eff: cards.Effect,
    trigger: str,
) -> None:
    """Tuck 1 from hand (optional); if accepted, optionally lay 1 egg on this bird."""
    bird = pb.bird
    if not player.hand:
        engine.log_skipped_decision(player.id, "no choices")
        return
    gate_ch = engine.ask(
        agent,
        decisions.ActivateTuckDecision(
            player_id=player.id,
            prompt=f"[{player.name}] tuck 1 card behind {bird.name}? (or skip)",
            choices=[
                decisions.TuckActivateChoice(label="tuck 1 card", cards_to_tuck=1),
                decisions.SkipChoice(label="skip"),
            ],
        ),
    )
    if isinstance(gate_ch, decisions.SkipChoice):
        return
    choices = [decisions.BirdChoice(label=card.name, bird=card) for card in player.hand]
    ch = engine.ask(
        agent,
        decisions.BirdPowerTuckFromHandDecision(
            player_id=player.id,
            prompt=f"[{player.name}] tuck 1 card behind {bird.name}",
            choices=choices,
        ),
    )
    player.hand.remove(ch.bird)
    pb.tucked_cards += 1
    engine.log(f"  {bird.name}: tucked {ch.bird.name}")

    # Offer the optional lay-on-this-bird.
    cap = bird.egg_limit - pb.eggs
    if cap <= 0:
        engine.log_skipped_decision(player.id, "no choices")
        return
    row = player.board[habitat]
    slot = next(idx for idx, slot_pb in enumerate(row) if slot_pb is pb)
    lay_choices: list[decisions.BoardTargetChoice | decisions.SkipChoice] = [
        decisions.BoardTargetChoice(
            label=f"{bird.name}@{habitat.value}[{slot}]({pb.eggs}/{bird.egg_limit})",
            habitat=habitat,
            slot=slot,
        ),
        decisions.SkipChoice(label="skip"),
    ]
    lay_ch = engine.ask(
        agent,
        decisions.LayEggDecision(
            player_id=player.id,
            prompt=f"[{player.name}] optionally lay 1 egg on {bird.name} (or skip)",
            choices=lay_choices,
        ),
    )
    if isinstance(lay_ch, decisions.SkipChoice):
        return
    pb.eggs += 1
    engine.log(f"  {bird.name}: laid 1 egg on itself")


@registry.handles(cards.EffectKind.TUCK_FROM_HAND_THEN_LAY_ANY)
def _h_tuck_from_hand_then_lay_any(
    engine: "core.Engine",
    agent: "core.Agent",
    player: state.Player,
    pb: state.PlayedBird,
    habitat: cards.Habitat,
    eff: cards.Effect,
    trigger: str,
) -> None:
    """Tuck 1 from hand (optional); if accepted, lay N eggs on any bird(s)."""
    from wingspan.engine import actions

    bird = pb.bird
    if not player.hand:
        engine.log_skipped_decision(player.id, "no choices")
        return
    gate_ch = engine.ask(
        agent,
        decisions.ActivateTuckDecision(
            player_id=player.id,
            prompt=f"[{player.name}] tuck 1 card behind {bird.name}? (or skip)",
            choices=[
                decisions.TuckActivateChoice(label="tuck 1 card", cards_to_tuck=1),
                decisions.SkipChoice(label="skip"),
            ],
        ),
    )
    if isinstance(gate_ch, decisions.SkipChoice):
        return
    choices = [decisions.BirdChoice(label=card.name, bird=card) for card in player.hand]
    ch = engine.ask(
        agent,
        decisions.BirdPowerTuckFromHandDecision(
            player_id=player.id,
            prompt=f"[{player.name}] tuck 1 card behind {bird.name}",
            choices=choices,
        ),
    )
    player.hand.remove(ch.bird)
    pb.tucked_cards += 1
    engine.log(f"  {bird.name}: tucked {ch.bird.name}")
    for _ in range(eff.amount):
        actions.lay_one_egg(engine, agent, player)


@registry.handles(cards.EffectKind.TUCK_FROM_HAND_THEN_GAIN_FOOD_SUPPLY)
def _h_tuck_from_hand_then_gain_food_supply(
    engine: "core.Engine",
    agent: "core.Agent",
    player: state.Player,
    pb: state.PlayedBird,
    habitat: cards.Habitat,
    eff: cards.Effect,
    trigger: str,
) -> None:
    """Tuck 1 from hand (optional); if accepted, gain N [food] from supply."""
    bird = pb.bird
    if not player.hand:
        engine.log_skipped_decision(player.id, "no choices")
        return
    gate_ch = engine.ask(
        agent,
        decisions.ActivateTuckDecision(
            player_id=player.id,
            prompt=f"[{player.name}] tuck 1 card behind {bird.name}? (or skip)",
            choices=[
                decisions.TuckActivateChoice(label="tuck 1 card", cards_to_tuck=1),
                decisions.SkipChoice(label="skip"),
            ],
        ),
    )
    if isinstance(gate_ch, decisions.SkipChoice):
        return
    choices = [decisions.BirdChoice(label=card.name, bird=card) for card in player.hand]
    ch = engine.ask(
        agent,
        decisions.BirdPowerTuckFromHandDecision(
            player_id=player.id,
            prompt=f"[{player.name}] tuck 1 card behind {bird.name}",
            choices=choices,
        ),
    )
    player.hand.remove(ch.bird)
    pb.tucked_cards += 1
    engine.log(f"  {bird.name}: tucked {ch.bird.name}")

    st = engine.state
    if eff.food and st.food_supply.get(eff.food, 0) >= eff.amount:
        st.food_supply[eff.food] -= eff.amount
        player.food[eff.food] += eff.amount
        engine.log(f"  {bird.name}: +{eff.amount} {eff.food.value} from supply")


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
