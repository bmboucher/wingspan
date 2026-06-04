"""Pink-power reactor hooks.

Pink ("once between turns") powers fire when *another* player takes a
specific action. The engine calls into this module after every Lay Eggs
action and after every successful predator hunt.
"""

from __future__ import annotations

import typing

from wingspan import cards, decisions, state

if typing.TYPE_CHECKING:
    from wingspan.engine import core


def trigger_pink_lay_eggs_reactors(
    engine: "core.Engine",
    active_player: state.Player,
) -> None:
    """Called after ``active_player`` completes a Lay Eggs action. Each
    OTHER player's ``PINK_LAY_EGG_ON_NEST`` birds fire in clockwise order
    from ``active_player.id + 1``."""
    st = engine.state
    num_players = len(st.players)
    for offset in range(1, num_players):
        other_player = st.players[(active_player.id + offset) % num_players]
        for habitat, row in other_player.board.items():
            for pb in row:
                if pb.bird.color != cards.PowerColor.PINK:
                    continue
                for eff in pb.bird.power.effects:
                    if eff.kind != cards.EffectKind.PINK_LAY_EGG_ON_NEST:
                        continue
                    fire_pink_lay_egg(engine, other_player, pb, habitat, eff)


def trigger_pink_predator_success(
    engine: "core.Engine",
    hunter_player: state.Player,
) -> None:
    """Called after a ``PREDATOR_HUNT`` succeeds (a card was tucked). Each
    OTHER player's ``PINK_PREDATOR_FEEDER`` birds gain 1 die of the reacting
    player's choice from the birdfeeder (the take entry point offers any
    single-face reset to them first)."""
    # Local import to keep main_actions/reactors decoupled at module load.
    from wingspan.engine import actions

    st = engine.state
    num_players = len(st.players)
    for offset in range(1, num_players):
        other_player = st.players[(hunter_player.id + offset) % num_players]
        for _, row in other_player.board.items():
            for pb in row:
                if pb.bird.color != cards.PowerColor.PINK:
                    continue
                for eff in pb.bird.power.effects:
                    if eff.kind != cards.EffectKind.PINK_PREDATOR_FEEDER:
                        continue
                    gained = actions.take_one_from_feeder(
                        engine,
                        engine.agent_for(other_player),
                        other_player,
                        prompt=(
                            f"[{other_player.name}] pick 1 from birdfeeder "
                            f"for {pb.bird.name}"
                        ),
                    )
                    assert gained is not None  # unrestricted menu, post-reset
                    engine.log(f"  {pb.bird.name}: +1 {gained.value} from birdfeeder")


def fire_pink_lay_egg(
    engine: "core.Engine",
    other_player: state.Player,
    pb: state.PlayedBird,
    habitat: cards.Habitat,
    eff: cards.Effect,
) -> None:
    assert eff.nest is not None
    nest = eff.nest
    eligible: list[decisions.BoardTargetChoice | decisions.SkipChoice] = []
    for habitat, row in other_player.board.items():
        for slot, target in enumerate(row):
            if target is pb:
                continue  # "another bird"
            if target.bird.nest != nest:
                continue
            if target.eggs >= target.bird.egg_limit:
                continue
            eligible.append(
                decisions.BoardTargetChoice(
                    label=(
                        f"{target.bird.name}@{habitat.value}[{slot}]"
                        f"({target.eggs}/{target.bird.egg_limit})"
                    ),
                    habitat=habitat,
                    slot=slot,
                )
            )
    if not eligible:
        engine.log(
            f"  {pb.bird.name} (pink): no other [{nest.value}] bird with room; skipped"
        )
        return
    eligible.append(decisions.SkipChoice(label="skip"))
    ch = engine.ask(
        engine.agent_for(other_player),
        decisions.LayEggDecision(
            player_id=other_player.id,
            prompt=f"[{other_player.name}] lay 1 egg on a [{nest.value}] bird ({pb.bird.name}) (or skip)",
            choices=eligible,
        ),
    )
    if isinstance(ch, decisions.SkipChoice):
        engine.log(f"  {pb.bird.name} (pink): [{other_player.name}] declined")
        return
    other_player.board[ch.habitat][ch.slot].eggs += 1
    engine.log(
        f"  {pb.bird.name} (pink): [{other_player.name}] laid 1 egg on "
        f"{other_player.board[ch.habitat][ch.slot].bird.name}@{ch.habitat.value}[{ch.slot}]"
    )


def trigger_pink_play_bird_reactors(
    engine: "core.Engine",
    active_player: state.Player,
    played_habitat: cards.Habitat,
) -> None:
    """Called after ``active_player`` plays a bird into ``played_habitat``. Each
    OTHER player's pink "when another player plays a bird in their [habitat]"
    power whose habitat matches fires now: gain a food from the supply (Belted
    Kingfisher / Eastern Kingbird) or tuck a card from hand (Horned Lark). Birds
    are scanned clockwise from ``active_player.id + 1``."""
    st = engine.state
    num_players = len(st.players)
    for offset in range(1, num_players):
        other_player = st.players[(active_player.id + offset) % num_players]
        for _, row in other_player.board.items():
            for pb in row:
                if pb.bird.color != cards.PowerColor.PINK:
                    continue
                for eff in pb.bird.power.effects:
                    if eff.habitat != played_habitat:
                        continue
                    if eff.kind == cards.EffectKind.PINK_PLAY_BIRD_GAIN:
                        _react_gain_from_supply(engine, other_player, pb, eff)
                    elif eff.kind == cards.EffectKind.PINK_PLAY_BIRD_TUCK:
                        _react_tuck_from_hand(engine, other_player, pb, eff)


def trigger_pink_gain_food_reactors(
    engine: "core.Engine",
    active_player: state.Player,
    gained_foods: set[cards.Food],
) -> None:
    """Called after ``active_player`` completes a Gain Food action having gained
    the foods in ``gained_foods``. Each OTHER player's pink "when another player
    gains [food]" power (Loggerhead Shrike) caches one of that food from the
    supply when the food was gained."""
    st = engine.state
    num_players = len(st.players)
    for offset in range(1, num_players):
        other_player = st.players[(active_player.id + offset) % num_players]
        for _, row in other_player.board.items():
            for pb in row:
                if pb.bird.color != cards.PowerColor.PINK:
                    continue
                for eff in pb.bird.power.effects:
                    if (
                        eff.kind == cards.EffectKind.PINK_GAIN_FOOD_CACHE
                        and eff.food is not None
                        and eff.food in gained_foods
                    ):
                        _react_cache_from_supply(engine, pb, eff)


def _react_gain_from_supply(
    engine: "core.Engine",
    other_player: state.Player,
    pb: state.PlayedBird,
    eff: cards.Effect,
) -> None:
    assert eff.food is not None
    st = engine.state
    if st.food_supply.get(eff.food, 0) < eff.amount:
        return
    st.food_supply[eff.food] -= eff.amount
    other_player.food[eff.food] += eff.amount
    engine.log(
        f"  {pb.bird.name} (pink): [{other_player.name}] +{eff.amount} "
        f"{eff.food.value} from supply"
    )


def _react_cache_from_supply(
    engine: "core.Engine", pb: state.PlayedBird, eff: cards.Effect
) -> None:
    assert eff.food is not None
    st = engine.state
    if st.food_supply.get(eff.food, 0) < eff.amount:
        return
    st.food_supply[eff.food] -= eff.amount
    pb.cached_food[eff.food] += eff.amount
    engine.log(f"  {pb.bird.name} (pink): cached {eff.amount} {eff.food.value}")


def _react_tuck_from_hand(
    engine: "core.Engine",
    other_player: state.Player,
    pb: state.PlayedBird,
    eff: cards.Effect,
) -> None:
    """The reacting player may tuck ``eff.amount`` card(s) from hand behind ``pb``.
    A gate ask is offered for each card; the player may decline at any point.
    No-op once the hand is empty."""
    trigger_habitat = eff.habitat.value if eff.habitat else "?"
    for _ in range(eff.amount):
        if not other_player.hand:
            return

        # Gate: does the reacting player want to activate the tuck?
        gate_ch = engine.ask(
            engine.agent_for(other_player),
            decisions.ActivateTuckDecision(
                player_id=other_player.id,
                prompt=(
                    f"[{other_player.name}] tuck 1 card behind {pb.bird.name}? "
                    f"(reacting to bird played in [{trigger_habitat}]) (or skip)"
                ),
                choices=[
                    decisions.TuckActivateChoice(label="tuck 1 card", cards_to_tuck=1),
                    decisions.SkipChoice(label="skip"),
                ],
            ),
        )
        if isinstance(gate_ch, decisions.SkipChoice):
            engine.log(f"  {pb.bird.name} (pink): [{other_player.name}] declined")
            return

        # Card selection: mandatory once activated.
        choices = [
            decisions.BirdChoice(label=card.name, bird=card)
            for card in other_player.hand
        ]
        ch = engine.ask(
            engine.agent_for(other_player),
            decisions.BirdPowerTuckFromHandDecision(
                player_id=other_player.id,
                prompt=(
                    f"[{other_player.name}] tuck 1 card behind {pb.bird.name} "
                    f"(reacting to bird played in [{trigger_habitat}])"
                ),
                choices=choices,
            ),
        )
        other_player.hand.remove(ch.bird)
        pb.tucked_cards += 1
        engine.log(
            f"  {pb.bird.name} (pink): [{other_player.name}] tucked {ch.bird.name}"
        )
