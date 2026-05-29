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
    OTHER player's ``PINK_PREDATOR_FEEDER`` birds gain 1 die from the
    birdfeeder."""
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
                    avail = [
                        food
                        for food, count in st.birdfeeder.counts.items()
                        if count > 0
                    ]
                    if not avail:
                        engine.log(
                            f"  {pb.bird.name} (pink): birdfeeder empty; skipped"
                        )
                        continue
                    actions.take_one_from_feeder(
                        engine,
                        engine.agent_for(other_player),
                        other_player,
                        pb,
                        avail,
                        reason="pink_predator_feeder",
                    )


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
        decisions.LayEggPickBirdDecision(
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
