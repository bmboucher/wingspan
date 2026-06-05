# pyright: reportUnusedFunction=false
# (every function here is a power handler registered via @registry.handles;
# none is called by name, so pyright's unused-function check is a false positive)
"""Predator hunts, rightmost-bird movement, and power-repeat handlers.

Each handler registers itself with ``@registry.handles`` and is imported by the
``powers`` package ``__init__`` so the dispatch table is populated on load.
"""

from __future__ import annotations

import typing

from wingspan import cards, decisions, state
from wingspan.engine import reactors
from wingspan.engine.powers import dispatch, registry

if typing.TYPE_CHECKING:
    from wingspan.engine import core


@registry.handles(cards.EffectKind.PREDATOR_HUNT)
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

    # Veto gate: offered when opponent(s) have PINK_PREDATOR_FEEDER birds that
    # would gain food on a successful hunt (gap #17).
    n_feeders = dispatch.count_opposing_pink_predator_feeders(engine, player)
    if n_feeders > 0:
        accepted = dispatch.offer_activation_veto(
            engine,
            agent,
            player,
            f"[{player.name}] activate {bird.name}? (opponents may gain food on success)",
            decisions.PayCostChoice(
                label="hunt",
                gained_tuck_count=1,
                opp_gained_food_count=n_feeders,
            ),
        )
        if not accepted:
            engine.log(f"  {bird.name}: [{player.name}] declined predator hunt")
            return

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


@registry.handles(cards.EffectKind.MOVE_BIRD_IF_RIGHTMOST)
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

    # Include the current habitat as "stay" so the agent can decline the move.
    all_choices = [
        decisions.HabitatChoice(label=f"stay in {habitat.value}", habitat=habitat),
        *(decisions.HabitatChoice(label=c.value, habitat=c) for c in targets),
    ]
    ch = engine.ask(
        agent,
        decisions.BirdPowerPickHabitatDecision(
            player_id=player.id,
            prompt=f"[{player.name}] move {bird.name} to which habitat? (or stay)",
            choices=all_choices,
            moving_bird=pb,
            from_habitat=habitat,
        ),
    )
    if ch.habitat == habitat:
        engine.log(f"  {bird.name}: chose to stay in [{habitat.value}]")
        return
    row.pop()
    player.board[ch.habitat].append(pb)
    engine.log(f"  {bird.name}: moved from [{habitat.value}] to [{ch.habitat.value}]")


@registry.handles(cards.EffectKind.REPEAT_BROWN_POWER)
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
    # ``ask``'s single-choice guard auto-resolves (and logs) the forced pick
    # when only one repeatable bird is present.
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
        dispatch.apply_effect(
            engine, agent, player, target_pb, habitat, sub, trigger="repeat"
        )


@registry.handles(cards.EffectKind.REPEAT_PREDATOR_POWER)
def _h_repeat_predator_power(
    engine: "core.Engine",
    agent: "core.Agent",
    player: state.Player,
    pb: state.PlayedBird,
    habitat: cards.Habitat,
    eff: cards.Effect,
    trigger: str,
) -> None:
    # Dice-roll predators (ROLL_NOT_IN_FEEDER_CACHE) are repeatable alongside
    # deck-draw predators (PREDATOR_HUNT) — both can succeed and both trigger
    # the pink-predator-success reactor on success.
    _PREDATOR_KINDS = (
        cards.EffectKind.PREDATOR_HUNT,
        cards.EffectKind.ROLL_NOT_IN_FEEDER_CACHE,
    )
    bird = pb.bird
    others = [
        other
        for other in player.board[habitat]
        if other is not pb
        and other.bird.predator
        and any(effect.kind in _PREDATOR_KINDS for effect in other.bird.power.effects)
    ]
    if not others:
        engine.log(f"  {bird.name}: no other predator here to repeat; skipped")
        return
    # ``ask``'s single-choice guard auto-resolves (and logs) the forced pick
    # when only one repeatable predator is present.
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
            dispatch.apply_effect(
                engine, agent, player, target_pb, habitat, sub, trigger="repeat"
            )
