"""Bird power dispatch entry points.

``dispatch_power`` iterates a played bird's parsed ``Power`` effects and
forwards each to ``apply_effect``, which looks the effect kind up in the
handler registry. Pink (between-turn) effects and ``UNIMPLEMENTED`` are no-ops
here; the handlers themselves live in the sibling submodules.
"""

from __future__ import annotations

import typing

from wingspan import cards, decisions, state
from wingspan.engine.powers import registry

if typing.TYPE_CHECKING:
    from wingspan.engine import core


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
    handler = registry.handler_for(eff.kind)
    if handler is None:
        # Pink reactor effects are not dispatched from here — they fire from
        # the engine's reactor hooks after the triggering action.
        if eff.kind in (
            cards.EffectKind.PINK_LAY_EGG_ON_NEST,
            cards.EffectKind.PINK_PREDATOR_FEEDER,
            cards.EffectKind.PINK_PLAY_BIRD_GAIN,
            cards.EffectKind.PINK_PLAY_BIRD_TUCK,
            cards.EffectKind.PINK_GAIN_FOOD_CACHE,
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
    ``nest`` (star nests are wild — ``cards.nest_matches``) and whose
    ``eggs < egg_limit`` and add 1 egg there. No-op if none match. If
    ``optional`` is True, the player may also choose to skip."""
    eligible: list[decisions.BoardTargetChoice | decisions.SkipChoice] = [
        decisions.BoardTargetChoice(
            label=f"{pb.bird.name}@{habitat.value}[{slot}]({pb.eggs}/{pb.bird.egg_limit})",
            habitat=habitat,
            slot=slot,
        )
        for habitat, row in target_player.board.items()
        for slot, pb in enumerate(row)
        if cards.nest_matches(pb.bird.nest, nest) and pb.eggs < pb.bird.egg_limit
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
        decisions.LayEggDecision(
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
