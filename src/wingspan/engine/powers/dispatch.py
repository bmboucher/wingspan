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


def offer_activation_veto(
    engine: "core.Engine",
    agent: "core.Agent",
    player: state.Player,
    prompt: str,
    accept_choice: decisions.PayCostChoice,
) -> bool:
    """Offer a SKIP_OPTIONAL AcceptExchangeDecision veto gate.

    Returns ``True`` if the player accepted (proceed), ``False`` if skipped."""
    ch = engine.ask(
        agent,
        decisions.AcceptExchangeDecision(
            player_id=player.id,
            prompt=prompt,
            choices=[accept_choice, decisions.SkipChoice(label="skip")],
        ),
    )
    return not isinstance(ch, decisions.SkipChoice)


def count_opposing_pink_predator_feeders(
    engine: "core.Engine",
    player: state.Player,
) -> int:
    """Count opposing players' not-yet-fired ``PINK_PREDATOR_FEEDER`` birds.

    Matches the reactor-loop predicate exactly so the veto ledger's
    ``opp_gained_food_count`` reflects the real opposing gain."""
    st = engine.state
    n_players = len(st.players)
    return sum(
        1
        for offset in range(1, n_players)
        for _, row in st.players[(player.id + offset) % n_players].board.items()
        for other_pb in row
        if other_pb.bird.color == cards.PowerColor.PINK
        and not other_pb.pink_fired
        and any(
            eff.kind == cards.EffectKind.PINK_PREDATOR_FEEDER
            for eff in other_pb.bird.power.effects
        )
    )


def lay_one_egg_on_nest(
    engine: "core.Engine",
    target_player: state.Player,
    nest: cards.NestType,
    label: str,
    exclude: state.PlayedBird | None = None,
) -> state.PlayedBird | None:
    """Ask ``target_player`` to place 1 egg on a matching-nest bird with room.

    Returns the ``PlayedBird`` that received the egg, or ``None`` if no
    eligible bird existed or the player declined the ``birds_no_eggs`` gate.

    ``exclude`` skips one specific bird (e.g. the one that already received the
    mandatory base egg in the same power activation).

    Outside the ``birds_no_eggs`` goal the lay is forced; when that goal is
    active an ``AcceptExchangeDecision`` gate is offered first so the
    SKIP_OPTIONAL head can decline."""
    eligible: list[decisions.BoardTargetChoice | decisions.SkipChoice] = [
        decisions.BoardTargetChoice(
            label=f"{pb.bird.name}@{habitat.value}[{slot}]({pb.eggs}/{pb.bird.egg_limit})",
            habitat=habitat,
            slot=slot,
        )
        for habitat, row in target_player.board.items()
        for slot, pb in enumerate(row)
        if cards.nest_matches(pb.bird.nest, nest)
        and pb.eggs < pb.bird.egg_limit
        and (exclude is None or pb is not exclude)
    ]
    if not eligible:
        engine.log(
            f"  {label}: [{target_player.name}] has no [{nest.value}] bird with room; skipped"
        )
        return None

    # When the anti-egg-goal is active, gate before the mandatory pick.
    anti_egg_goal = (
        engine.state.round_goals[engine.state.round_idx].category == "birds_no_eggs"
    )
    if anti_egg_goal:
        accepted = offer_activation_veto(
            engine,
            engine.agent_for(target_player),
            target_player,
            f"[{target_player.name}] lay 1 egg on a [{nest.value}] bird ({label})? (or skip)",
            decisions.PayCostChoice(label="lay 1 egg", gained_egg_count=1),
        )
        if not accepted:
            engine.log(f"  {label}: [{target_player.name}] declined to lay egg")
            return None

    ch = engine.ask(
        engine.agent_for(target_player),
        decisions.LayEggDecision(
            player_id=target_player.id,
            prompt=f"[{target_player.name}] lay 1 egg on a [{nest.value}] bird ({label})",
            choices=eligible,
        ),
    )
    if isinstance(ch, decisions.SkipChoice):
        # Unreachable — no skip row in choices; isinstance guard for type narrowing.
        engine.log(f"  {label}: [{target_player.name}] skipped optional extra egg")
        return None
    chosen = target_player.board[ch.habitat][ch.slot]
    chosen.eggs += 1
    engine.log(
        f"  {label}: [{target_player.name}] laid 1 egg on "
        f"{chosen.bird.name}@{ch.habitat.value}[{ch.slot}]"
    )
    return chosen
