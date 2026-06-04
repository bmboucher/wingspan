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
        decisions.BirdPowerPickGainOrderDecision(
            player_id=player.id,
            prompt=f"[{player.name}] pick who gains food first for {bird.name}",
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
    (the take entry point offers any single-face reset before each die),
    crediting each to their supply."""
    from wingspan.engine import actions

    responder = engine.agent_for(target_player)
    for _ in range(amount):
        gained = actions.take_one_from_feeder(
            engine,
            responder,
            target_player,
            prompt=f"[{target_player.name}] take 1 die from birdfeeder ({bird.name})",
        )
        assert gained is not None  # unrestricted menu, post-reset
        engine.log(f"  [{target_player.name}] +1 {gained.value} from birdfeeder")


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
    # "All players lay 1 [egg] on any 1 [<nest>] bird." The optional second
    # sentence ("You may lay 1 [egg] on 1 additional [<nest>] bird.") is
    # encoded as ``eff.amount`` extra optional layings the active player gets
    # after the main round.
    #
    # Sequence:
    #   1. P0 AcceptExchangeDecision — veto the whole power (exchange ledger
    #      shows what P0 gains and what eligible opponents gain).
    #   2. Non-active players, in turn order: if the active round goal rewards
    #      birds-without-eggs, each gets their own AcceptExchangeDecision;
    #      otherwise auto-yes and straight to LayEggDecision.
    #   3. P0's mandatory base LayEggDecision.
    #   4. P0's optional extra layings (existing ``eff.amount`` logic).
    st = engine.state
    bird = pb.bird
    assert eff.nest is not None, "ALL_PLAYERS_LAY_EGG_ON_NEST requires nest"
    nest = eff.nest
    extra_for_self = eff.amount
    n_players = len(st.players)
    active_idx = player.id

    # Eligibility check — who has a matching bird with room?
    p0_eligible = _has_eligible_bird_on_nest(player, nest)
    opp_eligible_count = sum(
        1
        for offset in range(1, n_players)
        if _has_eligible_bird_on_nest(
            st.players[(active_idx + offset) % n_players], nest
        )
    )
    if not p0_eligible and opp_eligible_count == 0:
        engine.log(
            f"  {bird.name}: no player has a [{nest.value}] bird with room; skipped"
        )
        return

    engine.log(
        f"  {bird.name}: all players lay 1 egg on a [{nest.value}] bird"
        + (
            f" (active player may lay {extra_for_self} additional)"
            if extra_for_self
            else ""
        )
    )

    # Step 1: P0 veto — they commit to activating or cancel for everyone.
    p0_commit = engine.ask(
        agent,
        decisions.AcceptExchangeDecision(
            player_id=player.id,
            prompt=f"[{player.name}] activate {bird.name}? (all players lay on [{nest.value}])",
            choices=[
                decisions.PayCostChoice(
                    label="activate",
                    gained_egg_count=1 if p0_eligible else 0,
                    opp_gained_egg_count=opp_eligible_count,
                ),
                decisions.SkipChoice(label="skip"),
            ],
        ),
    )
    if isinstance(p0_commit, decisions.SkipChoice):
        engine.log(f"  {bird.name}: [{player.name}] skipped activation")
        return

    # Step 2: non-active players, in turn order from P0+1.
    anti_egg_goal = st.round_goals[st.round_idx].category == "birds_no_eggs"
    for offset in range(1, n_players):
        other_player = st.players[(active_idx + offset) % n_players]
        if not _has_eligible_bird_on_nest(other_player, nest):
            engine.log(
                f"  {bird.name}: [{other_player.name}] has no [{nest.value}] bird with room; skipped"
            )
            continue
        if anti_egg_goal:
            opp_responder = engine.agent_for(other_player)
            opp_ch = engine.ask(
                opp_responder,
                decisions.AcceptExchangeDecision(
                    player_id=other_player.id,
                    prompt=(
                        f"[{other_player.name}] lay 1 egg on a [{nest.value}] bird? "
                        f"(or skip) ({bird.name})"
                    ),
                    choices=[
                        decisions.PayCostChoice(label="accept", gained_egg_count=1),
                        decisions.SkipChoice(label="skip"),
                    ],
                ),
            )
            if isinstance(opp_ch, decisions.SkipChoice):
                engine.log(f"  {bird.name}: [{other_player.name}] skipped optional egg")
                continue
        dispatch.lay_one_egg_on_nest(engine, other_player, nest, label=bird.name)

    # Step 3: P0's mandatory base egg.
    dispatch.lay_one_egg_on_nest(engine, player, nest, label=bird.name)

    # Step 4: P0's optional extra eggs.
    for _ in range(extra_for_self):
        dispatch.lay_one_egg_on_nest(
            engine, player, nest, label=bird.name, optional=True
        )


def _has_eligible_bird_on_nest(player: state.Player, nest: cards.NestType) -> bool:
    """Whether ``player`` has at least one bird of ``nest`` type with room for an egg."""
    return any(
        pb.bird.nest == nest and pb.eggs < pb.bird.egg_limit
        for row in player.board.values()
        for pb in row
    )
