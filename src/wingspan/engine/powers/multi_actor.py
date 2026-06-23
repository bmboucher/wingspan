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

    # Veto gate: all players gain a die, so offered when an opponent benefits (gap #16).
    n_opponents = n_players - 1
    if n_opponents > 0:
        accepted = dispatch.offer_activation_veto(
            engine,
            agent,
            player,
            f"[{player.name}] activate {bird.name}? (each player gains {eff.amount} die)",
            decisions.PayCostChoice(
                label="activate",
                gained_food_count=eff.amount,
                opp_gained_food_count=n_opponents * eff.amount,
            ),
        )
        if not accepted:
            engine.log(f"  {bird.name}: [{player.name}] skipped activation")
            return

    # Going first is strictly best unless the feeder shows exactly two faces.
    # With >2 faces neither seat can reset, so maximize our own selection by
    # going first; with <=1 face either seat could reset, so grab first pick.
    # Only the exactly-two-faces case is a genuine situational tradeoff worth
    # asking the model: going first guarantees one visible face, going second
    # leaves the reset available to us. Offering a single candidate in the other
    # cases lets Engine.ask auto-resolve it (no model call, no recorded step).
    starters = st.players if st.birdfeeder.distinct_faces() == 2 else [player]
    start_ch = engine.ask(
        agent,
        decisions.BirdPowerPickGainOrderDecision(
            player_id=player.id,
            prompt=f"[{player.name}] pick who gains food first for {bird.name}",
            choices=[
                decisions.PlayerIdChoice(
                    label=f"{candidate.name} (P{candidate.id})", player_id=candidate.id
                )
                for candidate in starters
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
    # encoded as ``eff.amount`` extra layings for the active player after the
    # main round.
    #
    # Sequence:
    #   1. P0 AcceptExchangeDecision — veto the whole power (ledger shows
    #      min(2, own_eligible) for P0 and eligible-opponent count).
    #   2. Non-active players in turn order — anti-egg-goal gate each one.
    #   3. P0's mandatory base LayEggDecision.
    #   4. P0's extra layings — mandatory unless anti-egg-goal is active;
    #      each excludes the previous egg's target (gap #13a/b/c).
    st = engine.state
    bird = pb.bird
    assert eff.nest is not None, "ALL_PLAYERS_LAY_EGG_ON_NEST requires nest"
    nest = eff.nest
    extra_for_self = eff.amount
    n_players = len(st.players)
    active_idx = player.id

    # Eligibility check — who has a matching bird with room? (gap #14: star nests
    # are wild and must be counted via cards.nest_matches, not ==.)
    own_eligible_count = sum(
        1
        for row in player.board.values()
        for target_pb in row
        if cards.nest_matches(target_pb.bird.nest, nest)
        and target_pb.eggs < target_pb.bird.egg_limit
    )
    opp_eligible_count = sum(
        1
        for offset in range(1, n_players)
        if _has_eligible_bird_on_nest(
            st.players[(active_idx + offset) % n_players], nest
        )
    )
    if own_eligible_count == 0 and opp_eligible_count == 0:
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

    # Step 1: P0 veto — commit to activating or cancel for everyone.
    p0_commit = engine.ask(
        agent,
        decisions.AcceptExchangeDecision(
            player_id=player.id,
            prompt=f"[{player.name}] activate {bird.name}? (all players lay on [{nest.value}])",
            choices=[
                decisions.PayCostChoice(
                    label="activate",
                    gained_egg_count=min(2, own_eligible_count),
                    opp_gained_egg_count=opp_eligible_count,
                ),
                decisions.SkipChoice(label="skip"),
            ],
        ),
    )
    if isinstance(p0_commit, decisions.SkipChoice):
        engine.log(f"  {bird.name}: [{player.name}] skipped activation")
        return

    anti_egg_goal = st.round_goals[st.round_idx].category == "birds_no_eggs"

    # Step 2: non-active players, in turn order from P0+1.
    _other_players_lay_optional(
        engine, player, nest, bird, anti_egg_goal, active_idx, n_players
    )

    # Step 3: P0's mandatory base egg.
    last_laid = dispatch.lay_one_egg_on_nest(engine, player, nest, label=bird.name)

    # Step 4: P0's extra eggs — each excludes the previous target (gap #13a).
    _active_player_extra_eggs(
        engine, agent, player, nest, bird, extra_for_self, anti_egg_goal, last_laid
    )


def _other_players_lay_optional(
    engine: "core.Engine",
    player: state.Player,
    nest: cards.NestType,
    bird: cards.Bird,
    anti_egg_goal: bool,
    active_idx: int,
    n_players: int,
) -> None:
    """Prompt each non-active player (in turn order) to lay one egg on a matching nest bird.

    When ``anti_egg_goal`` is active each player may skip; otherwise the lay is
    mandatory for eligible players."""
    st = engine.state
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


def _active_player_extra_eggs(
    engine: "core.Engine",
    agent: "core.Agent",
    player: state.Player,
    nest: cards.NestType,
    bird: cards.Bird,
    extra_for_self: int,
    anti_egg_goal: bool,
    last_laid: state.PlayedBird | None,
) -> None:
    """Lay the active player's extra eggs beyond the mandatory base egg.

    Each extra lay excludes the bird that received the previous egg so the same
    bird is not targeted twice in a row (gap #13a). Stops early if anti-egg-goal
    causes a skip, or if no eligible target remains."""
    for _ in range(extra_for_self):
        if anti_egg_goal:
            extra_ch = engine.ask(
                agent,
                decisions.AcceptExchangeDecision(
                    player_id=player.id,
                    prompt=(
                        f"[{player.name}] lay 1 more egg on a [{nest.value}] bird"
                        f" ({bird.name})? (or skip)"
                    ),
                    choices=[
                        decisions.PayCostChoice(
                            label="lay extra egg", gained_egg_count=1
                        ),
                        decisions.SkipChoice(label="skip"),
                    ],
                ),
            )
            if isinstance(extra_ch, decisions.SkipChoice):
                engine.log(f"  {bird.name}: [{player.name}] skipped extra egg")
                break
        last_laid = dispatch.lay_one_egg_on_nest(
            engine, player, nest, label=bird.name, exclude=last_laid
        )
        if last_laid is None:
            break


def _has_eligible_bird_on_nest(player: state.Player, nest: cards.NestType) -> bool:
    """Whether ``player`` has at least one bird matching ``nest`` with room for an egg.

    Uses ``cards.nest_matches`` so star-nest birds are counted as wild (gap #14)."""
    return any(
        cards.nest_matches(pb.bird.nest, nest) and pb.eggs < pb.bird.egg_limit
        for row in player.board.values()
        for pb in row
    )
