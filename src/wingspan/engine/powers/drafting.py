# pyright: reportUnusedFunction=false
# (every function here is a power handler registered via @registry.handles;
# none is called by name, so pyright's unused-function check is a false positive)
"""Extra-play and card-drafting handlers.

Each handler registers itself with ``@registry.handles`` and is imported by the
``powers`` package ``__init__`` so the dispatch table is populated on load.
"""

from __future__ import annotations

import typing

from wingspan import cards, decisions, state
from wingspan.engine.powers import registry

if typing.TYPE_CHECKING:
    from wingspan.engine import core


@registry.handles(cards.EffectKind.PLAY_ADDITIONAL_BIRD_HERE)
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


@registry.handles(cards.EffectKind.DRAW_N_PLUS_ONE_DRAFT)
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


@registry.handles(cards.EffectKind.DRAW_BONUS_KEEP)
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
