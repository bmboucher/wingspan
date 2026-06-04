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
    # American Oystercatcher (2-player): draw 3 cards into the active player's
    # hand. Active player (P0) passes 2 to the opponent (P1) via discard
    # decisions, then P1 returns 1. P0 ends with 2 net cards; P1 ends with 1.
    # Uses while-loops so edge cases (fewer cards drawn than expected) resolve
    # without crashing: fewer cards → fewer passes → fewer returns.
    st = engine.state
    bird = pb.bird
    n_players = len(st.players)

    # Skip-optional: power is always optional.
    skip_ch = engine.ask(
        agent,
        decisions.AcceptExchangeDecision(
            player_id=player.id,
            prompt=f"[{player.name}] activate {bird.name}?",
            choices=[
                decisions.PayCostChoice(
                    label="draw 3 cards, pass 2 to opponent, receive 1 back",
                    gained_card_count=2,
                    opp_gained_card_count=1,
                ),
                decisions.SkipChoice(label="skip"),
            ],
        ),
    )
    if isinstance(skip_ch, decisions.SkipChoice):
        engine.log(f"  {bird.name}: skipped")
        return

    # Draw (#players + 1) cards into P0's hand.
    drawn: list[cards.Bird] = []
    for _ in range(n_players + 1):
        drawn_card = st.draw_bird()
        if drawn_card is None:
            break
        drawn.append(drawn_card)
        player.hand.append(drawn_card)

    if not drawn:
        engine.log(f"  {bird.name}: deck empty; power skipped")
        return

    # P0 discards all-but-one of the drawn cards into a pass pile for P1.
    passable = list(drawn)
    pass_pile: list[cards.Bird] = []

    while len(passable) > 1:
        pass_ch = engine.ask(
            agent,
            decisions.BirdPowerDiscardFromHandDecision(
                player_id=player.id,
                prompt=f"[{player.name}] pass a card to opponent ({bird.name})",
                choices=[
                    decisions.BirdChoice(label=candidate.name, bird=candidate)
                    for candidate in passable
                ],
            ),
        )
        player.hand.remove(pass_ch.bird)
        passable.remove(pass_ch.bird)
        pass_pile.append(pass_ch.bird)
        engine.log(f"  {bird.name}: [{player.name}] passes {pass_ch.bird.name}")

    if not pass_pile:
        return

    # P1 receives the passed cards, then returns all-but-one back to P0.
    opponent = st.players[(player.id + 1) % n_players]
    for passed_card in pass_pile:
        opponent.hand.append(passed_card)

    returnable = list(pass_pile)
    cards_to_return: list[cards.Bird] = []

    while len(returnable) > 1:
        return_ch = engine.ask(
            engine.agent_for(opponent),
            decisions.BirdPowerDiscardFromHandDecision(
                player_id=opponent.id,
                prompt=f"[{opponent.name}] return a card to {player.name} ({bird.name})",
                choices=[
                    decisions.BirdChoice(label=candidate.name, bird=candidate)
                    for candidate in returnable
                ],
            ),
        )
        opponent.hand.remove(return_ch.bird)
        returnable.remove(return_ch.bird)
        cards_to_return.append(return_ch.bird)
        engine.log(f"  {bird.name}: [{opponent.name}] returns {return_ch.bird.name}")

    # P0 gains the cards P1 returned.
    for returned_card in cards_to_return:
        player.hand.append(returned_card)
        engine.log(f"  {bird.name}: [{player.name}] receives back {returned_card.name}")


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
