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
from wingspan.engine import ledger
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
    # House Wren: grants +1 extra play in this bird's habitat. Appended as a
    # single credit to the FIFO list; ``actions.consume_extra_plays`` pops and
    # enforces the habitat restriction.
    bird = pb.bird
    engine.state.turn_extra_plays.append(habitat)
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
    # American Oystercatcher: draw (#players + 1) cards into the active
    # player's hand, then draft them clockwise so the active player ends
    # with exactly 2 net cards and every opponent ends with exactly 1. At 2
    # players this is the familiar pass-and-return: the active player passes
    # 2 of the 3 drawn cards to the lone opponent, who returns 1 of them.
    # Uses while-loops throughout so edge cases (fewer cards drawn than
    # expected) resolve without crashing: fewer cards → fewer passes/asks.
    st = engine.state
    bird = pb.bird
    n_players = len(st.players)

    # The printed 2-player ledger text is preserved verbatim; n>=3 uses
    # generalized wording (the mechanic itself is identical in shape, just
    # drafted around more seats).
    veto_label = (
        "draw 3 cards, pass 2 to opponent, receive 1 back"
        if n_players == 2
        else f"draw {n_players + 1} cards, draft clockwise (you keep 2, each opponent 1)"
    )

    # Skip-optional: power is always optional.
    skip_ch = engine.ask(
        agent,
        decisions.AcceptExchangeDecision(
            player_id=player.id,
            prompt=f"[{player.name}] activate {bird.name}?",
            choices=[
                decisions.PayCostChoice(
                    label=veto_label,
                    gained_card_count=2,
                    opp_gained_card_count=n_players - 1,
                ),
                decisions.SkipChoice(label="skip"),
            ],
        ),
    )
    if isinstance(skip_ch, decisions.SkipChoice):
        engine.log(f"  {bird.name}: skipped")
        return

    # Draw (#players + 1) cards into the active player's hand. The draft pool
    # is revealed to the whole table in the physical game, so every drawn
    # card is face-up from the start.
    drawn: list[cards.Bird] = []
    for _ in range(n_players + 1):
        drawn_card = ledger.draw_from_deck(engine, player, face_up=True)
        if drawn_card is None:
            break
        drawn.append(drawn_card)

    if not drawn:
        engine.log(f"  {bird.name}: deck empty; power skipped")
        return

    pass_pile = _draft_pass_loop(engine, agent, player, bird, drawn)
    if not pass_pile:
        return

    _draft_return_loop(engine, player, bird, pass_pile, n_players)


def _draft_pass_loop(
    engine: "core.Engine",
    agent: "core.Agent",
    player: state.Player,
    bird: cards.Bird,
    drawn: list[cards.Bird],
) -> list[cards.Bird]:
    """The active player discards all-but-one drawn card into a pass pile
    for the clockwise draft that follows.

    Returns the cards passed (empty when nothing was passed, which signals the
    caller to skip the return loop)."""
    passable = list(drawn)
    pass_pile: list[cards.Bird] = []
    # The pile goes to the next seat clockwise, which _draft_return_loop hands
    # it to first. Naming the recipient here lets each card's departure and
    # arrival share one ledger record even though the pile is a transient.
    n_players = len(engine.state.players)
    recipient = engine.state.players[(player.id + 1) % n_players]
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
        ledger.take_into_pile(engine, player, recipient, pass_ch.bird)
        passable.remove(pass_ch.bird)
        pass_pile.append(pass_ch.bird)
        engine.log(f"  {bird.name}: [{player.name}] passes {pass_ch.bird.name}")
    return pass_pile


def _draft_return_loop(
    engine: "core.Engine",
    player: state.Player,
    bird: cards.Bird,
    pass_pile: list[cards.Bird],
    n_players: int,
) -> None:
    """Draft ``pass_pile`` clockwise from ``player``: each opponent in turn
    receives the pile into hand and passes all-but-one card onward (asking
    *their own* agent), keeping exactly one; whatever remains after the last
    opponent (at most one card) returns to ``player``.

    At 2 players this is ask-for-ask and card-for-card identical to the
    original pass-and-return: the lone opponent receives both cards, is
    asked once, and returns the untaken card. Short-deck edges degrade
    gracefully: a 1-card pile is kept by its holder without an ask, and an
    empty pile ends the draft early for every seat still downstream."""
    st = engine.state
    pile = list(pass_pile)
    for offset in range(1, n_players):
        if not pile:
            return
        opponent = st.players[(player.id + offset) % n_players]
        # Arrival only: each card's transfer was recorded when it left the
        # previous holder's hand (ledger.take_into_pile), which named this seat
        # as the recipient. The hand + known-hand mutation on arrival still
        # goes through the ledger (receive_passed_cards), it just records no
        # additional effect.
        ledger.receive_passed_cards(engine, opponent, pile)

        # Whoever holds the pile next: the following seat clockwise, or the
        # original player once the last opponent has been asked.
        next_holder = (
            player
            if offset == n_players - 1
            else st.players[(player.id + offset + 1) % n_players]
        )
        returnable = list(pile)
        next_pile: list[cards.Bird] = []
        while len(returnable) > 1:
            prompt = (
                f"[{opponent.name}] return a card to {player.name} ({bird.name})"
                if n_players == 2
                else f"[{opponent.name}] pass a card onward ({bird.name})"
            )
            pass_ch = engine.ask(
                engine.agent_for(opponent),
                decisions.BirdPowerDiscardFromHandDecision(
                    player_id=opponent.id,
                    prompt=prompt,
                    choices=[
                        decisions.BirdChoice(label=candidate.name, bird=candidate)
                        for candidate in returnable
                    ],
                ),
            )
            ledger.take_into_pile(engine, opponent, next_holder, pass_ch.bird)
            returnable.remove(pass_ch.bird)
            next_pile.append(pass_ch.bird)
            if n_players == 2:
                engine.log(
                    f"  {bird.name}: [{opponent.name}] returns {pass_ch.bird.name}"
                )
            else:
                engine.log(
                    f"  {bird.name}: [{opponent.name}] passes {pass_ch.bird.name} onward"
                )
        pile = next_pile

    # Arrival only, same as the mid-draft receive above: each card's departure
    # was already recorded by ledger.take_into_pile.
    ledger.receive_passed_cards(engine, player, pile)
    for returned_card in pile:
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
