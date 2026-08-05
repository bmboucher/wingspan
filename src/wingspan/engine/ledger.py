"""The mutate-and-record seam: every state change the game log accounts for.

Each function here performs one state mutation **and** records the matching
:class:`~wingspan.gamelog.models.Effect` in the same call, so the two can never
drift.  This is the by-construction alternative to asking every call site to
remember a separate ``record_effect`` line next to its mutation — a convention
that would silently rot the first time a new bird power forgot it.

The engine, its action modules, the power handlers and the pink reactors all
route their mutations through here.  ``tests/test_gamelog_ledger.py`` proves the
seam is complete by replaying the recorded ledger and reconciling it against the
final :class:`~wingspan.state.GameState`: any mutation that bypasses this module
shows up there as a mismatch.

Attribution: ``player`` is always the *owner* of the changed resource, not
necessarily the acting seat — a pink reactor mutates the reacting opponent's
board, and the effect is attributed to that opponent so per-seat reconciliation
adds up.

Every function takes the live ``Engine`` first, matching the free-function
convention of the sibling engine modules.
"""

from __future__ import annotations

import typing

from wingspan import cards, state
from wingspan.gamelog import models

if typing.TYPE_CHECKING:
    from wingspan.engine import core

# Returned by _locate_bird for a bird that is not on its owner's board — only
# reachable for a bird being placed, whose slot is recorded by place_bird itself.
_UNPLACED_SLOT = -1


# ---------------------------------------------------------------------------
# Food


def gain_food(
    engine: core.Engine,
    player: state.Player,
    food: cards.Food,
    amount: int = 1,
    *,
    source: models.FoodSource,
) -> None:
    """Credit ``amount`` of ``food`` to ``player``'s supply."""
    player.food[food] += amount
    engine.events.record_effect(
        models.GainFoodEffect(
            player_id=player.id, food=food.value, amount=amount, source=source
        )
    )


def spend_food(
    engine: core.Engine,
    player: state.Player,
    food: cards.Food,
    amount: int = 1,
    *,
    purpose: models.EffectPurpose,
) -> None:
    """Debit ``amount`` of ``food`` from ``player``'s supply."""
    player.food[food] -= amount
    engine.events.record_effect(
        models.SpendFoodEffect(
            player_id=player.id, food=food.value, amount=amount, purpose=purpose
        )
    )


def take_feeder_die(
    engine: core.Engine,
    player: state.Player,
    food: cards.Food,
    *,
    from_choice_die: bool = False,
) -> None:
    """Move one ``food`` die out of the birdfeeder into ``player``'s supply.

    The feeder side of the move is not itself an effect — the birdfeeder is
    public information whose contents are visible in every state snapshot, so
    only the reroll (which reveals hidden randomness) is recorded."""
    engine.state.birdfeeder.take(food, from_choice_die=from_choice_die)
    gain_food(engine, player, food, source=models.FoodSource.FEEDER)


def cache_food(
    engine: core.Engine,
    player: state.Player,
    played_bird: state.PlayedBird,
    food: cards.Food,
    amount: int = 1,
    *,
    from_supply: bool = False,
) -> None:
    """Place ``amount`` of ``food`` on ``played_bird``'s cache.

    ``from_supply=True`` means the food came out of ``player``'s own pool; the
    caller records that debit with :func:`spend_food` so both halves of the move
    reconcile independently."""
    played_bird.cached_food[food] += amount
    engine.events.record_effect(
        models.CacheFoodEffect(
            player_id=player.id,
            bird=played_bird.bird.name,
            food=food.value,
            amount=amount,
            from_supply=from_supply,
        )
    )


def uncache_food(
    engine: core.Engine,
    player: state.Player,
    played_bird: state.PlayedBird,
    food: cards.Food,
    amount: int = 1,
) -> None:
    """Remove ``amount`` of ``food`` from ``played_bird``'s cache."""
    played_bird.cached_food[food] -= amount
    engine.events.record_effect(
        models.UncacheFoodEffect(
            player_id=player.id,
            bird=played_bird.bird.name,
            food=food.value,
            amount=amount,
        )
    )


# ---------------------------------------------------------------------------
# Eggs


def lay_eggs(
    engine: core.Engine,
    player: state.Player,
    played_bird: state.PlayedBird,
    count: int = 1,
    *,
    purpose: models.EffectPurpose = models.EffectPurpose.ACTION,
) -> None:
    """Add ``count`` eggs to ``played_bird``."""
    played_bird.eggs += count
    habitat, slot = _locate_bird(player, played_bird)
    engine.events.record_effect(
        models.LayEggEffect(
            player_id=player.id,
            bird=played_bird.bird.name,
            habitat=habitat,
            slot=slot,
            count=count,
            purpose=purpose,
        )
    )


def remove_eggs(
    engine: core.Engine,
    player: state.Player,
    played_bird: state.PlayedBird,
    count: int = 1,
    *,
    purpose: models.EffectPurpose = models.EffectPurpose.ACTION,
) -> None:
    """Remove ``count`` eggs from ``played_bird``."""
    played_bird.eggs -= count
    habitat, slot = _locate_bird(player, played_bird)
    engine.events.record_effect(
        models.RemoveEggEffect(
            player_id=player.id,
            bird=played_bird.bird.name,
            habitat=habitat,
            slot=slot,
            count=count,
            purpose=purpose,
        )
    )


# ---------------------------------------------------------------------------
# Cards


def draw_from_deck(
    engine: core.Engine,
    player: state.Player,
    *,
    source: models.CardSource = models.CardSource.DECK,
) -> cards.Bird | None:
    """Draw the top deck card into ``player``'s hand — a reveal.

    ``None`` when the deck and discard are both exhausted."""
    drawn = engine.state.draw_bird()
    if drawn is None:
        return None
    player.hand.append(drawn)
    engine.events.record_effect(
        models.DrawCardEffect(player_id=player.id, card=drawn.name, source=source)
    )
    return drawn


def take_from_tray(
    engine: core.Engine, player: state.Player, tray_index: int
) -> cards.Bird | None:
    """Move the face-up tray card at ``tray_index`` into ``player``'s hand.

    The slot is left empty; refilling is the caller's (deferred) business."""
    drawn = engine.state.tray[tray_index]
    if drawn is None:
        return None
    engine.state.tray[tray_index] = None
    player.hand.append(drawn)
    engine.events.record_effect(
        models.DrawCardEffect(
            player_id=player.id,
            card=drawn.name,
            source=models.CardSource.TRAY,
            tray_slot=tray_index,
        )
    )
    return drawn


def discard_from_hand(
    engine: core.Engine,
    player: state.Player,
    card: cards.Bird,
    *,
    purpose: models.EffectPurpose = models.EffectPurpose.POWER,
) -> None:
    """Move ``card`` from ``player``'s hand to the bird discard pile."""
    player.hand.remove(card)
    engine.state.bird_discard.append(card)
    engine.events.record_effect(
        models.DiscardCardEffect(player_id=player.id, card=card.name, purpose=purpose)
    )


def reveal_from_deck(engine: core.Engine) -> cards.Bird | None:
    """Pop the top deck card without giving it to anyone.

    The dice-predator "draw the top card and check its cost" powers turn it face
    up before deciding whether it is tucked or discarded; the caller follows up
    with :func:`tuck_revealed` or :func:`discard_revealed`, either of which
    records the reveal."""
    return engine.state.draw_bird()


def tuck_revealed(
    engine: core.Engine,
    player: state.Player,
    played_bird: state.PlayedBird,
    card: cards.Bird,
) -> None:
    """Tuck an already-revealed deck card behind ``played_bird``."""
    played_bird.tucked_cards += 1
    engine.events.record_effect(
        models.TuckCardEffect(
            player_id=player.id,
            card=card.name,
            bird=played_bird.bird.name,
            source=models.CardSource.DECK,
        )
    )


def discard_revealed(
    engine: core.Engine,
    player: state.Player,
    card: cards.Bird,
    *,
    purpose: models.EffectPurpose = models.EffectPurpose.POWER,
) -> None:
    """Send an already-revealed deck card to the discard pile."""
    engine.state.bird_discard.append(card)
    engine.events.record_effect(
        models.DiscardCardEffect(player_id=player.id, card=card.name, purpose=purpose)
    )


def tuck_from_hand(
    engine: core.Engine,
    player: state.Player,
    card: cards.Bird,
    played_bird: state.PlayedBird,
) -> None:
    """Tuck ``card`` out of ``player``'s hand behind ``played_bird``."""
    player.hand.remove(card)
    played_bird.tucked_cards += 1
    engine.events.record_effect(
        models.TuckCardEffect(
            player_id=player.id,
            card=card.name,
            bird=played_bird.bird.name,
            source=models.CardSource.HAND,
        )
    )


def tuck_from_deck(
    engine: core.Engine, player: state.Player, played_bird: state.PlayedBird
) -> cards.Bird | None:
    """Tuck the top deck card behind ``played_bird`` — a reveal.

    ``None`` when the deck and discard are both exhausted."""
    drawn = reveal_from_deck(engine)
    if drawn is None:
        return None
    tuck_revealed(engine, player, played_bird, drawn)
    return drawn


def pass_card(
    engine: core.Engine,
    from_player: state.Player,
    to_player: state.Player,
    card: cards.Bird,
) -> None:
    """Move ``card`` directly from one player's hand to another's."""
    from_player.hand.remove(card)
    to_player.hand.append(card)
    engine.events.record_effect(
        models.PassCardEffect(
            player_id=from_player.id, card=card.name, to_player_id=to_player.id
        )
    )


def take_into_pile(
    engine: core.Engine,
    from_player: state.Player,
    to_player: state.Player,
    card: cards.Bird,
) -> None:
    """Remove ``card`` from ``from_player``'s hand, bound for ``to_player``.

    The card-drafting powers move a whole *pile* of cards at once, so a card
    leaves one hand into a transient list and joins the next hand a few lines
    later.  Recording the completed transfer at departure — naming the seat the
    pile is bound for — keeps one ledger record per transfer, so both hands
    reconcile even though the card is briefly in neither.  The receiving side
    is a bare ``hand.extend`` at the call site, deliberately unrecorded."""
    from_player.hand.remove(card)
    engine.events.record_effect(
        models.PassCardEffect(
            player_id=from_player.id, card=card.name, to_player_id=to_player.id
        )
    )


def place_bird(
    engine: core.Engine,
    player: state.Player,
    card: cards.Bird,
    habitat: cards.Habitat,
) -> state.PlayedBird:
    """Move ``card`` from ``player``'s hand onto the board and return its slot.

    The bird is appended to the habitat row, so its slot is the row's previous
    length."""
    player.hand.remove(card)
    played_bird = state.PlayedBird(bird=card)
    row = player.board[habitat]
    slot = len(row)
    row.append(played_bird)
    engine.events.record_effect(
        models.PlayBirdEffect(
            player_id=player.id, card=card.name, habitat=habitat.value, slot=slot
        )
    )
    return played_bird


def move_bird(
    engine: core.Engine,
    player: state.Player,
    played_bird: state.PlayedBird,
    from_habitat: cards.Habitat,
    to_habitat: cards.Habitat,
) -> None:
    """Relocate a bird already in play to another habitat row.

    Only the *rightmost* bird in a row can move (the printed restriction), so
    the departure is a plain pop; the bird keeps its eggs, tucked cards and
    cache because the same object is re-appended."""
    from_row = player.board[from_habitat]
    from_slot = len(from_row) - 1
    from_row.pop()
    to_row = player.board[to_habitat]
    to_slot = len(to_row)
    to_row.append(played_bird)
    engine.events.record_effect(
        models.MoveBirdEffect(
            player_id=player.id,
            card=played_bird.bird.name,
            from_habitat=from_habitat.value,
            from_slot=from_slot,
            to_habitat=to_habitat.value,
            to_slot=to_slot,
        )
    )


# ---------------------------------------------------------------------------
# Public randomness reveals


def reroll_feeder(engine: core.Engine, player: state.Player) -> None:
    """Reroll every birdfeeder die and record the fresh faces — a reveal.

    Attributed to ``player``: whoever caused the roll, which for a pink reactor
    is not necessarily the current player."""
    feeder = engine.state.birdfeeder
    feeder.reroll(engine.state.rng)
    engine.events.record_effect(
        models.FeederRerollEffect(
            player_id=player.id,
            faces=state.die_face_words(feeder.counts, feeder.choice_dice),
        )
    )


def record_dice_roll(
    engine: core.Engine,
    player: state.Player,
    played_bird: state.PlayedBird,
    counts: state.FoodPool,
    choice_dice: int,
) -> None:
    """Record a dice-predator's off-feeder roll — a reveal.

    The ``ROLL_NOT_IN_FEEDER_CACHE`` birds roll dice that never enter the
    birdfeeder, so nothing else in the state ever shows what they landed on.
    Recording only, no mutation: the caller owns the roll itself."""
    engine.events.record_effect(
        models.DiceRollEffect(
            player_id=player.id,
            bird=played_bird.bird.name,
            faces=state.die_face_words(counts, choice_dice),
        )
    )


def refill_tray(engine: core.Engine, player_id: int | None = None) -> None:
    """Restock the face-up tray from the deck, recording each card — a reveal."""
    _record_tray_reveals(engine, engine.state.refill_tray(), player_id)


def reset_tray(engine: core.Engine, player_id: int | None = None) -> None:
    """Discard the whole face-up tray and restock it, recording each card."""
    _record_tray_reveals(engine, engine.state.reset_tray(), player_id)


###### PRIVATE #######


def _record_tray_reveals(
    engine: core.Engine,
    revealed: typing.Sequence[tuple[int, cards.Bird]],
    player_id: int | None,
) -> None:
    """Record freshly turned-up tray cards inside their own event.

    A tray refill belongs to no player action — it runs after the turn's last
    bracket closes, and again when a round resets the tray — so without a
    bracket of its own it would fall into the anonymous ``LooseEvent`` bucket.
    Opened here rather than at the call sites so no future refill can skip it.
    A refill that turned up nothing opens no event at all."""
    if not revealed:
        return
    engine.events.begin_refill_tray(player_id)
    try:
        for slot, bird in revealed:
            engine.events.record_effect(
                models.TrayRefillEffect(player_id=player_id, slot=slot, card=bird.name)
            )
    finally:
        engine.events.end_event()


def _locate_bird(
    player: state.Player, played_bird: state.PlayedBird
) -> tuple[str, int]:
    """Find ``played_bird``'s ``(habitat, slot)`` on its owner's board.

    Identity-based: two copies of the same bird card are distinct
    :class:`~wingspan.state.PlayedBird` objects, and a board holds at most 15
    of them, so the scan is cheaper than threading the coordinates through every
    power handler that only ever holds the ``PlayedBird``."""
    for habitat, row in player.board.items():
        for slot, candidate in enumerate(row):
            if candidate is played_bird:
                return habitat.value, slot
    return "", _UNPLACED_SLOT
