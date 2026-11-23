"""Builder for the "yes" row of every tuck-from-hand activation gate.

Every ``ActivateTuckDecision`` (the six tuck-from-hand power gates in
``engine.powers.grants`` and Horned Lark's pink reaction in
``engine.reactors``) offers a ``TuckActivateChoice`` plus a bare ``SkipChoice``.
The yes row is an :class:`decisions.ExchangeLedger`, filled *optimistically*
— a "may" / "might" payoff is assumed to succeed — so the SKIP_OPTIONAL head
can weigh the whole trade from the one row.

The nominal ledger comes from ``engine.forecast.effect_exchange_ledger`` — the
one table mapping a power onto ledger fields, which the main-action forecast
and the card-attribute ``power_ex`` block read too — so the gate row and the
forecast price a tuck identically by construction. This module only adds what
needs board context or a typed food:

* every kind: ``paid_card_count = 1`` (the card leaving the hand) and
  ``gained_tuck_count = 1`` (the tuck itself: +1 VP and a tucked card);
* ``TUCK_FROM_HAND_THEN_DRAW``: ``gained_card_count = eff.amount``;
* ``TUCK_FROM_HAND_THEN_LAY_ON_THIS``: ``gained_egg_count`` capped at the egg
  room on the bird (0 when it is full);
* ``TUCK_FROM_HAND_THEN_LAY_ANY``: ``gained_egg_count`` capped at the total
  egg room on the player's board;
* ``TUCK_FROM_HAND_THEN_GAIN_FOOD_SUPPLY``: ``gained_food_count =
  eff.amount`` and ``gained_food = eff.food``;
* ``TUCK_FROM_HAND_THEN_GAIN_FOOD_CHOICE``: ``gained_food_count = 1`` and
  ``gained_food = None`` (the player picks between two foods later);
* plain ``TUCK_FROM_HAND`` and Horned Lark's ``PINK_PLAY_BIRD_TUCK``: just the
  paid card and the tuck.

A single function so no gate site can drift from the others (the
by-construction rule: fold the ledger into the one entry point).
"""

from __future__ import annotations

from wingspan import cards, decisions, state
from wingspan.engine import forecast

TUCK_GATE_LABEL = "tuck 1 card"
"""Label of every tuck-gate yes row (unchanged across eras; the ledger carries
the terms)."""


def tuck_activate_choice(
    player: state.Player, pb: state.PlayedBird, eff: cards.Effect
) -> decisions.TuckActivateChoice:
    """The optimistic yes-row for the tuck gate of effect ``eff`` on bird ``pb``
    (owned by ``player``) — see the module docstring for the per-kind values."""
    nominal = forecast.effect_exchange_ledger(
        eff, include_eot_discard=True, legacy_v1_5_mappers=False
    )

    gained_egg_count = nominal.gained_egg_count
    if eff.kind == cards.EffectKind.TUCK_FROM_HAND_THEN_LAY_ON_THIS:
        gained_egg_count = min(gained_egg_count, max(pb.bird.egg_limit - pb.eggs, 0))
    elif eff.kind == cards.EffectKind.TUCK_FROM_HAND_THEN_LAY_ANY:
        gained_egg_count = min(gained_egg_count, _total_egg_room(player))

    gained_food = (
        eff.food
        if eff.kind == cards.EffectKind.TUCK_FROM_HAND_THEN_GAIN_FOOD_SUPPLY
        else None
    )

    return decisions.TuckActivateChoice(
        label=TUCK_GATE_LABEL,
        paid_card_count=nominal.paid_card_count,
        gained_tuck_count=nominal.gained_tuck_count,
        gained_card_count=nominal.gained_card_count,
        gained_egg_count=gained_egg_count,
        gained_food_count=nominal.gained_food_count,
        gained_food=gained_food,
    )


###### PRIVATE #######


def _total_egg_room(player: state.Player) -> int:
    """Eggs the player's board can still take (sum of per-bird room)."""
    return sum(
        max(played_bird.bird.egg_limit - played_bird.eggs, 0)
        for row in player.board.values()
        for played_bird in row
    )
