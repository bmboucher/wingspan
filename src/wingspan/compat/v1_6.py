"""Pre-1.7 artifact compat shim: freeze two pre-1.7 encoding behaviors.

**Freeze #1 — static (egg-blind) bonus potential pricing (both nets).** The
bonus *potential* counters — how many not-yet-played birds could still
qualify a bonus card — became optimistic about the egg-counting dynamic
cards: a bird whose ``egg_limit`` reaches the card's threshold (Breeding
Manager at 4, Oologist at 1) now counts, via
:func:`wingspan.engine.scoring.bonus_potential_count`. Pre-1.7 encoders
counted only the static ``bonus_categories`` tag, which no dynamic card
carries — so the egg cards' potentials read identically 0. This moves stripe
*values* on both nets, at unchanged dims and offsets:

* main net — the ``bonus_value`` stripe's ``hand_potential`` /
  ``tray_potential`` scalars, on in-game ``BonusCardChoice`` rows and on
  bonus-carrying ``SetupChoice`` rows;
* setup net — the split-mode ``bonus_card_affinity`` min/max pair and the
  folded-mode ``kept_bonus_value`` 4-vector (its stepped/linear VP are priced
  at the qual count, so they move with it).

Board-side counts (``scoring.bonus_qualifying_count``) always handled the
dynamic cards and are unchanged in every era.

**Freeze #2 — spend-decision food routing (main net only).** Single-token
``FoodChoice`` rows offered by a spend decision (``SpendFoodDecision``'s
discard, ``SpendFoodForEggDecision``'s grassland trade) now route to the
``pay_food`` stripe (``1 / layout._PAYMENT_COUNT_SCALE`` per unit) instead of
the ``gain_food`` stripe, since the player is paying food away rather than
gaining it. Pre-1.7 encoders wrote a ``gain_food`` one-hot for every
``FoodChoice`` regardless of direction, spend included. Setup-candidate
encoding has no food-direction convention (it never offers a spend menu) and
is unaffected — ``SetupNetV1_6`` is untouched by this freeze.

**Shim strategy.** The behavior-only era shape (the v1_4 / ``SetupNetV1_5``
precedent): no dims-router branch, no offset or layout override — each class
overrides only the encoder whose values changed and regenerates the pre-1.7
values in place after live encoding, via the ``refill_*`` seams next to the
live fills (``refill_bonus_value_potentials_static`` for freeze #1,
``refill_spend_food_gain_routing`` for freeze #2).

**Routing.** ``PolicyValueNet.class_for_version`` and
``SetupNet.class_for_version`` route era 1.6 here.
``compat.v1_5.PolicyValueNetV1_5`` / ``SetupNetV1_5`` **inherit** these nets
(re-chained from the live classes), so every era <= 1.5 freezes both
behaviors too, composed with its own strips and refills via ``super()``
chaining. Both freezes target offsets (``_OFF_BONUS_VALUE``, ``_OFF_GAIN_FOOD``
/ ``_OFF_PAY``) that precede every column any downstream shim strips
(``v1_5``'s ``goal_delta_ignoring_eggs`` tail, ``v1_3``'s ``resets_feeder``,
``v1_0``'s ``becomes_unplayable``), so the whole chain composes with no
offset math.

**Fixture note.** A committed LFS checkpoint fixture is deferred (as for
v1.0 / v1.3 / v1.4 / v1.5): ``tests/test_compat_v1_6.py`` builds v1.6-era
nets (main and setup), saves them with a v1.6 stamp, and round-trip-loads
them through the production ``players.loaders.load_policy_net`` /
``load_setup_net`` paths.
"""

from __future__ import annotations

import typing

import numpy as np

from wingspan import decisions, setup_model, state
from wingspan.encode import choice_encode
from wingspan.model import core
from wingspan.setup_model import encode as setup_encode
from wingspan.training import setup_net


class PolicyValueNetV1_6(core.PolicyValueNet):
    """``PolicyValueNet`` with two pre-1.7 encoding behaviors frozen: (1)
    every bonus-carrying choice row prices ``hand_potential`` /
    ``tray_potential`` by the static ``bonus_categories`` tag alone, so the
    egg-counting dynamic cards read 0; (2) every spend-decision
    ``FoodChoice`` row (``SpendFoodDecision``, ``SpendFoodForEggDecision``)
    is a ``gain_food`` one-hot rather than a ``pay_food`` payment. Geometry
    is live — v1.7 is behavior-only, so this class overrides
    ``encode_choices`` and nothing else."""

    def encode_choices(
        self,
        decision: decisions.Decision[typing.Any],
        game_state: state.GameState,
    ) -> np.ndarray:
        """Encode at live geometry, then overwrite each affected row with its
        pre-1.7 values: bonus potentials on bonus-carrying rows, and
        gain/pay routing on spend-decision food rows.

        Both refills run at full live width — this is the deepest override
        above ``core`` — and target offsets (``_OFF_BONUS_VALUE``,
        ``_OFF_GAIN_FOOD`` / ``_OFF_PAY``) before every column any subclass
        strips (``v1_5``'s ``goal_delta_ignoring_eggs`` tail, ``v1_3``'s
        ``resets_feeder``, ``v1_0``'s ``becomes_unplayable``), so the whole
        chain composes with no offset math."""
        full = super().encode_choices(decision, game_state)
        player = game_state.players[decision.player_id]
        is_spend_menu = isinstance(
            decision, (decisions.SpendFoodDecision, decisions.SpendFoodForEggDecision)
        )
        for row, choice in zip(full, decision.choices):
            if isinstance(choice, decisions.BonusCardChoice):
                choice_encode.refill_bonus_value_potentials_static(
                    row, choice.bonus_card, player.hand, game_state.tray
                )
            elif (
                isinstance(choice, decisions.SetupChoice)
                and choice.bonus_card is not None
            ):
                choice_encode.refill_bonus_value_potentials_static(
                    row, choice.bonus_card, choice.kept_cards, game_state.tray
                )
            if is_spend_menu and isinstance(choice, decisions.FoodChoice):
                choice_encode.refill_spend_food_gain_routing(row, choice.food)
        return full


class SetupNetV1_6(setup_net.SetupNet):
    """``SetupNet`` with the pre-1.7 static bonus pricing: the split-mode
    ``bonus_card_affinity`` pair and the folded-mode ``kept_bonus_value``
    4-vector are priced by the static ``bonus_categories`` tag alone.
    Geometry is UNCHANGED — like :class:`wingspan.compat.v1_5.SetupNetV1_5`
    this class asserts nothing shape-related and joins no dims-router branch;
    it overrides only ``encode_candidate``."""

    def encode_candidate(
        self,
        candidate: setup_model.SetupCandidate,
        context: setup_model.SetupContext,
    ) -> np.ndarray:
        """Encode at live geometry, then overwrite the bonus pricing stripes
        with the pre-1.7 static counts (see
        :func:`wingspan.setup_model.encode.refill_bonus_pricing_static`)."""
        vec = super().encode_candidate(candidate, context)
        setup_encode.refill_bonus_pricing_static(vec, candidate, context, self.encoding)
        return vec
