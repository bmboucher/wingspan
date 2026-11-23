"""Pre-1.6 artifact compat shim: the era-1.5 SKIP_OPTIONAL accept-row pricing
and the era-<=1.5 ``engine.forecast`` effect mapper.

**What changed in v1.6 (behavior-only, no dims change).**

*Accept rows.* Every choice row the SKIP_OPTIONAL head sees
(``AcceptExchangeDecision``'s ``PayCostChoice``, ``ActivateTuckDecision``'s
``TuckActivateChoice``) now carries as much of the exchange ledger as possible,
filled optimistically:

1. ``TuckActivateChoice`` is an ``ExchangeLedger`` and featurizes through the
   same shared ledger featurizer as ``PayCostChoice`` — exchange stripe,
   hand-shrink ``bonus_delta``, egg ``goal_delta`` / ``becomes_playable`` —
   with the tuck gates filling the card paid, the tuck, and the power's payoff
   (``engine.tuck_gate``). Era 1.5 wrote only the ``cards_to_tuck`` exchange
   slot.
2. A new ``ExchangeLedger.gained_food`` (the type gained, when fixed) lights
   the matching GAIN_FOOD one-hot and prices ``becomes_playable`` at that
   exact food. Era 1.5 priced every food gain as an any-feeder-die gain and
   wrote no gain-food type.
3. An accept row that lays eggs prices the egg-counting bonus cards
   (``_fill_bonus_delta_best_case_for_eggs``) into ``bonus_delta``, summed with
   the hand-card flow. Era 1.5 wrote the hand-card flow only.

*Forecast effect mapper.* Eleven bugs (the eighth: ``FEWEST_FOREST_GAINS_DIE``
priced 0 food for Hermit Thrush, now 1 die for self and tied opponent; the ninth
to eleventh: ``PINK_LAY_EGG_ON_NEST`` priced 0 eggs, now 1; ``DRAW_N_PLUS_ONE_DRAFT``
priced 0 cards, now 2 self and 1 opponent; ``FEWEST_WETLAND_DRAWS_CARD`` also
credits the tied opponent) in ``engine.forecast.effect_exchange_ledger``
(the one table that maps a bird power onto exchange-ledger fields) were fixed:
the tuck-from-hand kinds now gain the tuck (``gained_tuck_count``);
``TUCK_FROM_DECK_PAID`` pays the *food* (was an egg); ``LAY_EGG_ALL_NEST`` no
longer credits the opponent; ``FEWEST_WETLAND_DRAWS_CARD`` gains a *card* (was
food, and the parsed amount is 0, so it contributed nothing); ``PREDATOR_HUNT``
is mapped (one optimistic tuck; was unmapped); ``PINK_PLAY_BIRD_TUCK`` also
pays the card it tucks from hand; ``EACH_PLAYER_GAINS_DIE_CHOOSE_ORDER`` uses
the effect amount (was a literal 1). The mapper feeds two net-side inputs:

* the card-attribute ``power_ex`` block of the frozen ``card_features`` buffer
  (both nets' shared card table, ``state_encode.card_feature_matrix``), and
* the ``exchange`` stripe of every habitat-row ``MainActionChoice``
  (``engine.forecast.habitat_action_exchange_forecast``).

**Shim strategy.** Refill-after-live-encode, the compat.v1_4 value-refill
precedent. ``encode_choices`` runs the live encoder, then
:func:`wingspan.encode.choice_encode.refill_exchange_rows_v1_5` zeroes each
``PayCostChoice`` / ``TuckActivateChoice`` row and re-featurizes it the
era-1.5 way, and
:func:`wingspan.encode.choice_encode.refill_main_action_exchange_legacy_v1_5`
rewrites each ``MainActionChoice`` row's exchange stripe from the forecast run
with ``legacy_v1_5_mappers=True`` (the mapper steers the forecast's
feasibility budgets, so the whole forecast is recomputed). ``_build_card_encoder``
(both nets) overwrites the frozen ``card_features`` buffer's power_ex columns via
:func:`wingspan.encode.state_encode.refill_card_features_power_exchange` with
``legacy_v1_5_mappers=True``. The state vector itself never reads the mapper
(the per-card power ledger rides the card table), so ``encode_state`` needs no
override. No geometry changes, so there are no ``_true_*_dim`` /
``*_embed_offsets`` / layout overrides and no dims-router branch; every column
this touches precedes the tail stripes the older shims strip, so
:class:`wingspan.compat.v1_4.PolicyValueNetV1_4` /
:class:`wingspan.compat.v1_4.SetupNetV1_4` simply **inherit** these classes and
their chains compose with no offset math.

A few accept-row ledger *values* also changed in the v1.6 engine (the
cache-vs-keep accept no longer sets both ``paid_food`` and ``paid_food_count``;
the all-players-lay and cache-roll accepts read ``eff.amount`` / ``1 + extra``
instead of literals). For every card in the core set those evaluate to the old
literals, and the featurizer reads only the cells the old values reached, so
the shim needs no reproduction for them.

**Routing.** ``PolicyValueNet.class_for_version`` and
``SetupNet.class_for_version`` route era 1.5 here (eras <= 1.4 route to
``compat.v1_4``, which inherits it).

**Fixture note.** As for every era, a committed LFS checkpoint is deferred:
``tests/test_compat_v1_5.py`` proves era-1.5 encoding byte-identical to the
pre-change behavior and round-trips an era-1.5 net through
``players.loaders.load_policy_net``.
"""

from __future__ import annotations

import typing

import numpy as np
import torch

from wingspan import architecture, decisions, state
from wingspan.encode import choice_encode, state_encode
from wingspan.model import core
from wingspan.training import setup_net


def refill_power_exchange_card_table(net_card_features: torch.Tensor) -> torch.Tensor:
    """The era-1.5 ``card_features`` table: ``net_card_features`` (a net's
    just-built live buffer) with the power_ex columns regenerated at the
    era-1.5 mapper (the EOT discard side included, the pre-v1.6 effect
    mapping). One helper so the main net and the setup net cannot drift."""
    matrix = net_card_features.numpy().copy()
    state_encode.refill_card_features_power_exchange(
        matrix, include_eot_discard=True, legacy_v1_5_mappers=True
    )
    return torch.tensor(matrix, dtype=torch.float32)


class PolicyValueNetV1_5(core.PolicyValueNet):
    """``PolicyValueNet`` with the era-1.5 accept-row pricing and forecast
    effect mapper.

    Overrides :meth:`encode_choices` and :meth:`_build_card_encoder`; geometry
    is the live geometry. Routes for era 1.5 exactly;
    :class:`wingspan.compat.v1_4.PolicyValueNetV1_4` (and through it v1_3 /
    v1_0) inherits this class, so every earlier era freezes the same pricing."""

    def encode_choices(
        self,
        decision: decisions.Decision[typing.Any],
        game_state: state.GameState,
    ) -> np.ndarray:
        """Encode at live dims, then regenerate every accept row and every
        habitat-row MAIN_ACTION exchange forecast at the era-1.5 pricing (see
        the module docstring)."""
        full = super().encode_choices(decision, game_state)
        choice_encode.refill_exchange_rows_v1_5(full, decision, game_state)
        player = game_state.players[decision.player_id]
        for row, choice in zip(full, decision.choices):
            if isinstance(choice, decisions.MainActionChoice):
                choice_encode.refill_main_action_exchange_legacy_v1_5(
                    row, player, game_state, choice.action
                )
        return full

    def _build_card_encoder(self, arch: architecture.ModelArchitecture) -> None:
        """Build the live card encoder, then freeze the ``card_features``
        buffer's power_ex columns at the era-1.5 effect mapper."""
        super()._build_card_encoder(arch)
        self.card_features = refill_power_exchange_card_table(self.card_features)


class SetupNetV1_5(setup_net.SetupNet):
    """``SetupNet`` with the era-1.5 card-table power_ex freeze — the setup-side
    twin of :class:`PolicyValueNetV1_5._build_card_encoder` (``SetupNet`` builds
    its own copy of the shared ``card_feature_matrix()`` table). Setup
    candidates never read the forecast otherwise, so nothing else changes.
    Routes for era 1.5 exactly; :class:`wingspan.compat.v1_4.SetupNetV1_4`
    inherits it."""

    def _build_card_encoder(self, main_arch: architecture.ModelArchitecture) -> None:
        """Build the live card encoder, then freeze the power_ex columns at the
        era-1.5 effect mapper."""
        super()._build_card_encoder(main_arch)
        self.card_features = refill_power_exchange_card_table(self.card_features)
