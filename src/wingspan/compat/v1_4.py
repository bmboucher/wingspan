"""Pre-1.5 artifact compat shim: merge of what were four provisionally-
numbered eras (1.5, 1.6, 1.7, 1.8) that landed on main in sequence but never
trained a run past 1.4 — collapsed into one era before any of them trained,
the same fold-in rule ``docs/VERSIONING.md`` already documents for the v1.4
two-changes-in-one-era precedent.

**What changed in v1.5 (four changes, folded together).**

1. *State* — a per-opponent ``known_hand_opp`` 180-wide identity multi-hot
   (the publicly-known subset of an opponent's hand;
   ``state.Player.known_hand``, maintained by ``engine.ledger``) is appended
   to the state vector's multi-hot region, immediately after
   ``hand_playable_eggs_me`` and before the trailing ``decision_type``
   one-hot. State width grows by 180 per opponent (N=2 base: 1129 -> 1309).
2. *Choice* — an 8-dim ``goal_delta_ignoring_eggs`` stripe is appended as
   the new last base choice-feature stripe (immediately after
   ``resets_feeder``): per round goal, a ``(count_delta, vp_delta)`` pair
   pricing the hypothesis that this row's bird is eventually played (a slot
   must be open) and egg-populated to whatever level best advances the
   goal. Choice width grows by 8 (N=2 base: 509 -> 517).
3. *Values* — the ``PlayBirdChoice`` featurizer's ``goal_delta`` stripe
   becomes conditioned on the row's committed landing habitat (a
   ``birds_<habitat>`` goal now moves only on the row that actually plays
   the bird into that habitat, instead of on every row of a multi-habitat
   bird), and the setup encoder's ``goal_affinity`` stripe switches from the
   play-instant count to the played-and-optimally-egg-populated bound.
4. *Values* — the bonus *potential* counters on both nets become optimistic
   about the egg-counting dynamic bonus cards (a not-yet-played bird whose
   ``egg_limit`` reaches a card's threshold now counts), and single-token
   ``FoodChoice`` rows offered by a spend decision route to the ``pay_food``
   stripe instead of ``gain_food`` (main net only).

**Shim strategy.** The geometry seams (state and choice width) derive their
narrow width absolutely from ``self.spec`` via ``_true_state_dim`` /
``_true_choice_dim`` rather than composing through a ``super()`` chain —
this is the shim closest to live, the same position the old (now-collapsed)
``v1_7`` module occupied on the state side. **A future MINOR bump that
narrows the state or choice vector again must convert both methods to the
composing form** (``super()._true_state_dim() - ...``), exactly as the old
v1.6 bump once converted ``compat.v1_3``'s ``_true_choice_dim`` — so a
further narrowing stacks instead of silently overwriting this one. The
value freezes (changes 3 and 4) are refill-after-live-encode: each targets
an offset that precedes the stripped ``goal_delta_ignoring_eggs`` tail, so
every column ``v1_3`` / ``v1_0`` strip on top of this class's output is
unaffected — the whole chain composes with no offset math.

**Routing.** ``PolicyValueNet.class_for_version`` and
``SetupNet.class_for_version`` both route era 1.4 here.
``compat.v1_3.PolicyValueNetV1_3`` inherits :class:`PolicyValueNetV1_4`, so
every pre-1.4 era freezes all four changes too, on top of its own stripe
strips. ``SetupNet.class_for_version`` also routes eras <= 1.3 to
:class:`SetupNetV1_4` — harmlessly, since a pre-1.3 setup artifact differs
in *shape* (the v1.3 two-tower restructure) and would fail at
``load_state_dict`` regardless of which class builds it;
``players.loaders.load_setup_net`` already turns that into a clear "retrain
the setup model" error.

**Fixture note.** A committed LFS checkpoint fixture is deferred, as for
every era: ``tests/test_compat_v1_4.py`` builds v1.4-era nets (main and
setup), saves them with a v1.4 stamp, and round-trip-loads them through the
production ``players.loaders.load_policy_net`` / ``load_setup_net`` paths.
"""

from __future__ import annotations

import typing

import numpy as np

from wingspan import architecture, decisions, encode, setup_model, state
from wingspan.encode import choice_encode, stripes
from wingspan.model import core
from wingspan.setup_model import encode as setup_encode
from wingspan.training import setup_net

# Layout names of the v1.5 stripes this shim strips (offsets/widths: the
# encode.STATE_KNOWN_HAND_OPP_* / encode.CHOICE_GOAL_DELTA_IGNORING_EGGS_*
# constants used below). Every compat-shimmed era is 2-player only
# (encoding_dims_for_era refuses spec.num_players != 2 up front), so there is
# never a suffixed known_hand_opp2 / known_hand_opp3 replica to strip here.
_V1_5_STATE_STRIPE_NAMES = ("known_hand_opp",)
_V1_5_CHOICE_STRIPE_NAME = "goal_delta_ignoring_eggs"


class PolicyValueNetV1_4(core.PolicyValueNet):
    """``PolicyValueNet`` with pre-1.5 geometry and value freezes — the merge
    of what were four provisionally-numbered eras (1.5-1.8) that landed on
    main but never trained a run past 1.4 (the fold-in rule; see
    ``docs/VERSIONING.md``):

    * **State** — the per-opponent ``known_hand_opp`` 180-wide multi-hot
      stripped from ``encode_state``, with the frozen pre-1.5
      ``StateEmbedOffsets`` (only ``decision_type`` shifts).
    * **Choice geometry** — the 8-dim ``goal_delta_ignoring_eggs`` tail
      stripe stripped from ``encode_choices``, with ``kept_multihot``
      shifted left to match.
    * **Choice values** — every bonus-carrying row's potentials and every
      spend-decision food row's direction regenerated at the pre-1.7 static
      / ``gain_food`` pricing, and every play-bird row's ``goal_delta``
      regenerated at the pre-1.5 habitat-agnostic pricing.

    Both geometry seams derive their width absolutely from ``self.spec``
    (``_true_state_dim`` / ``_true_choice_dim``) rather than composing via
    ``super()`` — this is the shim closest to live, mirroring the rationale
    the old ``v1_7`` module used before the (now-collapsed) v1.6 bump
    converted ``compat.v1_3``'s analogous method to the composing form. A
    future MINOR bump that narrows the state or choice vector again must
    convert these methods to the same composing form
    (``super()._true_state_dim() - ...``), so a further narrowing stacks
    instead of silently overwriting this one.

    Routes for era 1.4
    (:meth:`wingspan.model.core.PolicyValueNet.class_for_version`);
    :class:`wingspan.compat.v1_3.PolicyValueNetV1_3` inherits this class, so
    every pre-1.4 era freezes all of the above too, on top of its own
    stripe strips."""

    # --- state: strip the known_hand_opp stripe ---

    def _true_state_dim(self) -> int:
        """The state width this shim's ``encode_state`` actually produces —
        the live width minus the ``known_hand_opp`` stripe — derived from
        ``self.spec`` so it is independent of the ``state_dim`` passed to
        ``__init__``. Absolute form: this is the state-narrowing shim
        closest to live, so there is no ancestor era to compose through (see
        the module docstring's "future state-narrowing era" note)."""
        return encode.state_size(self.spec) - encode.STATE_KNOWN_HAND_OPP_DIM

    def _build_trunk(
        self, state_dim: int, arch: architecture.ModelArchitecture
    ) -> None:
        """Build the trunk at the pre-1.5 (narrow) state width.

        Ignores the passed ``state_dim`` in favour of ``_true_state_dim`` so
        the trunk matches what ``encode_state`` emits whether the
        constructor was handed live or era dims."""
        super()._build_trunk(self._true_state_dim(), arch)

    def encode_state(
        self,
        game_state: state.GameState,
        decision: decisions.Decision[typing.Any],
    ) -> np.ndarray:
        """Encode at live dims, then strip the ``known_hand_opp`` stripe.

        The live encoder writes the full live vector including the stripe;
        ``np.delete`` removes its contiguous columns so the result matches
        the width the pre-1.5 trunk expects."""
        full = super().encode_state(game_state, decision)
        start = encode.STATE_KNOWN_HAND_OPP_OFFSET
        end = start + encode.STATE_KNOWN_HAND_OPP_DIM
        return np.delete(full, slice(start, end), axis=0)

    def _state_embed_offsets(self) -> core.StateEmbedOffsets:
        """Return pre-1.5 offsets: ``decision_type`` shifted left by
        ``STATE_KNOWN_HAND_OPP_DIM`` because the stripe was never in the
        pre-1.5 state vector. ``card_index`` and ``hand_multihot`` are
        UNCHANGED — the new stripe is appended after both playability
        multi-hots, well past either offset, so neither moves (the opposite
        shift shape from ``compat.v1_3``, whose food-unlock stripe precedes
        ``card_index`` and so shifts all three)."""
        live = super()._state_embed_offsets()
        return core.StateEmbedOffsets(
            card_index=live.card_index,
            hand_multihot=live.hand_multihot,
            decision_type=live.decision_type - encode.STATE_KNOWN_HAND_OPP_DIM,
            hand_summary=live.hand_summary,
            hand_summary_end=live.hand_summary_end,
        )

    def raw_state_stripe_layout(self) -> stripes.VectorLayout:
        """The live state layout minus ``known_hand_opp`` — the layout of
        the (narrow) vector this shim's ``encode_state`` actually emits, so
        consumers that decode recorded vectors (the game-log encoding
        viewer) name the era's columns instead of the live ones."""
        return (
            super().raw_state_stripe_layout().without_stripes(_V1_5_STATE_STRIPE_NAMES)
        )

    # --- choice: strip the goal_delta_ignoring_eggs tail, refill values ---

    def _true_choice_dim(self) -> int:
        """The choice width this shim's ``encode_choices`` actually produces
        — the live width minus ``goal_delta_ignoring_eggs`` — derived from
        ``self.spec`` so it is independent of the ``choice_dim`` passed to
        ``__init__``. Subclasses (``v1_3``, reached via ``v1_4``) narrow
        further by overriding this and composing via ``super()`` (the v1_3
        pattern), rather than recomputing the width absolutely."""
        return (
            encode.choice_feature_dim(self.spec)
            - encode.CHOICE_GOAL_DELTA_IGNORING_EGGS_DIM
        )

    def _build_choice_encoder(
        self,
        choice_dim: int,
        arch: architecture.ModelArchitecture,
    ) -> None:
        """Build the choice encoder at the pre-1.5 (narrow) input width.

        Ignores the passed ``choice_dim`` in favour of ``_true_choice_dim`` —
        which is polymorphic, so a subclass instance builds at its own
        (further narrowed) width through this same method."""
        super()._build_choice_encoder(self._true_choice_dim(), arch)

    def encode_choices(
        self,
        decision: decisions.Decision[typing.Any],
        game_state: state.GameState,
    ) -> np.ndarray:
        """Encode at live dims, apply the pre-1.7-era bonus/spend-food value
        refills, strip the pre-1.6-era ``goal_delta_ignoring_eggs`` tail,
        then apply this era's own habitat-agnostic ``goal_delta`` refill —
        the chain the four provisionally-numbered eras (1.5-1.8) originally
        walked one hop at a time, now folded into one class (see the module
        docstring).

        Three phases, in the order the accumulated shims originally applied
        them:

        1. **Bonus-potential / spend-food refill** (was
           ``v1_6.PolicyValueNetV1_6.encode_choices``): every bonus-carrying
           row (``BonusCardChoice``, or a bonus-carrying ``SetupChoice``)
           gets its ``hand_potential`` / ``tray_potential`` scalars
           regenerated via the static (egg-blind) count; every
           spend-decision (``SpendFoodDecision`` / ``SpendFoodForEggDecision``)
           ``FoodChoice`` row is re-routed from ``pay_food`` back to a
           ``gain_food`` one-hot.
        2. **Tail strip** (was ``v1_5.PolicyValueNetV1_5.encode_choices``):
           ``np.delete`` the ``goal_delta_ignoring_eggs`` columns so the row
           matches the width the pre-1.5 choice encoder was built for.
        3. **goal_delta refill** (this class's own change, since v1.5
           itself): each ``PlayBirdChoice`` row's ``goal_delta`` stripe is
           overwritten with the habitat-agnostic pricing.

        Phase 1's and phase 3's refill offsets (``layout._OFF_BONUS_VALUE``,
        ``layout._OFF_GAIN_FOOD`` / ``layout._OFF_PAY``,
        ``layout._OFF_GOAL_DELTA``) all precede the tail phase 2 strips, and
        so precede every column ``v1_3`` / ``v1_0`` strip on top of this
        method's output — the whole chain composes with no offset math."""
        full = super().encode_choices(decision, game_state)

        # Phase 1 — pre-1.7-era bonus-potential and spend-food-direction
        # refills, at live column offsets (before the tail strip below).
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

        # Phase 2 — strip the pre-1.6-era goal_delta_ignoring_eggs tail.
        start = encode.CHOICE_GOAL_DELTA_IGNORING_EGGS_OFFSET
        end = start + encode.CHOICE_GOAL_DELTA_IGNORING_EGGS_DIM
        narrowed = np.delete(full, slice(start, end), axis=1)

        # Phase 3 — this era's own habitat-agnostic goal_delta refill, at an
        # offset well before the tail phase 2 just stripped.
        for row, choice in zip(narrowed, decision.choices):
            if isinstance(choice, decisions.PlayBirdChoice):
                choice_encode.refill_goal_delta_habitat_agnostic(
                    row, decision.player_id, choice.bird, game_state
                )
        return narrowed

    def _choice_embed_offsets(self) -> core.ChoiceEmbedOffsets:
        """Return pre-1.5 offsets: ``bird_id`` / ``becomes_playable`` /
        ``becomes_unplayable`` are unchanged (they precede the new stripe);
        ``kept_multihot`` shifts left by ``CHOICE_GOAL_DELTA_IGNORING_EGGS_DIM``
        because the stripe was never in the pre-1.5 stored choice vectors."""
        live = super()._choice_embed_offsets()
        kept = live.kept_multihot
        if kept is not None:
            kept = kept - encode.CHOICE_GOAL_DELTA_IGNORING_EGGS_DIM
        return core.ChoiceEmbedOffsets(
            bird_id=live.bird_id,
            becomes_playable=live.becomes_playable,
            becomes_unplayable=live.becomes_unplayable,
            kept_multihot=kept,
        )

    def raw_choice_stripe_layout(self) -> stripes.VectorLayout:
        """The live choice layout minus ``goal_delta_ignoring_eggs`` — the
        layout of the (narrow) rows this shim's ``encode_choices`` actually
        emits (the game-log encoding viewer's decode seam; see
        ``docs/VERSIONING.md``)."""
        return (
            super()
            .raw_choice_stripe_layout()
            .without_stripes((_V1_5_CHOICE_STRIPE_NAME,))
        )


class SetupNetV1_4(setup_net.SetupNet):
    """``SetupNet`` with both pre-1.5 setup value freezes: the bonus pricing
    stripe (split-mode ``bonus_card_affinity`` pair, or folded-mode
    ``kept_bonus_value`` 4-vector) at the pre-1.7 static (egg-blind) counts,
    and the ``goal_affinity`` stripe at the pre-1.6 static (egg-blind)
    per-kept-card pricing — the setup-side halves of the same four
    provisionally-numbered eras (1.5-1.8) :class:`PolicyValueNetV1_4` folds
    on the main-net side.

    Geometry is UNCHANGED — every setup-side change since the v1.3 two-tower
    restructure has been values-only — so this class asserts nothing
    shape-related and joins no dims-router branch; it overrides only
    ``encode_candidate``. The two refills write disjoint stripes at
    unchanged offsets, so their order is immaterial; kept in the order the
    two constituent shims originally applied them.

    The first :class:`wingspan.training.setup_net.SetupNet` compat shim
    (the ``compat/INDEX.md`` "PolicyValueNet/SetupNet subclass" hook).
    Routed by :meth:`wingspan.training.setup_net.SetupNet.class_for_version`
    for eras <= 1.4; eras <= 1.3 route here too, harmlessly — their setup
    net differs in *shape* (the v1.3 two-tower restructure) and would fail
    at ``load_state_dict`` regardless of which class builds them, and
    ``players.loaders.load_setup_net`` already turns that into a clear
    "retrain the setup model" error."""

    def encode_candidate(
        self,
        candidate: setup_model.SetupCandidate,
        context: setup_model.SetupContext,
    ) -> np.ndarray:
        """Encode at live geometry, then overwrite the bonus pricing stripe
        with the pre-1.7 static counts and the ``goal_affinity`` stripe with
        the pre-1.6 static pricing — disjoint stripes, so the two refills
        compose in either order (see
        :func:`wingspan.setup_model.encode.refill_bonus_pricing_static` /
        :func:`wingspan.setup_model.encode.refill_goal_affinity_static`)."""
        vec = super().encode_candidate(candidate, context)
        setup_encode.refill_bonus_pricing_static(vec, candidate, context, self.encoding)
        setup_encode.refill_goal_affinity_static(vec, candidate, context, self.encoding)
        return vec
