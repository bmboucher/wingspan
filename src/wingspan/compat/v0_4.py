# pyright: reportPrivateUsage=false
# (the v0.4 -> v0.6 shim reads the live layout's package-private stripe offsets
# and calls the frozen v0.4 state/choice encoders -- a deliberate compat coupling,
# pinned by the import-time layout-contract assertions below)
"""Frozen v0.4/v0.5 state and choice encoding: the shim that keeps pre-0.6 artifacts playable.

Artifact version 0.6 added two new hand-playability multi-hot stripes to the
state vector (``hand_playable_me`` and ``hand_playable_eggs_me``, each 180 dims)
and a ``becomes_playable`` stripe to every choice row (180 dims):

* Added to state: ``hand_playable_me`` and ``hand_playable_eggs_me`` immediately
  after the existing ``hand_multihot`` stripe, growing the state vector by
  2 × 180 = 360 dims (795 → 1155).
* Added to choice: ``becomes_playable`` 180-dim multi-hot appended after
  ``bonus_value``, growing each choice row by 180 dims.

Nets trained before 0.6 have a 795-dim state trunk input and a narrower choice
input that lacks the ``becomes_playable`` stripe; this module keeps them loadable:

* :func:`encode_state_v04` produces the complete 795-dim v0.4/v0.5 state vector
  (no ``hand_playable_me`` / ``hand_playable_eggs_me`` stripes, everything else
  live) for a pre-0.6 checkpoint.
* :func:`encode_choices_v04` featurizes each choice without filling the
  ``becomes_playable`` stripe, yielding the narrower pre-0.6 choice row the
  old checkpoint's choice encoder expects.
* :class:`PolicyValueNetV04` overrides :meth:`encode_state` and
  :meth:`encode_choices` so inference call sites drive the net with the
  795-dim state and the narrower choice vectors its weights expect. It also
  overrides :meth:`_state_embed_offsets` and :meth:`_choice_embed_offsets` so
  the trunk and choice encoder slice the vectors at their frozen pre-0.6 offsets
  (the live offsets include the new stripes, so using them would corrupt silently
  rather than crashing).
* :func:`state_embed_offsets_v04` is the frozen-geometry slice offsets for the
  pre-0.6 state vector.
* :func:`state_feature_dim_v04` is the frozen 795-dim state width.
* :func:`choice_feature_dim_v04` is the frozen (narrower) choice row width.
* :func:`uses_v0_4_encoding` identifies which artifact versions need this shim.

Per the compatibility policy (``CLAUDE.md``), this shim lives until a MAJOR
``MODEL_VERSION`` bump deletes it together with the pre-0.6 fixture set.
"""

from __future__ import annotations

import typing

import numpy as np

from wingspan import decisions, state, version
from wingspan.encode import choice_encode, layout, state_encode
from wingspan.model import core

PLAYABILITY_STRIPES_ADDED_IN = "0.6"
"""The artifact version that added the playability multi-hot stripes this shim undoes."""

# ---------------------------------------------------------------------------
# Frozen v0.4/v0.5 geometry constants.
#
# The v0.6 state vector adds N_HAND_PLAYABLE_MULTIHOTS extra 180-dim
# hand-playability stripes after ``hand_multihot``.  The v0.4/v0.5 vector
# therefore lacks those stripes entirely — the delta is exactly
# N_HAND_PLAYABLE_MULTIHOTS * HAND_MULTIHOT_DIM dims.
_N_FROZEN_PLAYABLE = 0  # pre-0.6 artifacts have 0 playability multihots
_PLAYABLE_STRIPE_DIM = layout.N_HAND_PLAYABLE_MULTIHOTS * layout.HAND_MULTIHOT_DIM
"""Total dims added to the state vector by the playability stripes (360 in v0.6)."""

# The hand_multihot offset in v0.4/v0.5: same as the live layout's value because
# both playability stripes are AFTER hand_multihot. No delta needed here — only
# the stripes that come AFTER the insertion point differ.
_V04_HAND_MULTIHOT_OFFSET = layout.OFF_HAND_MULTIHOT
"""Offset of ``hand_multihot`` in the frozen 795-dim v0.4 vector (unchanged from live)."""

# The decision-type stripe begins right after hand_multihot in v0.4 (no
# intervening playability stripes), but at OFF_DECISION_TYPE in v0.6 (which
# is OFF_HAND_MULTIHOT + (1 + N_HAND_PLAYABLE_MULTIHOTS) * HAND_MULTIHOT_DIM +
# decision_type_dim). Since the playability stripes sit between hand_multihot
# and the decision-type one-hot, the v0.4 decision-type starts earlier by
# exactly _PLAYABLE_STRIPE_DIM.
_V04_DECISION_TYPE_OFFSET = layout.OFF_DECISION_TYPE - _PLAYABLE_STRIPE_DIM
"""Offset of the decision-type one-hot in the frozen 795-dim v0.4 vector."""

# The choice vector's ``becomes_playable`` stripe (180 dims) was added at the end
# of the base choice spec in v0.6; pre-0.6 choice rows lack it entirely.
_V04_CHOICE_FEATURE_DIM_BASE = (
    layout.choice_feature_dim() - layout.CHOICE_BECOMES_PLAYABLE_DIM
)
"""The pre-0.6 (base-spec, setup-excluded) choice row width."""


def uses_v0_4_encoding(artifact_version: str) -> bool:
    """Whether ``artifact_version`` uses the pre-0.6 state/choice encoding and
    therefore needs this module's frozen geometry to load and play."""
    parsed = version.parse_version(artifact_version)
    changed = version.parse_version(PLAYABILITY_STRIPES_ADDED_IN)
    return (parsed.major, parsed.minor) >= (0, 4) and (
        parsed.major,
        parsed.minor,
    ) < (changed.major, changed.minor)


def encode_state_v04(
    game_state: state.GameState,
    decision: decisions.Decision[decisions.Choice] | None = None,
    spec: layout.EncodingSpec = layout.DEFAULT_SPEC,
) -> np.ndarray:
    """Produce the complete 795-dim v0.4/v0.5 state vector.

    Identical to the live ``encode_state`` except that the two hand-playability
    stripes (``hand_playable_me`` and ``hand_playable_eggs_me``) are omitted,
    keeping ``state_dim`` at 795.  Everything else (turn_state, board, hand
    multihot, round goals, decision-type) uses the live encoders unchanged."""
    pov = decision.player_id if decision is not None else game_state.current_player
    me = game_state.players[pov]
    opp = game_state.players[1 - pov] if len(game_state.players) > 1 else me

    # Collect stripes in the same order as encode_state, but omit the two
    # playability multi-hots that were added in v0.6.
    parts: list[np.ndarray] = [
        state_encode._summary_turn_state(game_state, me),
        state_encode._summary_food(me),
        state_encode._summary_food(opp),
        state_encode._board_slots_continuous(me),
        state_encode._board_slots_continuous(opp),
        state_encode._summary_board(me),
        state_encode._summary_board(opp),
        state_encode._summary_hand(me),
        state_encode._bonus_progress(me),
        state_encode._opp_bonus_count(opp),
        np.array([len(opp.hand) / layout._HAND_SIZE_SCALE], dtype=np.float32),
        state_encode._summary_birdfeeder(game_state),
        state_encode._summary_misc_scalars(game_state, me, opp),
        state_encode._round_goals_all_rounds(game_state, me),
        state_encode._card_index_block(me, opp, game_state),
        state_encode._hand_identity(me),
        # hand_playable_me and hand_playable_eggs_me are intentionally OMITTED
        state_encode._encode_decision_type(decision, spec),
    ]
    return np.concatenate(parts).astype(np.float32)


def encode_choices_v04(
    decision: decisions.Decision[typing.Any],
    game_state: state.GameState,
    spec: layout.EncodingSpec = layout.DEFAULT_SPEC,
) -> np.ndarray:
    """Featurize all choices in ``decision`` without the ``becomes_playable`` stripe.

    Produces the pre-0.6 choice matrix that a v0.4/v0.5 checkpoint's choice
    encoder expects: the base choice feature rows without the trailing 180-dim
    ``becomes_playable`` multi-hot. Every other stripe is filled identically to
    the live encoder."""
    return choice_encode.encode_choices(
        decision, game_state, spec, has_becomes_playable=False
    )


def state_embed_offsets_v04() -> core.StateEmbedOffsets:
    """The frozen slice offsets ``_embed_state`` uses for the 795-dim v0.4/v0.5
    state vector.

    The ``hand_multihot`` offset is unchanged (the playability stripes sit
    AFTER it, so its offset is not shifted). The ``decision_type`` offset is
    shifted back by ``_PLAYABLE_STRIPE_DIM`` (360 dims) because those stripes
    were inserted before the decision-type one-hot. The ``card_index`` and
    ``hand_summary`` offsets are also unchanged (both precede the insertion point).

    ``PolicyValueNetV04`` overrides ``_state_embed_offsets`` with this, so an
    old checkpoint's state vector is sliced at the columns it was written with."""
    return core.StateEmbedOffsets(
        card_index=layout.OFF_CARD_INDEX,
        hand_multihot=_V04_HAND_MULTIHOT_OFFSET,
        decision_type=_V04_DECISION_TYPE_OFFSET,
        hand_summary=layout.HAND_SUMMARY_OFFSET,
    )


def state_feature_dim_v04(spec: layout.EncodingSpec = layout.DEFAULT_SPEC) -> int:
    """The frozen v0.4/v0.5 state-vector width (795 under the default spec).

    The live width minus ``_PLAYABLE_STRIPE_DIM`` (360) — the size of
    :func:`encode_state_v04`'s output and the ``state_dim`` every pre-0.6 net
    was built with. The era-dims router (``compat.encoding_dims_for_era``) uses
    this so an era-pinned ``TrainConfig`` derives the dims its checkpoints
    actually carry."""
    return layout.state_feature_dim(spec) - _PLAYABLE_STRIPE_DIM


def choice_feature_dim_v04(spec: layout.EncodingSpec = layout.DEFAULT_SPEC) -> int:
    """The frozen v0.4/v0.5 choice-row width (without the ``becomes_playable`` stripe).

    The live width minus ``CHOICE_BECOMES_PLAYABLE_DIM`` (180) — the size of
    each row in :func:`encode_choices_v04`'s output."""
    return layout.choice_feature_dim(spec) - layout.CHOICE_BECOMES_PLAYABLE_DIM


class PolicyValueNetV04(core.PolicyValueNet):
    """A :class:`~wingspan.model.core.PolicyValueNet` frozen to the v0.4/v0.5
    state and choice geometry, for checkpoints written before artifact version 0.6.

    The state trunk was trained against 795-dim inputs (no hand-playability
    multi-hots); the choice encoder was trained without the ``becomes_playable``
    stripe. This subclass overrides :meth:`encode_state` and :meth:`encode_choices`
    to keep those widths, and :meth:`_state_embed_offsets` /
    :meth:`_choice_embed_offsets` to slice the narrower vectors at the correct
    frozen offsets.

    Constructed by the version-routing loaders (``PolicyValueNet.from_model_config``,
    ``players.loaders.load_policy_net``) — never by the training pipeline.
    """

    def encode_state(
        self,
        game_state: state.GameState,
        decision: decisions.Decision[decisions.Choice],
    ) -> np.ndarray:
        """Featurize ``game_state`` without the playability multi-hots,
        yielding a 795-dim vector this checkpoint's trunk expects."""
        return encode_state_v04(game_state, decision, self.spec)

    def encode_choices(
        self,
        decision: decisions.Decision[decisions.Choice],
        game_state: state.GameState,
    ) -> np.ndarray:
        """Featurize all choices without the ``becomes_playable`` stripe,
        yielding the narrower pre-0.6 choice matrix this checkpoint expects."""
        return encode_choices_v04(decision, game_state, self.spec)

    def _state_embed_offsets(self) -> core.StateEmbedOffsets:
        """Slice the 795-dim v0.4 state vector at its frozen offsets.

        Without this override the trunk would read ``decision_type`` 360 columns
        too far right (past the live playability stripes that the old checkpoint
        has never seen), corrupting its input silently rather than crashing."""
        return state_embed_offsets_v04()

    def _choice_embed_offsets(self) -> core.ChoiceEmbedOffsets:
        """Slice the pre-0.6 choice vector with ``becomes_playable=None``.

        Without this override the choice encoder would try to embed a
        ``becomes_playable`` multi-hot that isn't present, reading garbage data
        from the padding zone beyond the actual choice row width."""
        return core.ChoiceEmbedOffsets(
            board_idx=layout.CHOICE_BOARD_IDX_OFFSET,
            bird_id=layout.CHOICE_BIRD_ID_OFFSET,
            becomes_playable=None,  # not present in pre-0.6 choice rows
            kept_multihot=None,  # None when include_setup is False (same as live)
        )


###### PRIVATE #######


def _assert_live_layout_contract() -> None:
    """Import-time pins for the invariants the shim relies on.

    The shim omits the two playability multi-hot stripes that v0.6 inserts
    after ``hand_multihot`` in the state vector, and omits the
    ``becomes_playable`` stripe from the choice vector.  The frozen geometry
    is correct only while:

    1. ``N_HAND_PLAYABLE_MULTIHOTS`` remains 2 (exactly two extra stripes).
    2. The playability stripes sit between ``hand_multihot`` and the
       decision-type one-hot (i.e. after everything the shim keeps, before
       the suffix the shim computes fresh).
    3. ``CHOICE_BECOMES_PLAYABLE_DIM`` remains 180 (one per-bird entry).
    4. ``hand_multihot`` precedes the playability stripes in the live layout
       (so its offset is unchanged between the frozen and live vectors).
    """
    assert layout.N_HAND_PLAYABLE_MULTIHOTS == 2, (
        f"v0.4 shim expects N_HAND_PLAYABLE_MULTIHOTS == 2, "
        f"but found {layout.N_HAND_PLAYABLE_MULTIHOTS}; update the shim"
    )
    assert layout.CHOICE_BECOMES_PLAYABLE_DIM == layout.HAND_MULTIHOT_DIM, (
        f"v0.4 shim expects CHOICE_BECOMES_PLAYABLE_DIM == HAND_MULTIHOT_DIM "
        f"({layout.HAND_MULTIHOT_DIM}), but found {layout.CHOICE_BECOMES_PLAYABLE_DIM}; "
        "update the shim"
    )
    # Confirm the playability stripes sit after hand_multihot and before
    # the decision-type one-hot (OFF_DECISION_TYPE = total state continuous size).
    hand_mh_off = layout.STATE_CONT_LAYOUT.offset_of("hand_multihot")
    play_off = layout.STATE_CONT_LAYOUT.offset_of("hand_playable_me")
    assert hand_mh_off < play_off, (
        "v0.4 shim assumes hand_multihot precedes hand_playable_me in the live layout; "
        "the insertion point has moved — update the shim"
    )
    expected_total_delta = layout.N_HAND_PLAYABLE_MULTIHOTS * layout.HAND_MULTIHOT_DIM
    assert _PLAYABLE_STRIPE_DIM == expected_total_delta, (
        f"v0.4 shim's _PLAYABLE_STRIPE_DIM ({_PLAYABLE_STRIPE_DIM}) does not match "
        f"N_HAND_PLAYABLE_MULTIHOTS * HAND_MULTIHOT_DIM ({expected_total_delta}); "
        "update the shim"
    )


_assert_live_layout_contract()
