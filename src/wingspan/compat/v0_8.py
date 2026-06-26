# pyright: reportPrivateUsage=false
# (the v0.8 shim calls state_encode private sub-builders with old-behavior flags
# to freeze the 1155-dim v0.6–v0.8 geometry — a deliberate compat coupling,
# pinned by the import-time layout-contract assertions below)
"""Frozen v0.6–v0.8 state encoding: the shim that keeps pre-0.9 artifacts playable.

Artifact version 0.9 compacted the state vector from 1155→1119 dims by:

* Shrinking ``misc_scalars`` 4→2 dims (dropped round-goal VP scalars).
* Shrinking ``board_summary_me`` / ``board_summary_opp`` 18→6 dims each (kept
  only ``row_length`` and ``total_eggs`` per habitat).
* Removing ``hand_summary_me`` entirely (10 dims; now derived in-model).
* Zeroing already-scored ``round_goals`` slots (value only, width unchanged).

Nets trained in artifact eras 0.6–0.8 have a 1155-dim state trunk input; this
module keeps them loadable:

* :func:`encode_state_v08` reproduces the complete 1155-dim state vector by
  calling the live sub-builders with old-behavior flags (``full_stats=True``,
  ``include_goal_pts=True``, ``zero_passed_rounds=False``) and re-inserting the
  ``hand_summary_me`` stripe in its historical position.
* :func:`state_embed_offsets_v08` returns the frozen :class:`~wingspan.model.core.StateEmbedOffsets`
  for the 1155-dim vector, so ``_embed_state`` slices it at the columns it was
  written with — not the live v0.9 ones (which sit 36 columns earlier).
* :func:`state_feature_dim_v08` is the frozen 1155-dim width (under the default
  spec) — the value every pre-0.9 net was built with.
* :func:`uses_pre_v09_state_encoding` identifies artifact versions that need
  this shim (exactly 0.8 — 0.6 and 0.7 artifacts also use the 1155-dim state
  via delegating overrides in their own shim modules).
* :class:`PolicyValueNetV08` overrides :meth:`encode_state` and
  :meth:`_state_embed_offsets` to drive the net with its frozen geometry.

State has been 1155-dim since v0.6 (the playability-stripe bump), so v0.6 and
v0.7 shims delegate their ``encode_state`` / ``_state_embed_offsets`` overrides
here rather than duplicating the frozen vector logic.

Per the compatibility policy (``CLAUDE.md``), this shim lives until a MAJOR
``MODEL_VERSION`` bump deletes it together with the pre-0.9 fixture set.
"""

from __future__ import annotations

import numpy as np

from wingspan import decisions, state, version
from wingspan.encode import layout, state_encode
from wingspan.model import core

STATE_ENCODING_COMPACTED_IN = "0.9"
"""The artifact version that compacted the state vector this module undoes."""

# ---------------------------------------------------------------------------
# Frozen v0.6–v0.8 geometry constants.
#
# v0.9 compacts the continuous prefix by:
#   board_summary_me/opp: 18→6 each  (−12×2 = −24)
#   misc_scalars:         4→2         (−2)
#   hand_summary_me:      10→0        (−10)
#   Total continuous shrinkage:       −36
#
# All removed stripes sit BEFORE the card-index block, so card_index,
# hand_multihot, and decision_type all shift left by 36. The hand_summary
# stripe was between board_summary and misc_scalars (before card_index), so
# it still appears at its old position in the frozen 1155-dim vector.

_V08_CARD_INDEX = 562
"""Frozen offset of the card-index block in the 1155-dim v0.6–v0.8 state vector."""

_V08_HAND_MULTIHOT = 595
"""Frozen offset of the hand multi-hot in the 1155-dim v0.6–v0.8 state vector."""

_V08_DECISION_TYPE = 1135
"""Frozen offset of the decision-type one-hot in the 1155-dim v0.6–v0.8 state vector."""

_V08_HAND_SUMMARY = layout.HAND_SUMMARY_OFFSET
"""Frozen offset of the hand-summary stripe in the 1155-dim vector (343, unchanged from live)."""

_V08_HAND_SUMMARY_END = layout.HAND_SUMMARY_OFFSET + layout.HAND_SUMMARY_DIM
"""One-past-end of the hand-summary stripe in the 1155-dim vector (353)."""

_V08_STATE_DIM_BASE = 1155
"""State-vector width for the default spec in artifact eras 0.6–0.8."""


def uses_pre_v09_state_encoding(artifact_version: str) -> bool:
    """Whether ``artifact_version`` uses the pre-0.9 1155-dim state encoding and
    therefore needs :func:`encode_state_v08` to load and play.

    Covers exactly artifact version 0.8. Versions 0.6 and 0.7 also use the
    1155-dim state, but they are caught earlier in the routing chain by
    ``v0_6.uses_v0_6_card_feature_encoding`` and
    ``v0_7.uses_v0_7_becomes_playable_encoding`` respectively — those shims
    delegate their ``encode_state`` / ``_state_embed_offsets`` overrides here."""
    parsed = version.parse_version(artifact_version)
    compacted = version.parse_version(STATE_ENCODING_COMPACTED_IN)
    return (parsed.major, parsed.minor) == (compacted.major, compacted.minor - 1)


def encode_state_v08(
    game_state: state.GameState,
    decision: decisions.Decision[decisions.Choice] | None = None,
    spec: layout.EncodingSpec = layout.DEFAULT_SPEC,
) -> np.ndarray:
    """Produce the complete 1155-dim v0.6–v0.8 state vector.

    Reconstructs the pre-0.9 state by calling the live sub-builders with
    old-behavior flags:

    * ``_summary_board(..., full_stats=True)`` — 18 dims per seat (was 6 after
      v0.9; compacted to ``row_length`` + ``total_eggs`` only).
    * ``_summary_misc_scalars(..., include_goal_pts=True)`` — 4 dims (was 2 after
      v0.9; compacted by dropping the two round-goal VP scalars).
    * ``_round_goals_all_rounds(..., zero_passed_rounds=False)`` — all rounds
      filled regardless of scoring status (v0.9 zeros passed rounds).
    * ``_summary_hand(me)`` — re-inserted at its historical position between
      board_summary_opp and bonus_progress (removed from the live encoder in v0.9).

    Everything else (turn_state, food, per-slot board, hand multi-hot, bonus,
    tray, card-index) uses the live sub-builders unchanged."""
    pov = decision.player_id if decision is not None else game_state.current_player
    me = game_state.players[pov]
    opp = game_state.players[1 - pov] if len(game_state.players) > 1 else me

    # Collect stripes in the same order as the v0.6–v0.8 encode_state, restoring
    # the three compacted stripes and the removed hand_summary.
    parts: list[np.ndarray] = [
        state_encode._summary_turn_state(game_state, me),
        state_encode._summary_food(me),
        state_encode._summary_food(opp),
        state_encode._board_slots_continuous(me),
        state_encode._board_slots_continuous(opp),
        state_encode._summary_board(me, full_stats=True),  # 18 dims (old behavior)
        state_encode._summary_board(opp, full_stats=True),  # 18 dims (old behavior)
        state_encode._summary_hand(me),  # 10 dims (removed in v0.9)
        state_encode._bonus_progress(me),
        state_encode._opp_bonus_count(opp),
        np.array([len(opp.hand) / layout._HAND_SIZE_SCALE], dtype=np.float32),
        state_encode._summary_birdfeeder(game_state),
        state_encode._summary_misc_scalars(
            game_state, me, opp, include_goal_pts=True  # 4 dims (old behavior)
        ),
        state_encode._round_goals_all_rounds(
            game_state, me, zero_passed_rounds=False  # fill all rounds (old behavior)
        ),
        state_encode._card_index_block(me, opp, game_state),
        state_encode._hand_identity(me),
        state_encode._hand_playable(me),
        state_encode._hand_playable_eggs(me),
        state_encode._encode_decision_type(decision, spec),
    ]
    return np.concatenate(parts).astype(np.float32)


def state_embed_offsets_v08() -> core.StateEmbedOffsets:
    """The frozen slice offsets ``_embed_state`` uses for the 1155-dim v0.6–v0.8
    state vector.

    All four offsets are frozen so that ``_embed_state`` reads the correct
    columns from the old 1155-dim vector rather than the live v0.9 ones (which
    sit 36 columns earlier for card_index / hand_multihot / decision_type, and
    at 0/0 for hand_summary since the stripe was removed in v0.9).

    The non-zero ``hand_summary`` / ``hand_summary_end`` pair signals to
    ``_embed_state`` that the stripe is physically present in this frozen vector
    and must be excised from the continuous prefix and fed to the hand encoder
    (rather than derived in-model, which is the live v0.9 path)."""
    return core.StateEmbedOffsets(
        card_index=_V08_CARD_INDEX,
        hand_multihot=_V08_HAND_MULTIHOT,
        decision_type=_V08_DECISION_TYPE,
        hand_summary=_V08_HAND_SUMMARY,
        hand_summary_end=_V08_HAND_SUMMARY_END,
    )


def state_feature_dim_v08(spec: layout.EncodingSpec = layout.DEFAULT_SPEC) -> int:
    """The frozen v0.6–v0.8 state-vector width (1155 under the default spec).

    The spec only affects the decision-type one-hot; the delta from the default
    spec is computed from the live layout (the decision-type dim is unchanged
    between v0.8 and v0.9). The era-dims router
    (``compat.encoding_dims_for_era``) uses this so an era-pinned
    ``TrainConfig`` derives the dims its checkpoints actually carry."""
    decision_type_delta = layout.state_feature_dim(spec) - layout.state_feature_dim()
    return _V08_STATE_DIM_BASE + decision_type_delta


class PolicyValueNetV08(core.PolicyValueNet):
    """A :class:`~wingspan.model.core.PolicyValueNet` frozen to the v0.6–v0.8
    state geometry (1155-dim), for checkpoints written before artifact version 0.9.

    The state trunk was trained against 1155-dim inputs with the full board
    summary (18 dims per seat), 4-dim misc scalars, hand_summary stripe (10
    dims), and all ``round_goals`` slots filled regardless of scoring status.
    This subclass overrides :meth:`encode_state` and :meth:`_state_embed_offsets`
    to reproduce that geometry; choice encoding is identical to the live era.

    Constructed by the version-routing loaders (``PolicyValueNet.from_model_config``,
    ``players.loaders.load_policy_net``) — never by the training pipeline.
    """

    def encode_state(
        self,
        game_state: state.GameState,
        decision: decisions.Decision[decisions.Choice],
    ) -> np.ndarray:
        """Featurize ``game_state`` with the 1155-dim pre-0.9 state geometry."""
        return encode_state_v08(game_state, decision, self.spec)

    def _state_embed_offsets(self) -> core.StateEmbedOffsets:
        """Frozen slice offsets for the 1155-dim v0.6–v0.8 state vector.

        Without this override ``_embed_state`` would slice at the v0.9 live
        offsets (36 columns too far left for card_index / hand_multihot /
        decision_type, and would attempt to derive hand_summary in-model
        instead of reading the stripe from the frozen vector)."""
        return state_embed_offsets_v08()


###### PRIVATE #######


def _assert_live_layout_contract() -> None:
    """Import-time pins for the invariants the v0.8 shim relies on.

    The frozen offsets are correct only while:

    1. ``HAND_SUMMARY_OFFSET`` is 343 (the stripe was removed in v0.9 but the
       constant is kept as a frozen literal for shim use — verified here to
       catch any accidental change).
    2. ``HAND_SUMMARY_DIM`` is 10 (the stripe width is unchanged).
    3. The card-index block starts at 526 in the live v0.9 layout, i.e. 36
       columns earlier than the frozen ``_V08_CARD_INDEX = 562``.
    """
    assert layout.HAND_SUMMARY_OFFSET == _V08_HAND_SUMMARY, (
        f"v0.8 shim freezes HAND_SUMMARY_OFFSET at {_V08_HAND_SUMMARY}, "
        f"but live value is {layout.HAND_SUMMARY_OFFSET}; update the shim"
    )
    assert layout.HAND_SUMMARY_DIM == 10, (
        f"v0.8 shim freezes HAND_SUMMARY_DIM at 10, "
        f"but live value is {layout.HAND_SUMMARY_DIM}; update the shim"
    )
    # The frozen card_index offset (562) must be > the live offset (526) by
    # exactly the 36 dims that were removed in v0.9.
    live_card_index = layout.OFF_CARD_INDEX
    assert _V08_CARD_INDEX == live_card_index + 36, (
        f"v0.8 shim expects _V08_CARD_INDEX ({_V08_CARD_INDEX}) = "
        f"live OFF_CARD_INDEX ({live_card_index}) + 36; "
        "the compaction delta has changed — update the shim"
    )
    assert _V08_STATE_DIM_BASE == layout.state_feature_dim() + 36, (
        f"v0.8 shim expects _V08_STATE_DIM_BASE ({_V08_STATE_DIM_BASE}) = "
        f"live state_feature_dim() ({layout.state_feature_dim()}) + 36; "
        "the compaction delta has changed — update the shim"
    )


_assert_live_layout_contract()
