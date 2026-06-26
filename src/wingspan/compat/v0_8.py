# pyright: reportPrivateUsage=false
# (this shim reads the live layout's package-private stripe offsets to copy
# era-identical blocks from the live encoding, calls state_encode private
# sub-builders with old-behavior flags, and overrides the net's private
# builders — a deliberate compat coupling, pinned by the import-time
# layout-contract assertions below)
"""Frozen v0.6–v0.8 state and choice encoding: the shim that keeps pre-0.9 artifacts playable.

Artifact version 0.9 brought two simultaneous FRESH changes:

**State compaction (1155 → 1119 dims):**

* ``misc_scalars`` 4→2 dims (dropped round-goal VP scalars).
* ``board_summary_me`` / ``board_summary_opp`` 18→6 dims each (kept only
  ``row_length`` and ``total_eggs`` per habitat).
* ``hand_summary_me`` removed (10 dims; now derived in-model).
* Already-scored ``round_goals`` slots zeroed (width unchanged at 92 dims).

**Choice board simplification (``choice_dim`` 395 → 328):**

* ``board_target`` compressed 120 → 60 dims (4 scalars/slot instead of 8 —
  drops per-type cached food in favour of ``cached_total``).
* ``board_idx`` (15-slot embedded block) removed, replaced by ``board_hab``
  (3-dim habitat one-hot) and ``board_col`` (5-dim column one-hot).
* ``bird_id`` now also carries the targeted occupant on board-target rows.

This module keeps all pre-0.9 artifacts loadable:

State path:
* :func:`encode_state_v08` reproduces the 1155-dim state vector (old flags).
* :func:`state_embed_offsets_v08` returns the frozen slice offsets.
* :func:`state_feature_dim_v08` is the frozen 1155-dim width.
* :func:`uses_pre_v09_state_encoding` identifies artifact versions that need
  this path (exactly 0.8; 0.6/0.7 delegate here from their own shims).

Choice path:
* :func:`encode_choices_v08` rebuilds the v0.8 choice matrix from a live
  encoding plus game state (board_target 120, board_idx 15 restored).
* :func:`choice_feature_dim_v08` / :func:`choice_input_dim_v08` /
  :func:`choice_passthrough_dim_v08` are the frozen choice-width formulas.
* :func:`uses_v0_8_choice_encoding` identifies versions that need this path
  (0.1–0.8 exactly; pre-0.1 are caught by ``v0_0``).

:class:`PolicyValueNetV08` overrides all four axes simultaneously so a single
subclass covers every pre-0.9 artifact regardless of the state × choice pair it
was trained against.

State has been 1155-dim since v0.6; v0.6 and v0.7 shims delegate their
``encode_state`` / ``_state_embed_offsets`` overrides here. This is also the
first board-geometry change since v0.1; all earlier eras (v0.1–v0.7) shared
the same board_target 120 + board_idx 15 geometry, so every prior shim
re-routes its ``encode_choices`` through :func:`encode_choices_v08` and its
choice encoder through :func:`choice_input_dim_v08`.

Per the compatibility policy (``CLAUDE.md``), this shim lives until a MAJOR
``MODEL_VERSION`` bump deletes it together with the pre-0.9 fixture sets.
"""

from __future__ import annotations

import typing

import numpy as np
import torch

from wingspan import architecture, cards, decisions, encode, state, version
from wingspan.encode import choice_encode, layout, state_encode
from wingspan.model import core, mlp

BOARD_ENCODING_CHANGED_IN = "0.9"
"""The artifact version whose choice board encoding this module undoes."""

STATE_ENCODING_COMPACTED_IN = "0.9"
"""The artifact version that compacted the state vector this module undoes."""

# ---------------------------------------------------------------------------
# Frozen v0.8 choice geometry — the stripe chain every 0.1–0.8 checkpoint was
# trained against. Board slot format: 8 scalars (lay, pay, cached×5, tucked).

_BT_SLOT_SCALARS_V08 = 8
_BT_LAY_EGGS_V08 = 0
_BT_PAY_EGGS_V08 = 1
_BT_CACHED_START_V08 = 2  # per-type cached food: 5 values at offsets 2..6
_BT_TUCKED_V08 = 7
_BOARD_TARGET_DIM_V08 = layout._SLOTS_PER_BOARD * _BT_SLOT_SCALARS_V08  # 120
_BOARD_IDX_SLOTS_V08 = layout._SLOTS_PER_BOARD  # 15
_BECOMES_PLAYABLE_DIM_V08 = 180  # cards.n_birds() — frozen

_OFF_KIND_V08 = 0
_OFF_GAIN_FOOD_V08 = _OFF_KIND_V08 + layout._KIND_DIM  # 6
_OFF_PAY_V08 = _OFF_GAIN_FOOD_V08 + layout._GAIN_FOOD_DIM  # 13
_OFF_BOARD_V08 = _OFF_PAY_V08 + layout._PAY_FOOD_DIM  # 18
_OFF_MAIN_ACTION_V08 = _OFF_BOARD_V08 + _BOARD_TARGET_DIM_V08  # 138
_OFF_SPECIAL_V08 = _OFF_MAIN_ACTION_V08 + layout._MAIN_ACTION_DIM  # 142
_OFF_EXCHANGE_V08 = _OFF_SPECIAL_V08 + layout._SPECIAL_DIM  # 144
_OFF_BOARD_IDX_V08 = _OFF_EXCHANGE_V08 + layout._EXCHANGE_DIM  # 157
_OFF_BIRD_ID_V08 = _OFF_BOARD_IDX_V08 + _BOARD_IDX_SLOTS_V08  # 172
_OFF_BONUS_ID_V08 = _OFF_BIRD_ID_V08 + layout._CHOICE_BIRD_ID_DIM  # 173
_OFF_BONUS_DELTA_V08 = _OFF_BONUS_ID_V08 + layout._BONUS_ID_DIM  # 199
_OFF_GOAL_DELTA_V08 = _OFF_BONUS_DELTA_V08 + layout._BONUS_DELTA_DIM  # 202
_OFF_BONUS_VALUE_V08 = _OFF_GOAL_DELTA_V08 + layout._GOAL_DELTA_DIM  # 210
_OFF_BECOMES_PLAYABLE_V08 = _OFF_BONUS_VALUE_V08 + layout._BONUS_VALUE_DIM  # 215
_CHOICE_BASE_DIM_V08 = _OFF_BECOMES_PLAYABLE_V08 + _BECOMES_PLAYABLE_DIM_V08  # 395
_OFF_SETUP_V08 = _CHOICE_BASE_DIM_V08  # 395
_OFF_KEPT_MULTIHOT_V08 = _OFF_SETUP_V08 + layout._SETUP_DIM  # 399
_KEPT_MULTIHOT_DIM_V08 = _BECOMES_PLAYABLE_DIM_V08  # 180

# The extra dims that v0.8 rows carry vs live v0.9 rows (board delta):
# board_target +60 (120→60 collapse) plus board_idx +15 minus board_hab -3 minus board_col -5
_BOARD_DELTA = (_BOARD_TARGET_DIM_V08 - layout._BOARD_TARGET_DIM) + (
    _BOARD_IDX_SLOTS_V08 - layout._BOARD_HAB_DIM - layout._BOARD_COL_DIM
)  # (120-60) + (15-3-5) = 60 + 7 = 67

# ---------------------------------------------------------------------------
# Frozen v0.6–v0.8 state geometry constants.
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


# ---------------------------------------------------------------------------
# Choice-path predicates and width formulas


def uses_v0_8_choice_encoding(artifact_version: str) -> bool:
    """Whether ``artifact_version`` uses the pre-0.9 board choice encoding and
    therefore needs this module's frozen geometry to load and play.

    Covers 0.1–0.8 exactly (pre-0.1 artifacts use ``v0_0``'s own geometry;
    0.9+ use the live board-free encoder)."""
    parsed = version.parse_version(artifact_version)
    changed = version.parse_version(BOARD_ENCODING_CHANGED_IN)
    return (0, 1) <= (parsed.major, parsed.minor) < (changed.major, changed.minor)


def choice_feature_dim_v08(
    spec: layout.EncodingSpec = layout.DEFAULT_SPEC,
    *,
    has_becomes_playable: bool = True,
) -> int:
    """Width of one v0.8 choice row for ``spec``.

    The frozen row is wider than the live row by ``_BOARD_DELTA`` (67 dims):
    the per-type-cached board_target (120 vs 60) plus board_idx (15) minus the
    new board_hab (3) and board_col (5). The live formula drives the base;
    the delta adjusts it to the frozen geometry."""
    live_dim = layout.choice_feature_dim(spec)
    if not has_becomes_playable:
        live_dim -= _BECOMES_PLAYABLE_DIM_V08
    return live_dim + _BOARD_DELTA


def choice_input_dim_v08(
    choice_dim: int,
    card_embed_dim: int,
    *,
    include_setup: bool = False,
    has_becomes_playable: bool = True,
) -> int:
    """The v0.8 choice encoder's first-``Linear`` input width.

    The frozen formula restores the board-index embedding that v0.9 removed:
    the 15-slot board-index block becomes 15 card embeddings, plus one more for
    the candidate bird-index column. Everything else follows the same logic as
    the live ``layout.choice_input_dim``."""
    base = (
        choice_dim
        - layout._CHOICE_BIRD_ID_DIM  # candidate index column → one embedding
        - _BOARD_IDX_SLOTS_V08  # board-index block → per-slot embeddings
        + card_embed_dim  # candidate embedding
        + _BOARD_IDX_SLOTS_V08 * card_embed_dim  # board slot embeddings
    )
    if has_becomes_playable:
        base += card_embed_dim - _BECOMES_PLAYABLE_DIM_V08  # multi-hot → one embedding
    if include_setup:
        base += card_embed_dim - _KEPT_MULTIHOT_DIM_V08  # multi-hot → one embedding
    return base


def choice_passthrough_dim_v08(
    choice_dim: int,
    *,
    include_setup: bool = False,
    has_becomes_playable: bool = True,
) -> int:
    """The v0.8 choice columns that pass straight through to the encoder.

    The frozen formula subtracts both the candidate column and the 15-slot
    board-index block (both become embeddings) plus the multi-hot stripes."""
    extra = choice_dim - layout._CHOICE_BIRD_ID_DIM - _BOARD_IDX_SLOTS_V08
    if has_becomes_playable:
        extra -= _BECOMES_PLAYABLE_DIM_V08
    if include_setup:
        extra -= _KEPT_MULTIHOT_DIM_V08
    return extra


# ---------------------------------------------------------------------------
# State-path predicates and width formulas


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


def state_feature_dim_v08(spec: layout.EncodingSpec = layout.DEFAULT_SPEC) -> int:
    """The frozen v0.6–v0.8 state-vector width (1155 under the default spec).

    The spec only affects the decision-type one-hot; the delta from the default
    spec is computed from the live layout (the decision-type dim is unchanged
    between v0.8 and v0.9). The era-dims router
    (``compat.encoding_dims_for_era``) uses this so an era-pinned
    ``TrainConfig`` derives the dims its checkpoints actually carry."""
    decision_type_delta = layout.state_feature_dim(spec) - layout.state_feature_dim()
    return _V08_STATE_DIM_BASE + decision_type_delta


# ---------------------------------------------------------------------------
# Main encoding functions


def encode_choices_v08(
    decision: decisions.Decision[typing.Any],
    game_state: state.GameState,
    spec: layout.EncodingSpec = layout.DEFAULT_SPEC,
    *,
    has_becomes_playable: bool = True,
    food_playable_ignores_eggs: bool = True,
) -> np.ndarray:
    """Featurize every choice in ``decision`` as v0.8 rows.

    Calls the live (v0.9) encoder for era-identical content, then rebuilds the
    frozen board geometry (board_target 120, board_idx 15) from the game state
    directly. Stripes that are identical in both eras are copied at their frozen
    v0.8 offsets; the board region is filled from game_state.board.
    """
    live_rows = choice_encode.encode_choices(
        decision,
        game_state,
        spec,
        has_becomes_playable=has_becomes_playable,
        food_playable_ignores_eggs=food_playable_ignores_eggs,
    )
    n_rows = live_rows.shape[0]
    row_dim = choice_feature_dim_v08(spec, has_becomes_playable=has_becomes_playable)
    rows = np.zeros((n_rows, row_dim), dtype=np.float32)
    player = game_state.players[decision.player_id]

    # Copy era-identical prefix: kind + gain_food + pay_food (offsets 0..18).
    rows[:, _OFF_KIND_V08:_OFF_BOARD_V08] = live_rows[
        :, layout._OFF_KIND : layout._OFF_BOARD
    ]

    # Rebuild board_target (120 dims, 8 scalars/slot) from game_state board.
    _rebuild_board_target_v08(rows, live_rows, player, decision)

    # Copy main_action + special + exchange (live 78..97 → v08 138..157).
    rows[:, _OFF_MAIN_ACTION_V08:_OFF_BOARD_IDX_V08] = live_rows[
        :, layout._OFF_MAIN_ACTION : layout._OFF_BOARD_HAB
    ]

    # Rebuild board_idx (15 dims) from game_state board and landing-slot info.
    _rebuild_board_idx_v08(rows, live_rows, player)

    # bird_id: copy from live; zero it for board-target rows (v0.8 left it zero there).
    rows[:, _OFF_BIRD_ID_V08] = live_rows[:, layout._OFF_BIRD_ID]
    board_target_mask = (
        live_rows[:, layout._OFF_KIND + layout._KIND_BOARD_TARGET] == 1.0
    )
    rows[board_target_mask, _OFF_BIRD_ID_V08] = 0.0

    # Copy bonus_id through bonus_value (live 106..148 → v08 173..215).
    rows[:, _OFF_BONUS_ID_V08:_OFF_BECOMES_PLAYABLE_V08] = live_rows[
        :, layout._OFF_BONUS_ID : layout.CHOICE_BECOMES_PLAYABLE_OFFSET
    ]

    if has_becomes_playable:
        # Copy becomes_playable stripe (live 148..328 → v08 215..395).
        rows[:, _OFF_BECOMES_PLAYABLE_V08:_CHOICE_BASE_DIM_V08] = live_rows[
            :, layout.CHOICE_BECOMES_PLAYABLE_OFFSET : layout.CHOICE_FEATURE_DIM
        ]

    if spec.include_setup:
        # Copy setup_agg + kept_multihot (live 328..332 + 332..512 → v08 395..399 + 399..579).
        live_setup_end = layout._OFF_SETUP + layout._SETUP_DIM
        rows[:, _OFF_SETUP_V08:_OFF_KEPT_MULTIHOT_V08] = live_rows[
            :, layout._OFF_SETUP : live_setup_end
        ]
        rows[
            :, _OFF_KEPT_MULTIHOT_V08 : _OFF_KEPT_MULTIHOT_V08 + _KEPT_MULTIHOT_DIM_V08
        ] = live_rows[
            :,
            layout._OFF_KEPT_MULTIHOT : layout._OFF_KEPT_MULTIHOT
            + layout._KEPT_MULTIHOT_DIM,
        ]

    return rows


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


class PolicyValueNetV08(core.PolicyValueNet):
    """A :class:`~wingspan.model.core.PolicyValueNet` frozen to the v0.6–v0.8
    state and choice geometry, for checkpoints written before artifact version 0.9.

    v0.9 made two simultaneous FRESH changes — state compaction (1155→1119 dims)
    and choice board simplification (board_target 120→60, board_idx removed). This
    subclass restores both axes:

    * :meth:`encode_state` / :meth:`_state_embed_offsets` restore the 1155-dim
      state vector (full board summary, 4-dim misc, hand_summary stripe present,
      all round_goals filled).
    * :meth:`encode_choices`, :meth:`_build_choice_encoder`,
      :meth:`_embed_choices`, :meth:`_choice_embed_offsets` restore the 395-dim
      choice row (board_target 120, board_idx 15, bird_id zero on board-target
      rows).

    ``state_dim`` defaults to 1155 and ``choice_dim`` defaults to 395 so
    loaders that omit either get the correct frozen sizes automatically.

    Constructed by the version-routing loaders (``PolicyValueNet.from_model_config``,
    ``players.loaders.load_policy_net``) — never by the training pipeline.
    """

    def __init__(
        self,
        *,
        state_dim: int | None = None,
        choice_dim: int | None = None,
        num_families: int | None = None,
        arch: architecture.ModelArchitecture | None = None,
        spec: encode.EncodingSpec = encode.DEFAULT_SPEC,
    ) -> None:
        if state_dim is None:
            state_dim = state_feature_dim_v08(spec)
        if choice_dim is None:
            choice_dim = choice_feature_dim_v08(spec)
        super().__init__(
            state_dim=state_dim,
            choice_dim=choice_dim,
            num_families=num_families,
            arch=arch,
            spec=spec,
        )

    # ---- State overrides ----

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

    # ---- Choice overrides ----

    def encode_choices(
        self,
        decision: decisions.Decision[decisions.Choice],
        game_state: state.GameState,
    ) -> np.ndarray:
        """v0.8 rows for ``decision`` — frozen board geometry, never the live encoder."""
        return encode_choices_v08(decision, game_state, self.spec)

    def _build_choice_encoder(
        self, choice_dim: int, arch: architecture.ModelArchitecture
    ) -> None:
        """Register ``choice_encoder`` at the v0.8 input width (with board embedding)."""
        self.choice_encoder, _ = mlp.build_body(
            choice_input_dim_v08(
                choice_dim, arch.card_embed_dim, include_setup=self.include_setup
            ),
            arch.choice_layers,
            between_activation=arch.choice_between_activation_resolved,
            final_activation=arch.choice_final_activation_resolved,
            dropout=arch.dropout,
            layernorm=arch.layernorm,
        )

    def _embed_choices(
        self, choices: torch.Tensor, card_table: torch.Tensor
    ) -> torch.Tensor:
        """Frozen v0.8 board-bearing embedding. Delegates to the free function
        :func:`embed_choices_v08` so earlier-era shims can call it without a
        ``PolicyValueNetV08`` self-type."""
        return embed_choices_v08(self, choices, card_table)

    def _choice_embed_offsets(self) -> core.ChoiceEmbedOffsets:
        """Slice offsets for v0.8 rows: bird_id at frozen position 172.

        ``becomes_playable`` uses the frozen v0.8 offset (215) encoded via the
        live ChoiceEmbedOffsets; ``_embed_choices`` reads board offsets from the
        frozen literals directly."""
        return core.ChoiceEmbedOffsets(
            bird_id=_OFF_BIRD_ID_V08,
            becomes_playable=_OFF_BECOMES_PLAYABLE_V08,
            kept_multihot=_OFF_KEPT_MULTIHOT_V08 if self.include_setup else None,
        )


###### PRIVATE #######


def embed_choices_v08(
    net: core.PolicyValueNet,
    choices: torch.Tensor,
    card_table: torch.Tensor,
) -> torch.Tensor:
    """Frozen v0.8 board-bearing choice embedding.

    Accepts any :class:`~wingspan.model.core.PolicyValueNet` so earlier-era shims
    can call this without a ``PolicyValueNetV08`` self-type.  ``net`` is queried
    only via :meth:`~wingspan.model.core.PolicyValueNet._choice_embed_offsets` to
    decide which multi-hot stripes to collapse; all offsets within the v0.8 row
    use the frozen v0.8 constants directly.

    The 15-slot board-index block and the candidate bird-index column both
    become shared-embedding lookups; ``becomes_playable`` (when present) and
    the trailing ``kept_multihot`` (when ``include_setup``) are summed into
    one embedding each. Everything else passes through."""
    off_bird = _OFF_BIRD_ID_V08
    end_bird = off_bird + layout._CHOICE_BIRD_ID_DIM  # = 173

    board_idx = (
        choices[..., _OFF_BOARD_IDX_V08:off_bird]
        .long()
        .clamp_(0, encode.HAND_MULTIHOT_DIM)
    )
    cand_idx = (
        choices[..., off_bird].long().clamp_(0, encode.HAND_MULTIHOT_DIM)
    )  # (B, K)
    cand_mask = (cand_idx > 0).unsqueeze(-1).to(card_table.dtype)
    cand_emb = card_table[cand_idx] * cand_mask
    board_emb = card_table[board_idx].reshape(*board_idx.shape[:-1], -1)

    # becomes_playable offset in v0.8 rows; None = pre-0.6 (no stripe).
    offsets = net._choice_embed_offsets()
    off_becomes: int | None = (
        _OFF_BECOMES_PLAYABLE_V08 if offsets.becomes_playable is not None else None
    )
    off_kept: int | None = offsets.kept_multihot

    if off_becomes is not None and off_kept is not None:
        off_setup = off_becomes + _BECOMES_PLAYABLE_DIM_V08
        becomes_emb = choices[..., off_becomes:off_setup] @ card_table[1:]
        kept_emb = choices[..., off_kept:] @ card_table[1:]
        rest = torch.cat(
            [
                choices[..., :_OFF_BOARD_IDX_V08],
                choices[..., end_bird:off_becomes],
                choices[..., off_setup:off_kept],
            ],
            dim=-1,
        )
        return torch.cat([rest, cand_emb, board_emb, becomes_emb, kept_emb], dim=-1)
    elif off_becomes is not None:
        becomes_emb = choices[..., off_becomes:] @ card_table[1:]
        rest = torch.cat(
            [choices[..., :_OFF_BOARD_IDX_V08], choices[..., end_bird:off_becomes]],
            dim=-1,
        )
        return torch.cat([rest, cand_emb, board_emb, becomes_emb], dim=-1)
    elif off_kept is not None:
        kept_emb = choices[..., off_kept:] @ card_table[1:]
        rest = torch.cat(
            [choices[..., :_OFF_BOARD_IDX_V08], choices[..., end_bird:off_kept]],
            dim=-1,
        )
        return torch.cat([rest, cand_emb, board_emb, kept_emb], dim=-1)
    else:
        rest = torch.cat(
            [choices[..., :_OFF_BOARD_IDX_V08], choices[..., end_bird:]], dim=-1
        )
        return torch.cat([rest, cand_emb, board_emb], dim=-1)


def _assert_live_layout_contract() -> None:
    """Import-time pins for all invariants this shim relies on.

    Choice-path invariants: the block-copy offsets from the live encoding to
    the frozen v0.8 choice row depend on the live layout constants staying put.

    State-path invariants: the frozen state offsets (_V08_CARD_INDEX etc.)
    must equal the live offsets plus the 36-dim compaction delta."""

    # ---- Choice-path pins ----

    # kind + gain_food + pay_food: identical prefix in both eras.
    assert layout._OFF_KIND == _OFF_KIND_V08 == 0
    assert layout._OFF_GAIN_FOOD == _OFF_GAIN_FOOD_V08
    assert layout._OFF_PAY == _OFF_PAY_V08
    assert layout._OFF_BOARD == _OFF_BOARD_V08, (
        f"v0.8 shim expects board_target at offset {_OFF_BOARD_V08}; "
        f"live is at {layout._OFF_BOARD}"
    )
    # Board target: live is 60 dims, v0.8 is 120; delta is 60.
    assert layout._BOARD_TARGET_DIM == layout._SLOTS_PER_BOARD * 4
    # main_action + special + exchange are a contiguous 19-dim run in both eras.
    live_run = layout._OFF_BOARD_HAB - layout._OFF_MAIN_ACTION  # 97-78=19
    v08_run = _OFF_BOARD_IDX_V08 - _OFF_MAIN_ACTION_V08  # 157-138=19
    assert live_run == v08_run, (
        f"v0.8 shim expects main_action..exchange to be {v08_run} dims; "
        f"live has {live_run}"
    )
    # bird_id: single column in both eras.
    assert layout._CHOICE_BIRD_ID_DIM == 1
    # bonus_id through bonus_value: 42 dims in both eras.
    live_bonus_run = (
        layout.CHOICE_BECOMES_PLAYABLE_OFFSET - layout._OFF_BONUS_ID
    )  # 148-106=42
    v08_bonus_run = _OFF_BECOMES_PLAYABLE_V08 - _OFF_BONUS_ID_V08  # 215-173=42
    assert live_bonus_run == v08_bonus_run, (
        f"v0.8 shim expects bonus_id..bonus_value to be {v08_bonus_run} dims; "
        f"live has {live_bonus_run}"
    )
    # becomes_playable: 180 dims in both eras.
    assert _BECOMES_PLAYABLE_DIM_V08 == layout.CHOICE_BECOMES_PLAYABLE_DIM == 180
    # Board delta must equal the difference between v0.8 base dim and live base dim.
    live_base = layout.CHOICE_FEATURE_DIM
    v08_base = _CHOICE_BASE_DIM_V08
    assert v08_base - live_base == _BOARD_DELTA, (
        f"v0.8 shim's _BOARD_DELTA ({_BOARD_DELTA}) does not match "
        f"_CHOICE_BASE_DIM_V08 - live ({v08_base} - {live_base} = {v08_base - live_base}); "
        "update the shim"
    )

    # ---- State-path pins ----

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


def _rebuild_board_target_v08(
    rows: np.ndarray,
    live_rows: np.ndarray,
    player: state.Player,
    decision: decisions.Decision[typing.Any],
) -> None:
    """Fill the board_target stripe (120 dims, 8 scalars/slot) for all rows.

    Copies the lay_eggs and pay_eggs flags from the live encoding (they're at
    the same relative position within each slot). Fills per-type cached food and
    tucked from game_state. Non-targeted slots have no flags; targeted-slot flags
    already ride the live encoding at the live (4-scalar) offsets."""
    for hab_idx, habitat in enumerate(cards.ALL_HABITATS):
        row_data = player.board[habitat]
        for col in range(state.ROW_SLOTS):
            slot_idx = hab_idx * state.ROW_SLOTS + col
            # Lay/pay flags: live encodes them at live_scalar_base + 0 and + 1.
            live_scalar_base = layout._OFF_BOARD + slot_idx * layout._BT_SLOT_SCALARS
            v08_scalar_base = _OFF_BOARD_V08 + slot_idx * _BT_SLOT_SCALARS_V08
            rows[:, v08_scalar_base + _BT_LAY_EGGS_V08] = live_rows[
                :, live_scalar_base + layout._BT_LAY_EGGS
            ]
            rows[:, v08_scalar_base + _BT_PAY_EGGS_V08] = live_rows[
                :, live_scalar_base + layout._BT_PAY_EGGS
            ]
            if col >= len(row_data):
                continue
            pb = row_data[col]
            for food_idx, food in enumerate(cards.ALL_FOODS):
                rows[:, v08_scalar_base + _BT_CACHED_START_V08 + food_idx] = (
                    pb.cached_food[food] / layout._CACHED_FOOD_SCALE
                )
            rows[:, v08_scalar_base + _BT_TUCKED_V08] = (
                pb.tucked_cards / layout._TUCKED_SCALE
            )


def _rebuild_board_idx_v08(
    rows: np.ndarray,
    live_rows: np.ndarray,
    player: state.Player,
) -> None:
    """Fill the board_idx stripe (15 dims) for each row.

    Board-target rows: all 15 slots filled with occupant indices from game_state.
    All other rows with a board_hab/board_col signal (placements, move-bird):
    one slot filled with the candidate's bird_id from the live encoding.
    All other rows: zeros (left from allocation)."""
    # Identify board-target rows (kind[4] == 1).
    board_target_mask = (
        live_rows[:, layout._OFF_KIND + layout._KIND_BOARD_TARGET] == 1.0
    )

    # For board-target rows: fill all 15 slots from game_state board.
    if board_target_mask.any():
        for hab_idx, habitat in enumerate(cards.ALL_HABITATS):
            row_data = player.board[habitat]
            for col in range(state.ROW_SLOTS):
                slot_idx = hab_idx * state.ROW_SLOTS + col
                if col >= len(row_data):
                    continue
                bird_idx = cards.bird_index(row_data[col].bird) + 1
                rows[board_target_mask, _OFF_BOARD_IDX_V08 + slot_idx] = bird_idx

    # For other rows: fill the one slot indicated by board_hab/board_col.
    non_bt_mask = ~board_target_mask
    hab_one_hot = live_rows[
        :, layout._OFF_BOARD_HAB : layout._OFF_BOARD_HAB + layout._BOARD_HAB_DIM
    ]
    col_one_hot = live_rows[
        :, layout._OFF_BOARD_COL : layout._OFF_BOARD_COL + layout._BOARD_COL_DIM
    ]
    has_slot_signal = (hab_one_hot.max(axis=1) > 0) & non_bt_mask

    if has_slot_signal.any():
        hab_indices = hab_one_hot.argmax(axis=1)
        col_indices = col_one_hot.argmax(axis=1)
        bird_ids = live_rows[:, layout._OFF_BIRD_ID].astype(np.int64)
        for row_idx in np.nonzero(has_slot_signal)[0]:
            slot_idx = int(hab_indices[row_idx]) * state.ROW_SLOTS + int(
                col_indices[row_idx]
            )
            rows[row_idx, _OFF_BOARD_IDX_V08 + slot_idx] = int(bird_ids[row_idx])


_assert_live_layout_contract()
