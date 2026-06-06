# pyright: reportPrivateUsage=false
# (the live->v0.0 transform reads the live layout's package-private stripe
# offsets and overrides the net's private builders -- a deliberate compat
# coupling, pinned by the import-time layout-contract assertions below)
"""Frozen v0.0 choice encoding: the shim that keeps pre-0.1 artifacts playable.

Artifact version 0.1 reshaped the choice vector: placements gained a
landing-slot mark in the board-index block (replacing the 3-dim habitat
one-hot), ``bird_id`` collapsed from a 180-wide one-hot to a single integer
index column, and the setup keep multi-hot moved off ``bird_id`` onto a
dedicated trailing ``kept_multihot`` stripe. Nets trained before that change
consume 397-dim rows (401 with ``include_setup``) in the old geometry, so this
module regenerates them:

* :func:`encode_choices` re-encodes a decision with the live encoder and
  rearranges the result into the frozen v0.0 layout (block copies for the
  stripes whose contents are unchanged between the eras, regeneration from the
  decision itself for the reshaped placement / card-identity stripes).
* :class:`PolicyValueNetV00` overrides the two places the net itself bakes in
  the geometry: the choice encoder's input width (the pre-0.1
  ``choice_input_dim`` formula) and the ``_embed_choices`` card-region slicing.

The v0.0 stripe offsets below are frozen literals — they must never track the
live ``encode.layout`` (that is the point). The transform's *input* side reads
the live offsets so it moves with the live encoder, and
``_assert_live_layout_contract`` pins the block-copy invariants it relies on:
an incompatible future layout change fails this module's import (and the
compat fixture tests) instead of silently mis-slicing. State encoding, the
family-head order, and the setup model are unchanged between 0.0 and 0.1, so
only the choice path is shimmed.

Per the compatibility policy (``CLAUDE.md``), this shim lives until a MAJOR
``MODEL_VERSION`` bump deletes it together with the v0.0 fixture set.
"""

from __future__ import annotations

import typing

import numpy as np
import torch

from wingspan import architecture, cards, decisions, encode, version
from wingspan.encode import layout
from wingspan.model import core, mlp

if typing.TYPE_CHECKING:
    from wingspan import state

CHOICE_ENCODING_CHANGED_IN = "0.1"
"""The artifact version whose choice-vector reshape this module undoes."""

# ---------------------------------------------------------------------------
# Frozen v0.0 choice geometry — literal copies of the stripe chain every
# pre-0.1 checkpoint was trained against (commit 96bd70f, the v0.0 fixture's
# source). Catalog-derived sizes are frozen as literals deliberately: the
# v0.0 row format must not move even if the live catalog or layout does.

_KIND_DIM = 6
_GAIN_FOOD_DIM = 7
_HABITAT_DIM = 3  # the habitat one-hot the 0.1 reshape removed
_PAY_FOOD_DIM = 5
_BOARD_TARGET_DIM = 120
_MAIN_ACTION_DIM = 4
_SPECIAL_DIM = 2
_EXCHANGE_DIM = 13
_BOARD_IDX_SLOTS = 15
_BIRD_ID_DIM = 180  # cards.n_birds() at v0.0 — one-hot, or setup-keep multi-hot
_BONUS_ID_DIM = 26  # cards.n_bonus_cards() at v0.0
_BONUS_DELTA_DIM = 3
_GOAL_DELTA_DIM = 8
_BONUS_VALUE_DIM = 5
_SETUP_DIM = 4

_OFF_KIND = 0
_OFF_GAIN_FOOD = _OFF_KIND + _KIND_DIM
_OFF_HAB = _OFF_GAIN_FOOD + _GAIN_FOOD_DIM
_OFF_PAY = _OFF_HAB + _HABITAT_DIM
_OFF_BOARD = _OFF_PAY + _PAY_FOOD_DIM
_OFF_MAIN_ACTION = _OFF_BOARD + _BOARD_TARGET_DIM
_OFF_SPECIAL = _OFF_MAIN_ACTION + _MAIN_ACTION_DIM
_OFF_EXCHANGE = _OFF_SPECIAL + _SPECIAL_DIM
_OFF_BOARD_IDX = _OFF_EXCHANGE + _EXCHANGE_DIM
_OFF_BIRD_ID = _OFF_BOARD_IDX + _BOARD_IDX_SLOTS
_OFF_BONUS_ID = _OFF_BIRD_ID + _BIRD_ID_DIM
_OFF_BONUS_DELTA = _OFF_BONUS_ID + _BONUS_ID_DIM
_OFF_GOAL_DELTA = _OFF_BONUS_DELTA + _BONUS_DELTA_DIM
_OFF_BONUS_VALUE = _OFF_GOAL_DELTA + _GOAL_DELTA_DIM
_OFF_SETUP = _OFF_BONUS_VALUE + _BONUS_VALUE_DIM  # trailing; iff include_setup
_CHOICE_BASE_DIM = _OFF_SETUP  # 397 — the v0.0 fixture's choice_dim


def uses_v0_0_choice_encoding(artifact_version: str) -> bool:
    """Whether ``artifact_version`` predates the 0.1 choice-vector reshape and
    therefore needs this module's frozen geometry to load and play."""
    parsed = version.parse_version(artifact_version)
    changed = version.parse_version(CHOICE_ENCODING_CHANGED_IN)
    return (parsed.major, parsed.minor) < (changed.major, changed.minor)


def choice_feature_dim(spec: encode.EncodingSpec = encode.DEFAULT_SPEC) -> int:
    """Width of one v0.0 choice row for ``spec``: the frozen base plus the
    trailing ``setup_agg`` stripe when ``include_setup`` (the v0.0 layout had
    no separate kept-multi-hot stripe — keeps rode the ``bird_id`` stripe)."""
    return _CHOICE_BASE_DIM + (_SETUP_DIM if spec.include_setup else 0)


def choice_input_dim(choice_dim: int, card_embed_dim: int) -> int:
    """The v0.0 choice encoder's first-``Linear`` input width — the pre-0.1
    formula, with no ``include_setup`` axis: the candidate's 180-wide bird
    one-hot (doubling as the setup keep multi-hot) and the 15-slot board-index
    block are replaced by their shared-embedding lookups."""
    return (
        choice_dim
        - _BIRD_ID_DIM
        - _BOARD_IDX_SLOTS
        + card_embed_dim
        + _BOARD_IDX_SLOTS * card_embed_dim
    )


def encode_choices(
    decision: layout._AnyDecision,
    game_state: state.GameState,
    spec: encode.EncodingSpec = encode.DEFAULT_SPEC,
) -> np.ndarray:
    """Featurize every choice in ``decision`` as v0.0 rows.

    Returns a float32 array of shape ``(n_choices, choice_feature_dim(spec))``
    in the frozen v0.0 geometry. Every stripe both eras share carries unchanged
    contents (the 0.1 reshape moved offsets and re-encoded only the placement
    and card-identity stripes), so the bulk of each row is vectorized block
    copies of the live encoding; the reshaped stripes are regenerated from the
    decision itself.
    """
    live_rows = encode.encode_choices(decision, game_state, spec)
    rows = np.zeros((live_rows.shape[0], choice_feature_dim(spec)), dtype=np.float32)
    _copy_shared_blocks(rows, live_rows, spec)
    _rebuild_bird_identity(rows, live_rows, spec)
    for index, choice in enumerate(decision.choices):
        _fix_placement_stripes(rows[index], decision, choice)
    return rows


class PolicyValueNetV00(core.PolicyValueNet):
    """A :class:`~wingspan.model.core.PolicyValueNet` frozen to the v0.0 choice
    geometry, for checkpoints written before artifact version 0.1.

    State-side behaviour is identical to the base net (the 0.1 reshape touched
    only the choice vector); the three overrides keep the choice path in the
    frozen era: rows arrive via this module's :func:`encode_choices` transform,
    the choice encoder is sized by the pre-0.1 :func:`choice_input_dim`
    formula, and ``_embed_choices`` slices the frozen card regions (the
    180-wide bird one-hot / multi-hot and the board-index block at their v0.0
    offsets). Constructed by the version-routing loaders
    (``PolicyValueNet.from_model_config``, ``selfplay._load_policy_net``) —
    never by the training pipeline, which always runs the live era.
    """

    def __init__(
        self,
        *,
        state_dim: int | None = None,
        choice_dim: int | None = None,
        num_families: int | None = None,
        arch: architecture.ModelArchitecture | None = None,
        spec: encode.EncodingSpec = encode.DEFAULT_SPEC,
    ):
        # The frozen row width replaces the live default; explicit dims (the
        # model_config descriptor path) pass through unchanged.
        if choice_dim is None:
            choice_dim = choice_feature_dim(spec)
        super().__init__(
            state_dim=state_dim,
            choice_dim=choice_dim,
            num_families=num_families,
            arch=arch,
            spec=spec,
        )

    def encode_choices(
        self,
        decision: decisions.Decision[typing.Any],
        game_state: state.GameState,
    ) -> np.ndarray:
        """v0.0 rows for ``decision`` — the frozen transform, never the live
        encoder (whose geometry this net predates)."""
        return encode_choices(decision, game_state, self.spec)

    def _build_choice_encoder(
        self, choice_dim: int, arch: architecture.ModelArchitecture
    ) -> None:
        """Register ``choice_encoder`` at the v0.0 input width."""
        self.choice_encoder, _ = mlp.build_body(
            choice_input_dim(choice_dim, arch.card_embed_dim),
            arch.choice_layers,
            activation=arch.activation,
            dropout=arch.dropout,
            layernorm=arch.layernorm,
            final_activation=False,
        )

    def _embed_choices(
        self, choices: torch.Tensor, card_table: torch.Tensor
    ) -> torch.Tensor:
        """The frozen v0.0 embedding (verbatim pre-0.1 math over the frozen
        offsets): the board-index block becomes one ``card_embed_dim`` vector
        per slot, and the 180-wide bird one-hot — a setup pick's kept-set
        multi-hot sums their vectors — becomes one more via a matmul over the
        non-padding card table rows. Everything else passes through."""
        board_idx = (
            choices[..., _OFF_BOARD_IDX:_OFF_BIRD_ID].long().clamp_(0, _BIRD_ID_DIM)
        )
        bird_multihot = choices[..., _OFF_BIRD_ID:_OFF_BONUS_ID]
        rest = torch.cat(
            [choices[..., :_OFF_BOARD_IDX], choices[..., _OFF_BONUS_ID:]], dim=-1
        )
        cand_emb = bird_multihot @ card_table[1:]
        board_emb = card_table[board_idx].reshape(*board_idx.shape[:-1], -1)
        return torch.cat([rest, cand_emb, board_emb], dim=-1)


###### PRIVATE #######


def _assert_live_layout_contract() -> None:
    """Import-time pins for the block-copy invariants the transform relies on.

    The transform's input side reads the live offsets, so a future live-layout
    reshape that breaks a contiguity or width assumption fails loudly here (and
    in the compat fixture tests) instead of silently mis-slicing — extending
    this shim is then that change's job."""
    # kind + gain_food open the row in both eras; the live row goes straight to
    # pay_food where v0.0 interposed the habitat stripe.
    assert layout._OFF_KIND == _OFF_KIND == 0
    assert layout._OFF_GAIN_FOOD == _OFF_GAIN_FOOD
    assert layout._OFF_PAY == _OFF_HAB
    # pay_food -> board_target -> main_action -> special -> exchange ->
    # board_idx is one contiguous run of identical width in both eras.
    assert layout._OFF_BOARD_IDX - layout._OFF_PAY == _OFF_BOARD_IDX - _OFF_PAY
    assert layout._OFF_BIRD_ID - layout._OFF_BOARD_IDX == _BOARD_IDX_SLOTS
    # The live candidate column is a single index directly before the bonus
    # tail (bonus_id through bonus_value), one contiguous block of identical
    # width in both eras.
    assert layout._CHOICE_BIRD_ID_DIM == 1
    assert layout._OFF_BONUS_ID == layout._OFF_BIRD_ID + 1
    assert layout._CHOICE_BASE_DIM - layout._OFF_BONUS_ID == (
        _CHOICE_BASE_DIM - _OFF_BONUS_ID
    )
    # Trailing conditional stripes: setup_agg, then the kept multi-hot sized
    # like the frozen bird stripe it maps back onto.
    assert layout._SETUP_DIM == _SETUP_DIM
    assert layout._OFF_KEPT_MULTIHOT == layout._OFF_SETUP + layout._SETUP_DIM
    assert layout._KEPT_MULTIHOT_DIM == _BIRD_ID_DIM


def _copy_shared_blocks(
    rows: np.ndarray, live_rows: np.ndarray, spec: encode.EncodingSpec
) -> None:
    """Copy every stripe whose contents are unchanged between the eras:
    kind + gain_food, the pay_food..exchange run, the board-index block, the
    bonus_id..bonus_value tail, and (``include_setup``) the setup_agg stripe."""
    rows[:, _OFF_KIND:_OFF_HAB] = live_rows[:, layout._OFF_KIND : layout._OFF_PAY]
    rows[:, _OFF_PAY:_OFF_BOARD_IDX] = live_rows[
        :, layout._OFF_PAY : layout._OFF_BOARD_IDX
    ]
    rows[:, _OFF_BOARD_IDX:_OFF_BIRD_ID] = live_rows[
        :, layout._OFF_BOARD_IDX : layout._OFF_BIRD_ID
    ]
    rows[:, _OFF_BONUS_ID:_OFF_SETUP] = live_rows[
        :, layout._OFF_BONUS_ID : layout._CHOICE_BASE_DIM
    ]
    if spec.include_setup:
        rows[:, _OFF_SETUP : _OFF_SETUP + _SETUP_DIM] = live_rows[
            :, layout._OFF_SETUP : layout._OFF_SETUP + layout._SETUP_DIM
        ]


def _rebuild_bird_identity(
    rows: np.ndarray, live_rows: np.ndarray, spec: encode.EncodingSpec
) -> None:
    """Rebuild the v0.0 bird-identity stripe: the live single index column
    scatters to its one-hot bit, and (``include_setup``) the live kept
    multi-hot lands on the same stripe. The two sources are disjoint by
    construction — a setup row's candidate column is zero, and a candidate
    row's kept stripe is zero — so the addition never overlaps."""
    candidate = live_rows[:, layout._OFF_BIRD_ID].astype(np.int64)
    if int(candidate.max()) > _BIRD_ID_DIM:
        raise ValueError(
            "Choice candidate index outside the frozen v0.0 catalog "
            f"(got {int(candidate.max())}, v0.0 has {_BIRD_ID_DIM} birds): "
            "a v0.0-era net cannot encode this game."
        )
    with_bird = np.nonzero(candidate)[0]
    rows[with_bird, _OFF_BIRD_ID + candidate[with_bird] - 1] = 1.0
    if spec.include_setup:
        rows[:, _OFF_BIRD_ID:_OFF_BONUS_ID] += live_rows[
            :,
            layout._OFF_KEPT_MULTIHOT : layout._OFF_KEPT_MULTIHOT
            + layout._KEPT_MULTIHOT_DIM,
        ]


def _fix_placement_stripes(
    row: np.ndarray,
    decision: layout._AnyDecision,
    choice: decisions.Choice,
) -> None:
    """Regenerate the placement stripes the 0.1 reshape changed: a v0.0
    placement row (a play-bird candidate, its committed food payment, or a
    move-bird destination) carried the destination habitat as a one-hot and
    wrote nothing to the board-index block — the landing-slot mark copied from
    the live row is the 0.1 change being undone here."""
    if isinstance(choice, decisions.PlayBirdChoice):
        habitat = choice.habitat
    elif isinstance(choice, decisions.FoodPaymentChoice):
        # The committed-play context rode along only under PayBirdFoodDecision
        # in v0.0 (mirroring the live featurizer's own context condition).
        if not isinstance(decision, decisions.PayBirdFoodDecision):
            return
        habitat = decision.habitat
    elif isinstance(choice, decisions.HabitatChoice):
        habitat = choice.habitat
    else:
        return
    row[_OFF_BOARD_IDX:_OFF_BIRD_ID] = 0.0
    for index, candidate in enumerate(cards.ALL_HABITATS):
        if candidate == habitat:
            row[_OFF_HAB + index] = 1.0
            break


_assert_live_layout_contract()
