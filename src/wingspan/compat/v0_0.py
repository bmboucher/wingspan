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
* :func:`choice_stripe_layout` *describes* the frozen layout — the stripe
  registry the descriptor-driven reporting seam (``runmeta.choice_layout_for``)
  shows for pre-0.1 runs, mirroring the live
  ``encode.stripes.choice_stripe_layout``. Era-shared stripes reuse the live
  raw registry's descriptors (the same principle as the transform's block
  copies); only the reshaped placement / card-identity stripes are described
  here.

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
from wingspan.compat import v0_1
from wingspan.encode import layout
from wingspan.encode.stripes import choice as stripes_choice
from wingspan.encode.stripes import descriptors, embed_rules
from wingspan.model import mlp

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

_DEFAULT_CARD_EMBED_DIM = architecture.ModelArchitecture().card_embed_dim

# The stripes whose contents (and descriptions) are identical in both eras,
# mapped to their frozen v0.0 offsets — the layout twin of the transform's
# _copy_shared_blocks. The reshaped stripes (habitat / board_idx / bird_id)
# get era-specific descriptors instead.
_SHARED_STRIPE_OFFSETS: dict[str, int] = {
    "kind": _OFF_KIND,
    "gain_food": _OFF_GAIN_FOOD,
    "pay_food": _OFF_PAY,
    "board_target": _OFF_BOARD,
    "main_action": _OFF_MAIN_ACTION,
    "special": _OFF_SPECIAL,
    "exchange": _OFF_EXCHANGE,
    "bonus_id": _OFF_BONUS_ID,
    "bonus_delta": _OFF_BONUS_DELTA,
    "goal_delta": _OFF_GOAL_DELTA,
    "bonus_value": _OFF_BONUS_VALUE,
    "setup_agg": _OFF_SETUP,
}


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


def choice_passthrough_dim(choice_dim: int) -> int:
    """The v0.0 choice columns that pass straight through to the encoder —
    everything except the embedded card regions (the 180-wide bird one-hot /
    keep multi-hot and the 15-slot board-index block). The frozen twin of the
    live ``encode.choice_passthrough_dim``, with no ``include_setup`` axis."""
    return choice_dim - _BIRD_ID_DIM - _BOARD_IDX_SLOTS


def choice_stripe_layout(
    spec: encode.EncodingSpec = encode.DEFAULT_SPEC,
    card_embed_dim: int = _DEFAULT_CARD_EMBED_DIM,
) -> descriptors.VectorLayout:
    """Build the stripe registry for a v0.0 net's choice-encoder input vector.

    The frozen-era twin of the live ``encode.stripes.choice_stripe_layout``:
    every stripe of the v0.0 row at its frozen offset, with the board-index
    block and the 180-wide bird one-hot shown at their post-embedding width so
    the breakdown sums to :func:`choice_input_dim` — what a
    :class:`PolicyValueNetV00` actually sees. The trailing ``setup_agg`` stripe
    is present only when ``spec.include_setup`` (v0.0 had no separate
    ``kept_multihot`` stripe — keeps rode ``bird_id``).
    """
    raw = _raw_choice_stripes(spec)
    return embed_rules.embed_layout(
        raw,
        _choice_embed_rules(card_embed_dim),
        choice_input_dim(choice_feature_dim(spec), card_embed_dim),
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


class PolicyValueNetV00(v0_1.PolicyValueNetV01):
    """A :class:`~wingspan.model.core.PolicyValueNet` frozen to the v0.0 choice
    geometry, for checkpoints written before artifact version 0.1.

    Inherits from :class:`~wingspan.compat.v0_1.PolicyValueNetV01` to also
    pin the card encoder to the frozen 229-wide input — v0.0 artifacts predate
    both the 0.1 choice reshape and the 0.2 card-feature reshape. The three
    overrides keep the choice path in the frozen era: rows arrive via this
    module's :func:`encode_choices` transform, the choice encoder is sized by
    the pre-0.1 :func:`choice_input_dim` formula, and ``_embed_choices`` slices
    the frozen card regions (the 180-wide bird one-hot / multi-hot and the
    board-index block at their v0.0 offsets). Constructed by the
    version-routing loaders (``PolicyValueNet.from_model_config``,
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
            between_activation=arch.choice_between_activation_resolved,
            final_activation=arch.choice_final_activation_resolved,
            dropout=arch.dropout,
            layernorm=arch.layernorm,
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
    # width in both eras. The becomes_playable stripe was added AFTER bonus_value
    # in v0.6, growing _CHOICE_BASE_DIM by 180 — use CHOICE_BECOMES_PLAYABLE_OFFSET
    # as the boundary so this assertion does not include the new stripe.
    assert layout._CHOICE_BIRD_ID_DIM == 1
    assert layout._OFF_BONUS_ID == layout._OFF_BIRD_ID + 1
    assert layout.CHOICE_BECOMES_PLAYABLE_OFFSET - layout._OFF_BONUS_ID == (
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
    # Copy bonus_id through bonus_value; stop before the v0.6 becomes_playable
    # stripe (layout.CHOICE_BECOMES_PLAYABLE_OFFSET) which the v0.0 row lacks.
    rows[:, _OFF_BONUS_ID:_OFF_SETUP] = live_rows[
        :, layout._OFF_BONUS_ID : layout.CHOICE_BECOMES_PLAYABLE_OFFSET
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


#### v0.0 stripe registry ####


def _raw_choice_stripes(spec: encode.EncodingSpec) -> descriptors.VectorLayout:
    """The raw (pre-embedding) v0.0 stripe list: era-shared stripes reused from
    the live raw registry at their frozen offsets, plus the era-specific
    habitat / board_idx / bird_id descriptors. The contiguity loop pins every
    frozen offset against the inherited stripe widths — a live width change
    fails loudly here instead of silently shifting the v0.0 table."""
    shared = {
        stripe.name: stripe
        for stripe in stripes_choice.raw_choice_stripe_layout(spec).stripes
    }

    def shared_at(name: str) -> descriptors.StripeDescriptor:
        return shared[name].model_copy(update={"offset": _SHARED_STRIPE_OFFSETS[name]})

    assembled = [
        shared_at("kind"),
        shared_at("gain_food"),
        _habitat_stripe(),
        shared_at("pay_food"),
        shared_at("board_target"),
        shared_at("main_action"),
        shared_at("special"),
        shared_at("exchange"),
        _board_idx_stripe(),
        _bird_id_stripe(),
        shared_at("bonus_id"),
        shared_at("bonus_delta"),
        shared_at("goal_delta"),
        shared_at("bonus_value"),
    ]
    if spec.include_setup:
        assembled.append(shared_at("setup_agg"))

    offset = 0
    for stripe in assembled:
        assert stripe.offset == offset, (
            f"v0.0 stripe {stripe.name!r} lands at offset {offset} but the "
            f"frozen layout places it at {stripe.offset} — a live stripe width "
            "changed; extend the v0.0 shim"
        )
        offset += stripe.size
    return descriptors.VectorLayout(
        total_size=choice_feature_dim(spec), stripes=tuple(assembled)
    )


def _habitat_stripe() -> descriptors.StripeDescriptor:
    """The 3-wide destination-habitat one-hot the 0.1 reshape removed."""
    habitat_names = ", ".join(habitat.value for habitat in cards.ALL_HABITATS)
    return descriptors.StripeDescriptor(
        name="habitat",
        description=(
            "Destination-habitat one-hot for a placement row (a play-bird "
            "candidate, its committed food payment, or a move-bird habitat "
            "pick)."
        ),
        offset=_OFF_HAB,
        size=_HABITAT_DIM,
        encoding="one-hot",
        value_range="{0, 1}",
        notes=(
            f"Habitats in order: {habitat_names}. The 0.1 reshape replaced "
            "this with a landing-slot mark in board_idx. Zero for "
            "non-placement choices."
        ),
        sub_fields=_habitat_sub_fields(),
    )


def _habitat_sub_fields() -> tuple[descriptors.SubFieldDescriptor, ...]:
    """3 sub-fields for the v0.0 destination-habitat one-hot."""
    return tuple(
        descriptors.SubFieldDescriptor(
            name=f"habitat_{habitat.value}",
            description=f"This placement lands in {habitat.value}.",
            relative_offset=index,
            size=1,
            encoding="one-hot bit",
            value_range="{0, 1}",
        )
        for index, habitat in enumerate(cards.ALL_HABITATS)
    )


def _board_idx_stripe() -> descriptors.StripeDescriptor:
    """The v0.0 board-index block: board-target occupants only — placement rows
    wrote nothing here (the destination rode the habitat stripe)."""
    return descriptors.StripeDescriptor(
        name="board_idx",
        description=(
            "Bird indices for the deciding player's 15 board slots — the "
            "board_target stripe's occupants, looked up in the shared card "
            "table."
        ),
        offset=_OFF_BOARD_IDX,
        size=_BOARD_IDX_SLOTS,
        encoding="integer-index",
        value_range=f"int 0–{_BIRD_ID_DIM}",
        notes=(
            f"{_BOARD_IDX_SLOTS} integer indices (positional, ALL_HABITATS × "
            "ROW_SLOTS). bird_index + 1; 0 = empty slot. Board-target rows "
            "fill all occupants; v0.0 placement rows wrote nothing here (the "
            "0.1 landing-slot mark did not exist yet). Zero for other choices."
        ),
    )


def _bird_id_stripe() -> descriptors.StripeDescriptor:
    """The v0.0 bird-identity stripe: a 180-wide one-hot doubling as the
    setup keep multi-hot."""
    return descriptors.StripeDescriptor(
        name="bird_id",
        description=(
            f"The candidate bird's identity as a one-hot over all "
            f"{_BIRD_ID_DIM} core-set birds — doubling as a setup pick's "
            "kept-set multi-hot."
        ),
        offset=_OFF_BIRD_ID,
        size=_BIRD_ID_DIM,
        encoding="one-hot / multi-hot",
        value_range="{0, 1}",
        notes=(
            "One bit per core-set bird. One-hot for a single-bird candidate; "
            "a SetupChoice marks every kept bird (the 0.1 reshape collapsed "
            "this to one index column and moved the kept set onto a dedicated "
            "kept_multihot stripe). Zero for non-bird choices."
        ),
    )


def _choice_embed_rules(card_embed_dim: int) -> dict[str, embed_rules._EmbedRule]:
    """The v0.0 card-region stripes at embedded width: the board-index block as
    one embedding per slot, the bird one-hot / keep multi-hot summed through
    the shared card table into one embedding (the ``_embed_choices`` matmul)."""
    return {
        "board_idx": embed_rules._EmbedRule(
            new_size=_BOARD_IDX_SLOTS * card_embed_dim,
            encoding="card-embedding",
            value_range="learned",
            notes=(
                f"{_BOARD_IDX_SLOTS} board slots -> one {card_embed_dim}-dim "
                f"shared card embedding each ({_BOARD_IDX_SLOTS}x"
                f"{card_embed_dim}). Raw encoding stores {_BOARD_IDX_SLOTS} "
                "integer indices (bird_index + 1; 0 = empty)."
            ),
        ),
        "bird_id": embed_rules._EmbedRule(
            new_size=card_embed_dim,
            encoding="card-embedding (candidate / kept set, summed)",
            value_range="learned",
            notes=(
                f"The {_BIRD_ID_DIM}-wide bird one-hot (a setup keep's "
                f"multi-hot) -> one {card_embed_dim}-dim embedding, summed "
                "over the marked cards' shared card vectors."
            ),
        ),
    }


_assert_live_layout_contract()
