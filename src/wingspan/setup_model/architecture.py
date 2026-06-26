"""Setup network descriptors: encoding layout and topology.

:class:`SetupEncoding` describes the input-vector layout for a given flag
configuration — which stripes are present and where they start.  The vector
changes size when the ``split_setup_food`` / ``split_setup_bonus`` training flags
are active: deferred axes are removed rather than zeroed.

:class:`SetupArchitecture` is the small, torch-free analogue of
:class:`wingspan.architecture.ModelArchitecture` for the separately-trained setup
model: the readout MLP that scores one setup candidate at a time. The setup model
always trains actor-critic: two scalar outputs — a value head (MSE critic) and a
policy head (REINFORCE actor); candidate selection at collection time uses
policy-head logits.

The network's card-embedding blocks are *not* described here — they are frozen
copies of the main net's shared embedders, so their shapes come from the main
:class:`~wingspan.architecture.ModelArchitecture` (threaded into
:func:`setup_readout_input_dim` / :func:`count_setup_parameters`).

Kept torch-free (only ``pydantic`` / ``enum``) so ``config`` and ``setup_runmeta``
can import it without pulling in torch, mirroring why ``ModelArchitecture`` lives
at the package top level.
"""

from __future__ import annotations

import typing

import pydantic

from wingspan import architecture, cards, encode, state

# Fixed block sizes shared by SetupEncoding and encode.py.
_KEPT_CARDS_DIM = 180  # cards.n_birds() — stable core-set count
_KEPT_FOODS_DIM = 5  # cards.N_FOODS
_BONUS_DIM = 26  # cards.n_bonus_cards()
_BONUS_AFF_DIM = 2  # [min_bonus_card_affinity, max_bonus_card_affinity]
_TRAY_DIM = 3  # state.TRAY_SIZE
_FEEDER_DIM = 6  # N_FOODS + 1 choice die
_GOALS_DIM = 80  # 4 rounds × 20 goal categories
_KEPT_BONUS_VALUE_DIM = 4
_GOAL_AFFINITY_DIM = 4  # one scalar per round


class SetupEncoding(pydantic.BaseModel):
    """Input-vector layout for the setup net under a given flag configuration.

    When ``split_food`` is active the ``kept_foods`` stripe (5 dims) is omitted.
    When ``split_bonus`` is active the ``kept_bonus`` one-hot (26 dims) and
    ``kept_bonus_value`` (4 dims) are replaced by ``bonus_cards`` multi-hot
    (26 dims) + ``bonus_card_affinity`` min/max (2 dims). All offsets and the
    total dimension are derived from these two flags.

    ``SetupEncoding()`` — both flags ``False`` — reproduces the pre-0.2
    layout (308 dims), so old ``setup_config.json`` files that lack this field
    deserialize correctly via Pydantic's default.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    split_food: bool = False
    split_bonus: bool = False
    include_turn1_playable: bool = False
    include_playable_kept_cards: bool = False

    # ---- offset properties (all derived, no stored state) ----

    @property
    def kept_cards_dim(self) -> int:
        """Always 180 — the fixed number of core-set bird cards."""
        return _KEPT_CARDS_DIM

    @property
    def off_kept_cards(self) -> int:
        """Always 0: kept_cards is the first block."""
        return 0

    @property
    def _foods_dim(self) -> int:
        return 0 if self.split_food else _KEPT_FOODS_DIM

    @property
    def off_bonus_block(self) -> int:
        """Start of the bonus block (kept_bonus OR bonus_cards + affinity)."""
        return _KEPT_CARDS_DIM + self._foods_dim

    @property
    def bonus_block_dim(self) -> int:
        """28 when split_bonus (bonus_cards + affinity), 26 when not (kept_bonus only).

        ``kept_bonus_value`` (4 dims) is placed after goals, not in this block."""
        return (_BONUS_DIM + _BONUS_AFF_DIM) if self.split_bonus else _BONUS_DIM

    @property
    def off_tray(self) -> int:
        return self.off_bonus_block + self.bonus_block_dim

    @property
    def off_feeder(self) -> int:
        return self.off_tray + _TRAY_DIM

    @property
    def off_goals(self) -> int:
        return self.off_feeder + _FEEDER_DIM

    @property
    def off_bonus_value(self) -> int:
        """Start of the kept_bonus_value block (only present when ``split_bonus=False``)."""
        return self.off_goals + _GOALS_DIM

    @property
    def off_goal_affinity(self) -> int:
        bonus_value_dim = 0 if self.split_bonus else _KEPT_BONUS_VALUE_DIM
        return self.off_bonus_value + bonus_value_dim

    @property
    def off_turn1_playable(self) -> int:
        """Start of the turn1_playable multi-hot (only when ``include_turn1_playable``)."""
        return self.off_goal_affinity + _GOAL_AFFINITY_DIM

    @property
    def off_playable_kept_cards(self) -> int:
        """Start of the playable_kept_cards multi-hot (only when ``include_playable_kept_cards``)."""
        return self.off_turn1_playable + (
            _KEPT_CARDS_DIM if self.include_turn1_playable else 0
        )

    @property
    def total_dim(self) -> int:
        """Total feature-vector length for this encoding configuration."""
        base = self.off_goal_affinity + _GOAL_AFFINITY_DIM
        if self.include_turn1_playable:
            base += _KEPT_CARDS_DIM  # 180-dim multi-hot of turn-1-playable birds
        if self.include_playable_kept_cards:
            base += _KEPT_CARDS_DIM  # 180-dim multi-hot of food-agnostic playability
        return base


# A setup-net shape signature: the hidden-layer widths plus whether the policy
# head is present. Activation / dropout are excluded — they leave tensor shapes
# intact and a resumed run may change them without invalidating the saved weights.
type SetupShapeKey = tuple[tuple[int, ...], bool]


class SetupArchitecture(pydantic.BaseModel):
    """The reconstitutable topology of a :class:`wingspan.training.setup_net.SetupNet`'s
    readout MLP.

    ``hidden_layers`` is ordered input-to-output: ``(128, 64)`` is a two-layer
    MLP projecting to 64 before the final scalar readout. Reuses
    :data:`wingspan.architecture.Widths` and
    :class:`wingspan.architecture.ActivationName` so the configurator's layer /
    activation editors apply unchanged.

    ``between_activation`` is applied after each hidden-layer ``Linear``;
    ``final_activation`` (defaults to NONE) would be applied after the final
    ``Linear(·, 1)`` — keep NONE for the standard bare-scalar readout.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    hidden_layers: typing.Annotated[
        architecture.Widths, pydantic.Field(min_length=1)
    ] = (
        128,
        64,
    )
    between_activation: architecture.ActivationName = architecture.ActivationName.RELU
    final_activation: architecture.ActivationName = architecture.ActivationName.NONE
    dropout: typing.Annotated[float, pydantic.Field(ge=0.0, lt=1.0)] = 0.0
    # Always True: the setup net always trains actor-critic (value + policy heads).
    # Included in the shape key because it doubles the readout Linear layers.
    use_policy_head: bool = True

    @pydantic.model_validator(mode="before")
    @classmethod
    def _migrate_legacy_activation_field(cls, data: object) -> object:
        """Translate old ``activation`` field to ``between_activation``."""
        if not isinstance(data, dict) or "activation" not in data:
            return data
        raw = typing.cast("dict[str, typing.Any]", data)
        old_act: str = raw.pop("activation")
        raw.setdefault("between_activation", old_act)
        raw.setdefault("final_activation", "none")
        return raw

    @property
    def shape_key(self) -> SetupShapeKey:
        """The readout MLP's weight-compatibility signature (everything that
        changes one of *its* tensor shapes). The embedder copies' shapes ride the
        main architecture and are keyed separately
        (``TrainConfig.setup_architecture_key``)."""
        return (self.hidden_layers, self.use_policy_head)


def setup_readout_input_dim(
    feature_dim: int,
    main_arch: architecture.ModelArchitecture,
    *,
    include_turn1_playable: bool = False,
    include_playable_kept_cards: bool = False,
) -> int:
    """The setup readout MLP's first-``Linear`` input width: the raw
    ``feature_dim`` vector with the kept-cards multi-hot replaced by one
    ``N``-wide set embedding and the tray index columns replaced by
    ``TRAY_SIZE`` ``M``-wide card-table rows plus one more ``N``-wide tray-set
    embedding (``M = card_embed_dim``, ``N = hand_embed_width``). The single
    source of truth shared by ``SetupNet`` and the parameter accounting.

    When ``include_turn1_playable`` or ``include_playable_kept_cards`` is active,
    each trailing 180-dim multi-hot is embedded as one extra
    ``hand_embed_width``-wide set embedding."""
    passthrough = feature_dim - cards.n_birds() - state.TRAY_SIZE
    n_hand_sets = 2  # kept set + tray set
    for flag in (include_turn1_playable, include_playable_kept_cards):
        if flag:
            # Each appended 180-dim multi-hot is embedded as a card set:
            # subtract the raw dims from passthrough, add one set embedding.
            passthrough -= _KEPT_CARDS_DIM
            n_hand_sets += 1
    return (
        passthrough
        + n_hand_sets * main_arch.hand_embed_width
        + state.TRAY_SIZE * main_arch.card_embed_dim
    )


def count_setup_parameters(
    setup_arch: SetupArchitecture,
    *,
    feature_dim: int,
    main_arch: architecture.ModelArchitecture | None = None,
) -> architecture.BlockParam:
    """Analytic per-layer parameter accounting for the ``SetupNet`` that
    ``setup_arch`` describes.

    The setup net is a readout MLP (``setup_readout_input_dim → hidden… → 1``,
    Linears only, no LayerNorm) over the in-net embedding of the raw features,
    plus its two frozen embedder copies — the card encoder and the set (hand)
    encoder, whose shapes come from ``main_arch`` (``None`` = a bare
    :class:`~wingspan.architecture.ModelArchitecture`, matching ``SetupNet``'s
    default). The readout's per-layer counts are
    :func:`wingspan.architecture.readout_layers`; the embedder copies are rolled
    into ``extra`` (frozen parameters still count in ``numel``). Returns one
    :class:`wingspan.architecture.BlockParam` whose ``total`` equals
    ``sum(p.numel())`` of the built ``SetupNet`` — the architecture diagram's
    per-op and Σ source for the separate setup model.
    """
    if main_arch is None:
        main_arch = architecture.ModelArchitecture()
    embedder_params = sum(
        layer.params
        for layer in architecture.body_layers(
            encode.CARD_FEATURE_DIM,
            main_arch.card_encoder_layers + (main_arch.card_embed_dim,),
            main_arch,
        )
    ) + sum(
        layer.params
        for layer in architecture.body_layers(
            encode.HAND_ENCODER_INPUT_DIM,
            main_arch.hand_encoder_layers + (main_arch.hand_embed_width,),
            main_arch,
        )
    )
    readout_in = setup_readout_input_dim(
        feature_dim,
        main_arch,
        include_turn1_playable=False,
        include_playable_kept_cards=False,
    )
    # When the policy head is present there are two readout MLPs of identical
    # shape (value + policy), so their parameter count doubles.
    readout = architecture.readout_layers(readout_in, setup_arch.hidden_layers)
    n_heads = 2 if setup_arch.use_policy_head else 1
    return architecture.BlockParam(
        label="SETUP",
        layers=readout * n_heads,
        extra=embedder_params,
    )
