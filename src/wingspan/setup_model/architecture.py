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
    include_playable_kept_cards: bool = True

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


# A setup-net shape signature: the trunk widths, the per-head hidden widths,
# and whether the policy head is present. Activation / dropout are excluded —
# they leave tensor shapes intact and a resumed run may change them without
# invalidating the saved weights.
type SetupShapeKey = tuple[tuple[int, ...], tuple[int, ...], bool]


class SetupArchitecture(pydantic.BaseModel):
    """The reconstitutable topology of a :class:`wingspan.training.setup_net.SetupNet`'s
    readout MLP.

    ``trunk_layers`` is an optional shared trunk before the value/policy split:
    ``(128,)`` inserts one shared ``Linear → ReLU`` layer whose output feeds both
    heads. Empty (the default) means no trunk — both heads read the embedded
    input directly, exactly reproducing the pre-trunk behavior.

    ``hidden_layers`` is ordered input-to-output for each head: ``(64,)`` is a
    one-layer head projecting to 64 before the final scalar readout. Reuses
    :data:`wingspan.architecture.Widths` and
    :class:`wingspan.architecture.ActivationName` so the configurator's layer /
    activation editors apply unchanged.

    ``between_activation`` is applied after each hidden-layer ``Linear`` in both
    the trunk and the heads; ``final_activation`` (defaults to NONE) would be
    applied after the final ``Linear(·, 1)`` — keep NONE for the standard
    bare-scalar readout. The trunk always uses ``between_activation`` as its own
    final activation so no nonlinearity is skipped between trunk and heads.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    # Empty default keeps old artifacts loading identically (the sanctioned
    # "new fields default to reproduce old behavior" back-compat mechanism).
    trunk_layers: architecture.Widths = ()
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
        if not isinstance(data, dict):
            return data
        raw = typing.cast(dict[str, object], data)
        if "activation" not in raw:
            return raw
        old_act = str(raw.pop("activation"))
        raw.setdefault("between_activation", old_act)
        raw.setdefault("final_activation", "none")
        return raw

    @property
    def shape_key(self) -> SetupShapeKey:
        """The readout MLP's weight-compatibility signature (everything that
        changes one of *its* tensor shapes). The embedder copies' shapes ride the
        main architecture and are keyed separately
        (``TrainConfig.setup_architecture_key``)."""
        return (self.trunk_layers, self.hidden_layers, self.use_policy_head)


def setup_readout_input_dim(
    feature_dim: int,
    main_arch: architecture.ModelArchitecture,
    *,
    include_turn1_playable: bool = False,
    include_playable_kept_cards: bool = False,
) -> int:
    """The setup readout MLP's first-``Linear`` input width.

    The raw ``feature_dim`` vector is transformed by replacing each card-set
    multi-hot with a set embedding (pooled or encoder, matching the main net's
    path) and the tray index columns with ``TRAY_SIZE`` ``M``-wide card-table
    rows (``M = card_embed_dim``). The tray no longer carries a set embedding —
    only per-slot rows. The set embedding width ``W`` is
    ``main_arch.pooled_hand_width`` when not using a distinct hand model
    (the default), or ``main_arch.hand_embed_width`` otherwise.

    When ``include_turn1_playable`` or ``include_playable_kept_cards`` is active,
    each trailing 180-dim multi-hot is embedded as one extra ``W``-wide set
    embedding and its raw dims are subtracted from passthrough."""
    set_width = (
        main_arch.hand_embed_width
        if main_arch.use_distinct_hand_model
        else main_arch.pooled_hand_width
    )
    passthrough = feature_dim - cards.n_birds() - state.TRAY_SIZE
    n_card_sets = 1  # kept set
    for flag in (include_turn1_playable, include_playable_kept_cards):
        if flag:
            # Each appended 180-dim multi-hot is embedded as a card set:
            # subtract the raw dims from passthrough, add one set embedding.
            passthrough -= _KEPT_CARDS_DIM
            n_card_sets += 1
    return (
        passthrough
        + n_card_sets * set_width
        + state.TRAY_SIZE * main_arch.card_embed_dim
    )


class SetupParamReport(pydantic.BaseModel):
    """Per-block parameter accounting for a :class:`wingspan.training.setup_net.SetupNet`.

    Breaks the setup net into three tiers: the frozen embedder copies
    (``embedder_params``), the shared trunk (``trunk``, empty when
    ``trunk_layers=()``), and the per-head readout MLPs
    (``value_head`` / ``policy_head``). Built by
    :func:`count_setup_parameters`; its ``total`` equals ``sum(p.numel())``
    of the real module — the architecture diagram's per-op and Σ source.
    """

    embedder_params: int
    trunk: tuple[architecture.LayerParam, ...]
    value_head: tuple[architecture.LayerParam, ...]
    # None when use_policy_head=False (value-only mode).
    policy_head: tuple[architecture.LayerParam, ...] | None

    @property
    def trunk_params(self) -> int:
        """Total trainable parameters in the shared trunk (0 when no trunk)."""
        return sum(layer.params for layer in self.trunk)

    @property
    def head_in(self) -> int:
        """Input width to each head: trunk output width when a trunk is present,
        otherwise the embedded readout-input width."""
        if self.trunk:
            return self.trunk[-1].out_features
        return self.value_head[0].in_features

    @property
    def total(self) -> int:
        """Total trainable-parameter count (embedder + trunk + heads).

        Embedder copies are frozen at inference but their parameters are counted
        because ``sum(p.numel())`` includes all parameters regardless of
        ``requires_grad``."""
        head_params = sum(layer.params for layer in self.value_head)
        if self.policy_head is not None:
            head_params += sum(layer.params for layer in self.policy_head)
        return self.embedder_params + self.trunk_params + head_params


def count_setup_parameters(
    setup_arch: SetupArchitecture,
    *,
    feature_dim: int,
    main_arch: architecture.ModelArchitecture | None = None,
    encoding: SetupEncoding | None = None,
) -> SetupParamReport:
    """Analytic per-block parameter accounting for the ``SetupNet`` that
    ``setup_arch`` describes.

    Accounts for the frozen embedder copies (card + hand encoders, shaped by
    ``main_arch``), the optional shared trunk, and the per-head readout MLPs.
    Returns a :class:`SetupParamReport` whose ``total`` equals
    ``sum(p.numel())`` of the built ``SetupNet`` — the architecture diagram's
    per-op and Σ source for the separate setup model.

    ``encoding`` controls which optional stripes are included in the
    readout-input width calculation (default: ``SetupEncoding()``).
    """
    if main_arch is None:
        main_arch = architecture.ModelArchitecture()
    if encoding is None:
        encoding = SetupEncoding()

    # Frozen embedder copies (card + hand encoder).
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
        include_turn1_playable=encoding.include_turn1_playable,
        include_playable_kept_cards=encoding.include_playable_kept_cards,
    )

    # Optional shared trunk: body-block layers (no LayerNorm; uses
    # between_activation as its final activation so the trunk's output is
    # nonlinear before the heads' first Linear).
    trunk: tuple[architecture.LayerParam, ...] = ()
    trunk_out = readout_in
    if setup_arch.trunk_layers:
        trunk = architecture.body_layers(
            readout_in, setup_arch.trunk_layers, main_arch, layernorm=False
        )
        trunk_out = setup_arch.trunk_layers[-1]

    # Per-head readout MLPs reading the trunk output (or readout_in directly).
    value_head = architecture.readout_layers(trunk_out, setup_arch.hidden_layers)
    policy_head = value_head if setup_arch.use_policy_head else None

    return SetupParamReport(
        embedder_params=embedder_params,
        trunk=trunk,
        value_head=value_head,
        policy_head=policy_head,
    )
