"""The network topology descriptor: the editable *shape* of ``PolicyValueNet``.

:class:`ModelArchitecture` is the single, torch-free record of the network's
configurable topology — the per-block hidden-layer widths plus the between/final
activation, dropout, LayerNorm, and shared card-embedding handles. It is the one
vehicle that:

* the configurator edits (via the flat fields it mirrors on
  :class:`wingspan.training.config.TrainConfig`),
* :class:`model.PolicyValueNet` builds itself from, and
* ``model_config.json`` serializes so a run's full topology can be read at a
  glance and the network reconstituted later.

It lives at the package top level (beside ``model`` / ``encode`` / ``decisions``)
so the top-level ``model`` module can import it without dragging in the training
package; kept torch-free (only ``pydantic`` / ``enum``) so ``config`` and
``runmeta`` import it without pulling in torch. The four blocks it sizes are the
state trunk, the per-choice encoder, the per-family scorer heads, and the value
head. The trunk and the choice encoder may end at different widths ``M`` and
``N``; their outputs are concatenated to ``M+N`` before the scorer heads.
"""

from __future__ import annotations

import enum
import typing

import pydantic

# A per-block list of hidden-layer widths (each width >= 1). The body blocks
# (trunk / choice encoder) additionally require at least one layer; the head
# blocks may be empty (a direct linear readout).
type Widths = tuple[typing.Annotated[int, pydantic.Field(ge=1)], ...]

# The weight-shape signature of an architecture: the parts that, if changed,
# make previously-trained weights unloadable (activation / dropout are excluded —
# they leave every tensor shape intact, so a resumed run may change them).
# The four per-block layernorm bools replace the former single global layernorm
# slot; old configs resolve all four to the global, so matching is preserved.
type ShapeKey = tuple[
    tuple[int, ...],  # trunk_layers
    tuple[int, ...],  # choice_layers
    tuple[int, ...],  # head_layers
    tuple[int, ...],  # value_layers
    bool,  # card_layernorm_resolved
    bool,  # hand_layernorm_resolved
    bool,  # trunk_layernorm_resolved
    bool,  # choice_layernorm_resolved
    int,  # card_embed_dim
    tuple[int, ...],  # card_encoder_layers
    tuple[tuple[int, ...], ...] | None,  # per_family_head_layers
    bool,  # use_distinct_hand_model
    tuple[int, ...],  # hand_encoder_layers
    int,  # hand_embed_width (the *resolved* N, so None == explicit-equal)
    bool,  # tray_set_embedding
    bool,  # use_board_attention
    HandPooling | None,  # hand_pooling (None when use_distinct_hand_model)
]


class ActivationName(enum.StrEnum):
    """The activation functions selectable for every MLP block, plus NONE to
    drop the activation layer entirely."""

    RELU = "relu"
    GELU = "gelu"
    TANH = "tanh"
    SILU = "silu"
    LEAKY_RELU = "leaky_relu"
    NONE = "none"


class HandPooling(enum.StrEnum):
    """Permutation-invariant set-pooling mode for the hand embedding.

    Only meaningful when ``ModelArchitecture.use_distinct_hand_model`` is
    ``False``; inert (and silently carried) when the dedicated hand encoder is
    active. ``M = card_embed_dim`` throughout.

    - ``MEAN``: mean of selected card vectors (``M`` wide) — identical to the
      pre-pooling legacy behaviour; no count appended.
    - ``SUM``: sum of selected card vectors (``M`` wide); no count appended.
    - ``MAX``: element-wise max of selected card vectors, then a count scalar
      appended (``M+1`` wide).
    - ``CONCAT_MAX_SUM``: ``[max | sum | count]`` (``2M+1`` wide) — recommended
      default; max captures "best available per axis", sum captures multiplicity.
    """

    MEAN = "mean"
    SUM = "sum"
    MAX = "max"
    CONCAT_MAX_SUM = "concat_max_sum"


# Extra count-scalar dimension appended to pooling modes that include a max.
_HAND_POOL_COUNT_DIM = 1


class ModelArchitecture(pydantic.BaseModel):
    """The full, reconstitutable topology of a :class:`model.PolicyValueNet`.

    Width lists are ordered input-to-output: ``trunk_layers=(256, 128)`` is a
    two-layer trunk that projects to 128. The trunk ends at width ``M``
    (``trunk_embed_width``) and the choice encoder ends at width ``N``
    (``choice_embed_width``); both are independent and their outputs are
    concatenated to ``M+N`` before the scorer heads.

    Every MLP block has two activation slots: ``*_between_activation`` (applied
    after each non-final layer) and ``*_final_activation`` (applied after the
    last layer — ``NONE`` to skip it). Per-block ``None`` inherits the matching
    global (``between_activation`` or ``final_activation``). The trunk is the
    one exception: its ``*_final`` inherits ``between_activation`` (not
    ``final_activation``) so it always activates its output even when the global
    final defaults to ``NONE``.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    trunk_layers: typing.Annotated[Widths, pydantic.Field(min_length=1)] = (128, 128)
    choice_layers: typing.Annotated[Widths, pydantic.Field(min_length=1)] = (128, 128)
    head_layers: Widths = (128,)
    value_layers: Widths = ()
    between_activation: ActivationName = ActivationName.RELU
    final_activation: ActivationName = ActivationName.NONE
    dropout: typing.Annotated[float, pydantic.Field(ge=0.0, lt=1.0)] = 0.0
    layernorm: bool = False
    card_embed_dim: typing.Annotated[int, pydantic.Field(ge=1)] = 64
    # Hidden widths of the per-card encoder MLP, which maps each card's
    # [static attributes ⊕ identity one-hot] to its ``card_embed_dim``-wide vector.
    # Empty = a single linear projection to ``card_embed_dim``; a non-empty stack
    # makes the encoder genuinely nonlinear. Its output width is always
    # ``card_embed_dim`` (the model appends it), so this lists hidden layers only.
    # The default mirrors ``config.TrainConfig`` so a bare ``ModelArchitecture()``
    # (e.g. ``PolicyValueNet()``) matches a configured run's shape.
    card_encoder_layers: Widths = (128,)
    # Per-family scorer head widths, one entry per active decision family in stable
    # order. When set, each family's scoring MLP uses its own hidden-layer widths
    # instead of the shared ``head_layers``; ``head_layers_for(i)`` resolves this.
    # ``None`` (the default) = all families share ``head_layers`` (uniform mode).
    per_family_head_layers: tuple[Widths, ...] | None = None
    # When True a dedicated hand encoder MLP replaces the pooled hand embedding:
    # it takes [180-dim multi-hot ⊕ 10-dim hand summary] = 190 dims as input and
    # outputs a ``hand_embed_width``-wide hand vector. The 10 hand-summary dims are
    # redirected from the trunk's continuous input into this encoder, so the trunk
    # sees a correspondingly narrower continuous block. Defaults to False for new
    # runs — the pooled path (``hand_pooling``) is the recommended default. The
    # dedicated encoder is retained for back-compat; old checkpoints carry their
    # own value and load identically.
    use_distinct_hand_model: bool = False
    # Pooling mode for the hand set embedding (only meaningful when
    # ``use_distinct_hand_model`` is False). The hand multi-hot is pooled over the
    # shared card-encoder outputs (``card_table[1:]``) according to this mode; see
    # ``HandPooling`` for the per-mode output widths. Inert for old checkpoints
    # that carry ``use_distinct_hand_model=True``. Default = CONCAT_MAX_SUM
    # (recommended for new runs: max captures "best available per axis", sum
    # captures multiplicity, count disambiguates empty-hand).
    hand_pooling: HandPooling = HandPooling.CONCAT_MAX_SUM
    # Hidden widths of the hand encoder MLP (same shape convention as
    # ``card_encoder_layers``). Output is always ``hand_embed_width`` (the model
    # appends it), so this lists hidden layers only. Defaults mirror
    # ``card_encoder_layers`` so the default net shapes match when toggled on.
    hand_encoder_layers: Widths = (128,)
    # Output width ``N`` of the hand encoder — the multi-card *set* embedding's
    # width, distinct from the single-card ``card_embed_dim`` (``M``). ``None``
    # (the default) resolves to ``card_embed_dim``, so saved configs that predate
    # this field reconstitute their actual old shape. Only meaningful when
    # ``use_distinct_hand_model`` is on; resolved via ``hand_embed_width``.
    hand_embed_dim: typing.Annotated[int, pydantic.Field(ge=1)] | None = None
    # When True (requires ``use_distinct_hand_model``) the trunk input gains one
    # ``hand_embed_width``-wide embedding of the face-up tray *set*, produced by
    # the shared hand encoder from a tray multi-hot + summary the model derives
    # from the three tray index columns. The tray's three per-slot card-table
    # lookups are unchanged, so the tray contributes 3·M + N dims in total.
    # Defaults to False for new runs (REGIME change from the old True default —
    # saved configs carry their own value, so existing checkpoints are unaffected).
    tray_set_embedding: bool = False
    # When True, each player's 15 board slots are attended over as tokens
    # (card_embed_dim + 9 scalars wide) before the state trunk, using two
    # independent nn.MultiheadAttention modules (one per seat). Config-carried
    # (lives in model_config.json), default False → old artifacts rehydrate
    # identically with no compat shim. Joins ShapeKey so a True run won't try
    # to resume False weights (handled by the architecture_key gate, REGIME).
    use_board_attention: bool = False

    # Per-block between/final activation overrides plus dropout and LayerNorm
    # toggles. ``None`` means "inherit the matching global". Body blocks
    # (card/hand/trunk/choice) have all three knobs; readout blocks (value/head)
    # have between/final only. Old checkpoints that predate these fields
    # rehydrate via ``_migrate_legacy_activation_fields`` (REGIME).
    card_between_activation: ActivationName | None = None
    card_final_activation: ActivationName | None = None
    card_dropout: typing.Annotated[float, pydantic.Field(ge=0.0, lt=1.0)] | None = None
    card_layernorm: bool | None = None
    hand_between_activation: ActivationName | None = None
    hand_final_activation: ActivationName | None = None
    hand_dropout: typing.Annotated[float, pydantic.Field(ge=0.0, lt=1.0)] | None = None
    hand_layernorm: bool | None = None
    trunk_between_activation: ActivationName | None = None
    trunk_final_activation: ActivationName | None = None
    trunk_dropout: typing.Annotated[float, pydantic.Field(ge=0.0, lt=1.0)] | None = None
    trunk_layernorm: bool | None = None
    choice_between_activation: ActivationName | None = None
    choice_final_activation: ActivationName | None = None
    choice_dropout: typing.Annotated[float, pydantic.Field(ge=0.0, lt=1.0)] | None = (
        None
    )
    choice_layernorm: bool | None = None
    value_between_activation: ActivationName | None = None
    value_final_activation: ActivationName | None = None
    head_between_activation: ActivationName | None = None
    head_final_activation: ActivationName | None = None

    @pydantic.model_validator(mode="before")
    @classmethod
    def _migrate_legacy_activation_fields(cls, data: object) -> object:
        """Translate ≤0.8 ``activation`` + ``encoder_final_activation`` fields to
        the between/final pair scheme. Runs before field validation so old on-disk
        configs are silently upgraded without requiring a compat shim (REGIME)."""
        if not isinstance(data, dict):
            return data
        raw = typing.cast(dict[str, object], data)
        if "activation" not in raw:
            return raw

        global_act = str(raw.pop("activation"))
        encoder_final: bool = bool(raw.pop("encoder_final_activation", False))

        # New globals: between=old global, final always NONE on migrated runs.
        raw.setdefault("between_activation", global_act)
        raw.setdefault("final_activation", "none")

        # card/hand/choice encoders: between=old override (None = inherit),
        # final=resolved activation when encoder_final was True, else NONE.
        for block in ("card", "hand", "choice"):
            enc_raw = raw.pop(f"{block}_activation", None)
            old_act: str | None = str(enc_raw) if enc_raw is not None else None
            resolved = old_act if old_act is not None else global_act
            raw.setdefault(f"{block}_between_activation", old_act)
            raw.setdefault(
                f"{block}_final_activation", resolved if encoder_final else "none"
            )

        # trunk: was always finalled; bake the resolved activation in explicitly so
        # inheritance logic doesn't affect old configs when between ≠ final.
        trunk_raw = raw.pop("trunk_activation", None)
        old_trunk: str | None = str(trunk_raw) if trunk_raw is not None else None
        resolved_trunk = old_trunk if old_trunk is not None else global_act
        raw.setdefault("trunk_between_activation", old_trunk)
        raw.setdefault("trunk_final_activation", resolved_trunk)

        # value/head readouts: no final activation in old system.
        for block in ("value", "head"):
            vh_raw = raw.pop(f"{block}_activation", None)
            raw.setdefault(f"{block}_between_activation", vh_raw)
            raw.setdefault(f"{block}_final_activation", "none")

        return raw

    @pydantic.model_validator(mode="after")
    def _check_tray_set_embedding(self) -> "ModelArchitecture":
        """The tray-set embedding reuses the hand encoder, so it cannot exist
        without one — reject the combination as a normal validation error."""
        if self.tray_set_embedding and not self.use_distinct_hand_model:
            raise ValueError(
                "tray_set_embedding requires use_distinct_hand_model "
                "(the tray set is embedded through the hand encoder)"
            )
        return self

    @property
    def trunk_embed_width(self) -> int:
        """The trunk's output width ``M`` — what the scorer and value heads consume."""
        return self.trunk_layers[-1]

    @property
    def hand_embed_width(self) -> int:
        """The hand encoder's resolved output width ``N`` — ``hand_embed_dim``
        when set, else ``card_embed_dim`` (the pre-knob behavior)."""
        return (
            self.hand_embed_dim
            if self.hand_embed_dim is not None
            else self.card_embed_dim
        )

    @property
    def pooled_hand_width(self) -> int:
        """Output width of one pooled card-set embedding (``M = card_embed_dim``).

        MEAN / SUM → ``M``; MAX → ``M + 1`` (appends a count scalar); CONCAT_MAX_SUM
        → ``2M + 1`` (max ⊕ sum ⊕ count). Only meaningful when
        ``use_distinct_hand_model`` is ``False``; callers should prefer
        ``hand_embed_width`` for the distinct-encoder path."""
        m = self.card_embed_dim
        if self.hand_pooling in (HandPooling.MEAN, HandPooling.SUM):
            return m
        if self.hand_pooling == HandPooling.MAX:
            return m + _HAND_POOL_COUNT_DIM
        # CONCAT_MAX_SUM
        return 2 * m + _HAND_POOL_COUNT_DIM

    @property
    def choice_embed_width(self) -> int:
        """The choice encoder's output width ``N`` — concatenated with ``M`` for
        scoring."""
        return self.choice_layers[-1]

    def head_layers_for(self, family_index: int) -> Widths:
        """The scorer head hidden widths for the given family index.

        Returns the family's dedicated width list when ``per_family_head_layers``
        is set, or the shared ``head_layers`` in uniform mode."""
        if self.per_family_head_layers is not None:
            return self.per_family_head_layers[family_index]
        return self.head_layers

    # Per-block resolved activations. Each ``*_between_*`` inherits the global
    # ``between_activation``; each ``*_final_*`` inherits ``final_activation``.

    @staticmethod
    def _resolved(
        override: ActivationName | None, fallback: ActivationName
    ) -> ActivationName:
        """Return ``override`` when set (non-None), else ``fallback``."""
        return override if override is not None else fallback

    @property
    def card_between_activation_resolved(self) -> ActivationName:
        """Resolved between-layers activation for the card encoder."""
        return self._resolved(self.card_between_activation, self.between_activation)

    @property
    def card_final_activation_resolved(self) -> ActivationName:
        """Resolved final-layer activation for the card encoder."""
        return self._resolved(self.card_final_activation, self.final_activation)

    @property
    def card_dropout_resolved(self) -> float:
        """Resolved dropout for the card encoder."""
        return self.card_dropout if self.card_dropout is not None else self.dropout

    @property
    def card_layernorm_resolved(self) -> bool:
        """Resolved LayerNorm flag for the card encoder."""
        return (
            self.card_layernorm if self.card_layernorm is not None else self.layernorm
        )

    @property
    def hand_between_activation_resolved(self) -> ActivationName:
        """Resolved between-layers activation for the hand encoder."""
        return self._resolved(self.hand_between_activation, self.between_activation)

    @property
    def hand_final_activation_resolved(self) -> ActivationName:
        """Resolved final-layer activation for the hand encoder."""
        return self._resolved(self.hand_final_activation, self.final_activation)

    @property
    def hand_dropout_resolved(self) -> float:
        """Resolved dropout for the hand encoder."""
        return self.hand_dropout if self.hand_dropout is not None else self.dropout

    @property
    def hand_layernorm_resolved(self) -> bool:
        """Resolved LayerNorm flag for the hand encoder."""
        return (
            self.hand_layernorm if self.hand_layernorm is not None else self.layernorm
        )

    @property
    def trunk_between_activation_resolved(self) -> ActivationName:
        """Resolved between-layers activation for the state trunk."""
        return self._resolved(self.trunk_between_activation, self.between_activation)

    @property
    def trunk_final_activation_resolved(self) -> ActivationName:
        """Resolved final-layer activation for the state trunk."""
        return self._resolved(self.trunk_final_activation, self.final_activation)

    @property
    def trunk_dropout_resolved(self) -> float:
        """Resolved dropout for the state trunk."""
        return self.trunk_dropout if self.trunk_dropout is not None else self.dropout

    @property
    def trunk_layernorm_resolved(self) -> bool:
        """Resolved LayerNorm flag for the state trunk."""
        return (
            self.trunk_layernorm if self.trunk_layernorm is not None else self.layernorm
        )

    @property
    def choice_between_activation_resolved(self) -> ActivationName:
        """Resolved between-layers activation for the choice encoder."""
        return self._resolved(self.choice_between_activation, self.between_activation)

    @property
    def choice_final_activation_resolved(self) -> ActivationName:
        """Resolved final-layer activation for the choice encoder."""
        return self._resolved(self.choice_final_activation, self.final_activation)

    @property
    def choice_dropout_resolved(self) -> float:
        """Resolved dropout for the choice encoder."""
        return self.choice_dropout if self.choice_dropout is not None else self.dropout

    @property
    def choice_layernorm_resolved(self) -> bool:
        """Resolved LayerNorm flag for the choice encoder."""
        return (
            self.choice_layernorm
            if self.choice_layernorm is not None
            else self.layernorm
        )

    @property
    def value_between_activation_resolved(self) -> ActivationName:
        """Resolved between-layers activation for the value head."""
        return self._resolved(self.value_between_activation, self.between_activation)

    @property
    def value_final_activation_resolved(self) -> ActivationName:
        """Resolved final-layer activation for the value head."""
        return self._resolved(self.value_final_activation, self.final_activation)

    @property
    def head_between_activation_resolved(self) -> ActivationName:
        """Resolved between-layers activation for the scorer heads."""
        return self._resolved(self.head_between_activation, self.between_activation)

    @property
    def head_final_activation_resolved(self) -> ActivationName:
        """Resolved final-layer activation for the scorer heads."""
        return self._resolved(self.head_final_activation, self.final_activation)

    @property
    def shape_key(self) -> ShapeKey:
        """The weight-compatibility signature (everything that changes a tensor
        shape). Two architectures load each other's weights iff these agree."""
        return (
            self.trunk_layers,
            self.choice_layers,
            self.head_layers,
            self.value_layers,
            self.card_layernorm_resolved,
            self.hand_layernorm_resolved,
            self.trunk_layernorm_resolved,
            self.choice_layernorm_resolved,
            self.card_embed_dim,
            self.card_encoder_layers,
            self.per_family_head_layers,
            self.use_distinct_hand_model,
            self.hand_encoder_layers,
            self.hand_embed_width,
            self.tray_set_embedding,
            self.use_board_attention,
            # None when distinct (pooling inert) so old distinct artifacts'
            # keys are unaffected; the pooling mode only appears in the key
            # for the pooled path, where it determines the trunk input width.
            None if self.use_distinct_hand_model else self.hand_pooling,
        )


class LayerParam(pydantic.BaseModel):
    """The parameter count of one ``Linear`` layer and the ``LayerNorm`` that may
    follow it. ``linear`` is ``in*out + out`` (weight + bias); ``norm`` is
    ``2*out`` when a LayerNorm follows (its affine weight + bias), else 0. They are
    tracked apart so the diagram can annotate the Linear row with the dominant
    ``linear`` cost while ``params`` still rolls the LayerNorm into the block total."""

    in_features: int
    out_features: int
    linear: int
    norm: int = 0

    @property
    def params(self) -> int:
        """Total trainable parameters in this layer (Linear + any LayerNorm)."""
        return self.linear + self.norm


class BlockParam(pydantic.BaseModel):
    """One network block's parameter breakdown: its per-layer counts plus a flat
    ``extra`` (the embedding table), scaled by ``multiplier`` — the scorer bank
    instantiates one identical readout head per decision family."""

    label: str
    layers: tuple[LayerParam, ...] = ()
    multiplier: int = 1
    extra: int = 0

    @property
    def total(self) -> int:
        """The block's trainable-parameter count, including the family multiplier."""
        return self.multiplier * (
            sum(layer.params for layer in self.layers) + self.extra
        )


class ParamReport(pydantic.BaseModel):
    """The whole network's parameter accounting, block by block — the display
    source for the configurator's per-layer / per-block / total counts. Built by
    :func:`count_parameters`; its block totals sum to ``sum(p.numel())`` of the
    equivalent :class:`model.PolicyValueNet`."""

    embed: BlockParam
    trunk: BlockParam
    choice: BlockParam
    scorer: BlockParam
    value: BlockParam
    # Present only when ``ModelArchitecture.use_distinct_hand_model`` is active;
    # ``None`` in the default (mean-pool) configuration.
    hand: BlockParam | None = None
    # Present only when ``ModelArchitecture.use_board_attention`` is active;
    # ``None`` in the default (no-attention) configuration.
    board_attention: BlockParam | None = None

    @property
    def blocks(self) -> tuple[BlockParam, ...]:
        """The network blocks in flow order: embed, hand (if active),
        board_attention (if active), trunk, choice, scorer, value."""
        result: list[BlockParam] = [self.embed]
        if self.hand is not None:
            result.append(self.hand)
        if self.board_attention is not None:
            result.append(self.board_attention)
        result.extend([self.trunk, self.choice, self.scorer, self.value])
        return tuple(result)

    @property
    def total(self) -> int:
        """The network's total trainable-parameter count."""
        return sum(block.total for block in self.blocks)


def count_parameters(
    arch: ModelArchitecture,
    *,
    card_feat_in: int,
    trunk_in: int,
    choice_in: int,
    num_families: int,
    hand_feat_in: int = 0,
    slot_scalar_dim: int = 9,
) -> ParamReport:
    """Analytic per-block parameter accounting for the network ``arch`` describes.

    Torch-free — it reproduces the exact layer shapes ``mlp.build_body`` /
    ``mlp.build_readout`` would create, so the returned counts equal
    ``sum(p.numel())`` of the built net. The card-encoder input width
    (``card_feat_in``, ``encode.CARD_FEATURE_DIM``) and the effective post-embedding
    trunk / choice input widths (``trunk_in`` / ``choice_in``, from
    ``encode.{trunk,choice}_input_dim``) are passed in to keep this module free of
    the encoder / torch. When ``arch.use_distinct_hand_model`` is active,
    ``hand_feat_in`` (``encode.HAND_ENCODER_INPUT_DIM``) must also be supplied so
    the HAND block's parameter count is correct. ``slot_scalar_dim`` (default 9)
    is the mutable-scalar count per board slot, passed explicitly so this function
    stays encode-free.
    """
    # The trunk ends at width M and the choice encoder at width N; the scorer
    # heads read the M+N concat and the value head reads the trunk's M alone,
    # mirroring ``model.PolicyValueNet.__init__``. The EMBED block is the card
    # encoder MLP: card_feat_in -> card_encoder_layers -> card_embed_dim, built as
    # a body block (no final activation, like the choice encoder).
    trunk_m = arch.trunk_embed_width
    scorer_in = arch.trunk_embed_width + arch.choice_embed_width

    # Build the optional HAND block when the distinct hand encoder is active. Its
    # final width is the resolved set-embedding width N, not card_embed_dim.
    hand_block: BlockParam | None = None
    if arch.use_distinct_hand_model:
        hand_block = BlockParam(
            label="HAND",
            layers=body_layers(
                hand_feat_in,
                arch.hand_encoder_layers + (arch.hand_embed_width,),
                arch,
                layernorm=arch.hand_layernorm_resolved,
            ),
        )

    # Build the optional BOARD ATTENTION block. Each nn.MultiheadAttention with
    # embed_dim=W has: in_proj_weight[3W,W] + in_proj_bias[3W] + out_proj[W,W]
    # + out_proj.bias[W] = 4W² + 4W params. Two modules (own + opp) → 8W² + 8W.
    board_attn_block: BlockParam | None = None
    if arch.use_board_attention:
        token_width = arch.card_embed_dim + slot_scalar_dim
        attn_params = 4 * token_width * token_width + 4 * token_width
        board_attn_block = BlockParam(
            label="BOARD ATTN",
            layers=(),
            multiplier=1,
            extra=2 * attn_params,  # own board + opp board
        )

    return ParamReport(
        embed=BlockParam(
            label="EMBED",
            layers=body_layers(
                card_feat_in,
                arch.card_encoder_layers + (arch.card_embed_dim,),
                arch,
                layernorm=arch.card_layernorm_resolved,
            ),
        ),
        trunk=BlockParam(
            label="TRUNK",
            layers=body_layers(
                trunk_in,
                arch.trunk_layers,
                arch,
                layernorm=arch.trunk_layernorm_resolved,
            ),
        ),
        choice=BlockParam(
            label="CHOICE",
            layers=body_layers(
                choice_in,
                arch.choice_layers,
                arch,
                layernorm=arch.choice_layernorm_resolved,
            ),
        ),
        scorer=_scorer_block(arch, scorer_in, num_families),
        value=BlockParam(
            label="VALUE", layers=readout_layers(trunk_m, arch.value_layers)
        ),
        hand=hand_block,
        board_attention=board_attn_block,
    )


def _scorer_block(
    arch: ModelArchitecture, scorer_in: int, num_families: int
) -> BlockParam:
    # Uniform mode: all heads share one shape, expressed with a multiplier so the
    # arch diagram can show "× N families". Per-family mode: sum each family's
    # readout params individually into ``extra``; per-layer breakdown is omitted
    # (the diagram shows a single aggregate total).
    if arch.per_family_head_layers is not None:
        total = sum(
            sum(lp.params for lp in readout_layers(scorer_in, widths))
            for widths in arch.per_family_head_layers
        )
        return BlockParam(label="SCORER", layers=(), multiplier=1, extra=total)
    return BlockParam(
        label="SCORER",
        layers=readout_layers(scorer_in, arch.head_layers),
        multiplier=num_families,
    )


def _linear_params(in_features: int, out_features: int) -> int:
    """Parameter count of ``nn.Linear(in, out)`` — the weight plus the bias."""
    return in_features * out_features + out_features


def body_layers(
    in_dim: int,
    widths: Widths,
    arch: ModelArchitecture,
    *,
    layernorm: bool | None = None,
) -> tuple[LayerParam, ...]:
    """Per-layer counts for a body block (trunk / choice encoder / card or hand
    encoder): each width is a ``Linear`` followed — when layernorm — by a
    ``LayerNorm`` on every layer, mirroring ``mlp.build_body`` (activation /
    dropout add no params). Public because the setup net's parameter accounting
    (``setup_model.count_setup_parameters``) reuses it for the frozen embedder
    copies the setup net carries.

    ``layernorm`` overrides the block's per-block-resolved value; when ``None``
    the function falls back to ``arch.layernorm`` (the global default, used by
    the setup net and SVG diagram callers that don't carry per-block state)."""
    use_layernorm = layernorm if layernorm is not None else arch.layernorm
    layers: list[LayerParam] = []
    prev = in_dim
    for width in widths:
        norm = 2 * width if use_layernorm else 0
        layers.append(
            LayerParam(
                in_features=prev,
                out_features=width,
                linear=_linear_params(prev, width),
                norm=norm,
            )
        )
        prev = width
    return tuple(layers)


def readout_layers(in_dim: int, widths: Widths) -> tuple[LayerParam, ...]:
    """Per-layer counts for a readout block (scorer head / value head, and the
    separate setup net's MLP): the hidden ``widths`` as bare ``Linear`` layers
    (readouts never LayerNorm) then a final ``Linear(prev, 1)``, mirroring
    ``mlp.build_readout``."""
    layers: list[LayerParam] = []
    prev = in_dim
    for width in widths:
        layers.append(
            LayerParam(
                in_features=prev, out_features=width, linear=_linear_params(prev, width)
            )
        )
        prev = width
    layers.append(
        LayerParam(in_features=prev, out_features=1, linear=_linear_params(prev, 1))
    )
    return tuple(layers)
