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
:func:`setup_state_input_dim` / :func:`setup_choice_input_dim` /
:func:`count_setup_parameters`).

Kept torch-free (only ``pydantic`` / ``enum``) so ``config`` and ``setup_runmeta``
can import it without pulling in torch, mirroring why ``ModelArchitecture`` lives
at the package top level.
"""

from __future__ import annotations

import typing

import pydantic

from wingspan import architecture, encode, state

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
    def bonus_cards_dim(self) -> int:
        """Width of the on-offer ``bonus_cards`` multi-hot — the only bonus-block
        stripe that is action-independent *state* (the deal's bonus offer). It is
        the leading ``_BONUS_DIM`` of the bonus block when ``split_bonus`` is
        active (the trailing 2 dims are the keep-dependent affinity), and ``0``
        otherwise (the folded ``kept_bonus`` one-hot is an action stripe)."""
        return _BONUS_DIM if self.split_bonus else 0

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


# A setup-net shape signature mirroring the in-game two-tower net: the state-trunk
# widths, the choice-trunk widths, the policy-head widths, the value-head widths,
# and whether the policy head (and choice trunk) are present. Activation / dropout
# are excluded — they leave tensor shapes intact and a resumed run may change them
# without invalidating the saved weights.
type SetupShapeKey = tuple[
    tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...], bool
]


class SetupArchitecture(pydantic.BaseModel):
    """The reconstitutable topology of a :class:`wingspan.training.setup_net.SetupNet`,
    a two-tower actor-critic mirroring :class:`wingspan.architecture.ModelArchitecture`.

    The field names mirror the in-game net's descriptor exactly:

    * ``trunk_layers`` — the **state trunk**, encoding the action-independent
      stripes (``setup_state_input_dim``) into a shared ``state_enc``. Its output
      feeds *both* heads, so the value and policy heads share a learned state
      representation (the point of the two-tower design).
    * ``choice_layers`` — the **choice trunk**, encoding the action stripes
      (``setup_choice_input_dim``) into ``choice_enc``; feeds the policy head only.
    * ``head_layers`` — the **policy head** (actor), reading ``cat(state_enc,
      choice_enc)`` and producing the per-candidate selection logit.
    * ``value_layers`` — the **value head** (critic ``V(s)``), reading ``state_enc``
      only. Because ``V(s)`` is a function of state alone, the setup advantage
      ``target − V(s)`` does not self-cancel (``docs/TRAINING.md §6.5``).

    Each width tuple is ordered input-to-output. Reuses
    :data:`wingspan.architecture.Widths` and
    :class:`wingspan.architecture.ActivationName` so the configurator's layer /
    activation editors apply unchanged. ``between_activation`` is applied after
    each hidden ``Linear`` in the trunks and heads; the trunks additionally use it
    as their own final activation so ``state_enc`` / ``choice_enc`` are nonlinear
    before the heads' first ``Linear``. ``final_activation`` (default NONE) follows
    the heads' final ``Linear(·, 1)`` — keep NONE for the standard bare-scalar
    readout.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    # The state trunk (over the action-independent stripes); shared by both heads.
    trunk_layers: typing.Annotated[
        architecture.Widths, pydantic.Field(min_length=1)
    ] = (128,)
    # The choice trunk (over the action stripes); feeds the policy head only.
    choice_layers: typing.Annotated[
        architecture.Widths, pydantic.Field(min_length=1)
    ] = (128,)
    # The policy head (over cat(state_enc, choice_enc)).
    head_layers: architecture.Widths = (128,)
    # The value head (over state_enc). Empty → a single bare Linear(state_enc, 1).
    value_layers: architecture.Widths = ()
    between_activation: architecture.ActivationName = architecture.ActivationName.RELU
    final_activation: architecture.ActivationName = architecture.ActivationName.NONE
    dropout: typing.Annotated[float, pydantic.Field(ge=0.0, lt=1.0)] = 0.0
    # Always True: the setup net always trains actor-critic (value + policy heads).
    # When False (value-only) there is no choice trunk and no policy head.
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
        """The net's weight-compatibility signature (everything that changes one of
        its tensor shapes): the state trunk, the choice trunk, the policy head, the
        value head, and whether the policy head is present. The embedder copies'
        shapes ride the main architecture and are keyed separately
        (``TrainConfig.setup_architecture_key``)."""
        return (
            self.trunk_layers,
            self.choice_layers,
            self.head_layers,
            self.value_layers,
            self.use_policy_head,
        )


def setup_choice_input_dim(
    encoding: SetupEncoding, main_arch: architecture.ModelArchitecture
) -> int:
    """The setup **choice/policy** trunk's first-``Linear`` input width: the
    action-dependent (keep) stripes only.

    The choice trunk reads everything ``V(s)`` does not: the kept-cards set
    (embedded as one ``W``-wide set vector, ``W = main_arch.pooled_hand_width``
    when not using a distinct hand model, else ``main_arch.hand_embed_width``),
    kept foods (omitted in ``split_food`` mode), the bonus action (the folded
    ``kept_bonus`` one-hot, or the ``bonus_card_affinity`` min/max in
    ``split_bonus`` mode), ``kept_bonus_value`` (folded mode only), the per-round
    ``goal_affinity``, and each appended playability multi-hot (``turn1_playable``
    / ``playable_kept_cards``) embedded as one extra set vector.

    Together with :func:`setup_state_input_dim` this partitions the embedded
    candidate vector exactly: ``setup_choice_input_dim + setup_state_input_dim``
    equals the fused readout width the prior single-tower net used."""
    set_width = (
        main_arch.hand_embed_width
        if main_arch.use_distinct_hand_model
        else main_arch.pooled_hand_width
    )
    foods = 0 if encoding.split_food else _KEPT_FOODS_DIM
    # The bonus block minus its leading on-offer (state) stripe: the folded
    # kept_bonus one-hot (26) or the split bonus_card_affinity min/max (2).
    bonus_action = encoding.bonus_block_dim - encoding.bonus_cards_dim
    bonus_value = 0 if encoding.split_bonus else _KEPT_BONUS_VALUE_DIM
    n_card_sets = (
        1  # kept set
        + int(encoding.include_turn1_playable)
        + int(encoding.include_playable_kept_cards)
    )
    return (
        n_card_sets * set_width
        + foods
        + bonus_action
        + bonus_value
        + _GOAL_AFFINITY_DIM
    )


def setup_state_input_dim(
    encoding: SetupEncoding, main_arch: architecture.ModelArchitecture
) -> int:
    """The setup **value** head's first-``Linear`` input width: the
    action-independent (state) stripes only.

    ``V(s)`` reads the deal context that is identical across every keep candidate
    — the tray (embedded as ``TRAY_SIZE`` ``M``-wide card-table rows, ``M =
    card_embed_dim``), the raw birdfeeder + round-goal passthrough, and the
    bonus-cards-on-offer multi-hot (present only in ``split_bonus`` mode). It
    excludes every keep-dependent stripe (kept cards/foods/bonus, affinities,
    pricing, playability), so the value head cannot see the action — the property
    that makes it a true ``V(s)`` baseline rather than ``Q(s, a)``."""
    return (
        state.TRAY_SIZE * main_arch.card_embed_dim
        + _FEEDER_DIM
        + _GOALS_DIM
        + encoding.bonus_cards_dim
    )


class SetupParamReport(pydantic.BaseModel):
    """Per-block parameter accounting for a :class:`wingspan.training.setup_net.SetupNet`.

    Breaks the two-tower setup net into tiers: the frozen embedder copies
    (``embedder_params``), the **state trunk** (``state_trunk``, over the
    action-independent stripes; feeds both heads), the **choice trunk**
    (``choice_trunk``, over the action stripes; empty in value-only mode), and the
    per-head readout MLPs — ``value_head`` reads ``state_enc`` and ``policy_head``
    reads ``cat(state_enc, choice_enc)``. Built by :func:`count_setup_parameters`;
    its ``total`` equals ``sum(p.numel())`` of the real module — the architecture
    diagram's per-op and Σ source.
    """

    embedder_params: int
    # The state trunk (over the action-independent stripes); feeds both heads.
    state_trunk: tuple[architecture.LayerParam, ...]
    # The choice trunk (over the action stripes); empty when use_policy_head=False.
    choice_trunk: tuple[architecture.LayerParam, ...]
    value_head: tuple[architecture.LayerParam, ...]
    # None when use_policy_head=False (value-only mode).
    policy_head: tuple[architecture.LayerParam, ...] | None

    @property
    def state_trunk_params(self) -> int:
        """Total trainable parameters in the state trunk."""
        return sum(layer.params for layer in self.state_trunk)

    @property
    def choice_trunk_params(self) -> int:
        """Total trainable parameters in the choice trunk (0 in value-only mode)."""
        return sum(layer.params for layer in self.choice_trunk)

    @property
    def state_enc_dim(self) -> int:
        """Width of the shared state encoding (the state trunk's output)."""
        return self.state_trunk[-1].out_features

    @property
    def choice_enc_dim(self) -> int:
        """Width of the choice encoding (the choice trunk's output); 0 value-only."""
        return self.choice_trunk[-1].out_features if self.choice_trunk else 0

    @property
    def value_in(self) -> int:
        """Input width to the value head: the shared state encoding."""
        return self.state_enc_dim

    @property
    def policy_in(self) -> int:
        """Input width to the policy head: state encoding ⊕ choice encoding."""
        return self.state_enc_dim + self.choice_enc_dim

    @property
    def total(self) -> int:
        """Total trainable-parameter count (embedder + both trunks + heads).

        Embedder copies are frozen at inference but their parameters are counted
        because ``sum(p.numel())`` includes all parameters regardless of
        ``requires_grad``."""
        head_params = sum(layer.params for layer in self.value_head)
        if self.policy_head is not None:
            head_params += sum(layer.params for layer in self.policy_head)
        return (
            self.embedder_params
            + self.state_trunk_params
            + self.choice_trunk_params
            + head_params
        )


def count_setup_parameters(
    setup_arch: SetupArchitecture,
    *,
    main_arch: architecture.ModelArchitecture | None = None,
    encoding: SetupEncoding | None = None,
) -> SetupParamReport:
    """Analytic per-block parameter accounting for the ``SetupNet`` that
    ``setup_arch`` describes.

    Accounts for the frozen embedder copies (card + hand encoders, shaped by
    ``main_arch``), the state and choice trunks, and the per-head readout MLPs.
    Returns a :class:`SetupParamReport` whose ``total`` equals ``sum(p.numel())``
    of the built ``SetupNet`` — the architecture diagram's per-op and Σ source for
    the separate setup model.

    ``encoding`` controls which optional stripes are present in the state- and
    choice-input width calculations (default: ``SetupEncoding()``).
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

    # State trunk: over the action-independent stripes, feeding both heads. No
    # LayerNorm; uses between_activation as its own final activation so the output
    # is nonlinear before the heads' first Linear.
    state_in = setup_state_input_dim(encoding, main_arch)
    state_trunk = architecture.body_layers(
        state_in, setup_arch.trunk_layers, main_arch, layernorm=False
    )
    state_enc_dim = setup_arch.trunk_layers[-1]

    # Choice trunk: over the action stripes, feeding the policy head only. Present
    # only with a policy head (value-only mode has no choice trunk).
    choice_trunk: tuple[architecture.LayerParam, ...] = ()
    choice_enc_dim = 0
    if setup_arch.use_policy_head:
        choice_in = setup_choice_input_dim(encoding, main_arch)
        choice_trunk = architecture.body_layers(
            choice_in, setup_arch.choice_layers, main_arch, layernorm=False
        )
        choice_enc_dim = setup_arch.choice_layers[-1]

    value_head = architecture.readout_layers(state_enc_dim, setup_arch.value_layers)
    policy_head = (
        architecture.readout_layers(
            state_enc_dim + choice_enc_dim, setup_arch.head_layers
        )
        if setup_arch.use_policy_head
        else None
    )

    return SetupParamReport(
        embedder_params=embedder_params,
        state_trunk=state_trunk,
        choice_trunk=choice_trunk,
        value_head=value_head,
        policy_head=policy_head,
    )
