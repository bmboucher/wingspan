"""Training-run configuration.

``RunConfig`` is the single self-describing record of every hyperparameter a
run uses. It is stored verbatim inside every checkpoint (TRAINING.md §5.1) so a
run can be resumed and its results re-derived later. Config fields are organized
into six top-level sections:

* ``architecture`` — network topology + encoding shape toggles + synced dims
* ``run``          — loop shape, evaluation cadence, checkpoint / identity
* ``training``     — optimizer knobs, reward scheme, setup-net training
* ``opponent``     — bootstrap phase, self-play graduation, eval smoothing
* ``engine``       — placeholder for future encoding-independent game variants
* ``misc``         — seed, device, dashboard smoothing, instrumentation

The ``architecture`` section carries ``encoding_version`` which pins the run to
its artifact era: the dims are era-routed from it, every artifact the run writes
is stamped with it, and a resumed run adopts it from its checkpoint — so a FRESH
encoding change never orphans an in-flight run (``docs/VERSIONING.md``).

The defaults encode the TRAINING.md Phase-1 program: a synchronous
REINFORCE-with-value-baseline loop, advantage normalization, no epsilon-greedy,
sized by *games* per iteration, with a paired-game evaluation against the random
agent every few iterations.

On-disk, each training session writes one dated ``run_config_<stamp>.json``
file (a :class:`RunConfigFile` wrapper). Legacy ≤0.4 run directories still
carry the three-file layout (``model_config.json``, ``setup_config.json``,
``process_<stamp>.json``); the readers in ``runmeta`` / ``setup_runmeta``
dispatch on presence to remain backward-compatible.
"""

from __future__ import annotations

import enum
import typing

import pydantic

from wingspan import architecture, decisions, encode, setup_model, version
from wingspan.instrumentation import config as instrumentation_config


class RewardMode(enum.StrEnum):
    """How credit is assigned across decisions from a finished game.

    ``TERMINAL_MARGIN`` broadcasts the single end-of-game score (margin or
    own score, per ``RewardBasis``) to every decision. ``DECISION_DELTA``
    instead credits each decision with the change in the player's value from
    that decision onward, accumulated into a return discounted by
    ``reward_discount`` per unit of game-clock time between decisions
    (``Step.timestamp`` — one unit per game turn). ``GAE`` uses the
    Generalized Advantage Estimation kernel: the TD residuals
    ``δ_t = r_t + γV(s_{t+1}) − V(s_t)`` are accumulated backward with
    ``(γλ)^Δt`` decay (``gae_lambda`` controls λ); value targets are
    ``A_t + V(s_t)``. Requires ``behavior_logp`` / ``value_pred`` captured at
    collection time. All modes are shape-preserving (REGIME).
    """

    TERMINAL_MARGIN = "terminal_margin"
    DECISION_DELTA = "decision_delta"
    GAE = "gae"


class PolicyLoss(enum.StrEnum):
    """The policy-gradient objective used in the gradient update.

    ``REINFORCE`` applies the standard log-probability-weighted advantage loss
    ``−(log π · A).mean()``.  ``PPO`` uses the clipped surrogate
    ``−min(ratio·A, clip(ratio, 1±ε)·A).mean()`` (Schulman et al. 2017),
    where ``ratio = exp(logπ_new − logπ_old)`` and ``logπ_old`` is captured at
    collection time (``Step.behavior_logp``).  PPO enables safe multi-epoch
    reuse of each collected batch (``ppo_reuse_epochs``).  Both are
    shape-preserving (REGIME).
    """

    REINFORCE = "reinforce"
    PPO = "ppo"


class RewardBasis(enum.StrEnum):
    """What quantity is used as the reward signal after a finished game.

    ``MARGIN`` uses the score margin (own − opponent) so each seat gets
    opposite-sign rewards — gradient pushes toward beating the opponent.
    ``OWN_SCORE`` uses each player's own absolute final score so both seats
    receive positive rewards — gradient pushes toward maximizing raw score
    regardless of what the opponent scores. Shape-preserving (REGIME) —
    toggling never restarts the network.
    """

    MARGIN = "margin"
    OWN_SCORE = "own_score"


def _default_family_order() -> tuple[str, ...]:
    """The stable judgment-family head order, as strings, for the checkpoint
    descriptor (mirrors ``decisions.ALL_DECISION_FAMILIES``)."""
    return tuple(family.value for family in decisions.ALL_DECISION_FAMILIES)


# ---------------------------------------------------------------------------
# Section sub-models
# ---------------------------------------------------------------------------


class MainNetArchitecture(pydantic.BaseModel):
    """Topology fields for the main policy-value network."""

    # Body block hidden widths (input-to-output). The trunk ends at width M and
    # the choice encoder at width N; their outputs are concatenated to M+N for
    # the scorer heads. These flat fields mirror ``architecture.ModelArchitecture``
    # so the configurator can edit each one independently; ``RunConfig.arch``
    # assembles the frozen descriptor.
    trunk_layers: typing.Annotated[
        architecture.Widths, pydantic.Field(min_length=1)
    ] = (128, 128)
    choice_layers: typing.Annotated[
        architecture.Widths, pydantic.Field(min_length=1)
    ] = (128, 128)
    head_layers: architecture.Widths = (128,)
    value_layers: architecture.Widths = ()

    # "uniform" uses head_layers for every family; "per_family" uses the
    # per-family fields below. Fresh run when changed (head shapes differ).
    head_layers_mode: typing.Literal["uniform", "per_family"] = "uniform"

    # Per-family scorer heads — one per DecisionFamily in ALL_DECISION_FAMILIES
    # order. Active only in "per_family" mode; defaults match head_layers.
    head_layers_main_action: architecture.Widths = (128,)
    head_layers_draw_bird: architecture.Widths = (128,)
    head_layers_discard_bird: architecture.Widths = (128,)
    head_layers_gain_food: architecture.Widths = (128,)
    head_layers_spend_food: architecture.Widths = (128,)
    head_layers_lay_egg: architecture.Widths = (128,)
    head_layers_pay_egg: architecture.Widths = (128,)
    head_layers_skip_optional: architecture.Widths = (128,)
    head_layers_choose_bonus: architecture.Widths = (128,)
    head_layers_misc_rare: architecture.Widths = (128,)
    head_layers_play_bird: architecture.Widths = (128,)
    head_layers_reset_birdfeeder: architecture.Widths = (128,)
    head_layers_setup: architecture.Widths = (128,)

    activation: architecture.ActivationName = architecture.ActivationName.RELU
    dropout: typing.Annotated[float, pydantic.Field(ge=0.0, lt=1.0)] = 0.0
    layernorm: bool = False

    # Shared per-card embedding (one representation per core-set bird, reused
    # for every board / tray / hand / choice card slot).
    card_embed_dim: typing.Annotated[int, pydantic.Field(ge=1)] = 64
    card_encoder_layers: architecture.Widths = (128,)

    # Distinct hand-encoder MLP replacing mean-pool hand embedding. Fresh run.
    use_distinct_hand_model: bool = True
    hand_encoder_layers: architecture.Widths = (128,)
    hand_embed_dim: typing.Annotated[int, pydantic.Field(ge=1)] | None = None
    # Locked to False: new runs never embed the tray as a set (old checkpoints
    # carry their own value and continue working).
    tray_set_embedding: bool = False

    # When enabled, card/hand/choice encoders apply a final activation after
    # their last layer. Saved so old checkpoints keep their original behaviour.
    encoder_final_activation: bool = True

    # Per-block activation/dropout/layernorm overrides (None = inherit global).
    # Old run files that predate these fields rehydrate to None→global (REGIME).
    card_activation: architecture.ActivationName | None = None
    card_dropout: typing.Annotated[float, pydantic.Field(ge=0.0, lt=1.0)] | None = None
    card_layernorm: bool | None = None
    hand_activation: architecture.ActivationName | None = None
    hand_dropout: typing.Annotated[float, pydantic.Field(ge=0.0, lt=1.0)] | None = None
    hand_layernorm: bool | None = None
    trunk_activation: architecture.ActivationName | None = None
    trunk_dropout: typing.Annotated[float, pydantic.Field(ge=0.0, lt=1.0)] | None = None
    trunk_layernorm: bool | None = None
    choice_activation: architecture.ActivationName | None = None
    choice_dropout: typing.Annotated[float, pydantic.Field(ge=0.0, lt=1.0)] | None = (
        None
    )
    choice_layernorm: bool | None = None
    value_activation: architecture.ActivationName | None = None
    head_activation: architecture.ActivationName | None = None


class SetupNetArchitecture(pydantic.BaseModel):
    """Topology fields for the setup-model MLP."""

    hidden_layers: typing.Annotated[
        architecture.Widths, pydantic.Field(min_length=1)
    ] = (128, 64)
    activation: architecture.ActivationName = architecture.ActivationName.RELU
    dropout: typing.Annotated[float, pydantic.Field(ge=0.0, lt=1.0)] = 0.0
    # Locked to True: new runs always use actor-critic for the setup net.
    # Old checkpoints carry their own value (False = value-only) and still load.
    use_actor_critic: bool = True


class ArchitectureConfig(pydantic.BaseModel):
    """Network topology + encoding-shape toggles + era-synced dims.

    The ``main`` and ``setup`` sub-models carry the flat topology knobs the
    configurator edits; ``RunConfig.arch`` / ``RunConfig.setup_arch`` assemble
    these into the frozen ``ModelArchitecture`` / ``SetupArchitecture`` the
    model builders consume. The synced fields (``encoding_version``,
    ``state_dim``, ``choice_dim``, ``family_order``) are era-derived from the
    other fields and must not be edited directly.
    """

    main: MainNetArchitecture = pydantic.Field(default_factory=MainNetArchitecture)
    setup: SetupNetArchitecture = pydantic.Field(default_factory=SetupNetArchitecture)

    # Encoding-shape toggles (also affect main-net ``architecture_key``).
    use_setup_model: bool = True
    split_setup_bonus: bool = False
    split_setup_food: bool = False

    # Era-synced descriptor (derived, not freely editable).
    encoding_version: str = version.MODEL_VERSION
    state_dim: int = pydantic.Field(default_factory=encode.state_size)
    choice_dim: int = encode.CHOICE_FEATURE_DIM
    family_order: tuple[str, ...] = pydantic.Field(
        default_factory=_default_family_order
    )

    @pydantic.model_validator(mode="after")
    def _check_encoding_version(self) -> "ArchitectureConfig":
        """The run's era must be one this code can load."""
        try:
            version.check_artifact_compatible(
                self.encoding_version, what="encoding_version"
            )
        except version.IncompatibleArtifactError as error:
            raise ValueError(str(error)) from error
        return self

    @pydantic.model_validator(mode="after")
    def _sync_dims(self) -> "ArchitectureConfig":
        """Keep dims and family order in lockstep with ``use_setup_model`` and
        ``encoding_version``. See ``TrainConfig._sync_encoding_dims`` docs."""
        spec = encode.spec_for(self.use_setup_model)
        if self.encoding_version == version.MODEL_VERSION:
            self.state_dim = encode.state_size(spec)
            self.choice_dim = encode.choice_feature_dim(spec)
        else:
            from wingspan import compat  # noqa: PLC0415 — see live-era note

            self.state_dim, self.choice_dim = compat.encoding_dims_for_era(
                self.encoding_version, spec
            )
        self.family_order = tuple(
            family.value
            for family in decisions.active_decision_families(spec.include_setup)
        )
        return self


class RunSettings(pydantic.BaseModel):
    """Loop shape, evaluation cadence, and run identity."""

    # Training games collected (and learned from) per collect→update cycle.
    games_per_iter: typing.Annotated[int, pydantic.Field(ge=1)] = 256
    max_iterations: typing.Annotated[int, pydantic.Field(ge=0)] = 0  # 0 = forever
    # Pause at this iteration for a final evaluation + user acknowledgment.
    # 0 = no target. Must be ≤ max_iterations when both are > 0.
    target_iterations: typing.Annotated[int, pydantic.Field(ge=0)] = 0
    # Self-play games at the target milestone. 0 = auto: 10 × eval_games.
    target_eval_games: typing.Annotated[int, pydantic.Field(ge=0)] = 0
    # Eval cadence: run an eval block every N iterations (0 disables eval).
    eval_every: typing.Annotated[int, pydantic.Field(ge=0)] = 5
    # Held-out games per eval block (played as mirrored pairs).
    eval_games: typing.Annotated[int, pydantic.Field(ge=0)] = 128
    # Checkpoint / identity.
    checkpoint_dir: str = "checkpoints"
    run_name: str = "dashboard"
    resume: bool = True
    # In-memory iterations retained for the live convergence charts.
    history_len: typing.Annotated[int, pydantic.Field(ge=1)] = 1024


class SetupTrainingConfig(pydantic.BaseModel):
    """Training knobs for the setup network."""

    lr: typing.Annotated[float, pydantic.Field(gt=0.0)] = 1e-3
    policy_temperature: typing.Annotated[float, pydantic.Field(gt=0.0)] = 0.5
    policy_greedy: bool = False
    # (0, 0) = train from start; otherwise recording window must be non-empty.
    record_start_iter: typing.Annotated[int, pydantic.Field(ge=0)] = 0
    train_iter: typing.Annotated[int, pydantic.Field(ge=0)] = 0
    # Random-generation knobs (per batch).
    hand_combos: typing.Annotated[int, pydantic.Field(ge=1)] = 10
    food_sets: typing.Annotated[int, pydantic.Field(ge=1)] = 3
    tuples_per_batch: typing.Annotated[int, pydantic.Field(ge=1)] = 16
    # One-time offline fit epochs and minibatch size.
    offline_epochs: typing.Annotated[int, pydantic.Field(ge=1)] = 20
    offline_batch_size: typing.Annotated[int, pydantic.Field(ge=1)] = 256
    # Actor-critic loss weights (ignored when use_actor_critic is False).
    pg_coef: typing.Annotated[float, pydantic.Field(ge=0.0)] = 1.0
    value_coef: typing.Annotated[float, pydantic.Field(ge=0.0)] = 0.5
    entropy_coef: typing.Annotated[float, pydantic.Field(ge=0.0)] = 0.01


class TrainingConfig(pydantic.BaseModel):
    """Optimizer knobs, reward scheme, and setup-net training config."""

    lr: typing.Annotated[float, pydantic.Field(gt=0.0)] = 3e-4
    value_coef: typing.Annotated[float, pydantic.Field(ge=0.0)] = 0.5
    entropy_coef: typing.Annotated[float, pydantic.Field(ge=0.0)] = 0.01
    grad_clip: typing.Annotated[float, pydantic.Field(gt=0.0)] = 5.0
    score_norm: typing.Annotated[float, pydantic.Field(gt=0.0)] = 50.0
    reward_mode: RewardMode = RewardMode.TERMINAL_MARGIN
    reward_basis: RewardBasis = RewardBasis.MARGIN
    reward_discount: typing.Annotated[float, pydantic.Field(ge=0.0, le=1.0)] = 1.0
    end_game_bonus: float = 0.0
    setup: SetupTrainingConfig = pydantic.Field(default_factory=SetupTrainingConfig)
    # PPO / GAE algorithm knobs — REGIME (shape-preserving, no MODEL_VERSION bump).
    # Defaults reproduce today's REINFORCE behaviour exactly.
    policy_loss: PolicyLoss = PolicyLoss.REINFORCE
    ppo_clip_eps: typing.Annotated[float, pydantic.Field(gt=0.0)] = 0.2
    ppo_reuse_epochs: typing.Annotated[int, pydantic.Field(ge=1)] = 4
    gae_lambda: typing.Annotated[float, pydantic.Field(ge=0.0, le=1.0)] = 0.95


class OpponentConfig(pydantic.BaseModel):
    """Bootstrap phase, self-play graduation, and eval smoothing."""

    # "none" = no bootstrap; "random" = built-in random agent; <path> = ckpt.
    bootstrap_opponent: str = "random"
    random_phase_win_rate: typing.Annotated[float, pydantic.Field(gt=0.0, le=1.0)] = (
        0.65
    )
    opponent_reset_win_rate: typing.Annotated[float, pydantic.Field(ge=0.0, le=1.0)] = (
        0.95
    )
    opponent_max_iterations: typing.Annotated[int, pydantic.Field(ge=0)] = 500
    eval_ewma_alpha: typing.Annotated[float, pydantic.Field(gt=0.0, le=1.0)] = 0.3


class EngineConfig(pydantic.BaseModel):
    """Placeholder for future encoding-independent game-variant toggles.

    Every game-rule constant (``ROUND_CUBES``, ``ROW_SLOTS``, …) is hardcoded
    in ``state.py`` and is FRESH-coupled to the encoding. The only existing
    config-level rule toggles (``use_setup_model``, ``split_setup_*``) change
    tensor shapes, so they belong in ``architecture``. This section is reserved
    for future knobs that do not affect the encoding (e.g. player count)."""


class MiscConfig(pydantic.BaseModel):
    """Seed, device, dashboard smoothing, and instrumentation."""

    seed: typing.Annotated[int, pydantic.Field(ge=0)] = 0
    device: str = "cpu"
    # Decay for the PRODUCING band's EWMA (dashboard display only).
    produce_ewma_alpha: typing.Annotated[float, pydantic.Field(gt=0.0, le=1.0)] = 0.2
    # Custom event-callback recorders (see ``wingspan.instrumentation``).
    # Empty by default — no handlers, no overhead.
    instrumentation: instrumentation_config.InstrumentationConfig = pydantic.Field(
        default_factory=instrumentation_config.InstrumentationConfig
    )


class DaggerConfig(pydantic.BaseModel):
    """DAgger behavioral cloning configuration (7th RunConfig section).

    When ``expert_checkpoint`` names a ``.pt`` checkpoint, the first
    ``clone_iters`` training iterations collect games with the *student* policy
    but label each multi-option decision with the frozen expert's soft policy
    distribution.  The learner then minimizes cross-entropy to those targets
    instead of running the normal REINFORCE actor-critic update — pure imitation
    for ``clone_iters`` iterations, then the expert is dropped and training
    reverts to the standard RL loop (TRAINING.md DAgger section).

    This is a REGIME change: all fields are config-carried and training-only,
    add no tensor-shape geometry, and leave ``encode_state``/``encode_choices``/
    ``PolicyValueNet.forward`` untouched — so a rehydrated artifact computes
    identically at play time and no ``MODEL_VERSION`` bump is needed.
    """

    # ``"none"`` disables DAgger; a ``.pt`` path loads the expert checkpoint.
    # ``"random"`` is accepted by the cycling widget but treated as ``"none"``
    # by ``RunConfig.dagger_expert_checkpoint`` so the feature stays inactive.
    expert_checkpoint: str = "none"
    # Number of initial iterations to run in pure-imitation mode before
    # switching to the normal RL loop. 0 = never clone (default).
    clone_iters: typing.Annotated[int, pydantic.Field(ge=0)] = 0


# ---------------------------------------------------------------------------
# Top-level RunConfig
# ---------------------------------------------------------------------------


class RunConfig(pydantic.BaseModel):
    """Every hyperparameter for one training run, versioned and self-describing.

    Organized into seven sections; all computed properties (``arch``,
    ``setup_arch``, ``encoding_spec``, ``architecture_key``, etc.) stay at the
    top level so their heavily-used call sites do not churn.
    """

    architecture: ArchitectureConfig = pydantic.Field(
        default_factory=ArchitectureConfig
    )
    run: RunSettings = pydantic.Field(default_factory=RunSettings)
    training: TrainingConfig = pydantic.Field(default_factory=TrainingConfig)
    opponent: OpponentConfig = pydantic.Field(default_factory=OpponentConfig)
    engine: EngineConfig = pydantic.Field(default_factory=EngineConfig)
    misc: MiscConfig = pydantic.Field(default_factory=MiscConfig)
    dagger: DaggerConfig = pydantic.Field(default_factory=DaggerConfig)

    # ------------------------------------------------------------------
    # Cross-section validators (launch-time only — see validate_launchable)
    # ------------------------------------------------------------------

    @pydantic.model_validator(mode="after")
    def _check_architecture(self) -> "RunConfig":
        """Verify the topology descriptor assembles without error."""
        _ = self.arch
        return self

    # ------------------------------------------------------------------
    # Computed properties (top-level, delegating into sections)
    # ------------------------------------------------------------------

    @property
    def initial_vs_random(self) -> bool:
        """Whether the bootstrap phase is active."""
        return self.opponent.bootstrap_opponent != "none"

    @property
    def dagger_expert_checkpoint(self) -> str | None:
        """The checkpoint path to load as the DAgger expert, or ``None``.

        Clone always targets the bootstrap opponent — there is no separate
        expert-checkpoint field; ``dagger.expert_checkpoint`` is retained for
        old-file loading but ignored here. Clone is only possible when a
        checkpoint bootstrap opponent is configured (never "clone random")."""
        return self.bootstrap_opponent_checkpoint

    def dagger_active_at(self, iteration: int) -> bool:
        """Whether DAgger expert labeling is active for ``iteration``.

        True only when an expert checkpoint is configured and the iteration is
        within the first ``clone_iters`` iterations (the pure-imitation window).
        """
        return (
            self.dagger_expert_checkpoint is not None
            and iteration < self.dagger.clone_iters
        )

    @property
    def bootstrap_opponent_checkpoint(self) -> str | None:
        """The checkpoint path to load as the bootstrap opponent, or ``None``."""
        return (
            None
            if self.opponent.bootstrap_opponent in ("none", "random")
            else self.opponent.bootstrap_opponent
        )

    @property
    def encoding_spec(self) -> encode.EncodingSpec:
        """The state/choice encoding spec implied by ``use_setup_model``."""
        return encode.spec_for(self.architecture.use_setup_model)

    @property
    def split_setup_bonus_active(self) -> bool:
        """Whether the opening's bonus pick is deferred to the in-game head."""
        return self.architecture.split_setup_bonus and self.architecture.use_setup_model

    @property
    def split_setup_food_active(self) -> bool:
        """Whether the opening food pick is deferred to in-game decisions."""
        return self.architecture.split_setup_food and self.architecture.use_setup_model

    @property
    def eval_pairs(self) -> int:
        """Mirror-deal pairs per eval block."""
        return self.run.eval_games // 2

    @property
    def effective_target_eval_games(self) -> int:
        """Games run at the target milestone (explicit wins; 0 = 10 × eval_games)."""
        return (
            self.run.target_eval_games
            if self.run.target_eval_games > 0
            else 10 * self.run.eval_games
        )

    @property
    def arch(self) -> architecture.ModelArchitecture:
        """The network topology descriptor assembled from the flat topology
        fields. Named ``arch`` (not ``architecture``) so it never shadows the
        ``architecture`` section in this class's field annotations."""
        main = self.architecture.main
        spec = self.encoding_spec
        per_family: tuple[architecture.Widths, ...] | None = None
        if main.head_layers_mode == "per_family":
            active = decisions.active_decision_families(spec.include_setup)
            per_family = tuple(
                typing.cast(
                    architecture.Widths,
                    getattr(main, f"head_layers_{family.value}"),
                )
                for family in active
            )
        return architecture.ModelArchitecture(
            trunk_layers=main.trunk_layers,
            choice_layers=main.choice_layers,
            head_layers=main.head_layers,
            value_layers=main.value_layers,
            per_family_head_layers=per_family,
            activation=main.activation,
            dropout=main.dropout,
            layernorm=main.layernorm,
            card_embed_dim=main.card_embed_dim,
            card_encoder_layers=main.card_encoder_layers,
            use_distinct_hand_model=main.use_distinct_hand_model,
            hand_encoder_layers=main.hand_encoder_layers,
            hand_embed_dim=main.hand_embed_dim,
            tray_set_embedding=main.tray_set_embedding,
            encoder_final_activation=main.encoder_final_activation,
            card_activation=main.card_activation,
            card_dropout=main.card_dropout,
            card_layernorm=main.card_layernorm,
            hand_activation=main.hand_activation,
            hand_dropout=main.hand_dropout,
            hand_layernorm=main.hand_layernorm,
            trunk_activation=main.trunk_activation,
            trunk_dropout=main.trunk_dropout,
            trunk_layernorm=main.trunk_layernorm,
            choice_activation=main.choice_activation,
            choice_dropout=main.choice_dropout,
            choice_layernorm=main.choice_layernorm,
            value_activation=main.value_activation,
            head_activation=main.head_activation,
        )

    @property
    def setup_arch(self) -> setup_model.SetupArchitecture:
        """The setup network's topology descriptor."""
        setup = self.architecture.setup
        return setup_model.SetupArchitecture(
            hidden_layers=setup.hidden_layers,
            activation=setup.activation,
            dropout=setup.dropout,
            use_policy_head=setup.use_actor_critic,
        )

    @property
    def setup_encoding(self) -> setup_model.SetupEncoding:
        """The setup input-vector layout implied by the active split flags."""
        return setup_model.SetupEncoding(
            split_food=self.split_setup_food_active,
            split_bonus=self.split_setup_bonus_active,
        )

    @property
    def setup_architecture_key(
        self,
    ) -> tuple[
        int,
        setup_model.SetupShapeKey,
        tuple[tuple[int, ...], int, bool, bool, tuple[int, ...], int],
    ]:
        """The setup-net shape signature a ``setup.pt`` must match to be resumed."""
        arch = self.arch
        return (
            self.setup_encoding.total_dim,
            self.setup_arch.shape_key,
            (
                arch.card_encoder_layers,
                arch.card_embed_dim,
                arch.card_layernorm_resolved,
                arch.use_distinct_hand_model,
                arch.hand_encoder_layers,
                arch.hand_embed_width,
            ),
        )

    @property
    def trunk_hidden(self) -> int:
        """The trunk's output width ``M``."""
        return self.architecture.main.trunk_layers[-1]

    @property
    def choice_hidden(self) -> int:
        """The choice encoder's output width ``N``."""
        return self.architecture.main.choice_layers[-1]

    @property
    def architecture_key(
        self,
    ) -> tuple[str, int, int, tuple[str, ...], architecture.ShapeKey]:
        """The network-shape signature a checkpoint must match to be resumed."""
        arch_cfg = self.architecture
        return (
            arch_cfg.encoding_version,
            arch_cfg.state_dim,
            arch_cfg.choice_dim,
            arch_cfg.family_order,
            self.arch.shape_key,
        )

    # ------------------------------------------------------------------
    # Convenience flat-field shortcuts used heavily in the training loop
    # ------------------------------------------------------------------

    @property
    def encoding_version(self) -> str:
        """The artifact era (delegates to ``architecture.encoding_version``)."""
        return self.architecture.encoding_version

    @property
    def state_dim(self) -> int:
        """Synced state-vector width (delegates to ``architecture.state_dim``)."""
        return self.architecture.state_dim

    @property
    def choice_dim(self) -> int:
        """Synced choice-vector width (delegates to ``architecture.choice_dim``)."""
        return self.architecture.choice_dim

    @property
    def family_order(self) -> tuple[str, ...]:
        """Synced family-head order (delegates to ``architecture.family_order``)."""
        return self.architecture.family_order


# ---------------------------------------------------------------------------
# Launch-time validation (cross-field constraints)
# ---------------------------------------------------------------------------


def validate_launchable(cfg: RunConfig) -> list[str]:
    """Return a list of human-readable problems that would prevent a clean run start.

    These are the cross-field constraints previously enforced as ``@model_validator``
    methods on ``RunConfig``. Moving them here lets the configurator commit in-progress
    edits without false rejections while still catching misconfigurations before a
    multi-hour training session begins.  An empty list means the config is launchable.
    """
    problems: list[str] = []

    # A checkpoint-path bootstrap opponent requires device='cpu'.
    if cfg.opponent.bootstrap_opponent not in ("none", "random"):
        if cfg.misc.device != "cpu":
            problems.append(
                "a bootstrap checkpoint requires device='cpu' (mp_collect only)"
            )

    # Setup schedule: (0, 0) is always valid; otherwise train_iter must exceed
    # record_start_iter so the recording window is non-empty.
    setup = cfg.training.setup
    if setup.train_iter > 0 and setup.train_iter <= setup.record_start_iter:
        problems.append(
            f"training.setup.train_iter must exceed record_start_iter "
            f"(got {setup.train_iter} <= {setup.record_start_iter})"
        )

    # target_iterations must not exceed max_iterations when both are nonzero.
    run = cfg.run
    if run.target_iterations > 0 and run.max_iterations > 0:
        if run.target_iterations > run.max_iterations:
            problems.append(
                f"run.target_iterations must be ≤ max_iterations when both are > 0 "
                f"(got {run.target_iterations} > {run.max_iterations})"
            )

    return problems


# ---------------------------------------------------------------------------
# On-disk file wrapper (≥0.5 format)
# ---------------------------------------------------------------------------


class RunConfigFile(pydantic.BaseModel):
    """The dated per-session artifact written as ``run_config_<stamp>.json``.

    Wraps a :class:`RunConfig` with the session context fields that used to live
    in ``process_<stamp>.json`` (started_at, git_sha, resumed, etc.). The
    ``version`` field carries the artifact era so readers can detect old-format
    files and fall back to the legacy three-file layout.
    """

    version: str  # artifact era — "0.5"+ for this format
    saved_at: str
    started_at: str  # ISO-8601 local start time
    git_sha: str | None
    resumed: bool
    resumed_from_iteration: int
    config: RunConfig


# ---------------------------------------------------------------------------
# Artifact-rehydration helpers
# ---------------------------------------------------------------------------


def run_config_from_artifact(
    raw_config: typing.Any, artifact_version: str
) -> RunConfig:
    """Validate a checkpoint's embedded config at the artifact's own era.

    Handles two on-disk shapes:

    * **Nested (≥0.5):** the dict already has six top-level section keys
      (``architecture``, ``run``, ``training``, ``opponent``, ``engine``,
      ``misc``). Validated directly, with the era defaulted when absent.
    * **Flat (≤0.4):** the dict carries all fields at the top level (the old
      ``TrainConfig`` layout). Reshaped into the six-section structure before
      validation, preserving the legacy ``bootstrap_opponent`` migration.

    Raises ``pydantic.ValidationError`` exactly like ``RunConfig.model_validate``.
    """
    if not isinstance(raw_config, dict):
        return RunConfig.model_validate(raw_config)

    raw: dict[str, typing.Any] = dict(typing.cast("dict[str, typing.Any]", raw_config))

    # If the dict already has the nested shape, validate directly.
    if _is_nested_config(raw):
        raw.setdefault("architecture", {})
        arch_raw = raw["architecture"]
        if isinstance(arch_raw, dict):
            arch_typed = typing.cast("dict[str, typing.Any]", arch_raw)
            arch_typed.setdefault("encoding_version", artifact_version)
        return RunConfig.model_validate(raw)

    # --- Flat (≤0.4) → nested migration ---
    raw.setdefault("encoding_version", artifact_version)

    # Legacy bootstrap_opponent field migration.
    if "bootstrap_opponent" not in raw:
        old_initial_vs_random = raw.pop("initial_vs_random", True)
        old_checkpoint = raw.pop("bootstrap_opponent_checkpoint", None)
        if not old_initial_vs_random:
            raw["bootstrap_opponent"] = "none"
        elif old_checkpoint is not None:
            raw["bootstrap_opponent"] = old_checkpoint
        else:
            raw["bootstrap_opponent"] = "random"
    else:
        raw.pop("initial_vs_random", None)
        raw.pop("bootstrap_opponent_checkpoint", None)

    return RunConfig.model_validate(_reshape_flat_to_nested(raw))


def with_encoding_version(cfg: RunConfig, encoding_version: str) -> RunConfig:
    """A validated copy of ``cfg`` pinned to ``encoding_version``.

    Goes through full validation so the derived dims re-sync to the era."""
    data = cfg.model_dump()
    data["architecture"]["encoding_version"] = encoding_version
    return RunConfig.model_validate(data)


# ---------------------------------------------------------------------------
# Backward-compatibility alias
# ---------------------------------------------------------------------------

# Old name kept for any test or import that still spells it out.
TrainConfig = RunConfig
train_config_from_artifact = run_config_from_artifact


###### PRIVATE #######


_NESTED_SECTION_KEYS = frozenset(
    {"architecture", "run", "training", "opponent", "engine", "misc"}
)


def _is_nested_config(raw: dict[str, typing.Any]) -> bool:
    """True when ``raw`` already uses the six-section nested shape (≥0.5)."""
    return bool(_NESTED_SECTION_KEYS & raw.keys())


def _reshape_flat_to_nested(raw: dict[str, typing.Any]) -> dict[str, typing.Any]:
    """Map a flat ≤0.4 ``TrainConfig`` dict into the six-section structure."""

    # --- architecture ---
    main_keys = {
        "trunk_layers",
        "choice_layers",
        "head_layers",
        "value_layers",
        "head_layers_mode",
        "head_layers_main_action",
        "head_layers_draw_bird",
        "head_layers_discard_bird",
        "head_layers_gain_food",
        "head_layers_spend_food",
        "head_layers_lay_egg",
        "head_layers_pay_egg",
        "head_layers_skip_optional",
        "head_layers_choose_bonus",
        "head_layers_misc_rare",
        "head_layers_play_bird",
        "head_layers_reset_birdfeeder",
        "head_layers_setup",
        "activation",
        "dropout",
        "layernorm",
        "card_embed_dim",
        "card_encoder_layers",
        "use_distinct_hand_model",
        "hand_encoder_layers",
        "hand_embed_dim",
        "tray_set_embedding",
        "encoder_final_activation",
    }
    setup_arch_keys = {
        "setup_hidden_layers": "hidden_layers",
        "setup_activation": "activation",
        "setup_dropout": "dropout",
        "setup_use_actor_critic": "use_actor_critic",
    }
    arch_direct_keys = {
        "use_setup_model",
        "split_setup_bonus",
        "split_setup_food",
        "encoding_version",
        "state_dim",
        "choice_dim",
        "family_order",
    }

    main_arch: dict[str, typing.Any] = {
        key: raw.pop(key) for key in main_keys if key in raw
    }
    setup_arch: dict[str, typing.Any] = {
        new_key: raw.pop(old_key)
        for old_key, new_key in setup_arch_keys.items()
        if old_key in raw
    }
    arch: dict[str, typing.Any] = {
        key: raw.pop(key) for key in arch_direct_keys if key in raw
    }
    if main_arch:
        arch["main"] = main_arch
    if setup_arch:
        arch["setup"] = setup_arch

    # --- run ---
    run_keys = {
        "games_per_iter",
        "max_iterations",
        "target_iterations",
        "target_eval_games",
        "eval_every",
        "eval_games",
        "checkpoint_dir",
        "run_name",
        "resume",
        "history_len",
    }
    run: dict[str, typing.Any] = {key: raw.pop(key) for key in run_keys if key in raw}

    # --- training ---
    setup_training_keys = {
        "setup_lr": "lr",
        "setup_policy_temperature": "policy_temperature",
        "setup_policy_greedy": "policy_greedy",
        "setup_record_start_iter": "record_start_iter",
        "setup_train_iter": "train_iter",
        "setup_hand_combos": "hand_combos",
        "setup_food_sets": "food_sets",
        "setup_tuples_per_batch": "tuples_per_batch",
        "setup_offline_epochs": "offline_epochs",
        "setup_offline_batch_size": "offline_batch_size",
        "setup_pg_coef": "pg_coef",
        "setup_value_coef": "value_coef",
        "setup_entropy_coef": "entropy_coef",
    }
    training_direct_keys = {
        "lr",
        "value_coef",
        "entropy_coef",
        "grad_clip",
        "score_norm",
        "reward_mode",
        "reward_discount",
        "end_game_bonus",
    }
    setup_training: dict[str, typing.Any] = {
        new_key: raw.pop(old_key)
        for old_key, new_key in setup_training_keys.items()
        if old_key in raw
    }
    training: dict[str, typing.Any] = {
        key: raw.pop(key) for key in training_direct_keys if key in raw
    }
    if setup_training:
        training["setup"] = setup_training

    # --- opponent ---
    opponent_keys = {
        "bootstrap_opponent",
        "random_phase_win_rate",
        "opponent_reset_win_rate",
        "opponent_max_iterations",
        "eval_ewma_alpha",
    }
    opponent: dict[str, typing.Any] = {
        key: raw.pop(key) for key in opponent_keys if key in raw
    }

    # --- misc ---
    misc_keys = {"seed", "device", "produce_ewma_alpha", "instrumentation"}
    misc: dict[str, typing.Any] = {key: raw.pop(key) for key in misc_keys if key in raw}

    return {
        "architecture": arch,
        "run": run,
        "training": training,
        "opponent": opponent,
        "engine": {},
        "misc": misc,
        # Any leftover keys are silently dropped (extra-fields policy).
    }
