"""Training-run configuration.

``TrainConfig`` is the single self-describing record of every hyperparameter a
run uses. It is stored verbatim inside every checkpoint (TRAINING.md §5.1) so a
run can be resumed and its results re-derived later, and it carries an
architecture descriptor (``encoding_version`` / ``state_dim`` / ``choice_dim`` /
``family_order``) so a loader can detect an incompatible network before
misrouting heads. ``encoding_version`` pins the run to its artifact era: the
dims are era-routed from it, every artifact the run writes is stamped with it,
and a resumed run adopts it from its checkpoint — so a FRESH encoding change
never orphans an in-flight run (``docs/VERSIONING.md``).

The defaults encode the TRAINING.md Phase-1 program: a synchronous
REINFORCE-with-value-baseline loop, advantage normalization, no epsilon-greedy,
sized by *games* per iteration, with a paired-game evaluation against the random
agent every few iterations.
"""

from __future__ import annotations

import enum
import typing

import pydantic

from wingspan import architecture, decisions, encode, setup_model, version
from wingspan.instrumentation import config as instrumentation_config


class RewardMode(enum.StrEnum):
    """How a decision's REINFORCE return is computed from a finished game.

    ``TERMINAL_MARGIN`` broadcasts the single end-of-game score margin to every
    decision (the historical default). ``DECISION_DELTA`` instead credits each
    decision with the change in the player's score margin (own − opponent) over
    the interval until that player's next decision, accumulated into a return
    discounted by ``reward_discount`` per unit of game-clock time between
    decisions (``Step.timestamp`` — one unit per game turn) — a per-decision
    credit signal. Both are shape-preserving (REGIME), so toggling them never
    restarts the network.
    """

    TERMINAL_MARGIN = "terminal_margin"
    DECISION_DELTA = "decision_delta"


def _default_family_order() -> tuple[str, ...]:
    """The stable judgment-family head order, as strings, for the checkpoint
    descriptor (mirrors ``decisions.ALL_DECISION_FAMILIES``)."""
    return tuple(family.value for family in decisions.ALL_DECISION_FAMILIES)


class TrainConfig(pydantic.BaseModel):
    """Every hyperparameter for one training run, versioned and self-describing.

    Sized in *games* per iteration rather than steps because the reward is a
    single end-of-game margin shared across a game's ~140 decisions, so those
    decisions are correlated and one game is closer to one noisy label than to
    140 independent ones (TRAINING.md §3.2).
    """

    # ---- loop shape ----
    # Training games collected (and learned from) per collect-then-update cycle.
    # A multiple of the mp worker count (16) so the final collection wave has no
    # idle tail. 256 sits at the throughput knee (fixed per-iter overhead is
    # small, so larger buys little) and lowers REINFORCE gradient variance.
    games_per_iter: typing.Annotated[int, pydantic.Field(ge=1)] = 256
    max_iterations: typing.Annotated[int, pydantic.Field(ge=0)] = 0  # 0 = run forever
    # Pause at this iteration count: save a final checkpoint, run a large
    # fixed-model eval, and wait for the user to [C]ontinue or [E]nd the run.
    # 0 = no target. Must be ≤ max_iterations when both are > 0.
    target_iterations: typing.Annotated[int, pydantic.Field(ge=0)] = 0
    # Self-play games (model fixed, greedy both seats) run at the target milestone.
    # 0 = auto: 10 × eval_games.
    target_eval_games: typing.Annotated[int, pydantic.Field(ge=0)] = 0

    # ---- optimization (TRAINING.md §3.3) ----
    lr: typing.Annotated[float, pydantic.Field(gt=0.0)] = 3e-4
    value_coef: typing.Annotated[float, pydantic.Field(ge=0.0)] = 0.5
    entropy_coef: typing.Annotated[float, pydantic.Field(ge=0.0)] = 0.01
    grad_clip: typing.Annotated[float, pydantic.Field(gt=0.0)] = 5.0
    # Raw advantage scale before per-batch normalization. Kept for readable
    # value targets; the per-batch normalization (TRAINING.md §3.3) is what stabilizes the
    # gradient regardless of this constant.
    score_norm: typing.Annotated[float, pydantic.Field(gt=0.0)] = 50.0
    # How each decision's REINFORCE return is computed (see ``RewardMode``).
    # ``terminal_margin`` (default) broadcasts the end-of-game margin to every
    # decision; ``decision_delta`` credits each decision with its own discounted
    # margin change. Shape-preserving (REGIME): never restarts the network.
    reward_mode: RewardMode = RewardMode.TERMINAL_MARGIN
    # Discount γ for the ``decision_delta`` return, applied per unit of game-clock
    # time (one game turn): a decision's return is its own margin change plus
    # Σ γ^Δt·(future per-decision margin changes), where Δt is the game time
    # elapsed to each future checkpoint. γ=0 → the immediate change only; γ=1 →
    # the player's final margin minus the current margin (time-independent).
    # Inert in ``terminal_margin`` mode.
    reward_discount: typing.Annotated[float, pydantic.Field(ge=0.0, le=1.0)] = 1.0

    # ---- evaluation (TRAINING.md §7) ----
    # Run an eval block once every N training iterations (0 disables eval).
    # Evaluation is comparatively expensive, so it is amortized over several
    # cheap training cycles rather than run every cycle.
    eval_every: typing.Annotated[int, pydantic.Field(ge=0)] = 5
    # Total held-out games played per eval block. They are played as mirrored
    # (paired) deals to cancel Wingspan's first-player advantage, so this is
    # consumed as ``eval_pairs`` pairs => ``2 * eval_pairs`` games (an odd value
    # rounds down to the nearest even count).
    eval_games: typing.Annotated[int, pydantic.Field(ge=0)] = 128
    # Decay for the dashboard's EWMA of eval win-rate / margin: each new eval
    # contributes this fraction (higher = more responsive, lower = smoother), so
    # the trend readouts are not whipsawed by a single eval's sampling noise.
    # Must be in (0, 1]: at 0 the EWMA would never update and freeze on the
    # first value.
    eval_ewma_alpha: typing.Annotated[float, pydantic.Field(gt=0.0, le=1.0)] = 0.3
    # Once the EWMA win-rate against the current reference opponent reaches this,
    # the opponent is advanced: the current policy is frozen as the new "player
    # to beat" (saved to its own checkpoint) and the win-rate trend resets toward
    # 50%, so progress stays legible past the point of crushing the random agent.
    # 0 disables opponent advancement (always evaluate vs the random agent).
    opponent_reset_win_rate: typing.Annotated[float, pydantic.Field(ge=0.0, le=1.0)] = (
        0.95
    )
    # Force-advance the opponent after this many iterations even if the win-rate
    # threshold has not been reached yet. 0 disables the cap. Only applies in the
    # SELF_PLAY phase — the random-phase bootstrap uses its own graduation logic.
    opponent_max_iterations: typing.Annotated[int, pydantic.Field(ge=0)] = 500

    # ---- random-opponent bootstrap phase ----
    # Start a fresh run collecting against the random agent instead of self-play:
    # the net plays seat 0 and the random agent plays seat 1, only the net's
    # decisions are recorded (so a run yields half the steps but each game is
    # cheaper — only one seat queries the policy), and evaluation is paused since
    # the collection games already measure strength vs random. Once the smoothed
    # collection win-rate clears ``random_phase_win_rate`` the current policy is
    # frozen as "self·gen1" and the run switches to ordinary self-play + eval.
    # Only affects fresh runs; a resumed run keeps the phase stored in its
    # checkpoint.
    #
    # "none"   = no bootstrap phase — start directly in self-play
    # "random" = bootstrap against the built-in random agent (original behaviour)
    # <path>   = absolute path to a .pt.gz checkpoint to replay greedy as the
    #            bootstrap opponent. Known limitation: the opponent's *setup net*
    #            is not loaded. Only valid with device="cpu" (mp_collect path).
    bootstrap_opponent: str = "random"
    # Smoothed collection win-rate (vs random, EWMA over ``eval_ewma_alpha``) at
    # which a fresh run graduates from the random-opponent phase to self-play.
    random_phase_win_rate: typing.Annotated[float, pydantic.Field(gt=0.0, le=1.0)] = (
        0.65
    )

    @property
    def initial_vs_random(self) -> bool:
        """Whether the bootstrap phase is active (derived from ``bootstrap_opponent``)."""
        return self.bootstrap_opponent != "none"

    @property
    def bootstrap_opponent_checkpoint(self) -> str | None:
        """The checkpoint path to load as the bootstrap opponent, or ``None`` when
        using the random agent or no bootstrap phase at all."""
        return (
            None
            if self.bootstrap_opponent in ("none", "random")
            else self.bootstrap_opponent
        )

    # ---- "what the AI is producing" smoothing ----
    # Decay for the PRODUCING band's EWMA of the per-iteration score breakdown,
    # game length, and margins (folded once per finished iteration). In (0, 1]
    # for the same reason as ``eval_ewma_alpha``.
    produce_ewma_alpha: typing.Annotated[float, pydantic.Field(gt=0.0, le=1.0)] = 0.2

    # ---- runtime ----
    device: str = "cpu"
    seed: typing.Annotated[int, pydantic.Field(ge=0)] = 0

    # ---- network topology (see architecture.ModelArchitecture) ----
    # The four blocks' hidden-layer widths (input-to-output). The trunk ends at
    # width M and the choice encoder at width N; both are independent and their
    # outputs are concatenated to M+N for the scorer heads. Head blocks may be
    # empty for a direct readout. These flat fields mirror ``ModelArchitecture``
    # so the configurator can edit each one independently; ``self.arch``
    # assembles the descriptor.
    trunk_layers: typing.Annotated[
        architecture.Widths, pydantic.Field(min_length=1)
    ] = (
        128,
        128,
    )
    choice_layers: typing.Annotated[
        architecture.Widths, pydantic.Field(min_length=1)
    ] = (128, 128)
    head_layers: architecture.Widths = (128,)
    value_layers: architecture.Widths = ()
    # "uniform" uses head_layers for every family; "per_family" uses the
    # per-family fields below to give each scoring head its own hidden widths.
    # Fresh run when changed (head shapes differ between the two modes).
    head_layers_mode: typing.Literal["uniform", "per_family"] = "uniform"
    # Per-family scorer heads — one entry per DecisionFamily in ALL_DECISION_FAMILIES
    # order. Active only when head_layers_mode == "per_family"; defaults match
    # head_layers so switching modes starts from the same network shape.
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
    # Width of the shared per-card vector (one representation per core-set bird,
    # reused for every board / tray / hand / choice card slot) — the card encoder's
    # output width.
    card_embed_dim: typing.Annotated[int, pydantic.Field(ge=1)] = 64
    # Hidden widths of the card encoder MLP that maps each card's
    # [static attributes ⊕ identity one-hot] to its card_embed_dim vector. Empty =
    # a single linear projection; a non-empty stack makes it genuinely nonlinear.
    card_encoder_layers: architecture.Widths = (128,)
    # When enabled (the default), a dedicated hand encoder MLP replaces the
    # mean-pool hand embedding: it takes [180-dim multi-hot ⊕ 10-dim hand summary]
    # and outputs a card_embed_dim-wide vector, removing the hand summary from the
    # trunk's continuous feed. The mean-pool path remains for False configs, and
    # old checkpoints keep loading via their saved configs. Fresh run.
    use_distinct_hand_model: bool = True
    # Hidden widths of the hand encoder MLP. Active only when use_distinct_hand_model
    # is True; defaults match card_encoder_layers so toggling on starts from the same
    # structure. Fresh run.
    hand_encoder_layers: architecture.Widths = (128,)
    # Output width N of the hand encoder — the multi-card *set* embedding's width,
    # its own knob beside the single-card card_embed_dim (M). None = match
    # card_embed_dim (the pre-knob shape). Active only when use_distinct_hand_model
    # is on. Fresh run.
    hand_embed_dim: typing.Annotated[int, pydantic.Field(ge=1)] | None = None
    # Append one hand-encoder embedding of the face-up tray *set* to the trunk
    # input (the tray's three per-slot card-table lookups are unchanged, giving
    # 3·M + N tray dims). Requires use_distinct_hand_model; on by default
    # alongside it. Fresh run.
    tray_set_embedding: bool = True
    # When True (the default for new runs), the card, hand, and choice encoders
    # apply a final activation after their last layer, matching the trunk.
    # Saved into model_config.json so old checkpoints (which omit this field)
    # keep their original no-final-relu behaviour on load. REGIME.
    encoder_final_activation: bool = True

    # ---- setup model (TRAINING.md / DECISIONS.md §2.13: the start-of-game keep) ----
    # When enabled (the default), the start-of-game setup decision is pulled out of
    # the in-game policy into a separate value-regression bandit
    # (``wingspan.setup_model``): setups are drawn by the random generator early on,
    # recorded over a window, the setup net is fit once offline, then it drives
    # setup selection and trains on-policy. Pulling setup out also removes it from
    # the *main* net's encoding (``encode.EncodingSpec.include_setup``): the
    # decision-type one-hot's setup column, the SETUP scoring head, and the
    # setup_agg choice stripe all disappear. So this knob IS part of the main net's
    # ``architecture_key`` — ``_sync_encoding_dims`` derives ``state_dim`` /
    # ``choice_dim`` / ``family_order`` from it, so toggling it is a main-net-FRESH
    # change that restarts the main net (the setup net additionally has its own
    # ``setup_architecture_key`` and checkpoint).
    use_setup_model: bool = True
    # Split the opening keep into two judgments (only meaningful when
    # ``use_setup_model`` is on): the setup net picks cards while the bonus card
    # is deferred to the in-game ``CHOOSE_BONUS`` head. The setup candidates carry
    # ``bonus_card=None``; the setup feature vector gains a ``bonus_cards``
    # multi-hot (which bonuses are on offer) and a ``bonus_card_affinity`` pair
    # (min/max qualifier counts) in place of ``kept_bonus`` + ``kept_bonus_value``.
    # The vector SHRINKS by 2 dims vs. the full layout (30→28 bonus block), making
    # this a setup-FRESH knob that IS part of ``setup_architecture_key``.
    # Inert when the setup model is off; gate on ``split_setup_bonus_active``.
    split_setup_bonus: bool = False
    # Analogous to ``split_setup_bonus``: defers the opening food pick out of the
    # combined SetupDecision to sequential in-game food decisions resolved after the
    # card-keep applies (see ``engine.core.Engine._maybe_resolve_deferred_setup_food``).
    # The setup candidates carry ``kept_foods=()``, and the 5-dim ``kept_foods``
    # stripe is OMITTED from the feature vector, making this a setup-FRESH knob
    # that IS part of ``setup_architecture_key``.  Food decision schedule:
    #   0–2 birds → 2/1/0 SpendFoodDecision asks (discard N).
    #   3–5 birds → 0/1/2 GainFoodDecision asks (gain M).
    # ``setup_food_sets`` is ignored when this is active.
    # Inert when the setup model is off; gate on ``split_setup_food_active``.
    split_setup_food: bool = False
    # The setup net's MLP hidden widths (input-to-output) — a setup-FRESH change
    # (restarts only the setup net, never the main net).
    setup_hidden_layers: typing.Annotated[
        architecture.Widths, pydantic.Field(min_length=1)
    ] = (128, 64)
    setup_activation: architecture.ActivationName = architecture.ActivationName.RELU
    setup_dropout: typing.Annotated[float, pydantic.Field(ge=0.0, lt=1.0)] = 0.0
    setup_lr: typing.Annotated[float, pydantic.Field(gt=0.0)] = 1e-3
    # Softmax temperature over the 504 candidates' predicted margins when sampling
    # a setup during collection (eval always takes the argmax). Higher = more
    # exploration while predictions are near-flat early on. Ignored when
    # ``setup_policy_greedy`` is True.
    setup_policy_temperature: typing.Annotated[float, pydantic.Field(gt=0.0)] = 0.5
    # When True, collection uses hard argmax over predicted margins instead of
    # softmax sampling — ensures the in-game model always trains on the setup the
    # setup net currently considers best.
    setup_policy_greedy: bool = False
    # Schedule (cumulative/lifetime iterations): below ``record_start`` setups are
    # random and unrecorded; in ``[record_start, train)`` they are random and
    # recorded; at ``train`` the net is fit once offline and then drives selection
    # and trains on-policy.  Setting both to 0 (the default) is the "train from
    # start" sentinel: no warmup phases, the setup model is MODEL_DRIVEN from
    # iteration 0 and trains on-policy with REINFORCE immediately.  When either
    # is non-zero, ``train`` must exceed ``record_start``.
    setup_record_start_iter: typing.Annotated[int, pydantic.Field(ge=0)] = 0
    setup_train_iter: typing.Annotated[int, pydantic.Field(ge=0)] = 0
    # Random-generation knobs (per batch): joint keep-combos sampled, food keeps
    # per kept hand, and joint setup tuples sampled (= games per shared-deal batch).
    setup_hand_combos: typing.Annotated[int, pydantic.Field(ge=1)] = 10
    setup_food_sets: typing.Annotated[int, pydantic.Field(ge=1)] = 3
    setup_tuples_per_batch: typing.Annotated[int, pydantic.Field(ge=1)] = 16
    # The one-time offline fit's epochs over the recorded window, and the minibatch
    # size used for both the offline fit and the on-policy updates.
    setup_offline_epochs: typing.Annotated[int, pydantic.Field(ge=1)] = 20
    setup_offline_batch_size: typing.Annotated[int, pydantic.Field(ge=1)] = 256
    # Actor-critic training for the setup model (TRAINING.md §setup-actor-critic).
    # When True, a policy head is added to the setup net (setup-FRESH change) and
    # the MODEL_DRIVEN on-policy update uses REINFORCE (policy gradient + value
    # baseline + entropy bonus) instead of plain MSE on the value head only. The
    # offline fit at ``setup_train_iter`` always targets the value head via MSE
    # regardless of this setting; the policy head first trains on-policy.
    # Toggling this flag invalidates the existing ``setup.pt`` checkpoint because
    # ``use_policy_head`` enters the ``setup_architecture_key`` via shape_key.
    setup_use_actor_critic: bool = False
    # Loss-term weights for the actor-critic update (ignored when
    # ``setup_use_actor_critic`` is False). All must be ≥ 0.
    setup_pg_coef: typing.Annotated[float, pydantic.Field(ge=0.0)] = 1.0
    setup_value_coef: typing.Annotated[float, pydantic.Field(ge=0.0)] = 0.5
    setup_entropy_coef: typing.Annotated[float, pydantic.Field(ge=0.0)] = 0.01

    # ---- checkpointing (TRAINING.md §5) ----
    checkpoint_dir: str = "checkpoints"
    run_name: str = "dashboard"
    # Resume the network, optimizer, and run progress from ``last.pt`` in
    # ``checkpoint_dir`` when one is present (set False to always start fresh).
    resume: bool = True

    # ---- in-memory history cap (for the live convergence charts) ----
    history_len: typing.Annotated[int, pydantic.Field(ge=1)] = 1024

    # ---- instrumentation (event-callback recorders attached to this run) ----
    # Custom handlers fired at game events (see ``wingspan.instrumentation``).
    # Empty by default — no handlers, no overhead. Deliberately excluded from
    # ``architecture_key`` / the encoding key: attaching instrumentation never
    # changes a tensor shape, so it must not invalidate a checkpoint or trigger a
    # FRESH restart.
    instrumentation: instrumentation_config.InstrumentationConfig = pydantic.Field(
        default_factory=instrumentation_config.InstrumentationConfig
    )

    # ---- architecture descriptor (TRAINING.md §5.1) ----
    # The artifact era this run trains at: the encoding its vectors are produced
    # with and the version stamped on every artifact it writes. Fresh runs train
    # at the current MODEL_VERSION (``loop_resume.adopt_checkpoint_era`` re-keys
    # any stale era at launch); a resumed run adopts its checkpoint's era (the
    # configurator's saved-run seeding + live re-alignment via
    # ``configure.runs.align_era``, and the same ``adopt_checkpoint_era`` seam),
    # so a run started before a FRESH encoding change keeps training at its own
    # frozen geometry instead of being orphaned by it (docs/VERSIONING.md).
    # Deliberately not an editable configurator field — the era is a property of
    # the run directory, not a knob.
    encoding_version: str = version.MODEL_VERSION
    state_dim: int = pydantic.Field(default_factory=encode.state_size)
    choice_dim: int = encode.CHOICE_FEATURE_DIM
    family_order: tuple[str, ...] = pydantic.Field(
        default_factory=_default_family_order
    )

    @pydantic.model_validator(mode="after")
    def _check_bootstrap_opponent(self) -> "TrainConfig":
        """Validate the bootstrap-opponent checkpoint constraints.

        A checkpoint path is only reachable through the ``mp_collect`` path
        (CPU-only), so it must be paired with ``device="cpu"``."""
        if self.bootstrap_opponent not in ("none", "random"):
            if self.device != "cpu":
                raise ValueError(
                    "a bootstrap checkpoint requires device='cpu' (mp_collect only)"
                )
        return self

    @pydantic.model_validator(mode="after")
    def _check_architecture(self) -> TrainConfig:
        """Verify the topology descriptor assembles without error — surfaces any
        future cross-field invariants added to ``ModelArchitecture`` as normal
        validation errors so the configurator rejects them the same way it
        rejects out-of-range scalars."""
        _ = self.arch
        return self

    @pydantic.model_validator(mode="after")
    def _check_setup_schedule(self) -> TrainConfig:
        """Validate the setup-model warmup schedule.

        ``(0, 0)`` is the "train from start" sentinel and is always valid.
        For any other non-zero configuration the recording window must be
        non-empty: ``setup_train_iter`` must strictly exceed
        ``setup_record_start_iter``.
        """
        # Allow (0, 0): no warmup phases, MODEL_DRIVEN from iteration 0.
        if (
            self.setup_train_iter > 0
            and self.setup_train_iter <= self.setup_record_start_iter
        ):
            raise ValueError(
                "setup_train_iter must exceed setup_record_start_iter "
                f"(got {self.setup_train_iter} <= {self.setup_record_start_iter})"
            )
        return self

    @pydantic.model_validator(mode="after")
    def _check_target_iterations(self) -> "TrainConfig":
        """target_iterations must not exceed max_iterations when both are nonzero,
        since the target is a milestone *within* the run rather than a replacement
        for the hard cap."""
        if self.target_iterations > 0 and self.max_iterations > 0:
            if self.target_iterations > self.max_iterations:
                raise ValueError(
                    "target_iterations must be ≤ max_iterations when both are > 0 "
                    f"(got {self.target_iterations} > {self.max_iterations})"
                )
        return self

    @pydantic.model_validator(mode="after")
    def _check_encoding_version(self) -> "TrainConfig":
        """The run's era must be one this code can load: same MAJOR, MINOR at
        most the current. A newer era or a different MAJOR has no shims, so an
        era-pinned run could neither encode its vectors nor stamp its artifacts
        consistently."""
        try:
            version.check_artifact_compatible(
                self.encoding_version, what="TrainConfig.encoding_version"
            )
        except version.IncompatibleArtifactError as error:
            raise ValueError(str(error)) from error
        return self

    @pydantic.model_validator(mode="after")
    def _sync_encoding_dims(self) -> "TrainConfig":
        """Keep the encoding dims and family-head order in lockstep with
        ``use_setup_model`` and ``encoding_version``. The main model's shape is
        config-driven on the setup axis (``encode.EncodingSpec.include_setup`` =
        ``not use_setup_model``) and era-routed on the version axis
        (``compat.encoding_dims_for_era``), so ``state_dim`` / ``choice_dim`` /
        ``family_order`` are *derived*, not free knobs — and because they feed
        ``architecture_key``, toggling ``use_setup_model`` (or crossing a FRESH
        era boundary) correctly registers as a main-net-FRESH change. The
        family-head order has been stable across every 0.x era, so it stays
        live-derived."""
        spec = encode.spec_for(self.use_setup_model)
        if self.encoding_version == version.MODEL_VERSION:
            # The live era needs no shim — and must not import one: configs are
            # constructed at import time (field-spec defaults, loaders), some of
            # them *during* ``wingspan.compat``'s own import (compat → v0_1 →
            # training.setup_net → training.__init__ → … → fields). Era-pinned
            # configs only exist at runtime, when the late import below is safe.
            self.state_dim = encode.state_size(spec)
            self.choice_dim = encode.choice_feature_dim(spec)
        else:
            from wingspan import compat  # noqa: PLC0415 — see live-era note above

            self.state_dim, self.choice_dim = compat.encoding_dims_for_era(
                self.encoding_version, spec
            )
        self.family_order = tuple(
            family.value
            for family in decisions.active_decision_families(spec.include_setup)
        )
        return self

    @property
    def encoding_spec(self) -> encode.EncodingSpec:
        """The state/choice encoding spec implied by ``use_setup_model`` — the one
        config-driven axis of the main model's shape. Threaded into the encoders
        (so collected vectors match the net) and recorded in ``model_config.json``."""
        return encode.spec_for(self.use_setup_model)

    @property
    def split_setup_bonus_active(self) -> bool:
        """Whether the opening's bonus pick is deferred to the in-game
        ``CHOOSE_BONUS`` head this run. Only takes effect alongside the setup
        model, so it is gated on ``use_setup_model``; every collection / eval call
        site reads this rather than the raw ``split_setup_bonus`` flag."""
        return self.split_setup_bonus and self.use_setup_model

    @property
    def split_setup_food_active(self) -> bool:
        """Whether the opening food pick is deferred to sequential in-game
        GAIN_FOOD/SPEND_FOOD decisions this run. Only takes effect alongside the
        setup model, so it is gated on ``use_setup_model``; every collection / eval
        call site reads this rather than the raw ``split_setup_food`` flag."""
        return self.split_setup_food and self.use_setup_model

    @property
    def eval_pairs(self) -> int:
        """Mirror-deal pairs per eval block — ``eval_games`` games played as
        paired deals (rounded down so each deal is mirrored)."""
        return self.eval_games // 2

    @property
    def effective_target_eval_games(self) -> int:
        """The number of self-play games run at the target milestone.

        Explicit ``target_eval_games`` wins; 0 falls back to 10 × ``eval_games``
        so the default scales sensibly with the normal evaluation budget."""
        return (
            self.target_eval_games
            if self.target_eval_games > 0
            else 10 * self.eval_games
        )

    @property
    def arch(self) -> architecture.ModelArchitecture:
        """The network topology descriptor assembled from the flat topology
        fields — the single object the model builds from and ``model_config.json``
        serializes. Named ``arch`` (not ``architecture``) so it never shadows the
        imported ``architecture`` module in this class's field annotations."""
        # In per_family mode, assemble one Widths tuple per active family (the
        # active set respects use_setup_model, which removes SETUP from the main
        # net). In uniform mode, per_family_head_layers is None and every head
        # falls back to head_layers via ModelArchitecture.head_layers_for.
        per_family: tuple[architecture.Widths, ...] | None = None
        if self.head_layers_mode == "per_family":
            active = decisions.active_decision_families(
                self.encoding_spec.include_setup
            )
            per_family = tuple(
                typing.cast(
                    architecture.Widths,
                    getattr(self, f"head_layers_{family.value}"),
                )
                for family in active
            )
        return architecture.ModelArchitecture(
            trunk_layers=self.trunk_layers,
            choice_layers=self.choice_layers,
            head_layers=self.head_layers,
            value_layers=self.value_layers,
            per_family_head_layers=per_family,
            activation=self.activation,
            dropout=self.dropout,
            layernorm=self.layernorm,
            card_embed_dim=self.card_embed_dim,
            card_encoder_layers=self.card_encoder_layers,
            use_distinct_hand_model=self.use_distinct_hand_model,
            hand_encoder_layers=self.hand_encoder_layers,
            hand_embed_dim=self.hand_embed_dim,
            tray_set_embedding=self.tray_set_embedding,
            encoder_final_activation=self.encoder_final_activation,
        )

    @property
    def setup_arch(self) -> setup_model.SetupArchitecture:
        """The setup network's topology descriptor assembled from the flat setup
        fields — what ``SetupNet`` builds from and ``setup_config.json``
        serializes. Named ``setup_arch`` (mirroring ``arch``) so it never shadows
        the imported ``setup_model`` package in this class's annotations."""
        return setup_model.SetupArchitecture(
            hidden_layers=self.setup_hidden_layers,
            activation=self.setup_activation,
            dropout=self.setup_dropout,
            use_policy_head=self.setup_use_actor_critic,
        )

    @property
    def setup_encoding(self) -> setup_model.SetupEncoding:
        """The setup input-vector layout implied by the active split flags.

        Only meaningful when ``use_setup_model`` is on; gated via the
        ``*_active`` properties so the encoding always matches the engine's
        candidate generation."""
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
        """The setup-net shape signature a ``setup.pt`` must match to be resumed:
        the encoder's feature width, the readout MLP's hidden shape, and the
        frozen embedder copies' shape components from the main architecture
        (their widths size the setup net's tensors too). Independent of the main
        net's ``architecture_key`` — toggling the setup model never invalidates
        the main net's weights, though reshaping the shared embedders restarts
        both nets (each via its own key)."""
        arch = self.arch
        return (
            self.setup_encoding.total_dim,
            self.setup_arch.shape_key,
            (
                arch.card_encoder_layers,
                arch.card_embed_dim,
                arch.layernorm,
                arch.use_distinct_hand_model,
                arch.hand_encoder_layers,
                arch.hand_embed_width,
            ),
        )

    @property
    def trunk_hidden(self) -> int:
        """The trunk's output width ``M`` — the state-context and value-head input
        width."""
        return self.trunk_layers[-1]

    @property
    def choice_hidden(self) -> int:
        """The choice encoder's output width ``N`` — concatenated with ``M`` before
        the scorer heads."""
        return self.choice_layers[-1]

    @property
    def architecture_key(
        self,
    ) -> tuple[str, int, int, tuple[str, ...], architecture.ShapeKey]:
        """The network-shape signature a checkpoint must match to be resumed
        (TRAINING.md §5.1): two trained nets are weight-compatible iff their
        era (``encoding_version``), ``(state_dim, choice_dim, family_order)``,
        and full topology ``shape_key`` agree. The era leads the tuple so a
        shape-preserving FRESH change still reads as incompatible — coinciding
        widths across eras are exactly the silent-corruption case. Comparing
        this one derived tuple keeps the resume gate and the configurator's
        compatibility check from drifting apart."""
        return (
            self.encoding_version,
            self.state_dim,
            self.choice_dim,
            self.family_order,
            self.arch.shape_key,
        )


def train_config_from_artifact(
    raw_config: typing.Any, artifact_version: str
) -> TrainConfig:
    """Validate a checkpoint's embedded config at the artifact's own era.

    Configs written before the ``encoding_version`` field existed default it
    from the payload's ``version`` stamp — the field that has always carried the
    artifact's era — so a pre-field artifact rehydrates era-pinned: its dims
    re-derive to the era's widths and its ``architecture_key`` compares
    truthfully. Configs that already carry the field keep it (the two stamps
    agree by construction for artifacts written since). Raises
    ``pydantic.ValidationError`` exactly like ``TrainConfig.model_validate``.
    """
    if isinstance(raw_config, dict):
        adjusted = dict(typing.cast("dict[str, typing.Any]", raw_config))
        adjusted.setdefault("encoding_version", artifact_version)

        # Backwards compat: configs written before the unified bootstrap_opponent
        # field stored separate initial_vs_random + bootstrap_opponent_checkpoint
        # fields. Derive the new field and drop the old ones so model_validate
        # does not trip on unexpected keys.
        if "bootstrap_opponent" not in adjusted:
            old_initial_vs_random = adjusted.pop("initial_vs_random", True)
            old_checkpoint = adjusted.pop("bootstrap_opponent_checkpoint", None)
            if not old_initial_vs_random:
                adjusted["bootstrap_opponent"] = "none"
            elif old_checkpoint is not None:
                adjusted["bootstrap_opponent"] = old_checkpoint
            else:
                adjusted["bootstrap_opponent"] = "random"
        else:
            # Field already present — still remove any stale old-field keys
            # a mixed-era config might carry so they don't cause extra-fields errors.
            adjusted.pop("initial_vs_random", None)
            adjusted.pop("bootstrap_opponent_checkpoint", None)

        return TrainConfig.model_validate(adjusted)
    return TrainConfig.model_validate(raw_config)


def with_encoding_version(cfg: TrainConfig, encoding_version: str) -> TrainConfig:
    """A validated copy of ``cfg`` pinned to ``encoding_version``.

    Goes through full validation (not ``model_copy``) so the derived dims
    re-sync to the era — a plain attribute update would leave ``state_dim`` /
    ``choice_dim`` at the previous era's widths."""
    return TrainConfig.model_validate(
        {**cfg.model_dump(), "encoding_version": encoding_version}
    )
