"""Training-run configuration.

``TrainConfig`` is the single self-describing record of every hyperparameter a
run uses. It is stored verbatim inside every checkpoint (TRAINING.md §5.1) so a
run can be resumed and its results re-derived later, and it carries an
architecture descriptor (``state_dim`` / ``choice_dim`` / ``family_order``) so a
loader can detect an incompatible network before misrouting heads.

The defaults encode the TRAINING.md Phase-1 program: a synchronous
REINFORCE-with-value-baseline loop, advantage normalization, no epsilon-greedy,
sized by *games* per iteration, with a paired-game evaluation against the random
agent every few iterations.
"""

from __future__ import annotations

import typing

import pydantic

from wingspan import architecture, decisions, encode, setup_model


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
    # value targets; the per-batch normalization (§3.3) is what stabilizes the
    # gradient regardless of this constant.
    score_norm: typing.Annotated[float, pydantic.Field(gt=0.0)] = 50.0

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
    initial_vs_random: bool = True
    # Smoothed collection win-rate (vs random, EWMA over ``eval_ewma_alpha``) at
    # which a fresh run graduates from the random-opponent phase to self-play.
    random_phase_win_rate: typing.Annotated[float, pydantic.Field(gt=0.0, le=1.0)] = (
        0.65
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
    activation: architecture.ActivationName = architecture.ActivationName.RELU
    dropout: typing.Annotated[float, pydantic.Field(ge=0.0, lt=1.0)] = 0.0
    layernorm: bool = False
    # Width of the shared per-card embedding (one learned vector per core-set
    # bird, reused for every board / tray / hand / choice card slot).
    card_embed_dim: typing.Annotated[int, pydantic.Field(ge=1)] = 64

    # ---- setup model (TRAINING.md / DECISIONS.md: the start-of-game keep) ----
    # When enabled, the start-of-game setup decision is pulled out of the in-game
    # policy into a separate value-regression bandit (``wingspan.setup_model``):
    # setups are drawn by the random generator early on, recorded over a window,
    # the setup net is fit once offline, then it drives setup selection and trains
    # on-policy. Default OFF so existing checkpoints and behaviour are unchanged —
    # this knob does not touch the *main* net's ``architecture_key``; the setup net
    # has its own ``setup_architecture_key`` and its own checkpoint.
    use_setup_model: bool = True
    # The setup net's MLP hidden widths (input-to-output) — a setup-FRESH change
    # (restarts only the setup net, never the main net).
    setup_hidden_layers: typing.Annotated[
        architecture.Widths, pydantic.Field(min_length=1)
    ] = (128, 64)
    setup_activation: architecture.ActivationName = architecture.ActivationName.RELU
    setup_dropout: typing.Annotated[float, pydantic.Field(ge=0.0, lt=1.0)] = 0.0
    setup_lr: typing.Annotated[float, pydantic.Field(gt=0.0)] = 1e-3
    # Softmax temperature over the 504 candidates' predicted margins when sampling
    # a setup during collection (eval takes the argmax). Higher = more exploration
    # while predictions are near-flat early on.
    setup_policy_temperature: typing.Annotated[float, pydantic.Field(gt=0.0)] = 0.5
    # Schedule (cumulative/lifetime iterations): below ``record_start`` setups are
    # random and unrecorded; in ``[record_start, train)`` they are random and
    # recorded; at ``train`` the net is fit once offline and then drives selection
    # and trains on-policy. ``train`` must exceed ``record_start``.
    setup_record_start_iter: typing.Annotated[int, pydantic.Field(ge=0)] = 1000
    setup_train_iter: typing.Annotated[int, pydantic.Field(ge=1)] = 2000
    # Random-generation knobs (per batch): joint keep-combos sampled, food keeps
    # per kept hand, and joint setup tuples sampled (= games per shared-deal batch).
    setup_hand_combos: typing.Annotated[int, pydantic.Field(ge=1)] = 10
    setup_food_sets: typing.Annotated[int, pydantic.Field(ge=1)] = 3
    setup_tuples_per_batch: typing.Annotated[int, pydantic.Field(ge=1)] = 16
    # The one-time offline fit's epochs over the recorded window, and the minibatch
    # size used for both the offline fit and the on-policy updates.
    setup_offline_epochs: typing.Annotated[int, pydantic.Field(ge=1)] = 20
    setup_offline_batch_size: typing.Annotated[int, pydantic.Field(ge=1)] = 256

    # ---- checkpointing (TRAINING.md §5) ----
    checkpoint_dir: str = "checkpoints"
    run_name: str = "dashboard"
    # Resume the network, optimizer, and run progress from ``last.pt`` in
    # ``checkpoint_dir`` when one is present (set False to always start fresh).
    resume: bool = True

    # ---- in-memory history cap (for the live convergence charts) ----
    history_len: typing.Annotated[int, pydantic.Field(ge=1)] = 1024

    # ---- architecture descriptor (TRAINING.md §5.1) ----
    state_dim: int = pydantic.Field(default_factory=encode.state_size)
    choice_dim: int = encode.CHOICE_FEATURE_DIM
    family_order: tuple[str, ...] = pydantic.Field(
        default_factory=_default_family_order
    )

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
        """The setup model must start recording before it is fit/deployed, so the
        offline-fit window is non-empty. Enforced as a normal validation error so
        the configurator rejects an inconsistent edit the same way it rejects an
        out-of-range scalar."""
        if self.setup_train_iter <= self.setup_record_start_iter:
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
        return architecture.ModelArchitecture(
            trunk_layers=self.trunk_layers,
            choice_layers=self.choice_layers,
            head_layers=self.head_layers,
            value_layers=self.value_layers,
            activation=self.activation,
            dropout=self.dropout,
            layernorm=self.layernorm,
            card_embed_dim=self.card_embed_dim,
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
        )

    @property
    def setup_architecture_key(self) -> tuple[int, setup_model.SetupShapeKey]:
        """The setup-net shape signature a ``setup.pt`` must match to be resumed:
        the encoder's feature width and the MLP's hidden shape. Independent of the
        main net's ``architecture_key`` — toggling the setup model never
        invalidates the main net's weights."""
        return (setup_model.SETUP_FEATURE_DIM, self.setup_arch.shape_key)

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
    ) -> tuple[int, int, tuple[str, ...], architecture.ShapeKey]:
        """The network-shape signature a checkpoint must match to be resumed
        (TRAINING.md §5.1): two trained nets are weight-compatible iff their
        ``(state_dim, choice_dim, family_order)`` and full topology ``shape_key``
        agree. Comparing this one derived tuple keeps the resume gate and the
        configurator's compatibility check from drifting apart."""
        return (
            self.state_dim,
            self.choice_dim,
            self.family_order,
            self.arch.shape_key,
        )
