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

from wingspan import decisions, encode


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
    games_per_iter: typing.Annotated[int, pydantic.Field(ge=1)] = 64
    max_iterations: typing.Annotated[int, pydantic.Field(ge=0)] = 0  # 0 = run forever

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

    # ---- "what the AI is producing" smoothing ----
    # Decay for the PRODUCING band's EWMA of the per-iteration score breakdown,
    # game length, and margins (folded once per finished iteration). In (0, 1]
    # for the same reason as ``eval_ewma_alpha``.
    produce_ewma_alpha: typing.Annotated[float, pydantic.Field(gt=0.0, le=1.0)] = 0.2

    # ---- runtime ----
    device: str = "cpu"
    seed: typing.Annotated[int, pydantic.Field(ge=0)] = 0
    hidden: typing.Annotated[int, pydantic.Field(ge=1)] = 128

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

    @property
    def eval_pairs(self) -> int:
        """Mirror-deal pairs per eval block — ``eval_games`` games played as
        paired deals (rounded down so each deal is mirrored)."""
        return self.eval_games // 2

    @property
    def architecture_key(self) -> tuple[int, int, tuple[str, ...], int]:
        """The network-shape signature a checkpoint must match to be resumed
        (TRAINING.md §5.1): two trained nets are weight-compatible iff their
        ``(state_dim, choice_dim, family_order, hidden)`` agree. Comparing this
        one derived tuple keeps the resume gate and the configurator's
        compatibility check from drifting apart."""
        return (self.state_dim, self.choice_dim, self.family_order, self.hidden)
