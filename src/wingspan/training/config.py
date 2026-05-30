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
    games_per_iter: int = 64  # games collected per collect-then-update cycle
    max_iterations: int = 0  # 0 = run until interrupted

    # ---- optimization (TRAINING.md §3.3) ----
    lr: float = 3e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    grad_clip: float = 5.0
    # Raw advantage scale before per-batch normalization. Kept for readable
    # value targets; the per-batch normalization (§3.3) is what stabilizes the
    # gradient regardless of this constant.
    score_norm: float = 50.0

    # ---- evaluation (TRAINING.md §7) ----
    eval_every: int = 2  # run an eval block every N iterations (0 disables)
    eval_games: int = 32  # paired (mirror) games per eval => 2N full games

    # ---- runtime ----
    device: str = "cpu"
    seed: int = 0
    hidden: int = 128

    # ---- checkpointing (TRAINING.md §5) ----
    checkpoint_dir: str = "checkpoints"
    run_name: str = "dashboard"

    # ---- in-memory history cap (for the live convergence charts) ----
    history_len: int = 1024

    # ---- architecture descriptor (TRAINING.md §5.1) ----
    state_dim: int = pydantic.Field(default_factory=encode.state_size)
    choice_dim: int = encode.CHOICE_FEATURE_DIM
    family_order: tuple[str, ...] = pydantic.Field(
        default_factory=_default_family_order
    )
