"""The recorded self-play transition consumed by the training pipeline.

One :class:`Step` is recorded per multi-option decision during collection
(``wingspan.training.collect`` / ``batched_collect``) and consumed by the
length-bucketed REINFORCE update (``wingspan.training.learner``).
"""

from __future__ import annotations

import numpy as np
import pydantic


class Step(pydantic.BaseModel):
    """One recorded transition during self-play.

    ``choices`` is variable-length per step: shape ``(n_choices, F)``. The
    training loop pads across the batch when stacking.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    state: np.ndarray  # (state_dim,)
    choices: np.ndarray  # (n_choices, choice_dim)
    chosen_idx: int  # 0..n_choices-1
    player_id: int
    family_idx: int  # judgment-family scoring-head index (see decisions)
    # The deciding player's running score margin (own − opponent) right before
    # this decision. Consumed by the ``decision_delta`` reward mode with
    # ``MARGIN`` basis (``learner._flatten``); the default keeps
    # ``terminal_margin`` collection and any fixtures that omit it valid.
    margin_before: float = 0.0
    # The deciding player's own raw score right before this decision.
    # Consumed by the ``decision_delta`` reward mode with ``OWN_SCORE`` basis
    # (``learner._flatten``); the default keeps older fixtures valid.
    score_before: float = 0.0
    # Game-clock time of this decision: setup decisions at 0 / 1/3 / 2/3, turn
    # N's main action at exactly N, and mid-turn decisions interpolated into
    # (N, N+1) by ``timestamps.finalize_timestamps``. Consumed only by the
    # ``decision_delta`` reward mode's λ^Δt discounting; the default keeps
    # fixtures that omit it valid.
    timestamp: float = 0.0
    # DAgger expert soft-target probabilities: shape ``(n_choices,)``, aligned
    # to the ``choices`` rows so ``expert_probs[i]`` is the expert's probability
    # for candidate ``i``. ``None`` when no expert is active for this game or
    # when the step's ``family_idx >= expert_net.num_families`` (the SETUP-head
    # guard — SETUP is last in ``decisions.ALL_DECISION_FAMILIES``).
    expert_probs: np.ndarray | None = None
    # Log probability of ``chosen_idx`` under the collection-time policy
    # π_old — used by the PPO clipped surrogate ratio = exp(logπ_new − logπ_old).
    # Matches the degeneracy fallback in ``policy.sample_index_from_probs``:
    # ``−log(n)`` when the distribution is degenerate. Default keeps old
    # fixtures valid; fresh collection always fills it.
    behavior_logp: float = 0.0
    # Critic V(s) at this decision in normalized-return units (value head output,
    # trained against G/score_norm). Used as the PPO baseline and for GAE
    # bootstrapping. Default keeps old fixtures valid; fresh collection fills it.
    value_pred: float = 0.0
