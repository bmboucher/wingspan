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
