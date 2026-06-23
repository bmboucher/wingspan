"""The setup-model training sample.

A :class:`SetupSample` is one recorded ``(setup features, realized margin)`` pair:
the feature vector a seat's chosen setup encoded to, paired with the score margin
that seat ended the game with. These are the targets for actor-critic training —
``margin / score_norm`` is the reward signal. Each sample also carries
``chosen_idx`` and ``all_candidates`` so the learner can compute a REINFORCE
gradient over all candidates at training time.
"""

from __future__ import annotations

import numpy as np
import pydantic


class SetupSample(pydantic.BaseModel):
    """One ``(setup features, realized margin)`` actor-critic training sample.

    ``features`` is the :func:`wingspan.setup_model.encode.encode_setup_candidate`
    vector for the seat's chosen setup; ``margin`` is the seat's end-of-game
    ``own_total - opponent_total`` (the contextual-bandit reward), left
    unnormalized here so the learner can scale it by ``score_norm`` consistently
    with the in-game return.

    ``chosen_idx`` and ``all_candidates`` carry the data needed to compute a
    REINFORCE gradient over all candidates at training time:

    * ``chosen_idx`` — which row in ``all_candidates`` was selected.
    * ``all_candidates`` — the ``(K, feature_dim)`` matrix of every candidate's
      encoded features (K = 504 or 252 with split-bonus). Compressed to float16
      before IPC.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    features: np.ndarray
    margin: float
    iteration: int
    chosen_idx: int | None = None
    all_candidates: np.ndarray | None = None
