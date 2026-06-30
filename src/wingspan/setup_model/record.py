"""The setup-model training sample.

A :class:`SetupSample` is one recorded setup keep — the seat's chosen feature
vector plus everything needed to reproduce the **in-game return at the seat's
``t=0`` setup decision**. The setup keep is the ``t=0`` decision of the same game
whose in-game steps are ``t>0``, so its actor-critic target is the in-game return
kernel (``wingspan.training.returns.setup_return``) evaluated at that anchor —
consistent with the main learner under any ``reward_mode`` / discount / basis /
bonus rather than a separately-defined raw margin.
"""

from __future__ import annotations

import numpy as np
import pydantic


class SetupSample(pydantic.BaseModel):
    """One actor-critic training sample for a seat's setup keep.

    ``features`` is the :func:`wingspan.setup_model.encode.encode_setup_candidate`
    vector for the seat's chosen setup. ``chosen_idx`` / ``all_candidates`` carry
    the data needed to compute a REINFORCE gradient over all candidates:

    * ``chosen_idx`` — which row in ``all_candidates`` was selected.
    * ``all_candidates`` — the ``(K, feature_dim)`` matrix of every candidate's
      encoded features (K = 504 or 252 with split-bonus). Compressed to float16
      before IPC. The state stripes are byte-identical across these K rows, so
      ``V(s)`` is read from row 0 (any row).

    The remaining fields reproduce the in-game return at the ``t=0`` setup anchor
    (``returns.setup_return``); all have safe defaults so samples recorded before
    this field set still deserialize:

    * ``margin`` — the seat's end-of-game ``own_total − opponent_total`` (kept for
      the dashboard's realized-margin readout).
    * ``own_total`` / ``opp_total`` — the seat's and opponent's final scores.
    * ``won`` — the seat-relative outcome (``+1`` win, ``-1`` loss, ``0`` tie).
    * ``margin_checkpoints`` / ``score_checkpoints`` — the seat's per-in-game-decision
      ``margin_before`` / ``score_before`` value snapshots (used under
      ``DECISION_DELTA`` / ``GAE`` to discount the return back to ``t=0``).
    * ``decision_times`` — the matching in-game-decision game-clock timestamps.
    * ``final_timestamp`` — the terminal checkpoint's game-clock time.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    features: np.ndarray
    margin: float
    iteration: int
    chosen_idx: int | None = None
    all_candidates: np.ndarray | None = None
    # In-game-return-at-t=0 reproduction (safe defaults for older samples).
    own_total: float = 0.0
    opp_total: float = 0.0
    won: int = 0
    margin_checkpoints: list[float] = pydantic.Field(default_factory=list[float])
    score_checkpoints: list[float] = pydantic.Field(default_factory=list[float])
    decision_times: list[float] = pydantic.Field(default_factory=list[float])
    final_timestamp: float = 0.0
