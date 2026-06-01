"""State and per-choice encoders for RL.

``encode_state`` produces a fixed-size dense feature vector summarizing the game
from the perspective of the player about to decide; ``encode_choices`` produces a
``(n_choices, CHOICE_FEATURE_DIM)`` matrix describing each legal choice. The two
meet in the model. The package is split by concern:

- ``layout``        — feature dimensions, stripe offsets, normalization scales
- ``state_encode``  — ``encode_state`` / ``state_size`` + the state summaries
- ``choice_encode`` — ``encode_choices`` + the per-choice featurizers

The public constants below (read by ``model.py`` and the training pipeline) are
re-exported from ``layout``.
"""

from wingspan.encode.choice_encode import encode_choices, encode_decision
from wingspan.encode.layout import (
    CHOICE_BIRD_ID_DIM,
    CHOICE_BIRD_ID_OFFSET,
    CHOICE_BONUS_ID_OFFSET,
    CHOICE_FEATURE_DIM,
    DECISION_TYPE_DIM,
    GOAL_CATEGORIES,
    HAND_MULTIHOT_DIM,
    MAX_GOAL_CATEGORIES,
    N_CARD_INDEX_SLOTS,
    OFF_CARD_INDEX,
    OFF_DECISION_TYPE,
    OFF_HAND_MULTIHOT,
    RUNAWAY_CHOICE_THRESHOLD,
    SOFT_CHOICE_WARN_THRESHOLD,
)
from wingspan.encode.state_encode import encode_state, state_size

__all__ = [
    "CHOICE_BIRD_ID_DIM",
    "CHOICE_BIRD_ID_OFFSET",
    "CHOICE_BONUS_ID_OFFSET",
    "CHOICE_FEATURE_DIM",
    "DECISION_TYPE_DIM",
    "GOAL_CATEGORIES",
    "HAND_MULTIHOT_DIM",
    "MAX_GOAL_CATEGORIES",
    "N_CARD_INDEX_SLOTS",
    "OFF_CARD_INDEX",
    "OFF_DECISION_TYPE",
    "OFF_HAND_MULTIHOT",
    "RUNAWAY_CHOICE_THRESHOLD",
    "SOFT_CHOICE_WARN_THRESHOLD",
    "encode_choices",
    "encode_decision",
    "encode_state",
    "state_size",
]
