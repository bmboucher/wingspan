"""State and per-choice encoders for RL.

``encode_state`` produces a fixed-size dense feature vector summarizing the game
from the perspective of the player about to decide; ``encode_choices`` produces a
``(n_choices, choice_feature_dim(spec))`` matrix describing each legal choice. The
two meet in the model. The package is split by concern:

- ``layout``        — feature dimensions, stripe offsets, normalization scales
- ``state_encode``  — ``encode_state`` / ``state_size`` + the state summaries
- ``choice_encode`` — ``encode_choices`` + the per-choice featurizers

The encoders' *shape* is config-driven on one axis (``EncodingSpec.include_setup``);
the spec-dependent totals are functions (``state_size`` / ``choice_feature_dim`` /
``decision_type_dim`` / ``num_families``) and the module constants
``CHOICE_FEATURE_DIM`` / ``DECISION_TYPE_DIM`` are the default-spec values. The
public names below (read by ``model.py`` and the training pipeline) are
re-exported from ``layout`` / ``state_encode``.
"""

from wingspan.encode.choice_encode import encode_choices, encode_decision
from wingspan.encode.layout import (
    CARD_FEATURE_DIM,
    CHOICE_BIRD_ID_DIM,
    CHOICE_BIRD_ID_OFFSET,
    CHOICE_BOARD_IDX_OFFSET,
    CHOICE_BOARD_IDX_SLOTS,
    CHOICE_BONUS_ID_OFFSET,
    CHOICE_FEATURE_DIM,
    CHOICE_SETUP_OFFSET,
    DECISION_TYPE_DIM,
    DEFAULT_SPEC,
    GOAL_CATEGORIES,
    HAND_MULTIHOT_DIM,
    MAX_GOAL_CATEGORIES,
    N_CARD_INDEX_SLOTS,
    OFF_CARD_INDEX,
    OFF_DECISION_TYPE,
    OFF_HAND_MULTIHOT,
    RUNAWAY_CHOICE_THRESHOLD,
    SOFT_CHOICE_WARN_THRESHOLD,
    EncodingSpec,
    choice_feature_dim,
    choice_input_dim,
    decision_type_dim,
    num_families,
    spec_for,
    state_feature_dim,
    trunk_input_dim,
)
from wingspan.encode.state_encode import card_feature_matrix, encode_state, state_size

__all__ = [
    "CARD_FEATURE_DIM",
    "CHOICE_BIRD_ID_DIM",
    "CHOICE_BIRD_ID_OFFSET",
    "CHOICE_BOARD_IDX_OFFSET",
    "CHOICE_BOARD_IDX_SLOTS",
    "CHOICE_BONUS_ID_OFFSET",
    "CHOICE_FEATURE_DIM",
    "CHOICE_SETUP_OFFSET",
    "DECISION_TYPE_DIM",
    "DEFAULT_SPEC",
    "EncodingSpec",
    "GOAL_CATEGORIES",
    "HAND_MULTIHOT_DIM",
    "MAX_GOAL_CATEGORIES",
    "N_CARD_INDEX_SLOTS",
    "OFF_CARD_INDEX",
    "OFF_DECISION_TYPE",
    "OFF_HAND_MULTIHOT",
    "RUNAWAY_CHOICE_THRESHOLD",
    "SOFT_CHOICE_WARN_THRESHOLD",
    "card_feature_matrix",
    "choice_feature_dim",
    "choice_input_dim",
    "decision_type_dim",
    "encode_choices",
    "encode_decision",
    "encode_state",
    "num_families",
    "spec_for",
    "state_feature_dim",
    "state_size",
    "trunk_input_dim",
]
