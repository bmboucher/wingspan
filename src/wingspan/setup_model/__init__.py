"""The separately-trained setup model (a value-regression contextual bandit).

The start-of-game *setup* decision — which dealt cards / food / bonus to keep —
is one of the most consequential moves in Wingspan yet the most data-starved
(one decision per player per game). This package pulls it out of the unified
in-game policy into its own small network, trained to predict the expected
end-of-game score margin a setup leads to; the setup policy is a softmax over
the predicted margins of all 504 candidate keeps. The network evaluates
candidates through frozen, per-iteration-synced copies of the main net's two
shared card embedders (the single-card table and the multi-card set encoder),
so the data-starved setup MLP reads the representation the in-game model learns
from hundreds of decisions per game.

The torch-free pieces live here (descriptor, encoder, candidate enumeration,
random generator, sample store); the network and its learner live under
``wingspan.training`` (``setup_net`` / ``setup_learner``) so this package imports
without torch.
"""

from wingspan.setup_model.architecture import (
    SetupArchitecture,
    SetupEncoding,
    SetupParamReport,
    SetupShapeKey,
    count_setup_parameters,
    setup_readout_input_dim,
    setup_state_input_dim,
)
from wingspan.setup_model.candidates import (
    SetupCandidate,
    enumerate_setup_candidates,
    select_by_margins,
)
from wingspan.setup_model.encode import (
    SETUP_FEATURE_DIM,
    SETUP_GOAL_DIM,
    SetupContext,
    encode_setup_candidate,
)
from wingspan.setup_model.generate import JointSetup, RandomSetupGenerator, SeatDeal
from wingspan.setup_model.record import SetupSample
from wingspan.setup_model.stripes import (
    setup_readout_stripe_layout,
    setup_stripe_layout,
)

__all__ = [
    "JointSetup",
    "RandomSetupGenerator",
    "SETUP_FEATURE_DIM",
    "SETUP_GOAL_DIM",
    "SeatDeal",
    "SetupArchitecture",
    "SetupCandidate",
    "SetupContext",
    "SetupEncoding",
    "SetupParamReport",
    "SetupSample",
    "SetupShapeKey",
    "count_setup_parameters",
    "encode_setup_candidate",
    "enumerate_setup_candidates",
    "select_by_margins",
    "setup_readout_input_dim",
    "setup_readout_stripe_layout",
    "setup_state_input_dim",
    "setup_stripe_layout",
]
