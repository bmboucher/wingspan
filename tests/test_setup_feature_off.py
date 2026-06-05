"""With the setup model off, the main net carries the opening.

When ``use_setup_model=False`` the main net's encoding *includes* setup
(``EncodingSpec.include_setup=True``): ordinary self-play collection records the
setup decision as a step routed to the SETUP judgment-family head (so the main
net trains on it), and produces no separate setup samples.
"""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from wingspan import architecture, decisions, encode, model  # noqa: E402
from wingspan.training import collect  # noqa: E402

_SMALL_ARCH = architecture.ModelArchitecture(
    trunk_layers=(32, 32),
    choice_layers=(32, 32),
    head_layers=(),
    value_layers=(),
    card_embed_dim=8,
)


def test_self_play_records_setup_step_for_main_net():
    # include_setup=True is the setup-off fallback: the main net's encoding keeps
    # the SetupDecision column + SETUP head, so play_game scores the opening.
    net = model.PolicyValueNet(
        arch=_SMALL_ARCH, spec=encode.EncodingSpec(include_setup=True)
    )
    record = collect.play_game(net, torch.device("cpu"), random.Random(0), seed=7)
    setup_family = decisions.family_index_for(decisions.SetupDecision)
    assert any(step.family_idx == setup_family for step in record.steps)
    # The unchanged self-play path records no separate setup samples.
    assert record.setup_samples == []
