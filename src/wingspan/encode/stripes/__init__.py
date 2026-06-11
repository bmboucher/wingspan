"""Programmatic stripe registries for the state and choice vectors.

Each public function returns a :class:`VectorLayout` that lists every stripe in
the order they appear in the flat vector, with a short reference name, a human
description, size, encoding kind, value range, and optional sub-field notes.
All sizes are derived from the same ``layout`` constants the encoders use, so a
change to ``layout.py`` automatically flows through to this registry.

Both layouts take an :class:`layout.EncodingSpec`; the config-driven setup pieces
(the choice ``setup_agg`` stripe and the decision-type one-hot's setup column)
are present only when ``spec.include_setup``. ``wingspan-inspect`` passes the
run's spec so the report shows exactly the fields that run encodes.

Sub-modules:

- ``descriptors``   — :class:`StripeSpec`, :class:`SubFieldDescriptor`,
                      :class:`StripeDescriptor`, :class:`VectorLayout` models
- ``embed_rules``   — post-embedding rewrite logic shared by state and choice
- ``state``         — :func:`state_stripe_layout` and its sub-field builders
- ``choice``        — :func:`choice_stripe_layout` and its sub-field builders
- ``card_feature``  — :func:`card_feature_stripe_layout` and
                      :func:`hand_encoder_input_stripe_layout`
"""

from __future__ import annotations

from wingspan.encode.stripes.card_feature import (
    card_feature_stripe_layout,
    hand_encoder_input_stripe_layout,
)
from wingspan.encode.stripes.choice import (
    choice_stripe_layout,
    raw_choice_stripe_layout,
)
from wingspan.encode.stripes.descriptors import (
    StripeDescriptor,
    StripeSpec,
    SubFieldDescriptor,
    VectorLayout,
)
from wingspan.encode.stripes.state import state_stripe_layout

__all__ = [
    "StripeSpec",
    "SubFieldDescriptor",
    "StripeDescriptor",
    "VectorLayout",
    "state_stripe_layout",
    "choice_stripe_layout",
    "raw_choice_stripe_layout",
    "card_feature_stripe_layout",
    "hand_encoder_input_stripe_layout",
]
