# encode.stripes — Programmatic stripe registry

Descriptor models and builder functions for the state, choice, and card-feature
stripe layouts. The `__init__.py` re-exports the key types for callers in
`encode/` and `training/runmeta.py`.

The stripe system makes the encoder self-describing: each stripe carries its
name, dim, and normalization so the inspector and HTML report can render the
full vector layout without hard-coding field names.

## Modules

**`__init__.py`** — re-exports `SubFieldDescriptor`, `StripeDescriptor`,
`VectorLayout`, `state_stripe_layout`, `choice_stripe_layout`,
`card_feature_stripe_layout`.

**`descriptors.py`** — Core data models:
- `SubFieldDescriptor(name, dim, offset)` — one named feature slice within a stripe.
- `StripeDescriptor(name, dim, offset, sub_fields)` — one named block in the
  encoding vector; `sub_fields` makes it inspectable.
- `VectorLayout(stripes, total_dim)` — the full ordered list of stripes for a
  vector; exposes `stripe_by_name(name)` and iteration.

**`embed_rules.py`** — Post-embedding rewrite logic shared by state and choice
encoders. `apply_embed_rules(vec, rules, card_table)` replaces card-index columns
with their embedded representations after the base encoding pass.

**`state.py`** — `state_stripe_layout(spec: EncodingSpec) -> VectorLayout`.
Builder that assembles all state stripes in canonical order (board slots by
habitat, tray, birdfeeder, food cache, round goals, hand summary). Each stripe
builder is a private function returning a `StripeDescriptor`.

**`choice.py`** — `choice_stripe_layout(spec: EncodingSpec) -> VectorLayout`.
Builder for the per-choice row stripes (decision-type one-hot, choice-type
one-hot, per-`Choice` feature fields). Stripe order is part of the checkpoint
format.

**`card_feature.py`** — `card_feature_stripe_layout() -> VectorLayout` and
`hand_encoder_input_stripe_layout() -> VectorLayout`. Descriptor for the
per-card feature vector fed into the shared card embedding and the hand encoder.
