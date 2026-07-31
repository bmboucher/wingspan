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
`card_feature_stripe_layout`, `board_token_stripe_layout`.

**`descriptors.py`** — Core data models:
- `SubFieldDescriptor(name, dim, offset)` — one named feature slice within a stripe.
- `StripeDescriptor(name, dim, offset, sub_fields)` — one named block in the
  encoding vector; `sub_fields` makes it inspectable.
- `VectorLayout(stripes, total_dim)` — the full ordered list of stripes for a
  vector; exposes `offset_of(name) -> int`, `size_of(name) -> int`, iteration, and
  `without_stripes(names) -> VectorLayout` (drop named stripes, shifting later
  offsets left — how era compat nets derive their era's layout from the live one,
  mirroring the `np.delete` in their `encode_*` overrides).

**`embed_rules.py`** — Post-embedding rewrite rules for state, choice, and setup
vectors. `embed_layout(raw, rules, expected_total)` rewrites a raw `VectorLayout`
into the network's post-embedding view by expanding card-index / multi-hot stripes
to their embedded widths. `state_embed_rules`, `choice_embed_rules`, and
`setup_embed_rules` supply per-run rule dicts. For `setup_embed_rules(card_embed_dim,
set_width, *, use_distinct)`: `kept_cards` and optional `playable_kept_cards` /
`turn1_playable` stripes expand to `set_width = pooled_hand_width` (pooling path)
or `hand_embed_width` (distinct-encoder path); `tray` expands to
`TRAY_SIZE × card_embed_dim` (per-slot rows only — no tray-set embedding).
`state_embed_rules(..., use_board_attention, board_attention_positions)`: when
`use_board_attention`, `board_me`/`board_opp` expand to their attention-output
width (`SLOTS_PER_BOARD × (card_embed_dim + SLOT_SCALAR_DIM)`, `+
BOARD_POSITION_DIM` per slot when `board_attention_positions` is additionally
set) and `card_idx_board` is folded into them (`new_size=0`).

**`state.py`** — `state_stripe_layout(spec: EncodingSpec) -> VectorLayout`.
Builder that assembles all state stripes in canonical order (board slots by
habitat, tray, birdfeeder, food cache, round goals, hand summary). Each stripe
builder is a private function returning a `StripeDescriptor`. Also
`board_token_stripe_layout(card_embed_dim, *, board_attention_positions=False) ->
VectorLayout` — describes one board-attention input token (the shared card
embedding concatenated with the slot's 9 mutable scalars from
`_slot_scalar_sub_fields`, also reused by `_board_slot_sub_fields` for the
135-sub-field `board_me`/`board_opp` stripes); with `board_attention_positions`,
appends a trailing `position_habitat` (3) + `position_column` (5) constant block.

**`choice.py`** — `choice_stripe_layout(spec: EncodingSpec) -> VectorLayout`.
Builder for the per-choice row stripes (decision-type one-hot, choice-type
one-hot, per-`Choice` feature fields). Stripe order is part of the checkpoint
format.

**`card_feature.py`** — `card_feature_stripe_layout() -> VectorLayout` and
`hand_encoder_input_stripe_layout() -> VectorLayout`. Descriptor for the
per-card feature vector fed into the shared card embedding and the hand encoder.
