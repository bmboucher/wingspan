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
`state_embed_rules(card_embed_dim, ..., num_players=2)`: when
`use_board_attention`, `board_me` expands to its attention-output width
(`SLOTS_PER_BOARD × (card_embed_dim + SLOT_SCALAR_DIM)`, `+ BOARD_POSITION_DIM`
per slot when `board_attention_positions` is additionally set), and so does
*every* opponent board stripe (`board_opp`, `board_opp2`, ... one per opponent
clockwise, per `num_players`) — they all describe the same module applied
once per opponent board (`model.core`): `board_attn_opp` by default, or the
single shared `board_attn` under `board_attention_shared`, never one module
per opponent either way. Stripe SIZES are identical in both modes — sharing
changes which module computes the block, not its width. `card_idx_board` is
folded into them (`new_size=0`). `num_players` also sizes `card_idx_board`'s
board portion (`n_board = num_players * SLOTS_PER_BOARD`) on the non-attention
path.

**`state.py`** — `state_stripe_layout(spec: EncodingSpec) -> VectorLayout`.
Builder that assembles all state stripes in canonical order (board slots by
habitat, tray, birdfeeder, food cache, round goals, hand summary). Each stripe
builder is a private function returning a `StripeDescriptor`.
`_build_raw_state_stripes(spec)` is N-player-aware: at `spec.num_players >= 3`
it inserts a `turn_position` stripe after `turn_state` and, for each opponent
clockwise, a suffixed replica (`food_opp2`, `board_opp2`,
`board_summary_opp2`, `bonus_count_opp2`, `hand_size_opp2`, ...) directly
after its singular sibling via the `_opponent_description` /
`layout._opponent_suffix` helpers; `round_goals`' sub-fields
(`_round_goals_sub_fields(spec)`) report a `other_counts` vector sub-field
(width N−1) instead of the singular `opp_count` scalar at N>=3. Also
`board_token_stripe_layout(card_embed_dim, *, board_attention_positions=False) ->
VectorLayout` — describes one board-attention input token (the shared card
embedding concatenated with the slot's 9 mutable scalars from
`_slot_scalar_sub_fields`, also reused by `_board_slot_sub_fields` for the
135-sub-field per-board stripes); with `board_attention_positions`, appends a
trailing `position_habitat` (3) + `position_column` (5) constant block.

**`choice.py`** — `choice_stripe_layout(spec: EncodingSpec) -> VectorLayout`.
Builder for the per-choice row stripes (decision-type one-hot, choice-type
one-hot, per-`Choice` feature fields). Stripe order is part of the checkpoint
format: `resets_feeder` (v1.4) then `goal_delta_ignoring_eggs` (v1.6, the last
base stripe) are appended in that order. `raw_choice_stripe_layout(spec)`
inserts a `player_select` stripe (width `spec.num_players`) after
`goal_delta_ignoring_eggs`, before the conditional setup stripes, when
`spec.num_players >= 3` — absent at N=2.

**`card_feature.py`** — `card_feature_stripe_layout() -> VectorLayout` and
`hand_encoder_input_stripe_layout() -> VectorLayout`. Descriptor for the
per-card feature vector fed into the shared card embedding and the hand encoder.
