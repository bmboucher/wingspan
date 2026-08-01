# model — PyTorch policy-value network

Builds the actor-critic network from a `ModelArchitecture` descriptor. This package
is the only one with a hard PyTorch dependency; all topology description
lives in the torch-free `architecture.py` at the package root so training
config, runmeta, and the version checker can import it without torch.

## Modules

**`__init__.py`** — re-exports `PolicyValueNet`.

**`core.py`** — `PolicyValueNet(arch: ModelArchitecture, spec: EncodingSpec)`:
the main actor-critic network. Key structure:
- Shared trunk MLP: `state_dim → trunk_layers → M`.
- Choice encoder MLP: `choice_dim → choice_layers → N`.
- Concatenated scorer input: `M + N` → per-family head → logits.
- Value head: `M → value_layers → 1`.
- Optional shared card embedding (`nn.Embedding`) and hand encoder (if
  `arch.use_distinct_hand_model`).
- `PolicyValueNet.encode_state(gs, spec) -> Tensor` and
  `PolicyValueNet.encode_choices(gs, decision, spec) -> Tensor` — the sanctioned
  inference entry points; never pair the live encoder with a stale spec by hand.
- `PolicyValueNet.raw_state_stripe_layout()` / `raw_choice_stripe_layout()` —
  the `VectorLayout` describing this net's own `encode_*` output; compat-era
  subclasses override them (`without_stripes` of the columns they delete), so
  decode consumers (the game-log encoding viewer) never pair an era vector with
  the live layout.
- `PolicyValueNet.card_table() -> Tensor` — precomputes and caches the full
  card-feature matrix (all `N_BIRDS` embeddings); cached per forward pass
  (the card-table inference cache).
- `PolicyValueNet.from_model_config(config: ModelConfig) -> PolicyValueNet` —
  reconstitutes a net from a persisted descriptor; the only valid way to load
  a checkpoint for inference.
- `PolicyValueNet.class_for_version(artifact_version) -> type[PolicyValueNet]`
  — the single era-routing table (v1.0 → `compat.v1_0.PolicyValueNetV1_0`,
  all later same-MAJOR → live `PolicyValueNet`); used by `from_model_config`,
  the checkpoint loaders, and the era-pinned training pipeline.
- `StateEmbedOffsets(card_index, hand_multihot, decision_type)`
  — NamedTuple seam frozen by era shims so `_embed_state` slices each era's
  vector at its own offsets. Future shims override via `_state_embed_offsets()`.
- `ChoiceEmbedOffsets(board_idx, bird_id, becomes_playable, becomes_unplayable, kept_multihot)`
  — NamedTuple seam for the choice encoder; `becomes_unplayable` is `None` for
  v1.0 shims (v1.1 added it); `kept_multihot` is `None` outside setup.
  `_embed_choices` loops over whichever stripes are non-None, summing each through
  the shared card table, then splices out all card-index / multi-hot regions and
  concatenates the rest with the resulting embeddings.
- Board self-attention (`arch.use_board_attention`, optional): `board_attn_me` /
  `board_attn_opp` — two independent `nn.MultiheadAttention` modules, one per
  seat, each over that seat's 15 board-slot tokens (`card_embed_dim ⊕ 9` mutable
  scalars, `token_dim=73` by default). `arch.board_attention_heads` (default 1)
  sets the head count for both modules; since the true token width (73, or 81
  with positions) rarely divides it, `architecture.board_attention_embed_dim`
  pads the module's `embed_dim` up to the next multiple — `_apply_board_attention`
  zero-pads the token right before calling the module and slices the output back
  to `token_dim` right after, so the pad never escapes that one function (trunk
  input width is unaffected either way). `_embed_state_board_attention` builds
  the token sequences, `_apply_board_attention` (module-level) runs masked
  self-attention with a residual (empty slots stay exactly zero via
  `key_padding_mask` + an explicit `masked_fill`, unaffected by the pad since a
  zero token stays zero when padded), and the flattened outputs replace the
  standard per-slot card lookups + board continuous stripes.
  `arch.board_attention_positions` (optional, requires `use_board_attention`)
  concatenates a third, constant block onto every token — `board_position`, a
  non-persistent `[15, BOARD_POSITION_DIM]` habitat-one-hot ⊕ column-one-hot
  buffer built by the module-level `_board_position_matrix()` — so attention
  *mixing* (otherwise fully permutation-equivariant over the 15 slots) can
  condition on slot position; masked to zero on empty slots like the rest of the
  token. All three flags are REGIME (config-carried, join `ShapeKey` —
  `board_attention_heads` via its own field since padded shapes can coincide
  across head counts while the module still computes a different function); see
  `docs/reports/board-self-attention.md`.

**`mlp.py`** — Shared MLP building blocks used by both the policy net and the
setup net so they produce byte-identical stacks from the same width list:
- `build_body(in_dim, widths, *, between_activation, final_activation, dropout, layernorm) -> (nn.Sequential, int)` — body MLP; skips activation when value is `ActivationName.NONE`
- `build_readout(in_dim, widths, *, between_activation, final_activation, dropout) -> nn.Sequential` — readout MLP with optional final activation

**`hand_model.py`** — Stateless multi-card set-embedder helpers used by both the
main net and the setup net:
- `pool_card_set(multihot, card_rows, pooling) -> Tensor` — permutation-invariant
  pooling of a card set over the shared card table. `pooling` selects the mode
  (`MEAN`/`SUM`/`MAX`/`CONCAT_MAX_SUM`); output width is `M`, `M`, `M+1`, or `2M+1`.
  Used by `core.py` when `arch.use_distinct_hand_model` is `False` (new-run default).
- `embed_card_set(hand_encoder, multihot, summary) -> Tensor` — applies the dedicated
  hand encoder MLP to `[multi-hot ⊕ 10-dim summary]`. Used when
  `arch.use_distinct_hand_model` is `True` (retained for old-artifact back-compat).
- `set_summary_from_multihot`, `set_summary_from_indices`, `multihot_from_indices` —
  in-model derivation helpers for tray and setup-kept-set paths.
