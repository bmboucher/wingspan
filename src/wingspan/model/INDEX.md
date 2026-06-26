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
- `net.encode_state(gs, spec) -> Tensor` and `net.encode_choices(gs, decision,
  spec) -> Tensor` — the sanctioned inference entry points; never pair the live
  encoder with a stale spec by hand.
- `net.card_table() -> Tensor` — precomputes and caches the full card-feature
  matrix (all `N_BIRDS` embeddings); result is cached per forward pass for
  efficiency (the card-table inference cache).
- `PolicyValueNet.from_model_config(config: ModelConfig) -> PolicyValueNet` —
  reconstitutes a net from a persisted descriptor; the only valid way to load
  a checkpoint for inference.
- `PolicyValueNet.class_for_version(artifact_version) -> type[PolicyValueNet]`
  — the single era-routing table (pre-0.1 → `PolicyValueNetV00`, 0.1 →
  `…V01`, 0.2 → `…V02`, 0.3 → `…V03`, 0.4/0.5 → `…V04`, live otherwise);
  used by `from_model_config`, the checkpoint loaders, and the era-pinned
  training pipeline.
- `StateEmbedOffsets(card_index, hand_multihot, decision_type, hand_summary, hand_summary_end)`
  — NamedTuple seam frozen by era shims so `_embed_state` slices each era's
  vector at its own offsets (the v0.6 insertion of two playability stripes shifts
  `decision_type` by +360; earlier eras override via `_state_embed_offsets()`).
  The `hand_summary` / `hand_summary_end` pair is `(0, 0)` in the live v0.9 net
  (stripe absent, derived in-model) and `(343, 353)` for pre-0.9 shims (stripe
  physically present in the frozen 1155-dim vector and excised from the continuous
  prefix before the trunk).
- `ChoiceEmbedOffsets(board_idx, bird_id, becomes_playable, kept_multihot)`
  — NamedTuple seam for the choice encoder; `becomes_playable` is `None` for
  pre-0.6 shims that lack the stripe, `kept_multihot` is `None` outside setup.

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
