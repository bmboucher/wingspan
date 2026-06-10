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
  `…V01`, 0.2 → `…V02`, live otherwise); used by `from_model_config`, the
  checkpoint loaders, and the era-pinned training pipeline.

**`mlp.py`** — Shared MLP building blocks used by both the policy net and the
setup net so they produce byte-identical stacks from the same width list:
- `build_body(in_dim, hidden_widths, activation, dropout, layernorm) -> nn.Sequential`
- `build_readout(in_dim, out_dim) -> nn.Linear`

**`hand_model.py`** — Stateless multi-card set-embedder helpers for the optional
hand/tray/setup-kept-set encoders:
- `encode_set(card_indices, card_table, embed_dim) -> Tensor` — mean-pools a
  variable-length set of card embeddings into a fixed-width vector.
- Used by `core.py` when `arch.use_distinct_hand_model` is `True` to encode
  the player's hand, the tray, and the setup kept-set as separate inputs to
  the trunk rather than as part of the flat state vector.
