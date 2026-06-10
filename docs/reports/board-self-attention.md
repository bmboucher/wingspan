# Board Self-Attention Feasibility

**Question:** Would a self-attention layer over 15 board slots improve the model?

---

## Current board encoding path

The board encoding lives in `src/wingspan/model/core.py`, method `_embed_state`
(lines 403–475). The path is:

1. **Card-index block** — the encoder writes one integer index per board slot
   (bird_index + 1, 0 = empty) into a contiguous block of
   `N_BOARD_INDEX_SLOTS = 2 × 15 = 30` columns in the flat state vector
   (`src/wingspan/encode/layout.py`, line 403).
2. **Card-table lookup** — `core.py:428` gathers all 30 indices from the shared
   `[181, card_embed_dim]` embedding table in a single batched call
   (`card_table[card_idx]`) and immediately reshapes to a flat vector of
   `30 × card_embed_dim` dims.
3. **Mutable per-slot scalars** — 9 continuous values per slot (eggs,
   egg-capacity remaining, cached food ×5, tucked, activations) are encoded
   by `_board_slots_continuous` (`src/wingspan/encode/state_encode.py:271`) and
   concatenated into the flat continuous prefix before the card-index block.
4. **Trunk input** — all of the above (plus hand, tray, round-goal, and misc
   scalars) are concatenated into a single flat vector and fed directly into the
   trunk MLP; the trunk's first Linear takes **2,788 input dims** (`main.html`
   arch diagram, confirmed by `layout.trunk_input_dim`).

After step 4 the spatial structure of the 15-slot board is completely lost — the
trunk sees a single long vector with no slot axis. There is no attention, no
convolutional path, and no positional signal that groups "these 64 dims belong to
slot (forest, column 2)".

The trunk input width is computed in `src/wingspan/encode/layout.py:trunk_input_dim`
(lines 463–509). The relevant formula replaces `N_CARD_INDEX_SLOTS` raw int columns
with `N_CARD_INDEX_SLOTS × card_embed_dim` embedding dims.

`ShapeKey` is defined at `src/wingspan/architecture.py:38–51`. Any field that
changes a tensor shape must join it; adding self-attention would add at least two
new shape-governing fields.

---

## Per-slot token structure

Per-slot mutable scalars are defined in `src/wingspan/encode/layout.py` at lines
366–374 and written by `_write_slot_continuous` in
`src/wingspan/encode/state_encode.py` at lines 291–305.

| Component | Index | Width |
|-----------|-------|-------|
| eggs (normalized by `_EGG_COUNT_SCALE=6`) | `_SLOT_MUT_EGGS=0` | 1 |
| egg-capacity remaining | `_SLOT_MUT_EGG_CAP=1` | 1 |
| cached food per type (`cards.ALL_FOODS`, 5 types) | `_SLOT_MUT_CACHED=2..6` | 5 |
| tucked cards | `_SLOT_MUT_TUCKED=7` | 1 |
| activations this round | `_SLOT_MUT_ACTIVATIONS=8` | 1 |
| **Total mutable scalars per slot** (`_SLOT_MUT_DIM`) | | **9** |

The bird's static attributes (points, food cost, nest, habitat, wingspan, power
color, etc.) are not in the per-slot continuous block — they ride the shared card
table via the integer index column. The card encoder maps 224-dim
`[44 static attrs ⊕ 180-dim identity one-hot]` to `card_embed_dim=64` dims
(`src/wingspan/encode/layout.py:352–359`).

**Token width per slot:**

```
token_width = card_embed_dim + _SLOT_MUT_DIM = 64 + 9 = 73
```

An empty slot contributes the embedding table's padding row (index 0 → a learned
"empty" vector) plus all-zero mutable scalars.

---

## How self-attention would work

### Token construction

For each of 15 slots on one board, construct a token:

```
token[i] = concat(card_table[slot_card_index[i]],  # 64 dims
                  slot_scalars[i])                  # 9 dims
```

This gives a sequence `T ∈ ℝ^{15 × 73}`. Both boards could be treated
independently (two separate passes over 15 tokens each) or jointly as one
sequence of 30 tokens. The joint path lets slot `(forest, col 3)` of the active
player attend to `(grassland, col 1)` of the opponent, but mixes self-state with
opponent-state in a way the current architecture keeps separated. Starting with two
independent 15-token passes is simpler and matches the current encoder topology.

### Attention layer

- **Input:** `[B, 15, 73]` (or `[B, 30, 73]` for the joint variant).
- **Output:** same shape `[B, 15, 73]`.
- **Mechanism:** standard multi-head self-attention —
  `nn.MultiheadAttention(embed_dim=73, num_heads=h, batch_first=True)`.
  Each slot's output is a weighted sum of all 15 slots' value projections, where
  the weights are derived from the dot-product of query (what this slot is looking
  for) and key (what other slots offer).
- **Positional encoding:** optional. Board slots have an implicit 2-D structure
  (3 habitats × 5 columns) but the Wingspan rules treat slots as unordered within
  a habitat (you fill from left to right, but the choice of *which* column is
  irrelevant after placement). A learned 15-dim positional embedding added to each
  token would let the model distinguish column 1 from column 5, but may be
  unnecessary.
- **Depth:** one layer is likely sufficient; multiple stacked layers would be
  unusual at this token-count and could overfit given the small sequence length.

### What self-attention captures that the current MLP cannot

The trunk MLP sees the flat concat of all slot embeddings — it can, in principle,
learn interactions between any two slots, but only via the weights of its first
Linear layer. That layer maps `2,788 dims → 128` with `2,788 × 128 + 128 = 357,248`
parameters, and there is no structural prior that "slot 3 and slot 7 should interact
more than slot 3 and a food-inventory dim". A self-attention layer instead:

- Computes explicit pairwise slot interactions before the trunk sees the data.
- Can learn "this slot has 4 eggs and its neighbour has tucked-card power — together
  they form a combo" as an inductive bias rather than hoping the first MLP layer
  discovers it.
- Produces a slot-aware summary that feeds a smaller trunk (if the attention output
  is projected down before concatenation).

Specific Wingspan patterns this could help with:
- Habitat-full detection: the model would "see" that all 5 slots in the forest row
  are non-empty without needing the trunk to infer it from 5 × 64 scattered dims.
- Egg-laying synergy: a brown-power bird that benefits from neighbors' egg counts
  needs cross-slot awareness.
- Round-goal contribution: spotting that 3 forest birds already qualify for the
  current round goal and a 4th would change placement.

### Integration with the current architecture

Two options:

**Option A — Flat replacement (same downstream shape):**
After attention, flatten `[B, 15, 73] → [B, 1095]` and slot this into the same
position where the current `[B, 30 × 64]` board embedding lives. The trunk input
dim changes; this is a FRESH change (see below).

**Option B — Additive residual (optional module):**
Add a residual around the attention: `out[i] = token[i] + attn(token)[i]`. If
`use_board_attention=False` the residual path is the identity and the model is
exactly the current architecture. The flag must join `ShapeKey` only if the
attention output changes the downstream tensor shapes — with a residual that does
not change `token_width`, it does. The trunk still receives a different number of
dims (73 vs 64 per slot), so this remains FRESH.

**Option C — Board summary projection:**
Apply attention, then mean-pool the output across the 15 slot dimension to get one
`[B, 73]` board summary vector, concatenated alongside (not replacing) the existing
flat slot embeddings. This widens the trunk input by 73 dims per board and is also
a FRESH change.

In all cases the trunk receives the output, which then feeds the scorer heads as
today.

---

## Parameter and compute cost

Using `attn_dim = 64` and `token_width = 73`:

| Component | Formula | Count |
|-----------|---------|-------|
| Q projection | `token_width × attn_dim + attn_dim` | 73 × 64 + 64 = 4,736 |
| K projection | same | 4,736 |
| V projection | same | 4,736 |
| Output projection | `attn_dim × token_width + token_width` | 64 × 73 + 73 = 4,745 |
| **One 1-head attention layer** | | **18,953** |

With 4 heads (attn_dim per head = 16, total QKV the same since `nn.MultiheadAttention`
splits embed_dim across heads):

```
4 × (3 × 73 × 16 + 73 × 16) = 4 × (3,504 + 1,168) ≈ 18,688 params
```

(Slight variation depending on whether bias is included — the formula above is
bias-included. PyTorch's `nn.MultiheadAttention` uses in_proj and out_proj, giving
`4 × token_width² + 2 × token_width` total for 1-head;
for `h`-heads, same count because the head split is within `embed_dim`.)

A clean per-case breakdown for `embed_dim = token_width = 73`, any number of heads:

```
in_proj (Q, K, V combined): 3 × 73 × 73 + 3 × 73 = 16,350
out_proj:                       73 × 73 +     73 =  5,402
Total (one board):                                 21,752
```

Two boards (independent passes): 2 × 21,752 = **43,504 params**.

**Comparison to current model:**

| | Params |
|---|--------|
| Current model total | 1,013,901 |
| One-board attention layer (`embed_dim=73`) | 21,752 |
| Both boards | 43,504 |
| Increase | **+4.3%** |

This is modest. Reducing `embed_dim` (e.g. to 32, projecting 73 → 32 → 73) would
cut it to roughly `4 × 73 × 32 + 2 × 32 ≈ 9,408` per board, a ~1.8% increase.

**Compute cost:** self-attention on 15 tokens is `O(15² × 73) ≈ 16,000` FLOPs per
forward pass. The trunk's first Linear is `2,788 × 128 ≈ 357,000` FLOPs — roughly
22× more expensive. The attention is dominated by the trunk; the runtime overhead
would be negligible (under 5% of a forward pass).

---

## FRESH classification and ShapeKey implications

Self-attention changes `trunk_input_dim` in all integration options (the 15-slot
path feeds a different number of dims into the trunk). This is a **FRESH change**
by the definition in `docs/VERSIONING.md`: it changes tensor shapes, so old
checkpoints cannot be loaded against the new architecture.

`ShapeKey` is defined at `src/wingspan/architecture.py:38–51`. The current tuple
is 12 elements long; the fields that would need to be added for board attention:

| New field | Why it must join ShapeKey |
|-----------|--------------------------|
| `use_board_attention: bool = False` | False-path must produce the same trunk input dim as today so a False checkpoint loads; True-path changes trunk input dim |
| `board_attn_heads: int = 1` | changes the internal projection shapes |
| `board_attn_dim: int \| None = None` | if added as a projection bottleneck, changes the attention weight shapes |

With `use_board_attention=False` (the default) the trunk receives the same
`N_CARD_INDEX_SLOTS × card_embed_dim` flat board embedding as today. Old
checkpoints simply have the field absent from their `model_config.json`; Pydantic
assigns the default `False`, and `architecture_key` resolves to the same value as
the current model's key — so old checkpoints are **not broken by the addition** as
long as `use_board_attention=False` preserves the current tensor shapes exactly.

Enabling `use_board_attention=True` requires a `MODEL_VERSION` bump (see
`docs/VERSIONING.md`), a new compat shim in `wingspan.compat`, and a new LFS
fixture set — the standard FRESH path.

---

## Recommendation

**The mechanism is sound and the cost is low; the question is whether it is the
right bottleneck to address now.**

### In favour

- The inductive bias is a natural fit: boards have slot structure, slots interact
  (egg-laying birds next to cached-food birds, habitat-full detection), and the
  current MLP has no way to express "attend to the 5 slots in this habitat before
  deciding". The ~4% parameter overhead and negligible runtime cost make this a
  free lunch if it helps.
- Implementation is straightforward: `nn.MultiheadAttention` exists in PyTorch;
  the token construction (`card_table[slot_idx] ‖ slot_scalars`) is already broken
  out in `_embed_state` — the indices and scalars are just not kept as a `[15, 73]`
  tensor before flattening.

### Against

- No evidence yet that board synergy is a training bottleneck. The model can in
  principle learn pairwise slot correlations through the trunk MLP — this is
  a structural convenience, not a capability it lacks entirely.
- The FRESH versioning overhead (version bump, compat shim, fixture set) is
  non-trivial. Given that `docs/TRAINING.md` lists Phase 0 infrastructure fixes as
  the first priority, adding a FRESH architecture change now would force a version
  bump before the current training baseline is solid.
- Three other FRESH-adjacent features (per-decision models, reward shaping,
  delta-stripe gaps) are ahead in the research queue per `docs/RESEARCH.md`.

### Verdict

**Defer until Phase 1 (baseline established), then try as a single-flag
experiment.** The implementation is contained (one new `ModelArchitecture` field +
a dozen lines in `_embed_state`), so it can be turned on without touching the rest
of the codebase. The Phase 0 checkpoint will serve as the control.

### Minimal experiment sketch

1. Add `use_board_attention: bool = False` to `ModelArchitecture.shape_key`
   (`src/wingspan/architecture.py:38–51`).
2. In `_embed_state` (`core.py:403–475`), before flattening: if
   `use_board_attention`, reshape the own-board slice of `slot_emb` to
   `[B, 15, card_embed_dim]`, concatenate the 9 mutable scalars from the
   continuous prefix to get `[B, 15, 73]`, apply `nn.MultiheadAttention`, flatten,
   then substitute back into the concat.
3. Train two runs from the same random seed: one with `use_board_attention=False`
   (baseline) and one with `use_board_attention=True` (experiment). Hold all other
   hyperparameters fixed.
4. Compare: win rate vs. random after 500K games, win rate in self-play eval, and
   sample efficiency (games to 55% win rate).
5. If the attended model reaches the baseline win rate with fewer games, or
   surpasses it, the inductive bias is paying off and the FRESH version bump is
   warranted.

The experiment requires `MODEL_VERSION` bump + compat shim only for the
`use_board_attention=True` run; the control run loads existing checkpoints without
a shim.
