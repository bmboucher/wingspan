# Board Self-Attention Feasibility

**Question:** Would a self-attention layer over the 15 board slots improve the
model — and can the same mechanism be extended to the hand and tray as input
tokens?

*Last verified against the codebase at `MODEL_VERSION = 0.6`. All line numbers and
dimensions below were checked against the live source; if a future encoding change
moves them, re-derive from `encode.trunk_input_dim` and the offsets in
`encode/layout.py` rather than trusting the literals here.*

---

## Current board encoding path

The board encoding lives in `src/wingspan/model/core.py`, method `_embed_state`
(lines 507–607). The path is:

1. **Card-index block** — the encoder writes one integer index per board slot
   (`bird_index + 1`, 0 = empty) into a contiguous block of
   `N_BOARD_INDEX_SLOTS = 2 × 15 = 30` columns (own board + opponent board),
   followed by `state.TRAY_SIZE = 3` tray columns, for
   `N_CARD_INDEX_SLOTS = 33` index columns total in the flat state vector
   (`src/wingspan/encode/layout.py:478–479`).
2. **Card-table lookup** — `core.py:531` gathers all 33 indices from the shared
   `[181, card_embed_dim]` embedding table in a single batched call
   (`card_table[card_idx]`) and immediately reshapes to a flat vector of
   `33 × card_embed_dim` dims.
3. **Mutable per-slot scalars** — 9 continuous values per slot (eggs,
   egg-capacity remaining, cached food ×5, tucked, activations) are encoded
   by `_board_slots_continuous` / `_write_slot_continuous`
   (`src/wingspan/encode/state_encode.py:299` and `:319–333`) and concatenated
   into the flat continuous prefix (the `board_me` / `board_opp` stripes) before
   the card-index block.
4. **Trunk input** — all of the above (plus the hand set embedding, extra
   playability set embeddings, round-goal, and misc scalars) are concatenated
   into a single flat vector and fed into the trunk MLP; the trunk's first Linear
   takes **2,876 input dims** (`main.html` model-summary diagram, confirmed by
   `layout.trunk_input_dim` for the default architecture).

After step 4 the spatial structure of the 15-slot board is completely lost — the
trunk sees a single long vector with no slot axis. There is no attention, no
convolutional path, and no positional signal that groups "these 64 dims belong to
slot (forest, column 2)".

The trunk input width is computed in `src/wingspan/encode/layout.py:trunk_input_dim`
(lines 556–612). The relevant formula replaces `N_CARD_INDEX_SLOTS` raw int columns
with `N_CARD_INDEX_SLOTS × card_embed_dim` embedding dims.

`ShapeKey` is defined at `src/wingspan/architecture.py:40–56`. Any field that
changes a tensor shape must join it; adding self-attention would add at least two
new shape-governing fields.

### What the encoder *already* does with variable-size card collections

This matters for the extension question below, so it is worth stating up front:
the current architecture already reduces two variable-size card collections to
fixed-width vectors, by **two different mechanisms**, neither of which is attention:

- **The hand** is encoded as an *order-invariant set*. A 180-dim multi-hot (which
  birds are held) plus a 10-dim hand summary feeds the dedicated hand encoder MLP
  (`use_distinct_hand_model=True` is now the **default**, see
  `architecture.py:110`), producing one fixed `hand_embed_width`-wide vector. The
  multi-hot is fixed-width regardless of how many cards are held; duplicates and
  order are irrelevant by construction. The two extra hand-playability multi-hots
  (`N_HAND_PLAYABLE_MULTIHOTS = 2`) are reduced the same way. An optional
  `tray_set_embedding` flag (`architecture.py:129`, default False) gives the tray
  the same set treatment.
- **The choice set** is encoded as a *variable-length sequence scored per element*.
  `forward` takes `choices: (B, K, choice_dim)` padded to `K` with a `mask: (B, K)`;
  the per-choice encoder broadcasts over `K`, every candidate gets one logit, and
  padding rows are set to `-inf` so they receive no probability mass
  (`core.py:222–291`).

So "an arbitrary number of cards → a usable network input" is **not a new
problem** in this codebase — it is solved already, twice. Self-attention is a more
expressive version of the same idea (it lets the cards interact before being
pooled or scored), and the two existing patterns are exactly the tools the
extension below reuses.

---

## Per-slot token structure

Per-slot mutable scalars are defined in `src/wingspan/encode/layout.py` at lines
441–449 and written by `_write_slot_continuous` in
`src/wingspan/encode/state_encode.py` at lines 319–333.

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
table via the integer index column. The card encoder maps the 224-dim
`[44 static attrs ⊕ 180-dim identity one-hot]` feature row to `card_embed_dim=64`
dims (`src/wingspan/encode/layout.py:427–434`, `architecture.py:88`).

**Token width per slot:**

```
token_width = card_embed_dim + _SLOT_MUT_DIM = 64 + 9 = 73
```

An empty slot contributes the embedding table's padding row (index 0 → a forced
zero vector, `core.py:293–305`) plus all-zero mutable scalars.

---

## How self-attention would work (board-only)

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
- **Empty-slot masking:** empty slots should be masked out with `key_padding_mask`
  so they contribute nothing to the attention sums (see the variable-size
  discussion below) — without it, a board with 2 birds would let 13 learned-empty
  tokens dilute the signal.
- **Positional encoding:** optional. Board slots have an implicit 2-D structure
  (3 habitats × 5 columns) but the Wingspan rules treat slots as unordered within
  a habitat (you fill from left to right, but the choice of *which* column is
  irrelevant after placement). A learned positional embedding could distinguish
  habitats (which *do* matter), but column position within a habitat probably
  should not carry signal — so a per-habitat (not per-slot) position embedding is
  the principled choice if any is used at all.
- **Depth:** one layer is likely sufficient; multiple stacked layers would be
  unusual at this token-count and could overfit given the small sequence length.

### What self-attention captures that the current MLP cannot

The trunk MLP sees the flat concat of all slot embeddings — it can, in principle,
learn interactions between any two slots, but only via the weights of its first
Linear layer. That layer maps `2,876 dims → 128` with `2,876 × 128 + 128 = 368,256`
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

Three options:

**Option A — Flat replacement (same downstream shape):**
After attention, flatten `[B, 15, 73] → [B, 1095]` and slot this into the same
position where the current per-slot board embedding lives. The trunk input
dim changes; this is a FRESH change (see below).

**Option B — Additive residual (optional module):**
Add a residual around the attention: `out[i] = token[i] + attn(token)[i]`. If
`use_board_attention=False` the residual path is the identity and the model is
exactly the current architecture. Even with a residual the trunk now receives a
73-wide-per-slot block rather than the current 64-wide card embedding (the 9
mutable scalars are folded into the token instead of riding the flat continuous
prefix), so the trunk input dim still changes — this remains FRESH.

**Option C — Board summary projection (pool, don't replace):**
Apply attention, then pool the output across the 15-slot dimension (mean / max /
attention-pool — see the reduction discussion below) to get one `[B, 73]` (or
projected `[B, d]`) board summary, concatenated *alongside* the existing flat slot
embeddings. This widens the trunk input rather than replacing the slot block, and
is also a FRESH change. It is the cheapest way to add a slot-aware signal without
discarding the per-slot detail the trunk already consumes.

In all cases the trunk receives the output, which then feeds the scorer heads as
today.

---

## Extending to hand and tray tokens (the variable-size question)

The natural generalization the question raises: instead of board-only tokens,
build a token for **every card the player can see** — own board (15), opponent
board (15), tray (3), and hand (variable) — tag each with a **location encoding**
stripe, run one joint self-attention pass, and let cross-collection reasoning fall
out ("this hand card would complete my forest row"; "this tray card combos with my
cached-food bird"). Two sub-questions need answering: how variable hand size is
handled on the **input** side, and how the attention output is reduced to a fixed
size on the **output** side.

### Token layout with a location stripe

All tokens in one attention pass must share a width, but the collections carry
different mutable state (board slots have eggs/cached/tucked; hand and tray cards
have none). The fix is a **union token layout** with a location one-hot:

```
token = concat(card_table[card_index],   # 64  — shared card embedding
               mutable_block,            # 9   — board slots fill this; hand/tray zero it
               location_onehot)          # 4   — {board_me, board_opp, tray, hand}
```

Token width becomes `64 + 9 + 4 = 77`. The location one-hot is what lets a single
shared attention block handle heterogeneous tokens: the model learns that a token
tagged `hand` with a zero mutable block is a candidate-to-play, while a token
tagged `board_me` with 4 eggs is a deployed bird. The shared `card_table` row is
identical across locations, so the network reasons about *the same card* in
different roles — which is precisely the desired inductive bias.

### Input side: does the hand need a fixed max?

**Architecturally, no.** Self-attention has no built-in sequence-length limit — the
`QKᵀ` score matrix is `[L, L]` for whatever `L` you feed, and the learned
parameters (Q/K/V/output projections) are sized by the *token dimension*, not the
position count. A single forward pass accepts any number of tokens.

**For batched training, yes — a practical cap, handled by masking.** To batch `B`
decisions into one rectangular tensor `[B, L_max, 77]`, every sample must present
the same `L_max`, so you pad shorter samples and pass a
`key_padding_mask: [B, L_max]` (True = padding) so padding tokens contribute
nothing to any real token's attention and are excluded from pooling. **This is
exactly the pattern `forward` already uses for the choice set** — `choices` is
padded to `K` and `mask` `-inf`s the dead rows. You would:

- Size `L_max` from the fixed collections (15 + 15 + 3 = 33) plus a hand cap
  `H_max`. The board contributes a *constant* 33 tokens with empty slots masked;
  only the hand is genuinely variable.
- Choose `H_max` generously. Wingspan hands are unbounded in principle but small in
  practice; pick a cap above any realistic hand and, per the project's
  no-silent-caps rule (`CLAUDE.md`), `log()` a notice if a hand ever exceeds it
  rather than silently truncating.

**Or sidestep it entirely (recommended for the hand).** The hand is *already*
encoded order-invariantly as a fixed-width multi-hot set (above). Per-card hand
tokens only earn their keep if cards *within the hand* interact — and in Wingspan
they largely do not (hand cards combo with the **board**, not with each other).
The high-value place for attention is therefore the board, where deployed birds
genuinely interact, with the hand left as its existing set embedding (or, at most,
included as tokens that *attend to the board but are pooled back to one hand
summary*). The tray (3 fixed slots) is cheap to include as real tokens either way.

### Output side: reducing the attention output to a fixed size

The attention layer emits one vector per token: `[B, L, d]`. The trunk needs a
fixed width, so the `L` axis must be collapsed. Standard reductions, all of which
respect the `key_padding_mask` (pool over real tokens only):

1. **Mean / sum pool** → `[B, d]`. Simple, permutation-invariant, free handling of
   variable `L` (divide by the real-token count). This is *literally what the
   current set encoder does* — the mean-pool hand path is `hand_multihot @ card_table
   / count` (`core.py:579–581`). Attention-then-mean-pool is a strict generalization:
   the tokens interact first, then average.
2. **Max pool** → `[B, d]`. Good for "does *any* slot have property X" (habitat-full,
   any-predator-present, any-bird-qualifies-the-goal).
3. **Attention pooling / learned query** → `[B, d]`. One learned query vector
   attends over the `L` tokens and reads out a single weighted summary. More
   expressive than mean; one small extra parameter block.
4. **`[CLS]`/summary token** → `[B, d]`. Prepend one learned token to the sequence;
   after self-attention take *its* output row as the summary (BERT-style). The
   summary token attends to all cards and they to it. Equivalent in spirit to (3).

Any of these yields the fixed `[B, d]` you concatenate into the trunk input where
the flat board embedding lives today.

### The reduction question only applies to the *state/trunk* side

There are two consumers of card representations, and only one needs reduction:

- **State trunk (value head + per-decision state context):** wants a single fixed
  summary of the position → **pool** (options above).
- **Choice scoring (the pointer head):** already consumes a *variable-length* set
  of candidates and emits one logit each with a mask — **no reduction at all**. If
  you want attention to *help* choice scoring, you let the candidate tokens attend
  to the board/context tokens (self- or cross-attention) and read off the
  *per-candidate* output rows directly; the existing per-element scoring + mask
  consumes them unchanged. "Reduce to a fixed size" is a non-question on this path.

This is the cleanest framing of the whole concern: the model already answers
"variable number of cards → network" on both sides — pool for a summary, score
per-element for a decision — and self-attention slots into either without inventing
a new mechanism.

### Trade-offs of the unified (location-tagged) design

- **Pro:** one mechanism for all card collections; cross-collection reasoning the
  current separated encoders cannot express; the shared `card_table` means a card
  is reasoned about identically wherever it sits.
- **Con — POV hygiene:** mixing own-board, opponent-board, tray (public), and hand
  (private) tokens in one pass blends information the current encoder deliberately
  keeps in separate stripes. The location one-hot lets the model *re-separate* them,
  but it is now the model's job rather than the encoding's guarantee.
- **Con — scope:** this is strictly more than board-only attention. Each added
  collection is another shape-governing decision and more surface to get wrong.
  The board-only variant is the contained first step; widen to tray/hand only if
  board attention demonstrably pays off.

---

## Parameter and compute cost

Using `embed_dim = token_width = 73`. PyTorch's `nn.MultiheadAttention` splits
`embed_dim` across heads, so the head count does not change the parameter total —
but it requires `embed_dim % num_heads == 0`, and **73 is prime**, so multi-head
only works after projecting the token to a head-divisible width (e.g. 64 or 72).
A single head works at 73 directly; any multi-head variant implies the projection
bottleneck discussed below. Counts verified empirically against
`nn.MultiheadAttention(embed_dim=73, num_heads=1)`:

| Component | Formula | Count |
|-----------|---------|-------|
| in_proj (Q, K, V combined) | `3 × 73 × 73 + 3 × 73` | 16,206 |
| out_proj | `73 × 73 + 73` | 5,402 |
| **One attention layer (one board)** | | **21,608** |

Two boards (independent passes): 2 × 21,608 = **43,216 params**.

**Comparison to current model:**

| | Params |
|---|--------|
| Current model total (default arch) | 1,032,333 |
| One-board attention layer (`embed_dim=73`) | 21,608 |
| Both boards | 43,216 |
| Increase | **+4.2%** |

This is modest. Adding a projection bottleneck (`73 → 32 → 73` around an
`embed_dim=32` attention) roughly halves it to ~9,000 per board (~1.7% for both
boards). Extending to a unified 77-wide token over all collections is the same
order of magnitude — the cost is dominated by the per-token projections, and the
token count (≤ ~50) keeps the `O(L²)` term negligible.

**Compute cost:** self-attention on 15 tokens is `O(15² × 73) ≈ 16,000` multiply-adds
per forward pass. The trunk's first Linear is `2,876 × 128 ≈ 368,000` — roughly 23×
more. The attention is dominated by the trunk; the runtime overhead is negligible
(well under 5% of a forward pass). A unified ~50-token pass is `O(50² × 77) ≈
193,000` — still under the trunk's first layer, and the trunk itself shrinks if the
flat slot block is replaced by a pooled summary.

---

## FRESH classification and ShapeKey implications

Self-attention changes `trunk_input_dim` in all integration options (the slot path
feeds a different number of dims into the trunk). This is a **FRESH change**
by the definition in `docs/VERSIONING.md`: it changes tensor shapes, so old
checkpoints cannot be loaded against the new architecture.

`ShapeKey` is defined at `src/wingspan/architecture.py:40–56`. The current tuple
is **15 elements** long; the fields that would need to be added for board attention:

| New field | Why it must join ShapeKey |
|-----------|--------------------------|
| `use_board_attention: bool = False` | False-path must produce the same trunk input dim as today so a False checkpoint loads; True-path changes trunk input dim |
| `board_attn_heads: int = 1` | changes the internal projection shapes (only when the embed_dim is split unevenly; harmless to include for clarity) |
| `board_attn_dim: int \| None = None` | if added as a projection bottleneck, changes the attention weight shapes |

A unified hand/tray variant would add at least one more flag (e.g.
`card_attention_scope`) plus the hand cap `H_max` and the location-stripe width to
the shape signature.

With `use_board_attention=False` (the default) the trunk receives the same
per-slot board embedding it does today. Old checkpoints simply have the field
absent from their saved `model_config.json`; Pydantic assigns the default `False`,
and `shape_key` resolves to the same value as the current model's key — so old
checkpoints are **not broken by the addition** as long as `use_board_attention=False`
preserves the current tensor shapes exactly.

Enabling `use_board_attention=True` requires a `MODEL_VERSION` bump (currently
`0.6`; see `docs/VERSIONING.md`), a new compat shim in `wingspan.compat`, and a new
LFS fixture set — the standard FRESH path.

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
- The variable-size machinery the hand/tray extension needs (pad + `key_padding_mask`,
  pool-or-score) **already exists** in the codebase (choice-set masking; set
  embeddings), so the extension is reuse rather than new infrastructure.

### Against

- No evidence yet that board synergy is a training bottleneck. The model can in
  principle learn pairwise slot correlations through the trunk MLP — this is
  a structural convenience, not a capability it lacks entirely.
- The FRESH versioning overhead (version bump, compat shim, fixture set) is
  non-trivial. Given that `docs/TRAINING.md` lists Phase 0 infrastructure fixes as
  the first priority, adding a FRESH architecture change now would force a version
  bump before the current training baseline is solid.
- Other FRESH-adjacent features (per-decision models, reward shaping, delta-stripe
  gaps) are ahead in the research queue per `docs/RESEARCH.md`.
- The unified hand/tray variant additionally crosses the POV-separation line the
  current encoder maintains; it is the more speculative, later experiment.

### Verdict

**Defer until Phase 1 (baseline established), then try board-only attention as a
single-flag experiment first.** The board-only implementation is contained (one new
`ModelArchitecture` field + a dozen lines in `_embed_state`), so it can be turned on
without touching the rest of the codebase. The Phase 0 checkpoint serves as the
control. The location-tagged hand/tray unification is the **second** experiment,
attempted only if board attention shows signal — it is a larger shape change and
relaxes the encoder's POV separation, so it should not be the first thing tried.

### Minimal experiment sketch (board-only)

1. Add `use_board_attention: bool = False` to `ModelArchitecture` and to
   `shape_key` (`src/wingspan/architecture.py:40–56`).
2. In `_embed_state` (`core.py:507–607`), before flattening: if
   `use_board_attention`, reshape the own-board slice of `slot_emb` to
   `[B, 15, card_embed_dim]`, gather the matching 9 mutable scalars out of the
   `board_me` continuous stripe to get `[B, 15, 73]`, build a `key_padding_mask`
   for empty slots, apply `nn.MultiheadAttention`, pool (or flatten), then
   substitute back into the concat. (The mutable scalars and the embedded slot
   vectors live in *different* parts of the flat vector today — the scalars in the
   continuous prefix, the embeddings reconstructed from the index block — so the
   token build gathers from both.)
3. Train two runs from the same random seed: one with `use_board_attention=False`
   (baseline) and one with `use_board_attention=True` (experiment). Hold all other
   hyperparameters fixed.
4. Compare: win rate vs. random after 500K games, win rate in self-play eval, and
   sample efficiency (games to 55% win rate).
5. If the attended model reaches the baseline win rate with fewer games, or
   surpasses it, the inductive bias is paying off and the FRESH version bump is
   warranted — and the unified hand/tray variant becomes worth its larger cost.

The experiment requires a `MODEL_VERSION` bump + compat shim only for the
`use_board_attention=True` run; the control run loads existing checkpoints without
a shim.
