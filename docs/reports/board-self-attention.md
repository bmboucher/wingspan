# Board Self-Attention Feasibility

**Status: Board-only pass implemented, including optional per-slot position
features and multi-head attention.** `use_board_attention: bool = False`,
`board_attention_positions: bool = False`, and `board_attention_heads: int = 1`
are all configurable fields on `ModelArchitecture` (all three REGIME — no
`MODEL_VERSION` bump, no compat shim; see corrected classification below).
Enable them in the configurator under MODEL ARCHITECTURE ▸ STATE TRUNK
(`board_attention_positions` and `board_attention_heads` both require
`use_board_attention` and are hidden until it is on). The below analysis
stands; sections marked ✓DONE are reflected in the live code.

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
# with board_attention_positions=True:
token_width = card_embed_dim + _SLOT_MUT_DIM + BOARD_POSITION_DIM = 64 + 9 + 8 = 81
```

An empty slot contributes the embedding table's padding row (index 0 → a forced
zero vector, `core.py:293–305`) plus all-zero mutable scalars — and, with
positions on, a position block explicitly masked to zero for that slot (the
constant buffer itself has no notion of occupancy, so the model masks it the
same way `_apply_board_attention` masks the attention output).

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

**Weight sharing is a third, orthogonal axis.** Separate-vs-joint (above) is a
question about the *token sequence* — how many tokens one attention call
attends over. Whether the POV and opponent passes read one module's shared
*parameters*, or two independently-learned modules, is a separate question:
implemented sharing (✓DONE, `board_attention_shared`, see below) keeps the
passes exactly as separate as chosen above — 15 tokens each, no cross-board
attention — and ties only the parameters, not the token sequence.

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
- **Positional encoding:** ✓DONE (optional, `board_attention_positions`). Without
  it, attention is fully permutation-equivariant over the 15 slots — the
  *mixing* (which slots attend to which) cannot depend on habitat or column,
  even though the downstream flatten step still knows which output vector came
  from which slot. `board_attention_positions=True` concatenates a constant
  8-dim block — a 3-dim habitat one-hot ⊕ a 5-dim column one-hot
  (`encode.BOARD_POSITION_DIM`) — onto every token before attention, built once
  as a non-persistent buffer (`model/core.py::_board_position_matrix`) and
  masked to zero on empty slots so the "empty slot → exactly-zero token"
  invariant holds. This supersedes the speculation above that column shouldn't
  carry signal: concatenating a fixed one-hot ahead of a linear in-projection is
  mathematically equivalent to a learned, row/col-factorized additive bias in
  Q/K/V space, so including column costs nothing to *learn away* if it turns out
  not to matter, while omitting it would foreclose the option entirely. Token
  width becomes `73 + 8 = 81` (see the parameter-cost table below — multi-head
  attention (`board_attention_heads`) is implemented via zero-padding regardless
  of width, but `81 = 3⁴` is notable as the one case where several head counts —
  1, 3, 9, 27, 81 — divide it exactly and pad zero).
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

Using `embed_dim = token_width = 73` for the single-head (default) case.
PyTorch's `nn.MultiheadAttention` splits `embed_dim` evenly across heads, so it
requires `embed_dim % num_heads == 0` — and **73 is prime**, so multi-head only
works at a wider `embed_dim`.

**✓DONE — implemented as zero-padding, not a learned projection.**
`architecture.board_attention_embed_dim(token_width, num_heads)` pads up to the
next multiple of `num_heads` (`board_attention_heads` on
`ModelArchitecture`/`MainNetArchitecture`, default 1); `_apply_board_attention`
zero-pads the token immediately before calling the module and slices the
output back to `token_width` immediately after, so the pad never leaves that
one function — trunk input width is unaffected either way. This is equivalent
to a learned `token_width -> embed_dim` projection folded into the attention
module's own `in_proj` for free: the extra `in_proj` columns always see zero
input (inert, not a source of noise) and the corresponding `out_proj` rows are
exactly the ones sliced away, while every head still splits learned q/k/v of
the real signal — no head ever attends over only padding. A dedicated
projection layer would compose into a single linear transform anyway, so
nothing is gained by making it a separate module. `num_heads=1` always returns
`token_width` unchanged (pad 0), reproducing the original module exactly —
this is why the flag is REGIME (see below). Counts verified empirically
against `nn.MultiheadAttention(embed_dim=73, num_heads=1)`:

| Component | Formula | Count |
|-----------|---------|-------|
| in_proj (Q, K, V combined) | `3 × 73 × 73 + 3 × 73` | 16,206 |
| out_proj | `73 × 73 + 73` | 5,402 |
| **One attention layer (one board)** | | **21,608** |

Two boards (independent passes): 2 × 21,608 = **43,216 params**.

**With `board_attention_positions=True`** (`embed_dim = 81`):

| Component | Formula | Count |
|-----------|---------|-------|
| in_proj (Q, K, V combined) | `3 × 81 × 81 + 3 × 81` | 19,926 |
| out_proj | `81 × 81 + 81` | 6,642 |
| **One attention layer (one board)** | | **26,568** |

Two boards: 2 × 26,568 = **53,136 params** — an extra 9,920 params (+0.96% of the
current model total) over positions-off. The position block itself is a constant
buffer (no learned parameters); the added cost is entirely the wider Q/K/V/out
projections. Unlike the attention layer's own params, the trunk's first Linear
also grows — `N_BOARD_INDEX_SLOTS × BOARD_POSITION_DIM = 240` more input dims
(`encode.trunk_input_dim(..., board_position_dim=...)`), which dominates the
total parameter delta at typical trunk widths.

**✓DONE — with `board_attention_heads > 1`** (zero-padded `embed_dim`, per module
`4·E² + 4·E`; two boards double it, same base formula as above). At `W = 73`
(positions off):

| `board_attention_heads` | Padded `E` | Params/module | Δ vs. 1-head |
|---|---|---|---|
| 1 | 73 (no pad) | 21,608 | — |
| 2 | 74 | 22,200 | +592 |
| 4 | 76 | 23,408 | +1,800 |
| 8 | 80 | 25,920 | +4,312 |

At `W = 81` (`board_attention_positions=True`) — `81 = 3⁴`, so `heads=3` (or 9,
27, 81) divides it exactly and pads **zero**:

| `board_attention_heads` | Padded `E` | Params/module | Δ vs. 1-head |
|---|---|---|---|
| 1 | 81 (no pad) | 26,568 | — |
| 3 | 81 (no pad) | 26,568 | 0 |
| 2 | 82 | 27,224 | +656 |

The `heads=3`-at-`W=81` row is exactly why `board_attention_heads` must join
`ShapeKey` **as its own field**, not derived from the padded shape: it produces
a state_dict byte-identical in every tensor shape to `heads=1` (every
`nn.MultiheadAttention` parameter shape depends only on `embed_dim`, never
`num_heads`), yet computes a different function (a 3-way q/k/v split vs. a
single head) — shape equality alone would wrongly let the resume gate treat
them as compatible.

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

## ✓DONE REGIME classification and ShapeKey implications

*Correction from original draft: the initial analysis classified this as FRESH
(requiring a `MODEL_VERSION` bump, compat shim, and LFS fixture set). That was
an over-classification. The correct classification is **REGIME** — see rationale
below.*

The **residual in-place** integration (Option B) keeps `trunk_input_dim` unchanged:
the two board continuous stripes (270 dims) are excised from the continuous prefix
and re-folded into the flattened attended tokens (15×(E+9) per board = 270 dims each),
so the total width fed to the trunk's first Linear is identical with attention on or
off. `encode.trunk_input_dim` is not modified.

Because the encoding is byte-identical (same `encode_state` / `encode_choices` output
regardless of the flag), `use_board_attention` is **config-carried** — it lives in
`model_config.json`, defaults `False`, and old artifacts rehydrate to `False` and
run identically. No encoding change → no FRESH classification. Per
`docs/VERSIONING.md` (lines 394–406), config-carried topology flags are REGIME even
when they change architecture shapes, because they travel with the artifact.

`ShapeKey` is defined at `src/wingspan/architecture.py:36`. The class now carries
**19 fields**, three of them from board attention (`use_board_attention: bool`
and `board_attention_positions: bool` from the earlier board-only pass,
`board_attention_heads: int` added here), purely so a mismatched run refuses to
resume another's weights — handled gracefully by the `architecture_key` gate
(mismatch → fresh run, no crash). This is the same mechanism every other
topology knob uses; none of them triggered a version bump.

**`board_attention_positions` follows the identical reasoning.** The position
block is a constant tensor built inside the model
(`model/core.py::_board_position_matrix`) and never written into
`encode_state`'s output — `encode_state` / `encode_choices` are byte-identical
whether the flag is on or off, exactly like `use_board_attention` itself. It is
config-carried, defaults `False`, and joins `ShapeKey` for the same
refuse-instead-of-computing-garbage reason. `board_attention_positions=True`
combined with `use_board_attention=False` is *not* rejected by a
`@model_validator` on `ModelArchitecture` — that combination is simply inert
(`_build_board_attention` returns before ever reading the flag, so no
attention modules or position buffer are built either way) — because
`RunConfig._check_architecture` forces the full architecture to assemble on
*every* construction, including each single-field configurator commit; a hard
reject would surface as a "commit rejected" error the instant a user toggles
`use_board_attention` off while this was still on, before `reset_hidden_fields`
gets a chance to clear it (`training/config.py` calls this the "Workstream E"
lesson — the same class of bug previously hit `clone_iters` +
`bootstrap_opponent`). The combination is instead flagged as a launch blocker
by `config.validate_launchable`, and the configurator's `visible_when` +
`reset_hidden_fields` machinery clears the field back to its default the
moment `use_board_attention` is toggled off, so a user never sees it linger.

**`board_attention_heads` follows the same config-carried reasoning, with one
extra wrinkle.** The head count changes no tensor shape by itself — every
`nn.MultiheadAttention` parameter shape depends only on `embed_dim`, never
`num_heads` — so `ShapeKey` cannot infer incompatibility from shapes alone at a
pad-free width (`board_attention_positions=True` puts `embed_dim = 81 = 3⁴`,
which `heads=1` and `heads=3` both reach with zero padding; see the parameter
table above). `ShapeKey` therefore carries `board_attention_heads` (via
`board_attention_heads_active`, mirroring `board_attention_positions_active`)
as its own field rather than deriving it from any width. Default 1
(config-carried, REGIME, no compat shim) — `_build_board_attention` reproduces
the exact single-head module when unset — and the same not-a-`ModelArchitecture`-
validator, launch-blocker-instead reasoning applies (`config.validate_launchable`
rejects `board_attention_heads != 1` without `use_board_attention`). **One
downgrade caveat:** the rehydration guarantee is forward-only (an old *config*
loads safely under later code); it promises nothing about running a newer
`heads > 1` artifact under an *older checkout* that predates this field
entirely — at a pad-free width that older single-head code would load the
state_dict without error (the shapes coincide) and silently compute the wrong
function, so this is a hazard only for a code downgrade, never a normal
forward-compatible load.

A unified hand/tray variant would add at least one more flag (e.g.
`card_attention_scope`) plus the hand cap `H_max` and the location-stripe width to
the shape signature — still REGIME for the same reason.

---

## ✓DONE Shared board-attention weights

**Collapses the own/opponent pair into one shared module.**
`board_attention_shared: bool = False` (mirrored on
`training.config.MainNetArchitecture`) is the new flag; its resolver property
`board_attention_shared_active = use_board_attention and board_attention_shared`
mirrors `board_attention_positions_active` / `board_attention_heads_active`
exactly — every consumer resolves through it, never the raw field.

**Rationale: every board is public, so it is one function, not two.** POV and
opponent boards are the same kind of object wearing different labels —
nothing about a board's state is hidden from either player, so "how good is
this board" should be a single learned function rather than two
independently-trained copies. The unshared pair (`board_attn_me` /
`board_attn_opp`) starves each copy of half the training signal: across a
batch of games, `board_attn_me` only ever sees POV boards and `board_attn_opp`
only ever sees opponent boards, so neither benefits from what the other has
learned about the same kind of position. Sharing lets one module see every
seat's boards, doubling its effective training signal per step, since board
evaluation itself does not need to differ by seat. The sign a board should
carry (favorable because it's mine, unfavorable because it's my opponent's)
is not the attention module's job either way: it is learned downstream by the
trunk, which reads the flattened per-board blocks at fixed concat positions
(`own_flat`, then `opp_flats` in seat order) and can and does weight those
positions differently in its first `Linear`. **No POV marker token is added**
to the shared module's input — the concat position already carries that
signal exactly where the trunk needs it, so duplicating it into the token
would be redundant.

**Naming: a distinct third module, not an alias — because of a specific
`state_dict` hazard.** The shared path registers a new attribute, `board_attn`,
rather than aliasing `self.board_attn_opp = self.board_attn_me`. Aliasing
looks harmless but isn't: `nn.Module.state_dict()` does not dedupe two
attribute names bound to the same submodule instance — it walks the module
tree by name, so it would emit both `board_attn_me.*` and `board_attn_opp.*`
keys, pointing at identical tensors. A checkpoint saved that way would
`load_state_dict(strict=True)` cleanly into an *unshared* net, since the key
sets match exactly — silently leaving the unshared net's two modules
initialized to the same weights, with no error raised at all, and no way to
tell after the fact that they started as one tied module rather than two
independently-initialized ones. (`parameters()` *does* dedupe shared
submodules, so even a param-count regression test would pass over this bug.)
Registering the shared path under its own name instead makes the two modes'
key sets **disjoint** — `{board_attn.*}` vs. `{board_attn_me.*,
board_attn_opp.*}` — so a cross-mode `load_state_dict(strict=True)` fails
with both missing keys *and* unexpected keys, the loudest error torch can
give.

**Parameter count halves.** Using the same `attn_params = 4·E² + 4·E` per
module as the unshared case (`E` = the padded `embed_dim` from
`architecture.board_attention_embed_dim`), `architecture.count_parameters`
now scales the BOARD ATTN block by `1 if arch.board_attention_shared_active
else 2` in place of the previous unconditional `2` — one module's worth of
parameters instead of two, at every `board_attention_heads` /
`board_attention_positions` combination.

**Module count stays fixed at any `num_players`.** The N-player
simplification carries over unchanged: shared or not, the number of
`nn.MultiheadAttention` modules never grows with the number of opponents —
one module when shared, the historical two when not, regardless of whether
there are 1, 2, or 3 opponents at the table. Only the input-facing
`state_trunk.0` / `choice_encoder.0` `Linear` shapes move with `num_players`,
exactly as in the unshared case.

**Classification: REGIME, the same precedent as its siblings.**
`board_attention_shared` is config-carried (travels in the frozen
`RunConfig`), defaults `False` so every existing artifact rehydrates the
historical unshared pair unchanged, and joins `ShapeKey` via
`board_attention_shared_active` — exactly the `use_board_attention` /
`board_attention_positions` / `board_attention_heads` pattern above. No
`MODEL_VERSION` bump, no compat shim, no LFS fixture set. Verified directly:
`compat.v1_3.PolicyValueNetV1_3` subclasses `core.PolicyValueNet`, and
`compat.v1_0.PolicyValueNetV1_0` subclasses `v1_3.PolicyValueNetV1_3` in turn
— neither overrides `_build_board_attention` nor
`_embed_state_board_attention`, so at the default `False` both stay
byte-identical to today's shims. `configurator_defaults.json` seeds `true` so
all *new* runs start shared; existing artifacts are untouched and keep
loading unshared.

*A note on terminology: the configurator marks this field
`impact=ChangeImpact.FRESH`, since toggling it on a resumed run forces a
fresh run (it changes `architecture_key`). This does not contradict the
REGIME classification above — the two words answer different questions. The
configurator's `ChangeImpact.FRESH` means an in-progress run's weights cannot
survive the edit, the same reason toggling `card_embed_dim` carries that
impact. This file's REGIME means no `MODEL_VERSION` bump and no compat shim
— the artifact-format question, not the resume question. A field can need a
fresh run to take effect while needing no versioning machinery to load, and
`board_attention_shared` is exactly that case, like every other `ShapeKey`
member before it.*

**Rehydration is forward-only, and this is the good failure mode.** The
guarantee only ever promised that an *old* config loads safely under *later*
code — it says nothing about a shared artifact opened by a checkout that
predates this field entirely. That checkout has no `board_attn` attribute to
load into, so `load_state_dict` fails immediately on the unrecognized
`board_attn.*` keys: loud, immediate, and impossible to mistake for a healthy
load. Contrast the `board_attention_heads` downgrade hazard above, where a
pad-free width lets an old single-head checkout load a multi-head artifact's
`state_dict` *without* error and silently compute the wrong function. Sharing
fails the safe way — the naming choice above (a disjoint key set) is exactly
what turns what could have been a silent mismatch into a load-time crash
instead.

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

### ✓DONE Verdict / Experiment

**Board-only attention is implemented, with optional position-aware and
multi-head variants.** Toggle `use_board_attention` in the configurator under
MODEL ARCHITECTURE ▸ STATE TRUNK; once on, `board_attention_positions` and
`board_attention_heads` both become visible in the same group. No version bump
is needed for any of the three flags (REGIME classification; see above).

**Experiment protocol:**
1. Train two runs from the same random seed: `use_board_attention=False` (baseline)
   and `use_board_attention=True` (experiment). Hold all other hyperparameters fixed.
2. Compare: win rate vs. random after 500 K games, win rate in self-play eval, and
   sample efficiency (games to 55% win rate).
3. If the attended model reaches the baseline win rate with fewer games, or surpasses
   it, the inductive bias is paying off — and the unified hand/tray variant becomes
   worth its larger cost.
4. **Position follow-up (only once step 3 shows signal):** train a third run with
   `board_attention_positions=True` against the same seed and compare to the
   plain-attention run. Since the position block only changes what the *attention
   mixing* can condition on (the flatten step already preserves slot identity
   downstream either way), a null result here would suggest the trunk's first
   Linear layer was already recovering whatever positional information mattered
   from the fixed flatten order — useful negative evidence either way.

The location-tagged hand/tray unification remains a **second** experiment, attempted
only if board attention shows signal. It relaxes the encoder's POV separation and is
a larger change; it should not be the first thing tried.
