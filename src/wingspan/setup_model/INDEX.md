# setup_model — Setup model (actor-critic bandit)

Separately-trained model that scores initial bird-keep candidates at the start of
each game. Trained actor-critic alongside the main policy net: the value head
predicts the expected score margin for each setup; the policy head outputs
per-candidate logits used for REINFORCE selection. Has its own architecture
descriptor, encoder, and training sample type; the main policy net's checkpoint
format is not coupled to it.

## Modules

**`__init__.py`**

**`architecture.py`** — `SetupArchitecture(hidden_layers, activation, dropout,
layernorm, use_policy_head=True)` — frozen topology descriptor for the setup MLP,
analogous to `ModelArchitecture` for the policy net. `shape_key(arch) -> tuple` —
the checkpoint-invalidating subset of fields. `SetupEncoding` — the config-carried
encoding descriptor; `include_playable_kept_cards: bool = True` (default since
v1.1; enables a food-agnostic 180-dim playable-kept-cards multi-hot); total
default dim = 488. `include_turn1_playable: bool = False` appends a turn-1-only
playability multi-hot when enabled. `setup_readout_input_dim(feature_dim,
main_arch, ...)` computes the MLP's first-Linear input width after embedding: one
`pooled_hand_width`-wide set vector per included card-set stripe + `TRAY_SIZE ×
card_embed_dim` for per-slot tray rows only (no tray-set embedding) + passthrough
scalars. Default with `card_embed_dim=64`: 575 (= 125 + 2×129 + 192).

**`candidates.py`** — The keep-set options the setup model scores:
- `SetupCandidate(kept_cards, kept_foods, bonus_card)` — one keep option (a
  subset of the dealt hand, a food tuple, and an optional bonus card).
- `enumerate_setup_candidates(dealt_cards, dealt_bonus, *, include_bonus=True,
  include_food=True) -> list[SetupCandidate]` — enumerates all valid keep
  combinations. `include_bonus=False` drops the bonus axis (every candidate
  carries `bonus_card=None`). `include_food=False` collapses the food axis to
  a single deferred sentinel (`kept_foods=()`), producing 64 candidates for a
  5-card / 2-bonus deal instead of 504.

**`encode.py`** — `encode_setup_candidate(candidate: SetupCandidate, gs: GameState, encoding: SetupEncoding)
-> np.ndarray`: per-candidate feature encoder. Features include: kept bird
one-hots, habitat coverage, food-cost histogram, egg-limit sum, nest-type
mix, kept-food vector, and (when `encoding.include_turn1_playable`) a 180-dim
multi-hot of birds payable from `kept_foods` on turn 1. Output width matches
`encoding.total_dim`.

**`stripes.py`** — `setup_stripe_layout() -> VectorLayout` and
`setup_input_dim() -> int`. Programmatic stripe registry for the setup input
vector; analogous to `encode.stripes` for the main encoder.

**`generate.py`** — `RandomSetupGenerator(hand_combos, food_sets,
tuples_per_batch=16, *, split_food=False)` — generates random-setup candidates
for a game deal. `split_food=True` skips biased food sampling entirely and emits
`kept_foods=()` on every candidate (the engine resolves food via deferred in-game
GAIN_FOOD / SPEND_FOOD decisions). `generate_one` returns one `SetupBatch` for a
single game deal. `tuples_per_batch` defaults to 16 and is unused at runtime (was
used by the removed batch-deal random-phase path).

**`record.py`** — `SetupSample(features, margin, iteration, chosen_idx,
all_candidates)` — one actor-critic training sample. `features` is the encoded
feature vector for the chosen setup; `margin` is the seat's end-of-game score
margin; `chosen_idx` and `all_candidates` carry the data for REINFORCE gradient
computation over all candidates.
