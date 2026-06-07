# setup_model — Setup model (value-regression bandit)

Separately-trained model that scores initial bird-keep candidates at the start of
each game. Trained offline on random-setup samples before the main policy training
begins; updated online via MSE during self-play. Has its own architecture
descriptor, encoder, and training records so the main policy net's checkpoint
format is not coupled to it.

## Modules

**`__init__.py`**

**`architecture.py`** — `SetupArchitecture(hidden_layers, activation, dropout,
layernorm)` — frozen topology descriptor for the setup MLP, analogous to
`ModelArchitecture` for the policy net. `shape_key(arch) -> tuple` — the
checkpoint-invalidating subset of fields (used by the FRESH-restart gate in the
training loop).

**`candidates.py`** — The keep-set options the setup model scores:
- `SetupCandidate(kept_cards, kept_foods, bonus_card)` — one keep option (a
  subset of the dealt hand, a food tuple, and an optional bonus card).
- `enumerate_setup_candidates(dealt_cards, dealt_bonus, *, include_bonus=True,
  include_food=True) -> list[SetupCandidate]` — enumerates all valid keep
  combinations. `include_bonus=False` drops the bonus axis (every candidate
  carries `bonus_card=None`). `include_food=False` collapses the food axis to
  a single deferred sentinel (`kept_foods=()`), producing 64 candidates for a
  5-card / 2-bonus deal instead of 504.

**`encode.py`** — `encode_candidate(candidate: SetupCandidate, gs: GameState)
-> np.ndarray`: per-candidate feature encoder. Features include: kept bird
one-hots, habitat coverage, food-cost histogram, egg-limit sum, nest-type
mix, and kept-food vector. Output width matches `stripes.setup_input_dim()`.

**`stripes.py`** — `setup_stripe_layout() -> VectorLayout` and
`setup_input_dim() -> int`. Programmatic stripe registry for the setup input
vector; analogous to `encode.stripes` for the main encoder.

**`generate.py`** — `RandomSetupGenerator(hand_combos, food_sets,
tuples_per_batch, *, split_food=False)` — generates batches of random-setup
candidates for a game deal. `split_food=True` skips biased food sampling
entirely and emits `kept_foods=()` on every candidate (the engine resolves
food via deferred in-game GAIN_FOOD / SPEND_FOOD decisions). `generate_one`
returns one `SetupBatch` for a single game deal.

**`record.py`** — `SetupRecord(candidate, final_score)` — the setup training
sample. `SetupStore(path)` — an append-only on-disk store (JSONL) of
`SetupRecord`s; supports `append(record)`, `load_all() -> list[SetupRecord]`,
and `size() -> int`.
