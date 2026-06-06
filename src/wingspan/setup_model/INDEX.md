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
- `SetupCandidate(kept_birds, kept_food)` — one keep option (a subset of the
  dealt hand + food choices).
- `generate_candidates(hand, food_options) -> list[SetupCandidate]` — enumerates
  all valid keep combinations up to the starting-hand limit.
- `select_best(net: SetupNet, candidates, gs) -> SetupCandidate` — runs the
  setup net on each candidate's feature vector and returns the argmax.

**`encode.py`** — `encode_candidate(candidate: SetupCandidate, gs: GameState)
-> np.ndarray`: per-candidate feature encoder. Features include: kept bird
one-hots, habitat coverage, food-cost histogram, egg-limit sum, nest-type
mix, and kept-food vector. Output width matches `stripes.setup_input_dim()`.

**`stripes.py`** — `setup_stripe_layout() -> VectorLayout` and
`setup_input_dim() -> int`. Programmatic stripe registry for the setup input
vector; analogous to `encode.stripes` for the main encoder.

**`generate.py`** — `generate_random_setup_samples(n, seed) -> list[SetupRecord]`:
random-setup generation for the pre-model offline training phase (Phase 0).
Plays games with random keeps and records the final score as a regression target.

**`record.py`** — `SetupRecord(candidate, final_score)` — the setup training
sample. `SetupStore(path)` — an append-only on-disk store (JSONL) of
`SetupRecord`s; supports `append(record)`, `load_all() -> list[SetupRecord]`,
and `size() -> int`.
