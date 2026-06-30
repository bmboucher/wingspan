# setup_model — Setup model (actor-critic bandit)

Separately-trained model that scores initial bird-keep candidates at the start of
each game. Trained actor-critic alongside the main policy net: the **policy head**
reads the fused state ⊕ action candidate and ranks the keeps (REINFORCE); the
**value head** reads a state-only embedding, so it is the critic `V(s)` —
invariant to the chosen keep — rather than the post-keep `Q(s, a)` that made the
old advantage self-cancel. Has its own architecture descriptor, encoder, and
training sample type; the main policy net's checkpoint format is not coupled to it.

## Modules

**`__init__.py`**

**`architecture.py`** — `SetupArchitecture(trunk_layers=(), hidden_layers,
activation, dropout, use_policy_head=True, value_trunk_layers=(),
value_hidden_layers=())` — frozen topology descriptor for the setup net.
`trunk_layers` / `hidden_layers` describe the **policy** path (over the fused
candidate); `value_trunk_layers` / `value_hidden_layers` describe the separate
state-only **value** path (empty `value_hidden_layers` reuses `hidden_layers` via
`value_hidden_resolved`). `shape_key(arch) -> tuple` — the checkpoint-invalidating
subset of fields, covering both paths. `SetupParamReport` — typed accounting by
embedder / policy-trunk (`trunk`) / value-trunk (`value_trunk`) / `value_head` /
`policy_head`, with `value_in` / `policy_in` per-head input widths.
`SetupEncoding` — the config-carried encoding descriptor;
`include_playable_kept_cards: bool = True` (default since v1.1); total default
dim = 488. `include_turn1_playable: bool = False` appends a turn-1-only
playability multi-hot when enabled. `bonus_cards_dim` is the on-offer multi-hot
width (state, split-bonus only). `setup_readout_input_dim(feature_dim, main_arch,
...)` computes the **policy** head's input width (fused candidate); default with
`card_embed_dim=64`: 575. `setup_state_input_dim(encoding, main_arch)` computes
the **value** head's narrower state-only input width (tray rows + feeder + goals +
bonus-on-offer); 304 with split-bonus (= 192 + 6 + 80 + 26), 278 without (no
bonus-state stripe).

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

**`stripes.py`** — `setup_stripe_layout(encoding) -> VectorLayout` and
`setup_readout_stripe_layout(encoding) -> VectorLayout`. Programmatic stripe
registry for the setup input and readout vectors; analogous to `encode.stripes`
for the main encoder.

**`generate.py`** — `RandomSetupGenerator(hand_combos, food_sets,
tuples_per_batch=16, *, split_food=False)` — generates random-setup candidates
for a game deal. `split_food=True` skips biased food sampling entirely and emits
`kept_foods=()` on every candidate (the engine resolves food via deferred in-game
GAIN_FOOD / SPEND_FOOD decisions). `generate_one` returns one `SetupBatch` for a
single game deal. `tuples_per_batch` defaults to 16 and is unused at runtime (was
used by the removed batch-deal random-phase path).

**`record.py`** — `SetupSample(features, margin, iteration, chosen_idx,
all_candidates, own_total, opp_total, won, margin_checkpoints, score_checkpoints,
decision_times, final_timestamp)` — one actor-critic training sample. `features`
is the encoded chosen setup; `chosen_idx` / `all_candidates` carry the REINFORCE
data (V(s) is read from `all_candidates` row 0, whose state stripes are shared).
`margin` is the realized end-of-game margin (dashboard readout); the remaining
fields reproduce the in-game return at the `t=0` setup decision via
`returns.setup_return` (own/opponent totals, seat-relative `won`, and the seat's
per-decision checkpoint/time sequences for the discounted-return modes). The new
fields default safely so older samples still deserialize.
