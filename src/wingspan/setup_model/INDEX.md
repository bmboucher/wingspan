# setup_model — Setup model (actor-critic bandit)

Separately-trained model that scores initial bird-keep candidates at the start of
each game. A **two-tower actor-critic** mirroring the in-game `PolicyValueNet`: a
shared **state trunk** encodes the action-independent stripes (its `state_enc`
feeds both heads), a **choice trunk** encodes the action stripes (its `choice_enc`
feeds the policy head). The **value head** reads `state_enc` only, so it is the
critic `V(s)` — invariant to the chosen keep — rather than the post-keep `Q(s, a)`
that made the old advantage self-cancel; the **policy head** reads
`state_enc ⊕ choice_enc` and ranks the keeps (REINFORCE). Has its own
architecture descriptor, encoder, and training sample type; the main policy net's
checkpoint format is not coupled to it.

## Modules

**`__init__.py`**

**`architecture.py`** — `SetupArchitecture(trunk_layers=(128,),
choice_layers=(128,), head_layers=(128,), value_layers=(), between_activation,
final_activation, dropout, use_policy_head=True)` — frozen two-tower topology
descriptor, mirroring `ModelArchitecture`'s field names: `trunk_layers` = the
shared **state trunk** (over the action-independent stripes; feeds both heads),
`choice_layers` = the **choice trunk** (over the action stripes; feeds the policy
head), `head_layers` = the **policy head** (over `state_enc ⊕ choice_enc`),
`value_layers` = the **value head** (over `state_enc`). `trunk_layers` /
`choice_layers` are min-length 1 (mandatory trunks). `shape_key(arch) -> tuple` —
the checkpoint-invalidating 5-tuple `(trunk_layers, choice_layers, head_layers,
value_layers, use_policy_head)`. `SetupParamReport` — typed accounting by embedder
/ `state_trunk` / `choice_trunk` / `value_head` / `policy_head`, with
`state_enc_dim` / `choice_enc_dim` (trunk outputs) and `value_in` (= `state_enc`) /
`policy_in` (= `state_enc + choice_enc`) per-head input widths. `SetupEncoding` —
the config-carried encoding descriptor; `include_playable_kept_cards: bool = True`
(default since v1.1); total default dim = 488. `include_turn1_playable: bool =
False` appends a turn-1-only playability multi-hot when enabled. `bonus_cards_dim`
is the on-offer multi-hot width (state, split-bonus only).
`setup_state_input_dim(encoding, main_arch)` computes the **state** trunk's input
width (tray rows + feeder + goals + bonus-on-offer); 304 with split-bonus
(= 192 + 6 + 80 + 26), 278 without. `setup_choice_input_dim(encoding, main_arch)`
computes the **choice** trunk's input width (kept/playability sets + foods + bonus
action + affinities); 264 with split-bonus, 297 folded. The two partition the
embedded candidate: `setup_choice_input_dim + setup_state_input_dim` equals the
fused readout width (568 / 575).

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

**`stripes.py`** — `setup_stripe_layout(encoding) -> VectorLayout` (raw), plus
`setup_state_stripe_layout(encoding, main_arch)` and
`setup_choice_stripe_layout(encoding, main_arch)` — the post-embedding state and
choice vectors the two trunks receive, each in the net's concatenation order and
summing to `setup_state_input_dim` / `setup_choice_input_dim`. Analogous to
`encode.stripes` for the main encoder.

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
