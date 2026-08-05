# encode — State/choice tensor encoders

Converts `GameState` + `Decision` objects into fixed-width float tensors for the
policy-value network. The encoder is the primary checkpoint-format surface: every
stripe offset, normalization scale, and feature dim is part of the artifact format.

## Modules

**`__init__.py`**

**`layout.py`** — The single source of truth for all feature dimensions and
stripe offsets. Key exports:
- `EncodingSpec(include_setup: bool, num_players: int = 2)` — frozen
  config-driven shape descriptor. `include_setup` controls whether
  `SetupDecision` rows are included in the main model's choice head or
  delegated to the separate setup model. `num_players` (ge=2, le=5, default
  **2**) selects the seat count the encoding is built for — the checkpoint-compat
  anchor; N=2 output is byte-identical to the pre-N-player encoding (dims,
  offsets, stripe names, values). Carries an explicit `__hash__` (pydantic's
  `frozen=True` already generates one at runtime, but strict pyright doesn't
  see it, so every `@functools.lru_cache`-keyed layout function below would
  otherwise be flagged as non-`Hashable`).
- `spec_for(use_setup_model: bool, num_players: int = 2) -> EncodingSpec` —
  derive spec from run config.
- `state_feature_dim(spec) -> int`, `choice_feature_dim(spec) -> int`,
  `decision_type_dim(spec) -> int`, `num_families(spec) -> int` — spec-dependent
  totals consumed by `model.core.PolicyValueNet` at construction time.
- `state_cont_layout(spec)`, `choice_base_layout(spec)`, `choice_full_layout(spec)`
  — `functools.lru_cache`d, spec-parameterized stripe-layout builders (replace
  the pre-N-player module-literal stripe-spec lists). `STATE_CONT_LAYOUT` /
  `CHOICE_BASE_LAYOUT` / `CHOICE_FULL_LAYOUT` remain module constants — each is
  simply that function's `DEFAULT_SPEC` (N=2) result, so every pre-existing
  reader is untouched.
- N-player accessors — spec-aware counterparts of the N=2 module constants
  below, each equal to its constant when called on `DEFAULT_SPEC`:
  `n_board_index_slots(spec)`,
  `n_card_index_slots(spec)`, `off_card_index(spec)`, `off_hand_multihot(spec)`,
  `off_decision_type(spec)`, `off_board(spec, seat_offset)` (0 = POV, 1..N−1 =
  each opponent clockwise — generalizes `OFF_BOARD_ME`/`OFF_BOARD_OPP`),
  `off_player_select(spec) -> int | None` (`None` at N=2), `round_goal_slot_dim(spec)`,
  `round_goal_vp_offset(spec)`, `round_goals_stripe_dim(spec)`. Live code (model,
  encoders, runmeta reporting, the configurator) uses these; every compat-era
  path keeps passing the bare N=2 constants (the accessors' defaults), since
  compat artifacts are always 2-player (`compat.encoding_dims_for_era` refuses
  otherwise).
- `N_ROUNDS: int = 4` — one-hot dimension for round number (v0.3+).
- `MAX_ACTION_CUBES: int = 8` — one-hot dimension minus 1 for cube counts (v0.3+).
- `N_HAND_PLAYABLE_MULTIHOTS: int = 2` — number of playability-filtered hand
  multi-hots added in v0.6 (`hand_playable_me`, `hand_playable_eggs_me`).
- `CHOICE_BECOMES_PLAYABLE_OFFSET`, `CHOICE_BECOMES_PLAYABLE_DIM` — offset and
  width of the v0.6 `becomes_playable` stripe in each choice row.
- `CHOICE_BECOMES_UNPLAYABLE_OFFSET`, `CHOICE_BECOMES_UNPLAYABLE_DIM` — offset and
  width of the v1.1 `becomes_unplayable` stripe (immediately after `becomes_playable`;
  180 dims, same space). v1.0 artifacts lack this stripe; see `wingspan.compat.v1_0`.
- `STATE_HAND_FOOD_UNLOCK_OFFSET`, `STATE_TRAY_FOOD_UNLOCK_OFFSET`,
  `STATE_FOOD_UNLOCK_DIM` — offsets and width (5 each) of the two v1.4
  food-distance-to-playable **state** stripes (`hand_food_unlock_me`,
  `tray_food_unlock_me`), contiguous in the continuous prefix right after
  `food_opp`. Pre-1.4 artifacts lack them; see `wingspan.compat.v1_3`.
- `CHOICE_RESETS_FEEDER_OFFSET`, `CHOICE_RESETS_FEEDER_DIM` — offset and width (1) of
  the v1.4 `resets_feeder` stripe (after `becomes_unplayable`; set on a
  `combine_gain_food` `FoodSubsetChoice` that rerolls the birdfeeder). v1.0–1.3
  artifacts lack it; see `wingspan.compat.v1_3`.
- `CHOICE_GOAL_DELTA_IGNORING_EGGS_OFFSET`, `CHOICE_GOAL_DELTA_IGNORING_EGGS_DIM` —
  offset and width (8: 4 round goals × count/vp) of the v1.6 `goal_delta_ignoring_eggs`
  stripe — the last *base* stripe (after `resets_feeder`), pricing each round goal
  under the hypothesis that the row's bird is eventually played and egg-populated
  optimally. Pre-1.6 artifacts lack it; see `wingspan.compat.v1_5`.
- `SLOTS_PER_BOARD` (15), `SLOT_SCALAR_DIM` (9), `BOARD_CONT_STRIPE_DIM` (135),
  `OFF_BOARD_ME`/`OFF_BOARD_OPP` — live-encoding-only aliases consumed by the
  board self-attention path (`model.core`); a future FRESH shift must freeze
  them into `StateEmbedOffsets` rather than growing new live-offset readers.
- `BOARD_POSITION_HAB_DIM` (3), `BOARD_POSITION_COL_DIM` (5),
  `BOARD_POSITION_DIM` (8) — width of the optional constant per-token position
  block (`ModelArchitecture.board_attention_positions`); `trunk_input_dim(...,
  board_position_dim=...)` accepts the resulting extra width (`0` when the flag
  is off, the default).
- `_OFF_*` constants — the append-only offset chain (part of checkpoint format;
  reordering is a FRESH break).
- Normalization scales: `_POINTS_SCALE`, `_FOOD_COST_SCALE`, `_WINGSPAN_SCALE`, etc.
- `trunk_input_dim(..., n_card_index_slots=N_CARD_INDEX_SLOTS, n_board_index_slots=N_BOARD_INDEX_SLOTS)`
  — the two new kwargs default to the N=2 module constants (every existing
  caller, including every compat-era path, is unchanged); an N-player-aware
  caller passes `n_card_index_slots(spec)` / `n_board_index_slots(spec)`.

**`state_encode.py`** — `encode_state(gs: GameState, decision, spec) -> np.ndarray` and
`state_size(spec) -> int`. Encodes the full perceived game state into a 1-D
float vector (1129 dims at N=2 as of v1.4; 1299 at N=3, 1467 at N=4 — see
`docs/VERSIONING.md`'s `num_players` entry; was 1119 in v0.9–v1.3, 1155 in
v0.6–v0.8): per-habitat board slots, tray, per-type cached food, the two v1.4
food-unlock stripes (`hand_food_unlock_me`, `tray_food_unlock_me` — see
`engine.playability.min_food_to_unlock`), birdfeeder, round goals (scored rounds
zeroed), player hand + two playability multi-hots (`hand_playable_me`,
`hand_playable_eggs_me`) via the hand encoder, one-hot round number, one-hot
action cube counts, decision-type one-hot. The `hand_summary_me` stripe (10 dims) was removed in v0.9 — derived in-model
via `set_summary_from_multihot`; `board_summary_me/opp` compacted from 18→6 dims (only
`row_length` + `total_eggs` per habitat); `misc_scalars` compacted from 4→2 dims (dropped
round-goal VP scalars). Also exports per-aspect summary helpers (with `include_goal_pts`,
`full_stats`, `zero_passed_rounds` flags) used by the v0.8 compat shim and the dashboard.
At `spec.num_players >= 3`, `encode_state` inserts a `turn_position` one-hot
after `turn_state` and loops `GameState.opponents_clockwise` (Stage 1) to
repeat every per-opponent stripe group once per opponent — at N=2 the single
iteration reproduces the pre-N-player output byte-for-byte.

**`choice_encode.py`** — `encode_choices(gs, decision, spec, *, has_becomes_playable=True, food_playable_ignores_eggs=True) -> np.ndarray`
(shape `[n_choices, choice_dim]`). One row per offered choice; each row is the
concatenation of the decision-type one-hot, the choice featurizer output, and
the per-stripe filler outputs. The `becomes_playable` 180-dim stripe (v0.6) is
filled on gain-bearing rows; the `becomes_unplayable` 180-dim stripe (v1.1,
immediately after `becomes_playable`) is filled on spend-bearing rows. Both are
omitted when `has_becomes_playable=False` (pre-0.6 compat shims) — they are
always added and removed together. `food_playable_ignores_eggs=True` (default,
v0.8+) uses the eggs-agnostic food baseline and `ignore_eggs=True` in
`_bird_playable`; set to `False` for the v0.7 compat shim to restore
eggs-included semantics. `_featurize_food_subset` (registered for
`decisions.FoodSubsetChoice`, the `combine_gain_food` regime) fills the 7-slot
`gain_food` stripe as a count vector via `_fill_gain_food_vector` (raw counts, so
a non-resetting single-unit subset is byte-identical to the `FoodChoice` one-hot),
the combined `becomes_playable` via `playability.newly_playable_after_foods` on the
realized pool (`_combined_gain_pool`), and — when `choice.resets_birdfeeder` — the
1-dim `resets_feeder` stripe added in v1.4 (a FRESH bump; see `wingspan.compat.v1_3`).
At `spec.num_players >= 3`, `encode_choices` fills the `player_select` stripe at
one site (not per-featurizer) for every `decisions.PlayerIdChoice` row: a
`spec.num_players`-wide one-hot at `(choice.player_id − decision.player_id) %
spec.num_players`. Absent at N=2 (`layout.off_player_select(spec)` is `None`);
the existing `special.is_self` bit in `_featurize_player_id` is untouched.
The `goal_delta` stripe on `PlayBirdChoice` rows is conditioned on the row's
landing habitat (v1.5: `_fill_goal_delta`'s `play_habitat`, threaded to
`scoring.goal_count_delta_for_bird`); habitat-less candidate rows keep the
optimistic any-card-habitat bound. `refill_goal_delta_habitat_agnostic(feat,
player_id, bird, gs)` is the public compat seam `wingspan.compat.v1_4` uses to
re-fill a play-bird row with the pre-1.5 agnostic pricing.
`_fill_goal_delta_ignoring_eggs` (v1.6) fills the sibling `goal_delta_ignoring_eggs`
stripe at the same three bird-card row sites as `_fill_goal_delta`
(`_featurize_bird`, `_featurize_play_bird`, `_featurize_draw_source`): per round
goal, the delta under the hypothesis that the row's bird is eventually played
(slot-gated) and egg-populated to whatever level best advances the goal
(`scoring.goal_vp_delta_for_bird_with_eggs`) — nonzero on the egg-driven
categories that `goal_delta` always reads 0 for. `_write_goal_delta` takes an
explicit `base_offset` keyword so both stripes share one writer.

## Subpackage

**`stripes/`** — Programmatic stripe registry: descriptor models and builder
functions for all stripe layouts.
See [`stripes/INDEX.md`](stripes/INDEX.md).
