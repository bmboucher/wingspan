# encode — State/choice tensor encoders

Converts `GameState` + `Decision` objects into fixed-width float tensors for the
policy-value network. The encoder is the primary checkpoint-format surface: every
stripe offset, normalization scale, and feature dim is part of the artifact format.

## Modules

**`__init__.py`**

**`layout.py`** — The single source of truth for all feature dimensions and
stripe offsets. Key exports:
- `EncodingSpec(include_setup: bool)` — frozen config-driven shape descriptor;
  controls whether `SetupDecision` rows are included in the main model's
  choice head or delegated to the separate setup model.
- `spec_for(use_setup_model: bool) -> EncodingSpec` — derive spec from run config.
- `state_feature_dim(spec) -> int`, `choice_feature_dim(spec) -> int`,
  `decision_type_dim(spec) -> int`, `num_families(spec) -> int` — spec-dependent
  totals consumed by `model.core.PolicyValueNet` at construction time.
- `N_ROUNDS: int = 4` — one-hot dimension for round number (v0.3+).
- `MAX_ACTION_CUBES: int = 8` — one-hot dimension minus 1 for cube counts (v0.3+).
- `N_HAND_PLAYABLE_MULTIHOTS: int = 2` — number of playability-filtered hand
  multi-hots added in v0.6 (`hand_playable_me`, `hand_playable_eggs_me`).
- `CHOICE_BECOMES_PLAYABLE_OFFSET`, `CHOICE_BECOMES_PLAYABLE_DIM` — offset and
  width of the v0.6 `becomes_playable` stripe in each choice row.
- `CHOICE_BECOMES_UNPLAYABLE_OFFSET`, `CHOICE_BECOMES_UNPLAYABLE_DIM` — offset and
  width of the v1.1 `becomes_unplayable` stripe (immediately after `becomes_playable`;
  180 dims, same space). v1.0 artifacts lack this stripe; see `wingspan.compat.v1_0`.
- `_OFF_*` constants — the append-only offset chain (part of checkpoint format;
  reordering is a FRESH break).
- Normalization scales: `_POINTS_SCALE`, `_FOOD_COST_SCALE`, `_WINGSPAN_SCALE`, etc.

**`state_encode.py`** — `encode_state(gs: GameState, spec) -> np.ndarray` and
`state_size(spec) -> int`. Encodes the full perceived game state into a 1-D
float vector (1119 dims as of v0.9; was 1155 in v0.6–v0.8): per-habitat board
slots, tray, per-type cached food, birdfeeder, round goals (scored rounds zeroed),
player hand + two playability multi-hots (`hand_playable_me`, `hand_playable_eggs_me`)
via the hand encoder, one-hot round number, one-hot action cube counts, decision-type
one-hot. The `hand_summary_me` stripe (10 dims) was removed in v0.9 — derived in-model
via `set_summary_from_multihot`; `board_summary_me/opp` compacted from 18→6 dims (only
`row_length` + `total_eggs` per habitat); `misc_scalars` compacted from 4→2 dims (dropped
round-goal VP scalars). Also exports per-aspect summary helpers (with `include_goal_pts`,
`full_stats`, `zero_passed_rounds` flags) used by the v0.8 compat shim and the dashboard.

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
eggs-included semantics.

## Subpackage

**`stripes/`** — Programmatic stripe registry: descriptor models and builder
functions for all stripe layouts.
See [`stripes/INDEX.md`](stripes/INDEX.md).
