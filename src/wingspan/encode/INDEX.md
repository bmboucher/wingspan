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
- `_OFF_*` constants — the append-only offset chain (part of checkpoint format;
  reordering is a FRESH break).
- Normalization scales: `_POINTS_SCALE`, `_FOOD_COST_SCALE`, `_WINGSPAN_SCALE`, etc.

**`state_encode.py`** — `encode_state(gs: GameState, spec) -> np.ndarray` and
`state_size(spec) -> int`. Encodes the full perceived game state into a 1-D
float vector (790 dims as of v0.3): per-habitat board slots, tray, per-type
cached food, birdfeeder, all-4 round goals, player hand (via the hand encoder),
one-hot round number (4 dims), one-hot action cube counts (9 dims each × 2
players). Also exports per-aspect summary helpers used by the dashboard inspector.

**`choice_encode.py`** — `encode_choices(gs, decision, spec) -> np.ndarray`
(shape `[n_choices, choice_dim]`). One row per offered choice; each row is the
concatenation of the decision-type one-hot, the choice featurizer output, and
the per-stripe filler outputs. Per-`Choice` featurizer functions are registered
via `@featurizes(ChoiceClass)` and kept close to the stripe definitions in
`stripes/`.

## Subpackage

**`stripes/`** — Programmatic stripe registry: descriptor models and builder
functions for all stripe layouts.
See [`stripes/INDEX.md`](stripes/INDEX.md).
