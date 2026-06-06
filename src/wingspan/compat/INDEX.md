# compat — Version-specific artifact shims

One module per superseded artifact era. Deleted wholesale at a MAJOR version
bump. See `docs/VERSIONING.md` for the full compat policy (FRESH vs REGIME,
when a MINOR bump is required, fixture-set rules).

The inference loaders (`players.loaders`, `training.runmeta.read_model_config`)
route by the artifact's embedded `version` field and call the appropriate shim
before handing off to the live model. Never add shims directly to loaders —
each era belongs in its own module here.

## Modules

**`__init__.py`** — re-exports the shim-detection helpers used by loaders:
`uses_v0_0_choice_encoding(descriptor) -> bool`.

**`v0_0.py`** — Pre-0.1 (v0.0) choice geometry shim. The v0.0 encoding omitted
several choice-vector stripes that were added in v0.1.
- `uses_v0_0_choice_encoding(descriptor) -> bool` — True when the artifact's
  `version` is `"0.0"`.
- `regenerate_v0_0_choices(steps) -> steps` — Rebuilds choice rows from raw
  game state without the new fields (used during compat loading for eval).
- `PolicyValueNetV00` — Frozen-era subclass of `model.core.PolicyValueNet` that
  carries its own geometry; used for inference against v0.0 checkpoints so the
  live encoder is never paired with a stale spec by hand.
