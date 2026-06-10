# compat — Version-specific artifact shims

One module per superseded artifact era. Deleted wholesale at a MAJOR version
bump. See `docs/VERSIONING.md` for the full compat policy (FRESH vs REGIME,
when a MINOR bump is required, fixture-set rules).

The inference loaders (`players.loaders`, `training.runmeta.read_model_config`)
route by the artifact's embedded `version` field and call the appropriate shim
before handing off to the live model. Never add shims directly to loaders —
each era belongs in its own module here.

**Freeze _all_ geometry the net derives, not just `encode_state`.** A shim's job
is that the rehydrated net computes identically to the saved one. The
frozen-vector helpers (`encode_*_v0N`) are only half of it: the net also derives
*slice offsets* from the live layout (`_embed_state` / `_embed_choices`), and
those must move with the era too. When live and frozen widths coincide a stale
offset corrupts silently instead of crashing (the 2026-06-10 `_embed_state` bug).
Every code-carried value the loaded artifact's behavior depends on belongs in the
shim — see `docs/VERSIONING.md` (FRESH vs REGIME, config- vs code-carried).

**Shims also back era-pinned training.** A resumed run carries
`TrainConfig.encoding_version` and keeps producing artifacts at its own era
(see "Training resume: era pinning" in `docs/VERSIONING.md`): the training
pipeline constructs the era's net class via
`model.PolicyValueNet.class_for_version` and derives its dims via
`encoding_dims_for_era`. Superseded eras are therefore *producing* paths, not
read-only museums — a new training-side feature must work at every same-MAJOR
era or refuse one explicitly.

## Modules

**`__init__.py`** — imports the era modules and owns the package-level dims
router: `encoding_dims_for_era(artifact_version, spec) -> (state_dim,
choice_dim)` — the raw vector widths an era's encoders produce (pre-0.3 state
from `v0_2`, pre-0.1 choice from `v0_0`, live otherwise).

**`v0_0.py`** — Pre-0.1 (v0.0) choice geometry shim. The v0.0 encoding omitted
several choice-vector stripes that were added in v0.1.
- `uses_v0_0_choice_encoding(descriptor) -> bool` — True when the artifact's
  `version` is `"0.0"`.
- `regenerate_v0_0_choices(steps) -> steps` — Rebuilds choice rows from raw
  game state without the new fields (used during compat loading for eval).
- `PolicyValueNetV00` — Frozen-era subclass (extends `PolicyValueNetV01`) that
  carries its own choice geometry: overrides `encode_choices`, the choice-encoder
  input width, and `_embed_choices`' frozen card-region offsets. Inherits V01's
  card encoder and the pre-0.3 state geometry (including `_state_embed_offsets`).

**`v0_1.py`** — Pre-0.2 card-feature shim (`CARD_FEATURE_DIM` 229 → 224).
- `PolicyValueNetV01` — overrides `_build_card_encoder` (frozen 229-wide input +
  v0.1 feature table), `encode_state` (the 771-dim pre-0.3 vector via
  `v0_2.encode_state_v02`), and `_state_embed_offsets` (slices that vector at the
  v0.2 offsets, via `v0_2.state_embed_offsets_v02`).
- `SetupNetV01` — the setup-net twin of the frozen card encoder.

**`v0_2.py`** — Pre-0.3 state-misc shim (state vector 771 → 790 via the round /
cube one-hot reshape). All pre-0.3 nets feed the 771-dim vector.
- `PolicyValueNetV02` — overrides `encode_state` (frozen 7-scalar misc stripe via
  `encode_state_v02`) and `_state_embed_offsets` (the matching slice offsets).
- `encode_state_v02` / `state_embed_offsets_v02` — the frozen 771-dim vector and
  the `(card-index, hand, decision)` offsets it must be sliced at (live offsets
  shifted by `_MISC_DIM_DELTA` = −19). `state_stripe_layout_v02` is the reporting
  twin.
