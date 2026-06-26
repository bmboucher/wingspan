# compat — Version-specific artifact shims

**Currently empty.** The pre-1.0 shims (`v0_0` … `v0_7`) were dropped wholesale
at the 1.0 MAJOR version bump, along with their fixture sets. No 0.x artifact
loads under 1.0 code — `version.check_artifact_compatible` refuses any
different-MAJOR artifact. See `docs/VERSIONING.md` for the full compat policy
(FRESH vs REGIME, when a MINOR bump is required, fixture-set rules, the MAJOR
escape hatch).

The package is kept as the documented home for the seam. The next MINOR FRESH
change (the first `1.<N>` encoding reshape) re-introduces one module per
superseded era here, exactly as the v0.x shims worked:

- a `v1_<N>.py` module with a `uses_v1_<N>_*` version predicate and frozen
  `encode_*` / `*_embed_offsets` helpers,
- a `PolicyValueNet`/`SetupNet` subclass that overrides the reshaped geometry,
- a branch added to `model.PolicyValueNet.class_for_version` and (if widths
  change) to `encoding_dims_for_era` below.

**Freeze _all_ geometry the net derives, not just `encode_state`.** A shim's job
is that the rehydrated net computes identically to the saved one. The net also
derives *slice offsets* from the live layout (`_embed_state` / `_embed_choices`);
those must move with the era too (the 2026-06-10 / 2026-06-14 `_embed_state`
bugs). Every state-embed offset `_embed_state` reads is consolidated into
`model.StateEmbedOffsets`, which a shim overrides as one unit.

**Shims also back era-pinned training.** A resumed run carries
`RunConfig.encoding_version` and keeps producing artifacts at its own era: the
pipeline builds the era's net via `model.PolicyValueNet.class_for_version` and
derives its dims via `encoding_dims_for_era`. Superseded eras are *producing*
paths, not read-only museums — a new training-side feature must work at every
same-MAJOR era or refuse one explicitly.

## Modules

**`__init__.py`** — the package-level dims router:
`encoding_dims_for_era(artifact_version, spec) -> (state_dim, choice_dim)`. With
no shims present it validates the version and returns the live widths; a future
MINOR FRESH change adds its era branch here.
