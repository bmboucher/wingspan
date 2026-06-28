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
`encoding_dims_for_era(artifact_version, spec) -> (state_dim, choice_dim)`.
Routes v1.0 artifacts to the v1.0 frozen dims (narrower `choice_dim`: without
the `becomes_unplayable` stripe); later same-MAJOR artifacts get the live widths.

**`v1_0.py`** — v1.0 artifact compat shim:
- `PolicyValueNetV1_0` — `PolicyValueNet` subclass that restores the v1.0
  trunk-final-activation fallback (`trunk_final_activation=null` resolved to
  `between_activation` instead of `final_activation`) and strips the
  `becomes_unplayable` 180-dim choice stripe added in v1.1. Overrides
  `_build_trunk`, `encode_choices`, and `_choice_embed_offsets`.
- `uses_v1_0` — version predicate (`artifact.major == 1 and artifact.minor == 0`).
  Used by `model.PolicyValueNet.class_for_version` to route v1.0 artifacts here.
