# compat — Version-specific artifact shims

The pre-1.0 shims (`v0_0` … `v0_7`) were dropped wholesale at the 1.0 MAJOR version
bump, along with their fixture sets. No 0.x artifact loads under 1.x code —
`version.check_artifact_compatible` refuses any different-MAJOR artifact. The
same-MAJOR MINOR reshapes since then have re-populated the package: **`v1_0`** (the
v1.1 `becomes_unplayable` stripe + trunk-final-activation change) and **`v1_3`** (the
v1.4 `resets_feeder` stripe). See `docs/VERSIONING.md` for the full compat policy
(FRESH vs REGIME, when a MINOR bump is required, fixture-set rules, the MAJOR escape
hatch).

Each MINOR FRESH encoding reshape adds one module per superseded era:

- a `v1_<N>.py` module with frozen `encode_*` / `*_embed_offsets` overrides,
- a `PolicyValueNet`/`SetupNet` subclass that regenerates the era's geometry,
- a branch added to `model.PolicyValueNet.class_for_version` and (if widths change)
  to `encoding_dims_for_era` below.

When a later reshape supersedes an era that already lacked an earlier stripe, the
older shim *inherits* the newer one so the strips compose (e.g. `v1_0` inherits
`v1_3`: v1.0 vectors lack both `becomes_unplayable` and `resets_feeder`).

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
Narrows `choice_dim` by every stripe added after the artifact's era: minus the
1-dim `resets_feeder` stripe for every era with minor ≤ 3, and additionally minus
the 180-dim `becomes_unplayable` stripe for v1.0. Later same-MAJOR artifacts get the
live widths.

**`v1_3.py`** — v1.3 artifact compat shim:
- `PolicyValueNetV1_3` — `PolicyValueNet` subclass that strips the v1.4
  `resets_feeder` 1-dim choice stripe (the last base stripe, after
  `becomes_unplayable`). Overrides `_build_choice_encoder`, `encode_choices`, and
  `_choice_embed_offsets` (only `kept_multihot` shifts; `bird_id` /
  `becomes_playable` / `becomes_unplayable` precede the new stripe and are
  unchanged). No trunk override — v1.1–1.3 topology equals live. Routed for artifact
  minor 1–3 by `model.PolicyValueNet.class_for_version`.

**`v1_0.py`** — v1.0 artifact compat shim:
- `PolicyValueNetV1_0` — a `PolicyValueNetV1_3` subclass (so the `resets_feeder`
  strip is inherited) that additionally restores the v1.0 trunk-final-activation
  fallback (`trunk_final_activation=null` resolved to `between_activation` instead of
  `final_activation`) and strips the `becomes_unplayable` 180-dim choice stripe added
  in v1.1. Overrides `_build_trunk`, `_build_choice_encoder`, `encode_choices`, and
  `_choice_embed_offsets`, each chaining `super()` (the v1_3 strip) then removing
  `becomes_unplayable`. Routed for artifact minor 0.
