# compat — Version-specific artifact shims

The pre-1.0 shims (`v0_0` … `v0_7`) were dropped wholesale at the 1.0 MAJOR version
bump, along with their fixture sets. No 0.x artifact loads under 1.x code —
`version.check_artifact_compatible` refuses any different-MAJOR artifact. Two
same-MAJOR shims exist now: **`v1_0`** (v1.0 artifacts — the v1.1 `becomes_unplayable`
stripe + trunk-final-activation change) and **`v1_3`** (pre-1.4 geometry — the two
v1.4 food-unlock **state** stripes and the v1.4 `resets_feeder` **choice** stripe,
which shipped in one era). See `docs/VERSIONING.md` for the full compat policy
(FRESH vs REGIME, when a MINOR bump is required, fixture-set rules, the MAJOR escape
hatch).

Each MINOR FRESH encoding reshape adds one module per superseded era:

- a `v1_<N>.py` module with frozen `encode_*` / `*_embed_offsets` overrides,
- a `PolicyValueNet`/`SetupNet` subclass that regenerates the era's geometry,
- a branch added to `model.PolicyValueNet.class_for_version` and (if widths change)
  to `encoding_dims_for_era` below.

When a later reshape supersedes an era that already lacked an earlier stripe, the
older shim *inherits* the newer one so the strips compose (e.g. `v1_0` inherits
`v1_3`: v1.0 vectors lack the `becomes_unplayable`, `resets_feeder`, and both
food-unlock stripes).

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
Narrows the dims by every stripe added after the artifact's era. For every pre-1.4
same-MAJOR era: `state_dim -= 10` (the two food-unlock stripes — the **first**
state-dim branch) and `choice_dim -= 1` (the `resets_feeder` stripe). v1.0
additionally drops the 180-dim `becomes_unplayable` stripe from `choice_dim`. Later
same-MAJOR artifacts get the live widths.

**`v1_3.py`** — pre-1.4 geometry compat shim:
- `PolicyValueNetV1_3` — `PolicyValueNet` subclass that reverses **both** v1.4
  additions. On the **state** side it strips the two 5-wide food-unlock stripes
  (`hand_food_unlock_me`, `tray_food_unlock_me`) from `encode_state` and freezes the
  pre-1.4 `StateEmbedOffsets` (overrides `encode_state`, `_state_embed_offsets`,
  `_build_trunk`, `_true_state_dim`). On the **choice** side it strips the
  `resets_feeder` 1-dim stripe (the last base stripe, after `becomes_unplayable`)
  from `encode_choices` and shifts only `kept_multihot` (`bird_id` /
  `becomes_playable` / `becomes_unplayable` precede it and are unchanged); overrides
  `encode_choices`, `_choice_embed_offsets`, `_build_choice_encoder`,
  `_true_choice_dim`. Both `_build_*` derive their block width from `self.spec` via
  the `_true_*_dim` helpers (not the passed dims), so the shim is correct under both
  era-dim loads and live-dim test construction. Routes for eras 1.1-1.3 via
  `class_for_version`.

**`v1_0.py`** — v1.0 artifact compat shim:
- `PolicyValueNetV1_0` — subclass of `PolicyValueNetV1_3` (so it inherits the two
  strips above — v1.0 predates the state stripes and `resets_feeder` too) that
  additionally restores the v1.0 trunk-final-activation fallback
  (`trunk_final_activation=null` resolved to `between_activation` instead of
  `final_activation`) and strips the `becomes_unplayable` 180-dim choice stripe added
  in v1.1. Overrides `_build_trunk` (v1.0 activation fallback at the inherited narrow
  state width), `_true_choice_dim` (narrows the inherited v1.3 width by a further
  `becomes_unplayable`, read polymorphically by the inherited `_build_choice_encoder`),
  `encode_choices`, and `_choice_embed_offsets` — the last two chain `super()` (the
  v1_3 strip) then remove `becomes_unplayable`. Routes for era 1.0 via
  `class_for_version`.
