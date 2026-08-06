# compat — Version-specific artifact shims

The pre-1.0 shims (`v0_0` … `v0_7`) were dropped wholesale at the 1.0 MAJOR version
bump, along with their fixture sets. No 0.x artifact loads under 1.x code —
`version.check_artifact_compatible` refuses any different-MAJOR artifact. Five
same-MAJOR shims exist now: **`v1_0`** (v1.0 artifacts — the v1.1 `becomes_unplayable`
stripe + trunk-final-activation change), **`v1_3`** (pre-1.4 geometry — the two
v1.4 food-unlock **state** stripes and the v1.4 `resets_feeder` **choice** stripe,
which shipped in one era), **`v1_4`** (pre-1.5 behavior — the habitat-agnostic
play-bird `goal_delta` pricing; no shape change of its own), **`v1_5`**
(pre-1.6 geometry + behavior — the v1.6 `goal_delta_ignoring_eggs` main-net
**choice** stripe, *and* the first `SetupNet` shim: the static, egg-blind
pre-1.6 setup `goal_affinity` pricing, no shape change of its own), and
**`v1_6`** (pre-1.7 behavior, both nets — the static egg-blind bonus
*potential* pricing; no shape change of its own). See
`docs/VERSIONING.md` for the full compat policy (FRESH vs REGIME, when a MINOR
bump is required, fixture-set rules, the MAJOR escape hatch).

Each MINOR FRESH encoding reshape adds one module per superseded era:

- a `v1_<N>.py` module with frozen `encode_*` / `*_embed_offsets` overrides,
- a `PolicyValueNet`/`SetupNet` subclass that regenerates the era's geometry,
- a branch added to `model.PolicyValueNet.class_for_version` and (if widths change)
  to `encoding_dims_for_era` below.

When a later reshape supersedes an era that already lacked an earlier stripe, the
older shim *inherits* the newer one so the strips compose (e.g. `v1_0` inherits
`v1_3`: v1.0 vectors lack the `becomes_unplayable`, `resets_feeder`, `goal_delta_ignoring_eggs`,
and both food-unlock stripes; `v1_3` in turn inherits `v1_4`, which in turn
inherits `v1_5`, which in turn inherits `v1_6`, so every pre-1.4 era also
freezes the pre-1.5 `goal_delta` pricing, strips the v1.6 tail stripe, and
freezes the pre-1.7 bonus potentials). A **behavior-only** era (stripe
*values* changed, widths untouched — `v1_4`, `v1_6`) follows the same module
shape but overrides only the encoder that regenerates the old values; there is
no `encoding_dims_for_era` branch and no offset/layout override to add.

**Freeze _all_ geometry the net derives, not just `encode_state`.** A shim's job
is that the rehydrated net computes identically to the saved one. The net also
derives *slice offsets* from the live layout (`_embed_state` / `_embed_choices`);
those must move with the era too (the 2026-06-10 / 2026-06-14 `_embed_state`
bugs). Every state-embed offset `_embed_state` reads is consolidated into
`model.StateEmbedOffsets`, which a shim overrides as one unit.

**The layout descriptors are era geometry too.** Anything that *decodes* a
vector an era net produced (the game-log encoding viewer) must use that era's
stripe layout, so each shim also overrides `raw_state_stripe_layout` /
`raw_choice_stripe_layout` — `super()`'s layout `.without_stripes(...)` the same
names its `encode_*` deletes (the 2026-07-12 game.html phantom-cards bug: v1.3
vectors decoded at live v1.4 offsets). A new FRESH stripe change must extend
these overrides alongside the encoder strips.

**Shims also back era-pinned training.** A resumed run carries
`RunConfig.encoding_version` and keeps producing artifacts at its own era: the
pipeline builds the era's net via `model.PolicyValueNet.class_for_version` and
derives its dims via `encoding_dims_for_era`. Superseded eras are *producing*
paths, not read-only museums — a new training-side feature must work at every
same-MAJOR era or refuse one explicitly.

## Modules

**`__init__.py`** — the package-level dims router:
`encoding_dims_for_era(artifact_version, spec) -> (state_dim, choice_dim)`.
Narrows the dims by every stripe added after the artifact's era. For every
pre-1.6 same-MAJOR era: `choice_dim -= 8` (the `goal_delta_ignoring_eggs`
stripe — the **newest**, and so broadest, choice-narrowing branch). For every
pre-1.4 same-MAJOR era, additionally: `state_dim -= 10` (the two food-unlock
stripes — the **first** state-dim branch) and `choice_dim -= 1` (the
`resets_feeder` stripe). v1.0 additionally drops the 180-dim
`becomes_unplayable` stripe from `choice_dim`. Later same-MAJOR artifacts get
the live widths. Raises `version.IncompatibleArtifactError` outright when
`spec.num_players != 2` — every superseded era predates N-player support by
construction, so no shim ever needs to reproduce an N>=3 shape
(`docs/VERSIONING.md`'s `num_players` entry). Era 1.5 has no state branch and
no further choice branch beyond the v1.6 one — v1.5 itself changed stripe
values only.

**`v1_6.py`** — pre-1.7 behavior compat shim, both nets (the v1.7 optimistic
egg-bonus potential change — `scoring.bonus_potential_count` — is values-only):
- `PolicyValueNetV1_6` — `PolicyValueNet` subclass overriding `encode_choices`
  only: after live encoding, each bonus-carrying row (`BonusCardChoice`, or
  `SetupChoice` with a kept bonus) has its `bonus_value` `hand_potential` /
  `tray_potential` scalars re-filled via
  `choice_encode.refill_bonus_value_potentials_static` — the static-tag
  pricing every pre-1.7 net trained against (the egg-counting cards read 0;
  the hand-counting card's full-hand count is regenerated identically). The
  refill targets `layout._OFF_BONUS_VALUE`, before `becomes_playable` and so
  before every column any older shim strips — the v1_5 re-chain composes with
  no offset math. Routes for era 1.6 via `class_for_version`.
- `SetupNetV1_6` — `SetupNet` subclass overriding `encode_candidate` only:
  live encoding, then `setup_model.encode.refill_bonus_pricing_static`
  rewrites whichever bonus block the encoding carries (split-mode
  `bonus_card_affinity` pair, or folded-mode `kept_bonus_value` 4-vector) at
  unchanged offsets. Routes for era 1.6 via
  `SetupNet.class_for_version`.

**`v1_5.py`** — pre-1.6 geometry (main net) + behavior (setup net) compat shim
(both classes inherit their `v1_6` counterparts, so eras <= 1.5 freeze the
pre-1.7 bonus potentials too):
- `PolicyValueNetV1_5` — `PolicyValueNet` subclass that strips the v1.6
  `goal_delta_ignoring_eggs` 8-dim choice stripe (the last base stripe, after
  `resets_feeder`) from `encode_choices` and shifts only `kept_multihot`
  (`bird_id` / `becomes_playable` / `becomes_unplayable` precede it and are
  unchanged); overrides `encode_choices`, `_choice_embed_offsets`,
  `_build_choice_encoder`, `_true_choice_dim`. `_build_choice_encoder` derives
  its width from `self.spec` via `_true_choice_dim` (not the passed dim), so
  the shim is correct under both era-dim loads and live-dim test construction.
  Also overrides `raw_choice_stripe_layout` (live layout `.without_stripes(...)`
  the same name). No state-side overrides — v1.6's only main-net encoding
  change is this one choice stripe. Routes for era 1.5 via
  `model.core.PolicyValueNet.class_for_version`.
- `SetupNetV1_5` — `wingspan.training.setup_net.SetupNet` subclass, the
  **first `SetupNet` shim**. Overrides only `encode_candidate`: calls the live
  encoder via `super()`, then `setup_model.encode.refill_goal_affinity_static`
  overwrites the 4-scalar `goal_affinity` stripe with the pre-1.6 static
  (egg-blind) pricing, in place — no shape override, since v1.6's setup change
  is values-only. Routes for eras <= 1.5 via
  `wingspan.training.setup_net.SetupNet.class_for_version`.

**`v1_4.py`** — pre-1.5 behavior compat shim (inherits `v1_5.PolicyValueNetV1_5`,
so its geometry is no longer dims-equal-live):
- `PolicyValueNetV1_4` — `PolicyValueNetV1_5` subclass that freezes the pre-1.5
  habitat-agnostic play-bird `goal_delta` pricing: overrides `encode_choices`
  only — after its parent's (geometry-narrowing) encoding, each `PlayBirdChoice`
  row's `goal_delta` stripe is re-filled via
  `choice_encode.refill_goal_delta_habitat_agnostic` (a `birds_<habitat>` goal
  priced by the bird's *card* habitats, so a two-habitat bird claims the goal on
  both rows — what every pre-1.5 net trained against). The refill targets
  `layout._OFF_GOAL_DELTA`, an offset well before the stripped
  `goal_delta_ignoring_eggs` tail, so it is unaffected by the v1_5 re-chain.
  Routes for era 1.4 via `class_for_version`; `PolicyValueNetV1_3` inherits it,
  extending both the value freeze and the geometry narrowing to every earlier
  same-MAJOR era.

**`v1_3.py`** — pre-1.4 geometry compat shim (inherits `v1_4.PolicyValueNetV1_4`):
- `PolicyValueNetV1_3` — `PolicyValueNetV1_4` subclass that reverses **both** v1.4
  additions. On the **state** side it strips the two 5-wide food-unlock stripes
  (`hand_food_unlock_me`, `tray_food_unlock_me`) from `encode_state` and freezes the
  pre-1.4 `StateEmbedOffsets` (overrides `encode_state`, `_state_embed_offsets`,
  `_build_trunk`, `_true_state_dim`). On the **choice** side it strips its own
  `resets_feeder` 1-dim stripe (immediately after `becomes_unplayable`) from
  `encode_choices` and shifts only `kept_multihot` (`bird_id` /
  `becomes_playable` / `becomes_unplayable` precede it and are unchanged);
  `_true_choice_dim` composes via `super()` (the inherited v1_5 narrowing) minus
  `resets_feeder`, rather than recomputing absolutely from `self.spec` — so a
  further tail-stripe narrowing an ancestor era applies automatically. Overrides
  `encode_choices`, `_choice_embed_offsets`, `_build_choice_encoder`,
  `_true_choice_dim`. Both `_build_*` derive their block width from `self.spec` via
  the `_true_*_dim` helpers (not the passed dims), so the shim is correct under both
  era-dim loads and live-dim test construction. Also overrides
  `raw_state_stripe_layout` / `raw_choice_stripe_layout` (live layout
  `.without_stripes(...)` the same names, composed via the same `super()` chain)
  so decode consumers see the era's offsets. Routes for eras 1.1-1.3 via
  `class_for_version`.

**`v1_0.py`** — v1.0 artifact compat shim:
- `PolicyValueNetV1_0` — subclass of `PolicyValueNetV1_3` (so it inherits every
  strip above — v1.0 predates the state stripes, `resets_feeder`, and (via the
  v1_3 -> v1_4 -> v1_5 chain) `goal_delta_ignoring_eggs` too) that additionally
  restores the v1.0 trunk-final-activation fallback
  (`trunk_final_activation=null` resolved to `between_activation` instead of
  `final_activation`) and strips the `becomes_unplayable` 180-dim choice stripe added
  in v1.1. Overrides `_build_trunk` (v1.0 activation fallback at the inherited narrow
  state width), `_true_choice_dim` (narrows the inherited v1.3 width by a further
  `becomes_unplayable`, read polymorphically by the inherited `_build_choice_encoder`),
  `encode_choices`, and `_choice_embed_offsets` — the last two chain `super()` (the
  v1_3 strip) then remove `becomes_unplayable` — and `raw_choice_stripe_layout`
  (the inherited v1.3 layout minus `becomes_unplayable`; the state layout is
  inherited unchanged). Routes for era 1.0 via `class_for_version`.
