# compat — Version-specific artifact shims

The pre-1.0 shims (`v0_0` … `v0_7`) were dropped wholesale at the 1.0 MAJOR version
bump, along with their fixture sets. No 0.x artifact loads under 1.x code —
`version.check_artifact_compatible` refuses any different-MAJOR artifact. Three
same-MAJOR shims exist now: **`v1_0`** (v1.0 artifacts — the v1.1 `becomes_unplayable`
stripe + trunk-final-activation change), **`v1_3`** (pre-1.4 geometry — the two
v1.4 food-unlock **state** stripes and the v1.4 `resets_feeder` **choice** stripe,
which shipped in one era), and **`v1_4`** (pre-1.5 geometry + behavior, both nets —
the merge of what were four provisionally-numbered eras, 1.5-1.8, that landed on
main in sequence but never trained a run past 1.4, so they were collapsed into one
era before any of them trained: the v1.5 per-opponent `known_hand_opp` **state**
stripe, the v1.5 `goal_delta_ignoring_eggs` **choice** tail stripe, the
habitat-conditioned play-bird `goal_delta` / egg-aware setup `goal_affinity`
pricing, the egg-optimistic bonus-potential pricing (both nets) + spend-food
`pay_food` routing (main net), the optimistic `MainActionChoice` consequence
forecast (main net), and the card-attribute `power_ex` EOT-discard-side
freeze (both nets)). See `docs/VERSIONING.md` for the full compat policy
(FRESH vs REGIME, when a MINOR bump is required, fixture-set rules, the MAJOR
escape hatch, and the v1.5 renumbering note).

Each MINOR FRESH encoding reshape adds one module per superseded era:

- a `v1_<N>.py` module with frozen `encode_*` / `*_embed_offsets` overrides,
- a `PolicyValueNet`/`SetupNet` subclass that regenerates the era's geometry,
- a branch added to `model.PolicyValueNet.class_for_version` and (if widths change)
  to `encoding_dims_for_era` below.

When a later reshape supersedes an era that already lacked an earlier stripe, the
older shim *inherits* the newer one so the strips compose (e.g. `v1_0` inherits
`v1_3`: v1.0 vectors lack the `becomes_unplayable`, `resets_feeder`,
`goal_delta_ignoring_eggs`, both food-unlock stripes, and the `known_hand_opp`
state stripe; `v1_3` in turn inherits `v1_4`, so every pre-1.4 era also strips the
v1.5 `known_hand_opp` state stripe and `goal_delta_ignoring_eggs` choice tail
stripe, and freezes the pre-1.5 `goal_delta` / `goal_affinity` pricing, the
pre-1.7 bonus potentials and spend-food routing, the `MainActionChoice`
consequence forecast, and the pre-EOT `power_ex` card table). A
**behavior-only** value change (stripe *values* changed, widths untouched)
follows the same module shape but
overrides only the encoder that regenerates the old values; there is no
`encoding_dims_for_era` branch and no offset/layout override to add for that part
of the change — `v1_4` itself mixes both shapes (some of what it freezes is
geometry-narrowing, some is values-only) in one class.

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
pre-1.5 same-MAJOR era (`minor <= 4`): `state_dim -= 180` (the `known_hand_opp`
stripe) AND `choice_dim -= 8` (the `goal_delta_ignoring_eggs` stripe) together,
in one combined branch. For every pre-1.4 same-MAJOR era (`minor <= 3`),
additionally: `state_dim -= 10` (the two food-unlock stripes) and
`choice_dim -= 1` (the `resets_feeder` stripe). v1.0 additionally drops the
180-dim `becomes_unplayable` stripe from `choice_dim`. Later same-MAJOR
artifacts get the live widths. Raises `version.IncompatibleArtifactError`
outright when `spec.num_players != 2` — every superseded era predates N-player
support by construction, so no shim ever needs to reproduce an N>=3 shape
(`docs/VERSIONING.md`'s `num_players` entry).

**`v1_4.py`** — pre-1.5 geometry + behavior compat shim, both nets (the merged
shim: what were four provisionally-numbered eras, 1.5-1.8, collapsed into one
class before any of them ever trained a run, plus a later in-place amendment
(MAIN_ACTION consequence pricing) folded in the same way — see
`docs/VERSIONING.md`'s v1.5 entry for the full renumbering rationale and the
fold-in note):
- `PolicyValueNetV1_4` — `PolicyValueNet` subclass with two geometry strips,
  four value refills, and a card-table freeze:
  - **State strip.** Strips the `known_hand_opp` 180-dim state stripe (appended
    after both playability multi-hots, before `decision_type`) from
    `encode_state` and freezes the pre-1.5 `StateEmbedOffsets`: only
    `decision_type` shifts left by `STATE_KNOWN_HAND_OPP_DIM` — `card_index` /
    `hand_multihot` precede the new stripe and are unchanged (the opposite
    shift shape from `v1_3`'s food-unlock strip, which precedes both and
    shifts all three). Overrides `encode_state`, `_state_embed_offsets`,
    `_build_trunk`, `_true_state_dim` (absolute form, derived from `self.spec`
    — the shim closest to live) and `raw_state_stripe_layout`.
  - **Choice strip.** Strips the `goal_delta_ignoring_eggs` 8-dim choice tail
    stripe (the prior last base stripe, after `resets_feeder`) from
    `encode_choices` and shifts only `kept_multihot` (`bird_id` /
    `becomes_playable` / `becomes_unplayable` precede it and are unchanged).
    Overrides `_choice_embed_offsets`, `_build_choice_encoder`,
    `_true_choice_dim` (also absolute form) and `raw_choice_stripe_layout`.
  - **Four chained refills inside `encode_choices`,** in this order: (1) the
    bonus-potential / spend-food / MAIN_ACTION-forecast refill at live width
    (`choice_encode.refill_bonus_value_potentials_static` on every
    bonus-carrying row, `choice_encode.refill_spend_food_gain_routing` on
    every spend-decision `FoodChoice` row,
    `choice_encode.refill_main_action_forecast_zeros` on every
    `MainActionChoice` row — the whole `exchange` stripe, plus `bonus_delta`
    except DRAW_CARDS and `goal_delta` except LAY_EGGS, whose scalars predate
    this amendment and regenerate identically live); (2) the
    `goal_delta_ignoring_eggs` tail strip; (3) the habitat-agnostic
    `goal_delta` refill (`choice_encode.refill_goal_delta_habitat_agnostic`)
    on the narrowed rows.
    Every refill offset (`layout._OFF_BONUS_VALUE`, `layout._OFF_GAIN_FOOD` /
    `layout._OFF_PAY`, `layout._OFF_EXCHANGE`, `layout._OFF_BONUS_DELTA`,
    `layout._OFF_GOAL_DELTA`) precedes the stripped tail, so the chain
    composes with no offset math and `v1_3` / `v1_0`'s own strips run
    correctly after this method returns. Routes for era 1.4 via
    `model.core.PolicyValueNet.class_for_version`.
  - **Card-table freeze.** Overrides `_build_card_encoder`: builds the live
    card encoder via `super()`, then overwrites the frozen `card_features`
    buffer's power_ex columns with the pre-v1.5-amend (EOT-discard-omitted)
    values (`state_encode.refill_card_features_power_exchange_pre_eot`) —
    outside the per-row `encode_choices` chain, since the buffer is built
    once at construction and shared by every row.
- `SetupNetV1_4` — `wingspan.training.setup_net.SetupNet` subclass, the first
  `SetupNet` compat shim. Overrides `encode_candidate`: calls the live
  encoder via `super()`, then `setup_model.encode.refill_bonus_pricing_static`
  and `setup_model.encode.refill_goal_affinity_static` overwrite the bonus
  pricing and `goal_affinity` stripes with their pre-1.5 static (egg-blind)
  pricing, in place — disjoint stripes, so the two refills compose in either
  order. Also overrides `_build_card_encoder` — the same power_ex freeze as
  the main net's, since `SetupNet` builds its own copy of the shared
  `state_encode.card_feature_matrix()` table. No choice/candidate-geometry override —
  every setup-side change since the v1.3 two-tower restructure has been
  values-only, so this class joins no dims-router branch. Routes for eras
  <= 1.4 via `wingspan.training.setup_net.SetupNet.class_for_version`; eras
  <= 1.3 route here too, harmlessly (a pre-1.3 setup artifact differs in
  *shape* and would fail at `load_state_dict` regardless of which class
  builds it — `players.loaders.load_setup_net` already turns that into a
  clear "retrain the setup model" error).

**`v1_3.py`** — pre-1.4 geometry compat shim (inherits `v1_4.PolicyValueNetV1_4`):
- `PolicyValueNetV1_3` — `PolicyValueNetV1_4` subclass that reverses **both** v1.4
  additions. On the **state** side it strips the two 5-wide food-unlock stripes
  (`hand_food_unlock_me`, `tray_food_unlock_me`) from `encode_state` and freezes the
  pre-1.4 `StateEmbedOffsets` (overrides `encode_state`, `_state_embed_offsets`,
  `_build_trunk`, `_true_state_dim`). On the **choice** side it strips its own
  `resets_feeder` 1-dim stripe (immediately after `becomes_unplayable`) from
  `encode_choices` and shifts only `kept_multihot` (`bird_id` /
  `becomes_playable` / `becomes_unplayable` precede it and are unchanged);
  `_true_choice_dim` composes via `super()` (the inherited v1_4 narrowing) minus
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
  v1_3 -> v1_4 chain) `known_hand_opp` and `goal_delta_ignoring_eggs` too) that
  additionally restores the v1.0 trunk-final-activation fallback
  (`trunk_final_activation=null` resolved to `between_activation` instead of
  `final_activation`) and strips the `becomes_unplayable` 180-dim choice stripe added
  in v1.1. Overrides `_build_trunk` (v1.0 activation fallback at the inherited narrow
  state width), `_true_choice_dim` (narrows the inherited v1.3 width by a further
  `becomes_unplayable`, read polymorphically by the inherited `_build_choice_encoder`),
  `encode_choices`, and `_choice_embed_offsets` — the last two chain `super()` (the
  v1_3 strip) then remove `becomes_unplayable` — and `raw_choice_stripe_layout`
  (the inherited v1.3 layout minus `becomes_unplayable`; the state layout is
  inherited unchanged). Routes for era 1.0 via `class_for_version`.
