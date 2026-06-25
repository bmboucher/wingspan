# Artifact versioning and checkpoint compatibility

Every persisted artifact (`model_config.json`, `setup_config.json`, and every
`.pt` payload) is stamped with a `MAJOR.MINOR` **artifact version**
(`wingspan.version.MODEL_VERSION`). This is distinct from the package release
version (`wingspan.__version__`) — one tracks the codebase, the other the
on-disk artifact format.

The **June 2026 compatibility cutoff**: loaders tolerate no artifact written
before it. From the cutoff on, compatibility is governed by the artifact
version below — a deliberate, versioned guarantee, never ad-hoc tolerance.

Update this file in the same commit that bumps `MODEL_VERSION`.

---

## The rehydration guarantee (the whole point)

A run freezes its config to disk when it starts. Loading those files under any
later same-MAJOR code must reconstitute a model that **computes identically** to
the one that was saved — same encoding, same parameters, same inference logic.
An old artifact never silently adopts new behavior: a 0.3 model loaded under 0.7
code runs the 0.3 encoding and the 0.3 code paths, producing the same numbers it
always did.

So the trigger for an era shim is **any code change that would make a rehydrated
artifact behave differently**, not merely one that alters a tensor shape. A
shape change (FRESH) is the *most visible* kind — the weights won't even load —
but it is one sufficient trigger, not the defining one. A shape-preserving change
to how a feature value is computed, or new logic added in 0.5 that wasn't in 0.4,
is just as much a break: under 0.5 code, a 0.4 model must keep taking the 0.4
path. The one unavoidable exception is the engine (see below).

## Changelog

### v0.8 — Food-gain `becomes_playable` ignores eggs (current)

**FRESH change (code-carried)** — changed the `becomes_playable` computation on
**food-gain** choice rows so the egg-cost gate is no longer applied. A hand bird
is now flagged as "becomes playable" when gaining the offered food meets its food
cost AND an open habitat slot exists, regardless of whether the egg cost is also
met. The egg-gain path (`LAY_EGGS`, egg exchanges) is unchanged.

No tensor widths change (state dim, choice dim, and `CARD_FEATURE_DIM` are all
the same). This is a **code-carried** FRESH change: the same vector slot changes
its computed value, making inference on a 0.7 checkpoint diverge if it ran under
live 0.8 code without a shim.

Shim: `wingspan.compat.v0_7` — covers **exactly v0.7 artifacts**.
`PolicyValueNetV07` overrides `encode_choices` to call `encode_choices_v07`,
which passes `food_playable_ignores_eggs=False` to restore the eggs-included
semantics. `uses_v0_7_becomes_playable_encoding(v)` predicate is True iff
`(major, minor) == (0, 7)`.

`PolicyValueNetV06` (covering v0.2–v0.6 artifacts) also gains an `encode_choices`
override that delegates to `encode_choices_v07` — v0.6 artifacts predate the 0.8
fix exactly as 0.7 artifacts do, so they must compute the same eggs-included bits.

`encoding_dims_for_era` is unchanged: the 0.8 change does not affect `state_dim`
or `choice_dim`, so era-pinned v0.7 training resumes against the live
state/choice dims.

Fixture set: `tests/data/compat/v0.8/` — deferred per existing pattern.

### v0.7 — OR-cost flag in card features (superseded by v0.8)

**FRESH change** — added a 1-dim `or_cost` flag to the per-card attribute vector,
growing `CARD_FEATURE_DIM` by 1 (224 → 225). State and choice vector widths are unchanged.

The flag is `1.0` for the 31 core-set birds whose printed food cost is an OR choice
("pay 1 invertebrate OR 1 seed") and `0.0` for birds whose cost is AND ("pay 1
invertebrate AND 1 seed"). The `or_cost` stripe is appended last in `CARD_ATTR_LAYOUT`
so the earlier attribute block (dims 0..43) is identical between v0.6 and v0.7,
simplifying the compat shim.

The card parsing fix that introduced `BirdCost.is_or_cost` landed in the preceding
commit (OR-cost payment logic and display); this FRESH change closes the remaining gap
by surfacing the flag to the model.

Shim: `wingspan.compat.v0_6` — covers **v0.2 through v0.6 artifacts** (all eras with
the 224-wide card encoder). `PolicyValueNetV06` and `SetupNetV06` override
`_build_card_encoder` to build the 224-wide MLP input and register the pre-0.7 feature
table (via `card_feature_matrix_v06()`). Earlier shims (v0_2, v0_3, v0_4) also
override `_build_card_encoder` to delegate to `_install_v06_card_encoder_main`.
`uses_v0_6_card_feature_encoding(v)` predicate covers `(0,2) <= (major,minor) < (0,7)`.

`encoding_dims_for_era` is unchanged: the 0.7 change does not affect `state_dim` or
`choice_dim`, so era-pinned v0.6 training resumes against the live state/choice dims.

Fixture set: `tests/data/compat/v0.7/` — to be captured after a short 0.7 training run,
same pattern as prior eras.

**REGIME change: configurable hand pooling; `use_distinct_hand_model` default flipped** —
`HandPooling(StrEnum)` added to `architecture.py`; `hand_pooling: HandPooling =
HandPooling.CONCAT_MAX_SUM` field added to `ModelArchitecture`;
`use_distinct_hand_model` default changed `True → False` (new runs use the pooled path).
The five dedicated-hand-encoder configurator fields (`hand_embed_dim`,
`hand_encoder_layers`, `hand_activation`, `hand_dropout`, `hand_layernorm`) are removed
from the new-run UI but retained on the model for old-artifact back-compat; a single
`hand_pooling` ChoiceField replaces them. `HandPooling | None` appended to `ShapeKey`
(valued `None` for distinct-encoder runs so their key is unchanged). Pooling is entirely
network-internal (`model._embed_state`); raw `encode_state` and stripe offsets are
unchanged. Old distinct-encoder checkpoints carry `use_distinct_hand_model=True` → pooling
inert → identical rehydration. **No `MODEL_VERSION` bump, no compat shim.**
Mirrors the `use_board_attention` / `tray_set_embedding` REGIME precedents.

### v0.6 — playability-aware hand copies (superseded by v0.7)

**FRESH change** — added two playability-filtered hand multi-hots to the state vector
and a `becomes_playable` multi-hot to every choice row, growing the state by 360 dims
(795 → 1155) and each choice row by 180 dims:

1. **State: `hand_playable_me` (180 dims)** — birds in the active player's hand that
   are playable right now (food affordable + open habitat slot + eggs sufficient).
   Inserted immediately after `hand_multihot`, before the decision-type one-hot.
2. **State: `hand_playable_eggs_me` (180 dims)** — birds in hand where food is
   affordable and a habitat slot is open, but the player lacks the required eggs
   ("egg-blocked"). Inserted right after `hand_playable_me`.
3. **Choice: `becomes_playable` (180 dims)** — birds in hand that transition from
   not-playable to playable as a direct result of the food or egg(s) this choice
   grants. Appended after `bonus_value` in each choice row. Filled on:
   - `FoodChoice` (`GainFoodDecision`) — exact, one food gained.
   - `PayCostChoice` skip_optional exchanges — optimistic best-case: feeder food
     when `gained_food_count > 0`, egg unlock when `gained_egg_count > 0`.
   - `GAIN_FOOD` and `LAY_EGGS` `MainActionChoice` rows — optimistic best-case
     (feeder foods / `lay_eggs_count()` eggs respectively).
   - **Not** on `BoardTargetChoice` (`LayEggDecision`): the egg is already committed
     and the gain is constant across slots — no choice-relevant signal there.
4. **Each new multi-hot is embedded through the shared hand encoder** (derived 10-dim
   summary via `card_summary_matrix` when `use_distinct_hand_model` is on, mean-pool
   otherwise), so the trunk sees a learned dense summary rather than a raw bit vector.

**REGIME change: tray `set_embedding` default flipped** — `tray_set_embedding` default
changed `True → False` in `ModelArchitecture`. Old checkpoints carry `tray_set_embedding=True`
in their config and load unchanged via the config-carry mechanism; the code and validator
are retained for backward compat (removal at next MAJOR bump). New 0.6 runs never enable
the tray hand model.

**Config-carried setup change: `include_turn1_playable`** — `SetupEncoding` gains
`include_turn1_playable: bool = False`. When enabled, a 180-dim `turn1_playable`
multi-hot (birds from `kept_cards` payable from `kept_foods` on turn 1) is appended to
the setup feature vector and embedded through the hand encoder. Existing setup configs
deserialize with the flag absent → `False` → old `total_dim` preserved → no setup shim
needed. New 0.6 runs enable it by default.

Shim: `wingspan.compat.v0_4` — covers **both 0.4 and 0.5 artifacts** (encoding-identical
pair; the 0.5 bump was config-container only). `PolicyValueNetV04` overrides
`encode_state` / `encode_choices` / `_state_embed_offsets` / `_choice_embed_offsets` to
run the frozen 795-dim state (no playability stripes) and narrower choice rows (no
`becomes_playable`). `uses_v0_4_encoding(v)` predicate covers `(0,4) <= (major,minor)
< (0,6)`.

Fixture set: `tests/data/compat/v0.6/` — to be captured after this change merges
(requires a short 0.6 training run), same pattern as v0.4/v0.5.

### v0.5 — unified run-config file (REGIME additions)

#### DAgger behavioral cloning (REGIME)

Added `DaggerConfig` as a 7th `RunConfig` section (`dagger.expert_checkpoint`,
`dagger.clone_iters`). Config-carried and training-only: no tensor shape changes,
no featurizer change, no `MODEL_VERSION` bump, no compat shim. A run resumed
past `clone_iters` simply finds `dagger_active_at(iteration) == False` and trains
normally — the field survives the round-trip via `run_config_from_artifact`'s
default `DaggerConfig()`. Mirrors `reward_mode` in versioning classification.

`Step.expert_probs` is IPC/in-memory only: the `games.jsonl` serializer writes
`metrics.GameOutcome` (via `loop_metrics.game_outcome`), never `Step`, so there
is no on-disk format change from this field.

#### Clone + bootstrap unification (accepted loop drift)

`dagger_expert_checkpoint` now derives from `bootstrap_opponent_checkpoint` instead
of `dagger.expert_checkpoint`. Old runs that set `dagger.expert_checkpoint` with
`bootstrap_opponent="none"` will no longer clone on resume (the derived property
returns `None`). This is accepted drift under the engine's shared-seat exemption:
the training loop is shared by both seats and cannot fork by era, so loop-level
behavior changes are accepted drift. The `dagger.expert_checkpoint` field is
retained in the model so old artifacts load without errors.

#### Per-block activation/dropout/layernorm overrides (REGIME)

Added 14 optional per-block override fields to `ModelArchitecture` and
`MainNetArchitecture`: `{card,hand,trunk,choice}_{activation,dropout,layernorm}`,
`{value,head}_activation`. All default to `None` = "inherit the global".
Old artifacts (which carry no per-block keys) rehydrate with all 14 as `None`,
resolved identically to the global — **REGIME, no `MODEL_VERSION` bump, no compat
shim.** The `ShapeKey` now includes 4 resolved per-block layernorm bools (replacing
the single `layernorm` bool at position 4); old artifacts produce an identical key
because `None` → global for all four.

#### Locked-in defaults flip (REGIME)

`MainNetArchitecture.tray_set_embedding` default flipped `True → False`;
`SetupNetArchitecture.use_actor_critic` default flipped `False → True`. Old
artifacts carry their own saved values and are unaffected. New runs get the
locked value.

---

**Config-container change, NOT an encoding change.** This MINOR bump is unusual:
the encoding is byte-for-byte identical to 0.4 — `state_dim` / `choice_dim` / the
card feature vector are unchanged — so **no `wingspan.compat.v0_4` shim exists or
is needed.** A 0.4 artifact is encoding-identical to live and already falls
through the live paths (`encoding_dims_for_era`, `PolicyValueNet.class_for_version`,
the `uses_v0_*` gates) with no 0.4 entry. The MINOR bump exists only so the old
and new on-disk *config-file* formats can never share a version.

What changed is the per-run config layout. The three files a run used to scatter
its configuration across —

* `process_<stamp>.json` (the flat `TrainConfig` + session context),
* `model_config.json` (the weight-compat descriptor), and
* `setup_config.json` (the setup-net descriptor)

— collapse into **one dated `run_config_<stamp>.json`** per session
(`config.RunConfigFile`), wrapping the new hierarchical `RunConfig`. `RunConfig`
(formerly `TrainConfig`, kept as an alias) groups every hyperparameter into six
nested sections: `architecture` (by submodel) · `run` · `training` · `opponent` ·
`engine` · `misc`. `ModelConfig` / `SetupConfig` survive as the *in-memory*
inference descriptors; only their on-disk source changes.

Backward compatibility is two presence-dispatched seams, both **outside** the
`compat` package (this is a config-format reader dispatch, not an encoding shim):

1. **Run-directory reads** — `runmeta.read_model_config` /
   `setup_runmeta.read_setup_config` derive the descriptor from
   `run_config_<stamp>.json` when present, else fall back to the legacy file.
   The v0.0–v0.2 fixture dirs carry only legacy files → legacy branch → the
   existing compat tests pass unmodified.
2. **Embedded `.pt` reads** — `config.run_config_from_artifact(raw, version)`
   validates a ≥0.5 nested dict directly and reshapes a ≤0.4 flat dict into the
   six sections (preserving the legacy `bootstrap_opponent` migration).

Shim: none (config-container only — see above).

Fixture set: a v0.5 run directory (the unified format) is to be captured after
this change merges (requires a short training run), as with v0.4. The format
itself is covered by `tests/test_run_config.py` (flat→nested migration + dated
file round-trip) and by the live-loop write/read in `test_training_dashboard.py`.

### v0.4 — turn-state stripe and first-player flag

**FRESH change** — replaced the round one-hot and both cube one-hots in
`misc_scalars` with a new leading `turn_state` stripe, growing the state vector
by 5 dims (790 → 795):

1. **New `turn_state` stripe (27 dims) prepended first** — a 26-dim player-turn
   one-hot (which of the player's 26 personal turns across all 4 rounds they are
   on) plus a 1-bit `is_first_player` flag (1.0 when the POV player goes first in
   the current round). All-zeros during setup (`turn_counter == 0`). Turn index
   formula: `_ROUND_CUBE_OFFSETS[round_idx] + (ROUND_CUBES[round_idx] - action_cubes_left)`.
2. **`misc_scalars` shrank from 26 → 4 dims** — the 4-dim round one-hot and
   both 9-dim cube one-hots were dropped; only the 4 trailing scalars remain
   (goal pts × 2, tray size, deck size).

New constants: `N_PLAYER_TURNS = 26`, `_ROUND_CUBE_OFFSETS = [0, 8, 15, 21]`.
Constants `N_ROUNDS` and `MAX_ACTION_CUBES` are retained for backward-compat shims.

Shim: `wingspan.compat.v0_3` — `PolicyValueNetV03` (overrides `encode_state`
with frozen 26-dim one-hot misc stripe, no turn_state), `encode_state_v03`
(790-dim frozen vector), `state_stripe_layout_v03` (frozen stripe registry).

Fixture set: `tests/data/compat/v0.4/` — to be captured after this change
merges (requires a short training run).

### v0.3 — one-hot round number and cube counts

**FRESH change** — replaced three raw scalars in `_summary_misc_scalars` with
one-hot vectors, growing the state vector by 19 dims (771 → 790):

1. **Round scalar replaced** — `round_idx / 3.0` (1 dim) → 4-dim one-hot over
   rounds 0–3 (`N_ROUNDS = 4`).
2. **Cube-me scalar replaced** — `action_cubes_left / 8.0` (1 dim) → 9-dim
   one-hot over cube counts 0–8 (`MAX_ACTION_CUBES + 1 = 9`).
3. **Cube-opp scalar replaced** — same as cube-me. Net: +8 dims.

Total misc-scalars stripe: 1 + 1 + 1 = 3 raw scalars → 4 + 9 + 9 = 22 one-hot
dims + the 4 unchanged scalars (goal pts × 2, tray size, deck size) = 26 dims.

Both new constants (`N_ROUNDS`, `MAX_ACTION_CUBES`) live in `encode/layout.py`.

Shim: `wingspan.compat.v0_2` — `PolicyValueNetV02` (overrides `encode_state`
with frozen 7-scalar misc stripe), `encode_state_v02` (the 771-dim frozen
vector), `state_stripe_layout_v02` (frozen stripe registry for reporting).

Fixture set: `tests/data/compat/v0.2/` — carries `version: "0.2"` explicitly.

### v0.2 — card feature vector redesign

**FRESH change** — reshaped the card feature vector (no longer current; superseded by v0.3) (`CARD_FEATURE_DIM` 229 → 224)
in three ways:

1. **`bonus_categories` pruned** — trimmed from 26 dims (one per bonus card, keyed
   to `cards.bonus_index()`) to 7 dims covering only intrinsic-property categories
   not already expressed by other stripes: Anatomist, Backyard Birder, Cartographer,
   Historian, Large Bird Specialist, Passerine Specialist, Photographer. Dropped:
   state-dependent (Breeding Manager, Ecologist, Oologist, Visionary Leader),
   food-cost duplicates (Bird Feeder, Fishery Manager, Food Web Expert, Omnivore
   Specialist, Rodentologist, Viticulturalist), nest duplicates (Enclosure Builder,
   Nest Box Builder, Platform Builder, Wildlife Gardener), habitat duplicates
   (Forester, Prairie Manager, Wetland Scientist, Bird Bander), and flag duplicates
   (Bird Counter, Falconer).
2. **`caches_food` flag added** — 1-dim binary flag set when any power effect
   caches food on the bird (CACHE_FOOD, GAIN_FOOD_FEEDER_MAY_CACHE,
   ROLL_NOT_IN_FEEDER_CACHE, PINK_GAIN_FOOD_CACHE).
3. **`power_exchange` stripe added** — 13-dim vector encoding the bird's power's
   resource exchange, using the same slot semantics and `_EXCHANGE_SCALE` as the
   choice-row exchange stripe.

Shim: `wingspan.compat.v0_1` — `PolicyValueNetV01` (frozen 229-wide card encoder
with v0.1 feature table), `card_feature_matrix()` (v0.1 [181, 229] feature table).

Fixture set: `tests/data/compat/v0.2/` — carries `version: "0.2"` explicitly.

### v0.1 — choice-vector encoding redesign

**FRESH change** — reshaped choice row layout in three ways:

1. **Landing-slot mark in board indices** — v0.0 used a 3-wide habitat
   one-hot stripe for placement choices (where a bird lands); v0.1 replaced it
   with a single-slot landing mark inside the board-index block.
2. **Bird-identity index collapse** — v0.0 stored the candidate bird as a
   180-wide one-hot (one bit per core-set bird); v0.1 collapsed it to a single
   integer index column.
3. **Dedicated kept-multihot stripe** — v0.0 doubled the bird-identity one-hot
   as the setup-pick kept-set multi-hot (same stripe, two interpretations);
   v0.1 moved the kept set onto a separate trailing `kept_multihot` stripe.

Choice rows shrank from 397 dims (base, 401 with `include_setup`) to a
different footprint. State encoding, family-head ordering, and setup model
were not changed.

Shim: `wingspan.compat.v0_0` — `PolicyValueNetV00` (frozen v0.0 choice
encoder geometry), `encode_choices()` (v0.0 row layout from live state),
`choice_stripe_layout()` (v0.0 layout for reporting surfaces).

Fixture set: `tests/data/compat/v0.1/` — carries `version: "0.1"` explicitly.

### v0.0 — initial versioned era

The first artifact era. Artifacts from this era may have the `version` field
absent; loaders default-read a missing field as
`version.PRE_VERSIONING_VERSION` (`"0.0"`, pinned forever as `MODEL_VERSION`
advances). Choice rows were 397 dims (base) / 401 dims (with `include_setup`).

Fixture set: `tests/data/compat/v0.0/` — deliberately omits the `version`
field to exercise the default-to-`"0.0"` load path.

---

## The guarantee: load + play

At code version X.Y, artifacts with the same MAJOR and MINOR ≤ Y must load
and play games (inference / eval / tournament). A different MAJOR, or a MINOR
newer than the code, is refused cleanly with `version.IncompatibleArtifactError`.

**Enforcement is deliberately asymmetric.** The hard version check guards the
*inference* loaders (`runmeta.read_model_config`,
`setup_runmeta.read_setup_config`, and the `players.loaders` trio
`load_policy_net` / `load_setup_net` / `load_policy_net_from_run_dir` behind
`cli.main_play` and `tournament.participants.load_player`). The *resume*
loaders (`loop_resume`, `loop_setup`, `loop_eval.load_opponent`) keep the
graceful `architecture_key` gate — a genuine mismatch starts fresh, never
crashes. A run whose only mismatch is its era is not a mismatch at all: it
resumes **era-pinned** (next section).

The guarantee extends to *describing*: reporting surfaces (`wingspan inspect`,
the run-start `model_inspect.json` / `model_summary.html`) derive every
layout, width, and parameter count through the descriptor seam in `runmeta`
(`choice_layout_for`, `param_report_for`, `build_model_summary_html`, …),
which routes by the descriptor's version the same way the loaders do — never
compute a report value from the live encoder when a descriptor is in hand.

## Training resume: era pinning

A run records the era it trains at in its config
(`TrainConfig.encoding_version`) and never leaves it: its dims derive from the
era (`compat.encoding_dims_for_era`), its net is constructed as the era's
class (`model.PolicyValueNet.class_for_version`) — in the main loop, the eval
clone, and every `mp_collect` worker — collection encodes through that net's
frozen encoders, and **every artifact the run writes is stamped with the
run's era**, never the live `MODEL_VERSION`: `last.pt` / `best.pt` /
`opponent.pt` / `setup.pt`, `model_config.json`, `setup_config.json`. An
era-pinned run's directory is indistinguishable from one still being written
by its own era's code — the rehydration guarantee applied to training, so a
FRESH encoding change no longer orphans an in-flight run.

Pinning is adopted from the run directory, never configured by hand
(`encoding_version` is deliberately not an editable configurator field):

- The configurator seeds the working config from the saved run's embedded
  config, rehydrated at the payload's own version stamp
  (`config.train_config_from_artifact`), so an old-era run reads RESUMABLE
  and launches pinned. It then keeps the era *aligned* on every edit
  (`configure.runs.align_era`): while the working config stays
  architecture-compatible with the saved run it keeps the run's era, but an
  edit that forces a fresh run bumps it to the live `MODEL_VERSION` (and
  reverting the edit re-pins) — the era line in the run-management panel
  tracks this live.
- `TrainingLoop.__init__` calls `loop_resume.adopt_checkpoint_era` before
  building anything: when adopting the checkpoint's era is exactly what makes
  the saved and current `architecture_key`s agree, the config is pinned —
  covering headless entry points (cloud runner, direct construction) by
  construction. Every other situation starts fresh, and **a fresh launch is
  re-keyed at the live `MODEL_VERSION`** — a new run never inherits a stale
  era from a working config seeded off an old run. (Regenerating old-era
  artifacts therefore means building the era net directly, the
  `tests/test_era_pinned_resume.py` pattern, not training fresh through
  `TrainingLoop`.)

`architecture_key` itself now leads with the era, so a shape-preserving FRESH
change still reads as incompatible (coinciding widths across eras are the
silent-corruption case), and configs written before `encoding_version` existed
derive it from the payload's `version` stamp — the field that has always
carried the era.

The cost of pinning is deliberate: an era-pinned run never gains later
encodings' features (that is the point), and superseded eras become
*producing* paths — a new training feature must either work at every
same-MAJOR era or refuse one explicitly. Moving a line of work onto a new
encoding still means a fresh run at the live era, optionally bootstrapped
against the old model via `opponent.bootstrap_opponent` (a checkpoint path,
which loads through the shims).

## Compat shims — the one sanctioned mechanism

Each MINOR bump adds one module to the `wingspan.compat` package, one per
superseded era. Shape: `if artifact older than the change: regenerate the
encoding without the new field`. Inference call sites must encode through
the net (`net.encode_state` / `net.encode_choices`), never by pairing the live
encoder with a spec by hand — that is what lets a compat-era net carry its own
geometry.

**Compat is version-number-specific checks, never config flags.** Do not add
`TrainConfig` axes to toggle old behaviors.

## MINOR bumps (FRESH changes)

A MINOR bump is required for every FRESH-type change — any change that alters
a tensor shape — and must:

1. Bump `MODEL_VERSION` in `wingspan/version.py`.
2. Add the version-specific shim in `wingspan/compat/v<X_Y>.py`.
3. Capture a new fixture set under `tests/data/compat/v<X.Y>/` from a run at
   the new version (see that directory's READMEs for the expected shape).
4. Extend the compat tests so **every retained fixture set still loads and
   plays**. All same-MAJOR fixture sets are retained.

## MAJOR bumps (escape hatch)

A MAJOR bump drops the accumulated shims and deletes the old fixture sets. It
must be its own called-out, user-approved decision — never a side effect of
another change.

## FRESH vs REGIME

`architecture_key` / `ShapeKey` (and the setup twins) cover everything that
changes a tensor shape. A mismatch refuses the weights and restarts cleanly
(**FRESH**) — and shipping such a change requires the MINOR bump + shim +
fixture set described above. Shape-preserving knobs (`activation`, `dropout`,
learning rates, cadences) stay out of the key, resume freely, and need no
version bump (**REGIME**).

But shape is a proxy, and an incomplete one. The real fault line is
**config-carried vs code-carried behavior**:

- **Config-carried** behavior — `activation`, `dropout`, every dim and flag in
  `model_config.json` — travels *with the artifact*. It rehydrates exactly
  because the value is read back from the frozen file, so it needs no version
  gate. This is why those knobs are safely REGIME: not because they preserve
  shape, but because the artifact carries its own copy.
- **Code-carried** behavior — featurizer arithmetic, the slice offsets a net
  derives from `encode.layout`, an inference branch, a new computation added in
  a later version — lives in the *live codebase*, not the artifact. Any change
  to it must be era-gated (MINOR bump + shim) so an old artifact keeps the old
  path, **even when no tensor shape changes**.

A shape-preserving code-carried change is the dangerous case: it loads without
complaint and silently misbehaves. The 2026-06-10 `_embed_state` bug was exactly
this — a v0.2 net fed its 771-dim vector but sliced it with the live 790-dim
offsets; the widths coincided, so nothing crashed while 19 columns of the trunk
input were wrong. The fix freezes the slice offsets per era
(`compat.v0_2.state_embed_offsets_v02`, overriding `_state_embed_offsets`), the
same way `encode_state` is already frozen. So when adding an era shim, freeze
**all** geometry the net derives from the layout, not just `encode_state` —
every code-carried value the loaded artifact's behavior depends on.

The same class recurred on 2026-06-14: the 0.4 `turn_state` stripe (27 dims,
new at the front) shifted the hand-summary stripe 27 columns, but `_embed_state`
still read it from the live `encode.HAND_SUMMARY_OFFSET` — a layout offset the
era seam did not yet cover. Every pre-0.4 net with a distinct hand model had its
hand summary mis-sliced; `encode_state` itself was byte-correct, so only the
forward pass was wrong, and sharp checkpoints dropped to random-level play while
their self-play training metric — which never round-trips through the shim —
stayed healthy. The structural fix makes the seam exhaustive: `_state_embed_offsets`
returns a `model.StateEmbedOffsets` named tuple carrying *all four* offsets
`_embed_state` reads (card-index, hand, decision, hand-summary), and each shim
freezes the whole tuple — different stripes precede each, so they do not share
one delta. Import-time assertions pin the hand-summary delta to the live
`turn_state` width, so the next inserted stripe is caught, not silently absorbed.

## The one accepted source of drift: the engine

`engine.core.Engine` is shared by both players in a game and cannot fork its
behavior by either player's model version — a single game runs one engine while
the two seats may carry different-version nets. So a change to how the engine
**calculates, applies, or presents** choices changes the inputs every model
sees, and an old model will play slightly differently under newer engine code.
This is accepted and unavoidable; it is *not* a versioned guarantee and never
gets a shim.

The seam is clean: the engine produces `GameState` and the menu of `Choice`s —
that **may** drift across versions. The per-artifact featurization of that state
into tensors — `encode_state` / `encode_choices` / `_embed_state` /
`_embed_choices` and every offset, width, and feature value the net derives —
**may not**: it is frozen per era by the compat shims. If you find yourself
wanting to version-gate something inside the engine, that's the signal it
belongs on the net side of the seam instead.

## Format rules

- **Every artifact is self-describing; loaders refuse what isn't.** Every
  checkpoint embeds its `config` and its `version`. A ≥0.5 run directory carries
  one dated `run_config_<stamp>.json` (the `ModelConfig` / `SetupConfig`
  descriptors are *derived* from it); a ≤0.4 directory carries the legacy
  `model_config.json` (+ `setup_config.json` when the setup model is on), read
  via presence dispatch. Never add an "assume compatible" branch, a second
  on-disk location for the same datum, or a ghost entry kept only for index
  stability.
- **The stable orders are part of the checkpoint format.**
  `ALL_DECISION_CLASSES`, `ALL_DECISION_FAMILIES`, the `encode/layout.py`
  offset chain, and the `cards.parse.catalog` card-index maps are append-only;
  reordering, renumbering, or removing an entry is a FRESH break for every
  checkpoint and must be a deliberate, called-out decision.
- **New fields on persisted models default — the one sanctioned back-compat
  mechanism** so current-era artifacts keep loading; comment why the default
  exists. Required fields stay required. (The `version` field itself works
  this way: absence reads as `version.PRE_VERSIONING_VERSION`, pinned `"0.0"`
  forever while `MODEL_VERSION` advances.)
- **Fixture sets are the only checkpoints in git**: gzip-compressed (`*.pt.gz`)
  and **Git LFS**-tracked via `.gitattributes`, with the config JSONs committed
  plain. New fixture sets must follow the same shape (see the v0.0 README in
  `tests/data/compat/`).
- Crash-survivability tolerance is fine and stays (e.g. `metrics_log` skipping
  a truncated final line): it guards the *current* format against interruption,
  not an old format against age.
