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

### v0.4 — turn-state stripe and first-player flag (current)

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
against the old model via `bootstrap_opponent_checkpoint` (which loads
through the shims).

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
  checkpoint embeds its `config` and its `version`; every run directory carries
  `model_config.json` (+ `setup_config.json` when the setup model is on).
  Never add an "assume compatible" branch, a second on-disk location for the
  same datum, or a ghost entry kept only for index stability.
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
