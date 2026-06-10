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

## Changelog

### v0.3 — one-hot round number and cube counts (current)

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
graceful `architecture_key` gate — mismatch starts fresh, never crashes.
Training resume across versions is **not** promised.

The guarantee extends to *describing*: reporting surfaces (`wingspan inspect`,
the run-start `model_inspect.json` / `model_summary.html`) derive every
layout, width, and parameter count through the descriptor seam in `runmeta`
(`choice_layout_for`, `param_report_for`, `build_model_summary_html`, …),
which routes by the descriptor's version the same way the loaders do — never
compute a report value from the live encoder when a descriptor is in hand.

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
