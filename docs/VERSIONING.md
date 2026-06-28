# Artifact versioning and checkpoint compatibility

Every persisted artifact (the dated `run_config_<stamp>.json` run descriptor and
every `.pt` payload) is stamped with a `MAJOR.MINOR` **artifact version**
(`wingspan.version.MODEL_VERSION`, currently **`1.1`**). This is distinct from
the package release version (`wingspan.__version__`) — one tracks the codebase,
the other the on-disk artifact format.

**1.0 is the clean-break baseline.** The 1.0 MAJOR bump dropped every pre-1.0
compat shim and fixture set, so `check_artifact_compatible` refuses every pre-1.0
(0.x) artifact as a different MAJOR. From here on, compatibility is governed by
the artifact version below — a deliberate, versioned guarantee, never ad-hoc
tolerance.

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

### v1.1 — uniform final-activation inheritance + `becomes_unplayable` stripe + setup-encoding pooling (current)

Three independent changes landed in v1.1:

**1. Uniform final-activation inheritance (architecture).** `ModelArchitecture.trunk_final_activation_resolved` now inherits
`final_activation` when `trunk_final_activation` is `None` — the same rule every
other block uses. Previously the trunk fell back to `between_activation`, giving
it an implicit final relu whenever `final_activation=none` (the default). This
was a silent asymmetry: `trunk_final_activation=null` in a config meant
"inherit relu" while `choice_final_activation=null` meant "inherit none".

**2. `becomes_unplayable` choice-encoding stripe (encoding).** A 180-dim
multi-hot stripe, embedded through the shared card table exactly like
`becomes_playable`, was appended to the base choice feature vector immediately
after `becomes_playable`. It flags which currently-playable hand birds a choice
would make unplayable by spending food, eggs, or a board slot. This adds one
`card_embed_dim`-wide embedding to the choice encoder's first-Linear input.

- **Behavioral change — architecture.** Any v1.0 artifact with
  `trunk_final_activation=null` (the default) would compute differently if
  reloaded under v1.1 code. The shim
  `wingspan.compat.v1_0.PolicyValueNetV1_0` (routed by
  `PolicyValueNet.class_for_version`) restores the old fallback for those
  artifacts.
- **Shape change — encoding.** v1.0 choice vectors are 180 dims narrower (no
  `becomes_unplayable` stripe). The same shim overrides `encode_choices` (strips
  the stripe via `np.delete` after live encoding) and `_choice_embed_offsets`
  (returns `becomes_unplayable=None`; shifts `kept_multihot` offset left by 180
  when `include_setup`). `encoding_dims_for_era` now branches: v1.0 returns
  `choice_dim = choice_feature_dim(spec) − CHOICE_BECOMES_UNPLAYABLE_DIM`; v1.1+
  returns live widths. This is an encoding FRESH change; no separate `v1_1.py`
  shim was needed because no v1.1 training run existed at the time.
- **No LFS fixture.** The only in-production v1.0 artifacts at the bump had
  `trunk_final_activation=null` and were discarded in favour of a fresh training
  run. `tests/test_compat_v1_0.py` exercises both shim behaviors via a
  freshly-built weight tensor rather than a saved checkpoint.
- **User action required.** To get the intended relu after both trunk and choice
  encoders, set `final_activation = "relu"` globally in `TrainConfig` before
  starting a new training run; no config-format change is needed.

**3. Setup-encoding pooling migration (setup encoding).** The setup net's
card-set embeddings are migrated to match the main net's hand-pooling path:

- **Kept-card set** (`kept_cards`): was embedded via `hand_model.embed_card_set`
  using the setup net's own hand encoder (`hand_embed_width = N`). Now embedded
  via `hand_model.pool_card_set` using the shared card table (`pooled_hand_width =
  2N+1 = 129` for CONCAT_MAX_SUM). When `use_distinct_hand_model=True`, the prior
  distinct-encoder path is preserved.
- **Tray** (`tray`): the hardcoded tray-set embedding (`hand_model.embed_card_set`
  over the tray multihot) is dropped. The tray now contributes only the three
  per-slot card-table rows: `TRAY_SIZE × N = 3N = 192` dims, matching the main
  net's state tray with `tray_set_embedding=False`.
- **`SetupEncoding.include_playable_kept_cards`** defaults to `True`: the
  food-agnostic playable-kept-card set embedding (embedded the same pooling way)
  is now included by default. `total_dim` of a default `SetupEncoding()` is
  `488`; `setup_readout_input_dim` with a default main arch is `575`
  (= 125 passthrough + 2×129 sets + 3×64 tray).

These are **setup-artifact-only** shape changes. No main-net compat shim is
needed. Any existing v1.0/v1.1 setup checkpoints (`setup_config.json` +
`setup_*.pt`) are incompatible and must be discarded — no v1.1 setup training runs
existed at the time of this change.

### v1.0 — clean-break baseline

The 1.0 MAJOR bump. A MAJOR bump is the sanctioned escape hatch that drops the
accumulated shims and deletes the old fixture sets wholesale; it is its own
user-approved decision, never a side effect of another change. What 1.0 did:

- **Dropped every pre-1.0 compat shim and fixture set.** The
  `wingspan.compat.v0_0` … `v0_7` modules and the `tests/data/compat/v0.*/`
  fixtures are gone. `check_artifact_compatible` now refuses every pre-1.0 (0.x)
  artifact as a different MAJOR — there is no 0.x → 1.0 load path. The full
  per-version 0.1–0.8 changelog that used to live here is recoverable from git
  history (it ran from "0.0 initial era" through "0.8 food-gain `becomes_playable`
  ignores eggs").
- **Removed the dead code paths the shims existed to support.** The distinct-hand
  encoder and `tray_set_embedding` — together with the `use_distinct_hand_model`
  flag, the `_check_tray_set_embedding` validator, and their two `ShapeKey` slots
  — are gone. The main net now always takes the pooled hand path (`HandPooling`,
  unconditional), and `StateEmbedOffsets` dropped its `hand_summary` field (now
  three offsets). The setup net's own `hand_encoder_layers` / `hand_embed_dim` /
  `hand_embed_width` are **kept** (it still builds a hand encoder; they remain in
  `setup_architecture_key`).
- **Deleted the unused `BirdPowerPickBirdFromHandDecision` slot** from
  `ALL_DECISION_CLASSES` and `_DECISION_FAMILY` — a real FRESH change that shrinks
  the decision-type one-hot by 1. `num_families` is unchanged (`DRAW_BIRD` stays,
  now serving `DrawCardsPickSourceDecision` alone).
- **Removed pre-1.0 on-disk tooling.** The flat (≤0.4) config format and its
  reshape/migration (`_reshape_flat_to_nested` / `_is_nested_config`), and the
  legacy `model_config.json` / `setup_config.json` / `process_*.json` sidecar
  readers + writers (and their name constants `MODEL_CONFIG_JSON`,
  `SETUP_CONFIG_JSON`, `SETUP_CONFIG_JSON_LEGACY`, `PROCESS_PREFIX`,
  `PROCESS_GLOB`), are gone — the unified `run_config_<stamp>.json` is the only
  run-dir config artifact. The compat-only constants `N_ROUNDS` / `MAX_ACTION_CUBES`
  were dropped from `encode/layout.py` (live game constants like `N_PLAYER_TURNS`
  stay). The in-memory descriptors `runmeta.ModelConfig` / `setup_runmeta.SetupConfig`
  are **kept** (they describe a loaded run).

The versioning *machinery* is intact: the `compat` package
(`compat.encoding_dims_for_era`), the `PolicyValueNet.class_for_version` and
`version.adapt_encoding_for_version` seams, and `RunConfig.encoding_version`
era-pinning are all wired up. With 1.0 being the first same-MAJOR era, `class_for_version`
fell straight through to the live encoders — the first v1.x shim is `compat.v1_0`,
introduced in v1.1.

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
`opponent.pt` / `setup.pt`, and the dated `run_config_<stamp>.json`. An
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

The `wingspan.compat` package is **currently empty** — the 1.0 MAJOR bump dropped
every pre-1.0 shim, leaving only the inert dims-router seam
(`compat.encoding_dims_for_era`). Each future MINOR bump adds one module back
(`v1_<N>.py`), one per superseded same-MAJOR era. Shape: `if artifact older than
the change: regenerate the encoding without the new field`. Inference call sites
must encode through the net (`net.encode_state` / `net.encode_choices`), never by
pairing the live encoder with a spec by hand — that is what lets a compat-era net
carry its own geometry.

**Compat is version-number-specific checks, never config flags.** Do not add
`TrainConfig` axes to toggle old behaviors.

## MINOR bumps (FRESH changes)

A MINOR bump is required for every FRESH-type change — any change that alters
a tensor shape — and must:

1. Bump `MODEL_VERSION` in `wingspan/version.py`.
2. Add the version-specific shim in `wingspan/compat/v<X_Y>.py`.
3. Capture a new fixture set under `tests/data/compat/v<X.Y>/` from a run at
   the new version (the first `v1.<N>` set re-establishes the expected shape and
   its README, since the pre-1.0 sets were deleted at the MAJOR bump).
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
  the embedded `RunConfig` (the `run_config_<stamp>.json` payload) — travels
  *with the artifact*. It rehydrates exactly because the value is read back from
  the frozen file, so it needs no version gate. This is why those knobs are
  safely REGIME: not because they preserve shape, but because the artifact
  carries its own copy.
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
returns a `model.StateEmbedOffsets` named tuple carrying every offset
`_embed_state` reads, and each shim freezes the whole tuple — different stripes
precede each, so they do not share one delta. (At 1.0 that tuple is three
offsets — card-index, hand, decision; the fourth, `hand_summary`, was retired
with the distinct hand model, since the pooled-only main net no longer slices the
hand-summary stripe out of its continuous trunk input.)

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
  checkpoint embeds its `config` and its `version`. A run directory carries one
  dated `run_config_<stamp>.json` (the in-memory `ModelConfig` / `SetupConfig`
  descriptors are *derived* from it); the pre-1.0 legacy sidecars
  (`model_config.json` / `setup_config.json` / `process_*.json`) and their
  presence-dispatch reader were removed at the 1.0 MAJOR bump. Never add an
  "assume compatible" branch, a second on-disk location for the same datum, or a
  ghost entry kept only for index stability.
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
  plain. The pre-1.0 fixture sets were deleted at the 1.0 MAJOR bump; the first
  `tests/data/compat/v1.<N>/` set (captured for the next FRESH MINOR, with its
  own README) re-establishes the shape every later set must follow.
- Crash-survivability tolerance is fine and stays (e.g. `metrics_log` skipping
  a truncated final line): it guards the *current* format against interruption,
  not an old format against age.
