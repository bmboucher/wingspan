# Artifact versioning and checkpoint compatibility

Every persisted artifact (the dated `run_config_<stamp>.json` run descriptor and
every `.pt` payload) is stamped with a `MAJOR.MINOR` **artifact version**
(`wingspan.version.MODEL_VERSION`, currently **`1.7`**). This is distinct from
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

### v1.7 — optimistic egg-bonus potential pricing (current)

A **behavior-only** MINOR FRESH bump — no tensor shape changes on either net —
that is the bonus-card twin of v1.6's `goal_affinity` change: the bonus
*potential* counters (how many not-yet-played birds could still qualify a
bonus card) became optimistic about the egg-counting dynamic cards. A bird
whose `egg_limit` reaches the card's threshold — Breeding Manager at 4,
Oologist at 1 — now counts as potentially qualifying, via the new shared
counter `scoring.bonus_potential_count` (its `hand_sized` flag carries the
pre-existing Visionary Leader full-hand special case, which applies to
hand-like sources only — the tray asymmetry is deliberately preserved).
Pre-1.7 encoders counted only the static `bonus_categories` tag, which no
dynamic card carries, so the egg cards' potentials read identically 0 —
during setup, a keep of two 4-egg-capacity birds showed `min_affinity` 0
against a dealt Breeding Manager. Board-side counts
(`scoring.bonus_qualifying_count`) were always dynamic-aware and are
unchanged.

Values move at unchanged dims and offsets on both nets:

- **Main net** — the `bonus_value` stripe's `hand_potential` /
  `tray_potential` scalars, on in-game `BonusCardChoice` rows and
  bonus-carrying `SetupChoice` rows (`choice_encode._fill_bonus_value`). The
  board trio (qual/stepped/linear) reads actual state and is unchanged.
- **Setup net** — the split-mode `bonus_card_affinity` min/max pair and the
  folded-mode `kept_bonus_value` 4-vector (`setup_model.encode`, via
  `_kept_qual_for_bonus`); its stepped/linear VP are priced at the qual
  count, so they move with it.

**Shim.** `wingspan.compat.v1_6` follows the behavior-only era shape (the
v1_4 / `SetupNetV1_5` precedent): `PolicyValueNetV1_6` overrides only
`encode_choices` — live encode, then
`choice_encode.refill_bonus_value_potentials_static` rewrites the two
potential scalars of each bonus-carrying row with the static pricing (the
refill re-runs the generic static predicate, `bonus_potential_count_static`,
so a Visionary Leader row's full-hand count survives byte-identically) — and
`SetupNetV1_6` overrides only `encode_candidate`, applying
`setup_model.encode.refill_bonus_pricing_static` (both bonus-block shapes, at
unchanged offsets). No `encoding_dims_for_era` branch, no offset or layout
override. `compat.v1_5.PolicyValueNetV1_5` / `SetupNetV1_5` re-chain to
subclass the v1_6 classes, so every era <= 1.5 freezes the static potentials
too: the choice refill runs at full live width inside the `super()` chain and
targets `layout._OFF_BONUS_VALUE` — before `becomes_playable`, and therefore
before every column any older shim strips — so the whole v1_5→v1_4→v1_3→v1_0
chain composes with no offset math.

The golden fixture (`tests/data/golden_n2.json`) was recaptured — bonus-pick
and setup rows change hashes, exactly the v1.5/v1.6 pattern;
`state_dict_shape_n2.json` is untouched (no shapes move). A committed LFS
checkpoint fixture remains deferred, as for every prior era:
`tests/test_compat_v1_6.py` builds v1.6-stamped nets and round-trips them
through the production loaders. **User action: none** — pre-1.7 checkpoints
load and compute identically via the shim chain.

**Fold-in: spend-decision food routing (no version bump).** Folded into this
same v1.7 era after the fact — no MINOR bump, since no v1.7 artifact yet
exists to protect. Single-token `FoodChoice` rows offered by a spend decision
(`SpendFoodDecision`'s discard, `SpendFoodForEggDecision`'s grassland trade)
were being routed into the `gain_food` stripe like every other food row, even
though the player is paying food away; they now route to `pay_food` at the
usual `1 / _PAYMENT_COUNT_SCALE` per-unit scale, alongside a label fix in
`wingspan.reporting.humanize` (the grassland spend sub-event read "Gains
{food}" instead of "Spends {food}"). Era-gated by extending
`compat.v1_6.PolicyValueNetV1_6.encode_choices` with a second refill,
`choice_encode.refill_spend_food_gain_routing`, alongside the existing bonus
potential refill — same shape-preserving, offset-preceding-the-tail pattern,
so the v1_5→v1_3→v1_0 chain composes unchanged. `tests/data/golden_n2.json`
was regenerated in place (the 6 `SpendFoodForEggDecision` records in the
fixture change hash; no other record moves — the fixture carries no standalone
`SpendFoodDecision` rows). Folding into 1.7 without a bump is sanctioned
because no v1.7 run has ever been trained or checkpointed.

### v1.6 — `goal_delta_ignoring_eggs` choice stripe + setup `goal_affinity` egg pricing

A **shape + behavior** MINOR FRESH bump landing two views of the same scoring
upgrade — "assume this bird is eventually played and egg-populated to
whatever level best advances a round goal" — under one `MODEL_VERSION` bump
because they were developed together: a shape-changing addition on the main
net's choice side, and a behavior-only value change on the setup net's
`goal_affinity` stripe.

**Main-net half — new `goal_delta_ignoring_eggs` choice stripe (shape).** The
existing `goal_delta` stripe only prices the *play-instant* delta — a freshly
played bird has no eggs, so every egg-driven category (`eggs_<habitat>`,
`eggs_<nest>`, `*_birds_with_eggs`, `egg_sets_3habitats` — 12 categories in
all) reads 0 on every bird-card row, even when playing that bird is obviously
a step toward the goal. The new 8-dim `goal_delta_ignoring_eggs` stripe prices
the same 4 round goals under the played-and-optimally-egg-populated
hypothesis instead: a bird's contribution is priced as if it is eventually
played (a slot must be open in one of its card habitats — the guard lives in
`scoring.goal_vp_delta_for_bird_with_eggs`) and its eggs are set to whatever
level best advances the goal (up to `bird.egg_limit`; star nests wild via
`cards.nest_matches`). It is appended as the new *last base* stripe
(immediately after `resets_feeder`, the prior last base stripe), filled by a
separate featurizer (`choice_encode._fill_goal_delta_ignoring_eggs`, built on
`scoring.goal_count_delta_for_bird_with_eggs` /
`goal_vp_delta_for_bird_with_eggs`) at the same three bird-card row call sites
that already fill `goal_delta`: `_featurize_bird` (hand keeps /
`BirdChoice`), `_featurize_play_bird` (committed landing habitat /
`PlayBirdChoice`), and `_featurize_draw_source` (tray rows /
`DrawSourceChoice`). The existing `goal_delta` stripe's play-instant semantics
are completely untouched — this is a pure addition, not a value change to the
existing stripe. Choice width grows by 8: N=2 base 509 → 517, `include_setup`
693 → 701, N=3 base 512 → 520 (state dims unchanged).

**Setup-net half — `goal_affinity` egg pricing (behavior only).** The setup
encoder's `goal_affinity` stripe (`setup_model.encode.encode_setup_candidate`,
block 7) priced kept cards via `scoring.goal_count_delta_for_bird` summed per
card — the same play-instant pricing, zero for every one of those 12
egg-driven categories, since nothing has eggs at setup time. It now prices via
`scoring.goal_affinity_for_kept`: the played-and-optimally-egg-populated
hand-level bound (no committed habitat, no board context).
`egg_sets_3habitats` is handled specially — it is not additive across birds
(each bird can only land in one habitat), so `_best_kept_egg_sets`
brute-forces the best habitat assignment across the whole keep (kept hands
are capped at 5 birds, so at most 3⁵ = 243 combinations). Shape is unchanged
(still 4 scalars, same offsets); values may now exceed 1 (the ÷5
normalization is a heuristic, not a hard cap).

**Why an era gate for both halves.** The rehydration guarantee: a pre-1.6 net
*trained against* 509-wide choice vectors (main) or the egg-blind
`goal_affinity` pricing (setup) must keep computing identically on reload —
feeding it the new stripe, or the new pricing, would change its outputs in
ways it never learned to interpret.

- **Shim mechanics — main net.** `wingspan.compat.v1_5.PolicyValueNetV1_5`
  mirrors the `v1_3` narrowing shape, applied to one stripe: `encode_choices`
  calls the live encoder then `np.delete`s the appended columns;
  `_choice_embed_offsets` keeps `bird_id` / `becomes_playable` /
  `becomes_unplayable` (they precede the stripe) and shifts `kept_multihot`
  left by 8; `_build_choice_encoder` builds at `_true_choice_dim()`
  (`self.spec`-derived, not the passed `choice_dim`); `raw_choice_stripe_layout`
  drops the stripe via `VectorLayout.without_stripes`. `PolicyValueNetV1_4` —
  previously a live-geometry, value-only shim — now **re-chains** to subclass
  `PolicyValueNetV1_5` directly instead of the live net, so era 1.4's choice
  geometry is 8 dims narrower than live too; its own `goal_delta`
  habitat-agnostic refill still targets `layout._OFF_GOAL_DELTA`, an offset
  well before the stripped tail, so it is unaffected by the re-chain.
  `compat.v1_3.PolicyValueNetV1_3._true_choice_dim` now **composes via
  `super()`** (`super()._true_choice_dim() - CHOICE_RESETS_FEEDER_DIM`) instead
  of recomputing the width absolutely from `self.spec` — so the inherited v1_5
  narrowing is never silently dropped, and any future tail-stripe narrowing an
  ancestor era applies automatically. `compat.encoding_dims_for_era` gains a
  `minor <= 5` branch (`choice_dim -= CHOICE_GOAL_DELTA_IGNORING_EGGS_DIM`, 8)
  — the newest, and so broadest, choice-narrowing branch, composing with the
  existing `minor <= 3` (food-unlock state stripes + `resets_feeder`) and
  `== 0` (`becomes_unplayable`) branches. `class_for_version` routes era 1.5 to
  `PolicyValueNetV1_5`.
- **Shim mechanics — setup net, the first `SetupNet` compat seam.** Before
  this era, `players.loaders.load_setup_net` built the live `SetupNet`
  unconditionally ("no pre-1.0 shims remain") because no setup-net behavior
  had ever needed freezing — `SetupNet.class_for_version` did not exist (see
  the corrected v1.2 entry below). v1.6 is the first version where a setup
  artifact's *behavior* changes on rehydration even though its geometry does
  not, so this era adds the seam for real:
  - `training.setup_net.SetupNet.encode_candidate(candidate, context) ->
    np.ndarray` — a new instance method, the setup-side analogue of
    `model.core.PolicyValueNet.encode_state` / `encode_choices`. The base
    implementation just calls the free `setup_model.encode_setup_candidate`
    function with `self.encoding`; a compat subclass overrides it to carry
    its own frozen pricing. Every call site that holds a `SetupNet` instance
    now routes through this method instead of pairing the free function with
    an encoding by hand: `players.factory._compute_setup_scores_and_probs`
    (inference) and `training.collect.play_game_with_setup` / `_choose_setups`
    (recording and scoring during collection). The free function remains the
    fallback when no net is in hand (`setup_policy_net is None`).
  - `training.setup_net.SetupNet.class_for_version(artifact_version) ->
    type[SetupNet]` — mirrors `model.core.PolicyValueNet.class_for_version`;
    routes eras <= 1.5 to `wingspan.compat.v1_5.SetupNetV1_5`.
  - `wingspan.compat.v1_5.SetupNetV1_5` — overrides only `encode_candidate`:
    calls the live encoder via `super()`, then
    `setup_model.encode.refill_goal_affinity_static` overwrites the 4-scalar
    `goal_affinity` stripe with the pre-1.6 static (egg-blind) pricing, in
    place. No geometry override — the setup change is values-only, so this
    class joins no dims-router branch.
  - Setup-net **construction** is now era-routed everywhere a fresh instance
    is built for an existing run, mirroring how the main net is already
    era-pinned: `training.loop_setup.build_setup_net` (training/resume),
    `training.mp_collect._worker_init` (collection workers), and
    `players.loaders.load_setup_net` (inference) all call
    `SetupNet.class_for_version(...)` before constructing.
  - `training.config.RunConfig.setup_architecture_key` now **leads with
    `encoding_version`**, mirroring `architecture_key` (which already leads
    with the era for the same reason) — so a same-shape `setup.pt` from a
    different era (e.g. a v1.5 setup net under live code) reads as
    incompatible instead of a silent shape coincidence, and
    `loop_setup.setup_architecture_matches` catches it at resume too.
- **Golden fixture recaptured.** `tests/data/golden_n2.json` and
  `tests/data/state_dict_shape_n2.json` pin live-encoder bytes and shapes, so
  both were re-captured at v1.6.
- **No LFS fixture (deferred, as for v1.0 / v1.3 / v1.4), for both halves.**
  `tests/test_compat_v1_5.py` builds v1.5-era nets (main and setup), saves
  them with a v1.5 stamp, and round-trip-loads them through the production
  `load_policy_net` / `load_setup_net` paths.
- **User action.** None: a pre-1.6 run resumes era-pinned via both shims
  (main net choice geometry, setup net `goal_affinity` pricing); a run
  started on 1.6 gets both new signals.

### v1.5 — habitat-conditioned play-bird `goal_delta`

A **behavior-only** MINOR FRESH bump — the first era gate that changes **no
tensor shape**, only stripe *values*. The `PlayBirdChoice` featurizer's
`goal_delta` stripe is now conditioned on the row's committed landing habitat:
a `birds_<habitat>` round goal moves only on the row that actually plays the
bird into that habitat (`scoring.goal_count_delta_for_bird`'s new
`play_habitat` parameter, threaded through `_fill_goal_delta` from
`_featurize_play_bird`).

**The bug this fixes.** A `birds_<habitat>` goal was priced from the bird's
*card* habitats, not the row's landing habitat — so a two-habitat bird
(the Peregrine Falcon shape, grassland/wetland) claimed the "[bird] in
[wetland]" goal's `count_delta`/`vp_delta` on its grassland row too, making
the two placements indistinguishable on the goal stripe. Candidate rows with
no committed placement (hand / tray / setup keeps) keep the optimistic
any-card-habitat bound — those semantics were correct and are unchanged
(`play_habitat=None`).

- **Behavioral change — encoding (FRESH, shape-preserving).** Every dim,
  offset, and layout is unchanged, so `encoding_dims_for_era` has **no 1.4
  branch** (era 1.4 dims equal live) and the shim overrides nothing geometric.
  `compat.v1_4.PolicyValueNetV1_4` overrides `encode_choices` only: after live
  encoding, each play-bird row's `goal_delta` stripe is re-filled with the
  habitat-agnostic pricing (`choice_encode.refill_goal_delta_habitat_agnostic`)
  — the value-level analogue of the older shims' `np.delete` column strips.
  `class_for_version` routes era 1.4 there; `PolicyValueNetV1_3` now
  **inherits** `PolicyValueNetV1_4` (and `PolicyValueNetV1_0` inherits
  `V1_3`), so every pre-1.5 era freezes the old pricing — the refill runs at
  live column offsets inside the `super().encode_choices` chain, before the
  older shims' column strips shift anything.
- **Why an era gate for a bug fix.** The rehydration guarantee: a 1.4 net
  *trained against* the both-rows pricing, so feeding it corrected stripes
  would change its policy outputs. Old artifacts never adopt new behavior,
  buggy or not. `architecture_key` leads with the era, so a same-shape 1.4 run
  still reads as its own era and resumes era-pinned as the shim class.
- **Setup net unaffected.** The setup encoder prices keeps with no committed
  habitat (the unchanged `play_habitat=None` bound), so setup artifacts stay
  loadable and there is no setup-side shim.
- **Golden fixture recaptured.** `tests/data/golden_n2.json` pins live-encoder
  bytes, so it was re-captured at v1.5 (`python tests/golden_capture.py`): a
  subset of the recorded hashes changed; the decision sequence is unchanged.
- **No LFS fixture (deferred, as for v1.0 / v1.3).** `tests/test_compat_v1_4.py`
  builds a v1.4-era net, saves it with a v1.4 stamp, round-trip-loads it
  through the production `load_policy_net` path, and pins the frozen behavior
  (both rows of a two-habitat bird priced identically on `goal_delta`; every
  other stripe byte-identical to live).
- **User action.** None: a pre-1.5 run resumes era-pinned via the shim; a run
  started on 1.5 prices play-bird rows at their landing habitat.

### v1.4 — food-unlock state stripes + `resets_feeder` choice stripe

A **main-net encoding** MINOR FRESH bump that lands **two independent encoding
changes together** (developed in parallel, folded into one era because no v1.4
training runs existed yet). Both widen the main net; each is stripped for pre-1.4
artifacts by the same `compat.v1_3` shim.

**(a) Food-distance-to-playable state stripes** — the **first same-MAJOR change to
alter the state vector width**. Two 5-wide pass-through stripes are appended to the
continuous state prefix (immediately after `food_opp`):

- `hand_food_unlock_me` — per food, the smallest count that would newly unlock a
  bird in the POV player's **hand**.
- `tray_food_unlock_me` — the same over the face-up **tray**, scored as if those
  cards were in hand (against the POV player's own food + board).

Both are computed by `engine.playability.min_food_to_unlock`: a bird is
"unlockable" when it is currently unplayable for a food reason but has an open
matching habitat slot (egg cost ignored); affordability uses the full engine rule
(1-for-1, 2-for-1 substitution, wild). State width grows by 10.

**(b) `resets_feeder` choice stripe.** Under `combine_gain_food`,
`actions.combined_feeder_gain` folds the birdfeeder reroll into the
`FoodSubsetChoice` menu: a partial take (fewer than `n` dice), or a full take that
empties the feeder, commits the engine to a reroll and re-pick. Previously such an
option was byte-indistinguishable from a plain smaller gain — only a lower
`gain_food` count. v1.4 adds a 1-dim `resets_feeder` stripe as the last *base*
choice stripe (immediately after `becomes_unplayable`, before the conditional setup
stripes); the engine sets `FoodSubsetChoice.resets_birdfeeder` and the featurizer
lights the bit. Choice width grows by 1.

- **Shape change — encoding (FRESH).** Pre-1.4 state vectors are 10 dims narrower
  and choice vectors 1 dim narrower. `compat.v1_3.PolicyValueNetV1_3` strips **both**
  additions after live encoding (`np.delete`): the two state stripes (freezing the
  pre-1.4 `StateEmbedOffsets` — `card_index` / `hand_multihot` / `decision_type` each
  shifted left by 10) and the `resets_feeder` column (keeping `becomes_unplayable`,
  shifting only `kept_multihot`). Its `_build_trunk` / `_build_choice_encoder` derive
  the block widths from `self.spec` (live minus the stripes) via `_true_state_dim` /
  `_true_choice_dim` rather than the passed dims, so the shim is correct whether the
  constructor is handed era dims (the load path) or live dims (test default).
  `encoding_dims_for_era` gains its **first state-dim branch**: for every pre-1.4
  same-MAJOR era, `state_dim -= 10` and `choice_dim -= 1`.
- **Setup net unaffected.** The setup model's choice encoding is independent of the
  main choice width, so setup artifacts stay loadable and there is no setup-side shim.
- **Routing.** `class_for_version` routes eras 1.1-1.3 to `PolicyValueNetV1_3`.
  `PolicyValueNetV1_0` **inherits** `PolicyValueNetV1_3`, so v1.0 artifacts strip the
  state stripes and `resets_feeder` too (they predate all three), on top of the v1.0
  trunk-final fallback and `becomes_unplayable` choice-stripe removal — composed via
  `super()` chaining and `_true_choice_dim` narrowing.
- **No LFS fixture (deferred, as for v1.0).** No in-production v1.3 checkpoint was
  preserved. `tests/test_compat_v1_3.py` instead builds a v1.3-era net, saves it
  with a v1.3 stamp, and round-trip-loads it through the production
  `players.loaders.load_policy_net` path (era dims → constructor → `load_state_dict`
  → forward) — the load path a real checkpoint takes, which fails on any
  double-subtraction of either stripe width.
- **User action.** None for existing runs: a pre-1.4 run resumes era-pinned at its
  own era (main net unchanged there, no new signals) via the shim. A run started on
  1.4 gets both new signals.

### v1.3 — setup net → two-tower (state trunk + choice trunk)

A **setup-artifact-only** FRESH bump. v1.2 made the value head a state-only `V(s)`
but left the net asymmetric: the policy head read a single *fused* state ⊕ action
vector through one trunk, the value head read a state-only vector through a
separate trunk, and the two shared nothing. v1.3 restructures `SetupNet` into a
**two-tower actor-critic mirroring the in-game `PolicyValueNet`**: a shared **state
trunk** (`trunk_layers`) encodes the action-independent stripes into a `state_enc`
that feeds *both* heads, and a separate **choice trunk** (`choice_layers`) encodes
the action stripes into a `choice_enc`. The value head reads `state_enc` only
(still a true `V(s)`); the policy head reads `cat(state_enc, choice_enc)`. The
value and policy heads now share a learned state representation — the point of the
two-tower design — while the `V(s)` baseline property is preserved (the value head
output is a pure function of state). `SetupArchitecture`'s layer-width fields are
renamed to mirror `ModelArchitecture` exactly: `trunk_layers` (state trunk),
`choice_layers` (choice trunk), `head_layers` (policy head), `value_layers` (value
head), defaulting to `(128,)` state / `(128,)` choice trunks.

- **Main net unaffected.** No `compat.v1_2` shim and no `encoding_dims_for_era`
  entry — 1.3 main-net dims equal 1.2 equal live.
- **Setup checkpoints are discarded, not migrated.** The submodule set and the
  policy head's first `Linear` change, so a v1.2 `setup.pt` no longer fits. On
  resume, `loop_setup.maybe_resume_setup`'s shape-mismatch path rebuilds the setup
  net fresh (with an `ALARM`); `players.loaders.load_setup_net` refuses an
  incompatible `setup.pt` with a clear "retrain the setup model" error. The main
  net resumes normally.
- **In-flight runs.** A run already in progress keeps all main-net progress; only
  its setup model restarts fresh at the next resume.

### v1.2 — setup value head `Q(s,a)` → state-only `V(s)` + setup↔in-game return reconciliation

A **setup-artifact-only** FRESH bump. The separate setup model's value head was a
per-candidate critic reading the *fused* state ⊕ action vector — it learned
`Q(s, a_chosen)`, so the actor-critic advantage `margin − Q(s,a)` self-cancelled
(its conditional mean given the chosen action is ≈ 0) and the setup policy never
left a near-uniform band. v1.2 splits the value head into a **state-only** trunk
reading only the action-independent deal stripes (tray, birdfeeder, round goals,
bonus-on-offer) → a true `V(s)` baseline, so `advantage = target − V(s)` carries a
real gradient. The value head's first `Linear` therefore shrinks (≈304 vs ≈568 by
default), changing the setup-net weight shape.

A second, shape-preserving (REGIME) change reconciles the setup target with the
in-game return: the setup keep is the `t=0` decision of the same game whose
in-game steps are `t>0`, so its target is now the in-game return kernel
(`wingspan.training.returns.setup_return`) evaluated at that anchor — honoring
`reward_mode` / `reward_discount` / `reward_basis` / `end_game_bonus` exactly as
the main learner does. At the default config the target is byte-identical to the
old `margin / score_norm`.

- **Main net unaffected.** The main net's encoding and topology are unchanged, so
  there is **no `compat.v1_1` shim** and no `encoding_dims_for_era` entry — 1.2
  main-net dims equal 1.1 equal live (`encoding_dims_for_era` already returns live
  dims for any same-MAJOR era ≥ 1.1). A run pinned to an earlier same-MAJOR era
  keeps training its main net at that era.
- **Setup checkpoints are discarded, not migrated.** A `Q`-trained fused value
  head has no faithful `V(s)` reconstruction, so there was no setup shim at
  the time (no `SetupNet.class_for_version` seam existed until v1.6, when the
  first behavior-only setup change needed one). On resume,
  `loop_setup.maybe_resume_setup`'s shape-mismatch path rebuilds the setup net
  fresh (with an `ALARM`); `players.loaders.load_setup_net` refuses an
  incompatible `setup.pt` with a clear "retrain the setup model" error rather than
  an opaque torch size error. The main net resumes normally.
- **In-flight runs.** A run already in progress stays era-pinned at its own
  version and keeps all main-net progress; only its setup model restarts fresh at
  the next resume (the setup policy was near-random, so nothing of value is lost).

### v1.1 — uniform final-activation inheritance + `becomes_unplayable` stripe + setup-encoding pooling

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

The `wingspan.compat` package holds one module per superseded same-MAJOR era
(`v1_0`, `v1_3`, `v1_4`, `v1_5`) plus the dims-router seam
(`compat.encoding_dims_for_era`); each future MINOR bump adds one more
(`v1_<N>.py`). Shape: `if artifact older than the change: regenerate the
encoding without the new field` — or, for a behavior-only era (v1.5's main-net
value change; v1.6's setup-net value change), regenerate the prior stripe
values. Inference call sites must encode through the net (`net.encode_state` /
`net.encode_choices`, and — since v1.6, the first era a setup artifact's
behavior needed freezing — `SetupNet.encode_candidate`), never by pairing the
live encoder with a spec by hand — that is what lets a compat-era net carry
its own geometry.

**Compat is version-number-specific checks, never config flags.** Do not add
`TrainConfig` axes to toggle old behaviors.

## MINOR bumps (FRESH changes)

A MINOR bump is required for every FRESH-type change — any change that alters
a tensor shape, or any code-carried behavior change that would make a
rehydrated artifact compute differently even at unchanged shape (v1.5; see
"FRESH vs REGIME" below) — and must:

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

`combine_gain_food` (the collapsed multi-food gain;
`EngineConfig.combine_gain_food`, default off) is the textbook *safe* case. It is
**config-carried** (the flag travels in the frozen `RunConfig`) **and** purely
additive: it widens `GainFoodDecision` with a new `FoodSubsetChoice` shape and
adds a new featurizer (`_featurize_food_subset`) that fills the same 7-slot
`gain_food` stripe as a count vector — the existing `_fill_gain_food` /
`_featurize_food` one-hot path is untouched. An old artifact rehydrates with the
flag off, so the new featurizer is dead code for it and it computes identically.
No tensor shape changes (it stays out of `architecture_key`), no `MODEL_VERSION`
bump, no compat shim. This is the difference from the 2026-06-1x bugs below:
those mutated a shared code-carried path that *every* artifact reads;
`combine_gain_food` only ever runs new code behind a config-carried flag.

The v1.4 `resets_feeder` stripe is the instructive counter-case: it annotates the
*same* `combine_gain_food` `FoodSubsetChoice` rows, but it is a new always-present
choice-vector column, not config-carried geometry — so even though the bit only ever
carries a value for combined-gain rows, adding the column widened `choice_dim` for
every artifact and *did* take a FRESH `MODEL_VERSION` bump + `compat.v1_3` shim.
Config-carried *behavior* behind a flag is REGIME; a new *stripe* is FRESH no matter
how narrow its trigger.

### `num_players`: a third pattern — config-carried, shape-changing, still no bump

`ArchitectureConfig.num_players` / `EncodingSpec.num_players` /
`ModelArchitecture.num_players` (3-4 player support, default **2**) is a third
pattern, distinct from both cases above. Like `combine_gain_food` it is
config-carried and default-reproducing: the field travels in the frozen
`RunConfig`/`model_config.json`, defaults to 2, and at the default every dim,
offset, stripe name, and encoded value is byte-identical to the pre-`num_players`
encoding (proved by `tests/test_encoding_golden_n2.py`'s per-decision hash replay
and the frozen stripe table in `tests/test_encoding_layout_nplayers.py`) — so N=2
needs no `MODEL_VERSION` bump and no compat shim, exactly like
`combine_gain_food`. But unlike `combine_gain_food`, a *non-default* value **does**
change tensor shape: `state_dim`/`choice_dim` grow with per-opponent stripe
replicas (`food_opp2`, `board_opp2`, ...), a `turn_position` state stripe, and a
`player_select` choice stripe (all N>=3-only), and the board-attention path's
input-facing Linears widen accordingly. This is the `use_board_attention` /
`board_attention_heads` / `board_attention_shared` precedent: `num_players`
joins `ShapeKey` (via `ModelArchitecture.num_players`) and `architecture_key`
(via `ArchitectureConfig.state_dim`/`choice_dim`, which move with it), so a
mismatched seat count refuses cleanly through the existing resume/load gates —
never a silent shape coincidence. (Each field `ShapeKey` picks up this way —
most recently `board_attention_shared` — is safe to add as a *required* field
rather than a defaulted one: a `ShapeKey` is only ever constructed from a live
`ModelArchitecture` and compared with `==`, never serialized to disk on its
own; it would become its own compat surface only if something ever wrote one
out directly.)

The consequence for compat: **N>=3 artifacts are fresh nets, forever, never
compat-shimmed.** Every existing compat era (`v1_0`, `v1_3`, and every future
`v1_<N>`) predates `num_players` by construction — no historical checkpoint ever
set it to anything but the implicit 2 — so `compat.encoding_dims_for_era` raises
`version.IncompatibleArtifactError` outright for `spec.num_players != 2` rather
than attempting to freeze a shape no shim will ever need to reproduce. This is
also why `training.config.validate_launchable` carries a *temporary* blocker on
`num_players > 2` (removed once the training pipeline itself is seat-count-generic,
tracked in the N-player plan's later stages): the encoding and model layers accept
N>=3 today, but nothing downstream has been asked to *train* one yet.

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

And a third recurrence on 2026-07-12, this time on the *decode* side: the
game-log HTML encoding viewer displayed recorded state vectors by walking the
live `raw_state_stripe_layout`, so a v1.3 run's game rendered under v1.4 code
showed every stripe past the food-unlock insertion point shifted 10 columns —
phantom hand birds, the tray leaking into board slots, the decision-type bit
decoding as a bird. The model itself was fine (its shim encoders are
era-frozen); only the human-facing panel lied. The structural fix mirrors the
embed-offsets one: the net owns layout descriptors too
(`raw_state_stripe_layout` / `raw_choice_stripe_layout`, overridden per shim via
`VectorLayout.without_stripes`), agents stamp them onto the `PolicyAnnotation`,
and the viewer decodes with the stamped layout — plus a hard length check so a
mismatched pairing raises instead of silently mis-labelling. A new FRESH stripe
change must extend the shim's layout overrides alongside its encoder strips.

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
