# compat — Version-specific artifact shims

One module per superseded artifact era. Deleted wholesale at a MAJOR version
bump. See `docs/VERSIONING.md` for the full compat policy (FRESH vs REGIME,
when a MINOR bump is required, fixture-set rules).

The inference loaders (`players.loaders`, `training.runmeta.read_model_config`)
route by the artifact's embedded `version` field and call the appropriate shim
before handing off to the live model. Never add shims directly to loaders —
each era belongs in its own module here.

**Freeze _all_ geometry the net derives, not just `encode_state`.** A shim's job
is that the rehydrated net computes identically to the saved one. The
frozen-vector helpers (`encode_*_v0N`) are only half of it: the net also derives
*slice offsets* from the live layout (`_embed_state` / `_embed_choices`), and
those must move with the era too. When live and frozen widths coincide a stale
offset corrupts silently instead of crashing (the 2026-06-10 card-index and
2026-06-14 hand-summary `_embed_state` bugs). Every state-embed offset
`_embed_state` reads is consolidated into `model.StateEmbedOffsets` (card-index,
hand, decision, hand-summary), which each shim overrides as one unit — so a
newly-inserted stripe cannot desync an offset a shim forgot. Every code-carried
value the loaded artifact's behavior depends on belongs in the shim — see
`docs/VERSIONING.md` (FRESH vs REGIME, config- vs code-carried).

**Shims also back era-pinned training.** A resumed run carries
`TrainConfig.encoding_version` and keeps producing artifacts at its own era
(see "Training resume: era pinning" in `docs/VERSIONING.md`): the training
pipeline constructs the era's net class via
`model.PolicyValueNet.class_for_version` and derives its dims via
`encoding_dims_for_era`. Superseded eras are therefore *producing* paths, not
read-only museums — a new training-side feature must work at every same-MAJOR
era or refuse one explicitly.

## Modules

**`__init__.py`** — imports the era modules and owns the package-level dims
router: `encoding_dims_for_era(artifact_version, spec) -> (state_dim,
choice_dim)` — the raw vector widths an era's encoders produce (pre-0.3 state
from `v0_2`, pre-0.1 choice from `v0_0`, live otherwise).

**`v0_0.py`** — Pre-0.1 (v0.0) choice geometry shim. The v0.0 encoding omitted
several choice-vector stripes that were added in v0.1.
- `uses_v0_0_choice_encoding(descriptor) -> bool` — True when the artifact's
  `version` is `"0.0"`.
- `regenerate_v0_0_choices(steps) -> steps` — Rebuilds choice rows from raw
  game state without the new fields (used during compat loading for eval).
- `PolicyValueNetV00` — Frozen-era subclass (extends `PolicyValueNetV01`) that
  carries its own choice geometry: overrides `encode_choices`, the choice-encoder
  input width, and `_embed_choices`' frozen card-region offsets. Inherits V01's
  card encoder and the pre-0.3 state geometry (including `_state_embed_offsets`).

**`v0_1.py`** — Pre-0.2 card-feature shim (`CARD_FEATURE_DIM` 229 → 224).
- `PolicyValueNetV01` — overrides `_build_card_encoder` (frozen 229-wide input +
  v0.1 feature table), `encode_state` (the 771-dim pre-0.3 vector via
  `v0_2.encode_state_v02`), and `_state_embed_offsets` (slices that vector at the
  v0.2 offsets, via `v0_2.state_embed_offsets_v02`).
- `SetupNetV01` — the setup-net twin of the frozen card encoder.

**`v0_2.py`** — Pre-0.3 state-misc shim (the 0.3 round / cube one-hot reshape).
All pre-0.3 nets feed the 771-dim vector.
- `PolicyValueNetV02` — overrides `encode_state` (frozen 7-scalar misc stripe via
  `encode_state_v02`) and `_state_embed_offsets` (the matching slice offsets).
- `encode_state_v02` / `state_embed_offsets_v02` — the frozen 771-dim vector and
  the `StateEmbedOffsets` (card-index, hand, decision, hand-summary) it must be
  sliced at: the first three are live offsets shifted by `_MISC_DIM_DELTA` (−24
  vs. the live 0.4 vector); hand-summary — which precedes misc, so only the
  `turn_state` stripe sits ahead of it — shifts by `_HAND_SUMMARY_DIM_DELTA`
  (−27). `state_stripe_layout_v02` is the reporting twin.

**`v0_3.py`** — Pre-0.4 state shim (the 0.4 `turn_state` stripe + misc shrink,
state vector 790 → 795). All pre-0.4 nets feed the 790-dim vector.
- `PolicyValueNetV03` — overrides `encode_state` (frozen 26-dim one-hot misc, no
  `turn_state`, via `encode_state_v03`) and `_state_embed_offsets`.
- `encode_state_v03` / `state_embed_offsets_v03` — the frozen 790-dim vector and
  its `StateEmbedOffsets`: card-index / hand / decision shifted by
  `_TOTAL_DIM_DELTA` (−365 vs. live v0.6; −5 from the v0.4 misc change, −360 from
  the v0.6 playability stripes), hand-summary by `_HAND_SUMMARY_DIM_DELTA` (−27).
  `state_stripe_layout_v03` is the reporting twin.
- Import-time `_assert_live_layout_contract` pins the deltas and ordering.

**`v0_4.py`** — Pre-0.6 state+choice shim (the 0.6 playability multi-hots; covers
both 0.4 and 0.5, which are encoding-identical). State vector 795 → 1155; choice
row +180.
- `uses_v0_4_encoding(v) -> bool` — True when `(0,4) ≤ (major,minor) < (0,6)`.
- `PolicyValueNetV04` — overrides `encode_state` (no `hand_playable_me` /
  `hand_playable_eggs_me` stripes, via `encode_state_v04`), `encode_choices`
  (no `becomes_playable` stripe, via `encode_choices_v04`), `_state_embed_offsets`
  (`decision_type` shifted −360; card-index / hand / summary unchanged), and
  `_choice_embed_offsets` (`becomes_playable=None`).
- `encode_state_v04` / `state_embed_offsets_v04` / `state_feature_dim_v04` —
  frozen 795-dim state, frozen offsets, frozen dim.
- `encode_choices_v04` / `choice_feature_dim_v04` — frozen narrower choice rows.
- Import-time `_assert_live_layout_contract` pins N_HAND_PLAYABLE_MULTIHOTS==2,
  stripe ordering, and the −360/−180 deltas.

**`v0_6.py`** — Pre-0.7 card-feature shim (224-wide card encoder; no `or_cost` flag;
also restores v0.7 eggs-included food `becomes_playable` semantics; covers 0.2–0.6).
- `uses_v0_6_card_feature_encoding(v) -> bool` — True when `(0,2) ≤ (major,minor) < (0,7)`.
- `PolicyValueNetV06` — overrides `_build_card_encoder` (frozen 224-wide MLP) and
  `encode_choices` (delegates to `v0_7.encode_choices_v07` for eggs-included food encoding).
- `card_feature_matrix_v06()` — rebuilds the `[181, 224]` feature table without `or_cost`.
- `_install_v06_card_encoder_main` / `_install_v06_card_encoder_setup` — wire the frozen
  encoder into a `PolicyValueNet` or `SetupNet` instance.
- `SetupNetV06` — overrides `_build_card_encoder` for the setup net.
- Import-time `_assert_live_layout_contract` pins `_OR_COST_FLAG_DIM==1`, stripe offset,
  and bird catalog size.

**`v0_7.py`** — Pre-0.8 food `becomes_playable` shim (eggs-included food-gain path; covers
exactly 0.7).
- `uses_v0_7_becomes_playable_encoding(v) -> bool` — True iff `(major, minor) == (0, 7)`.
- `encode_choices_v07(decision, game_state, spec)` — calls live `encode_choices` with
  `food_playable_ignores_eggs=False` to restore eggs-included semantics.
- `PolicyValueNetV07` — overrides `encode_choices` to delegate to `encode_choices_v07`;
  also overrides `encode_state` and `_state_embed_offsets` to delegate to `v0_8` (v0.6–v0.8
  share the same 1155-dim state).

**`v0_8.py`** — Pre-0.9 state-compaction shim (1155 → 1119 dims; covers exactly 0.8; v0.6–v0.7
delegate their state overrides here).
- `uses_pre_v09_state_encoding(v) -> bool` — True iff `(major, minor) == (0, 8)`.
- `encode_state_v08(game_state, decision, spec)` — reproduces the 1155-dim pre-0.9 vector by
  calling live sub-builders with old-behavior flags (`full_stats=True`,
  `include_goal_pts=True`, `zero_passed_rounds=False`) and re-inserting the removed
  `hand_summary_me` stripe in its historical position.
- `state_embed_offsets_v08()` — the frozen `StateEmbedOffsets` for the 1155-dim vector:
  `card_index=562`, `hand_multihot=595`, `decision_type=1135`, `hand_summary=343`,
  `hand_summary_end=353` (all 36 columns right of the live v0.9 offsets).
- `state_feature_dim_v08(spec)` — frozen 1155-dim width (default spec).
- `PolicyValueNetV08` — overrides `encode_state` and `_state_embed_offsets` to drive the
  net with its frozen 1155-dim geometry; choice encoding is identical to live.
- Import-time `_assert_live_layout_contract` pins `HAND_SUMMARY_OFFSET==343`,
  `HAND_SUMMARY_DIM==10`, and `_V08_CARD_INDEX == live OFF_CARD_INDEX + 36`.

**`__init__.py` dims router** — `encoding_dims_for_era(artifact_version, spec)` routes:
`(0,6)–(0,8)` → `v0_8.state_feature_dim_v08(spec)` (1155-dim pre-compaction state);
`(0,3)–(0,5)` → `v0_4.state_feature_dim_v04(spec)` (795-dim, pre-playability);
`(0,0)–(0,2)` → `v0_2.state_feature_dim_v02(spec)` (771-dim pre-v0.3 misc);
choice dim is similarly era-routed.
