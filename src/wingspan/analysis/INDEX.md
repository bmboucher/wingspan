# analysis — Architecture-importance probes over a trained checkpoint

Offline diagnostics for a trained `PolicyValueNet`: how much does each part of
the network's architecture actually matter? Loads a checkpoint, self-plays a
sample of on-distribution decisions, then measures board-attention behavior
and ablation sensitivity, trunk/choice-encoder layer capacity, and trunk
input-energy shares. Exposed as `wingspan analysis probe`. See
`docs/TRAINING.md` "Representation diagnostics" for how to read the numbers
and the 2026-09-15 baseline, and `docs/RESEARCH.md`'s "General architecture
exploration" project for how this fits the research agenda.

## Modules

**`__init__.py`** — docstring only, no eager imports.

**`models.py`** — Every Pydantic model and enum the package produces or
consumes:
- `AblationMode` — `FULL` / `ZERO` / `UNIFORM` / `HEAD_KNOCKOUT`.
- `PolicyDeltaStats(n, mean_kl, p90_kl, flip_rate, value_shift_points)` — the
  shared "how much did this perturbation move the policy/value" shape, base
  class of `AblationEffect` (adds `mode`, `head`) and `ReferenceComparison`
  (adds `checkpoint`).
- `FamilyEffect` / `BoardFillEffect` — the zero/uniform effect sliced by
  judgment family or by `BOARD_FILL_BANDS` (`(0,0) (1,3) (4,6) (7,9) (10,15)`).
- `LayerSummary(name, in_features, out_features, rank95, rank95_over_width,
  linear_r2, dead_fraction)` — the training-loop-sized subset; `LayerStats`
  extends it with `rows`, `participation_ratio`, `rare_fraction`,
  `weight_stable_rank` and a `.summary` property that projects back down to
  `LayerSummary`.
- `AttentionHeadStats` / `AttentionBlockStats` — per-head entropy/self-weight
  and the block-level contribution-ratio/cosine spread.
- `ParamCensusEntry`, `InputGroupShare`, `HeadToHeadResult`.
- `RepresentationReport` — the full per-checkpoint report `representation.measure`
  returns; `.effect(mode)` looks up the block-level (not per-head)
  `AblationEffect` for a mode. `RepresentationMetrics` — the lightweight
  Stage-2-sized projection `representation.summarize_for_loop` builds (this
  package does not itself write one anywhere).

**`attention_probe.py`** — `ProbeAttention(nn.Module)`: a duck-typed,
hand-reimplemented stand-in for one board-attention `nn.MultiheadAttention`,
installed via `install(net) -> list[ProbeAttention]` in place of `board_attn`
(shared) or `board_attn_me`/`board_attn_opp` (POV wrapper first); `uninstall`
restores the originals; `set_mode(wrappers, mode, head=-1)` flips every
installed wrapper together. Matches the original module exactly in FULL mode;
ZERO/UNIFORM/HEAD_KNOCKOUT substitute a perturbed computation without
touching `model/core.py` — the padding, key-masking, and empty-slot residual
stay owned by `_apply_board_attention`, which reads a wrapper exactly like the
real module (`embed_dim`, `num_heads`, the `(query, key, value,
key_padding_mask, need_weights)` call contract). `AttentionStatsCollector`
accumulates FULL-mode per-head entropy/self-weight and block-level
contribution-ratio/cosine across every board a pass visits (one or two
wrappers may share one collector), rolled up via `.summarize()`.

**`layer_probe.py`** — `LinearLayerProbe(prefix, sequential)`: forward-hooks
every `nn.Linear` in an `nn.Sequential` (named `f"{prefix}.L{index}"`),
capturing inputs and pre-activation outputs; `.remove()` detaches the hooks;
`.stats()` computes each layer's `models.LayerStats` on post-ReLU activations
(row-capped to 20,000 with a fixed seed). Public linear-algebra primitives:
`participation_ratio_and_rank95(activations)` (effective-rank pair),
`ridge_r2(inputs, outputs, lam_rel=1e-3)` (linear-predictability of the
layer's own nonlinearity), `weight_stable_rank(weight)`
(`||W||_F^2/||W||_2^2`). The three `torch.linalg.*` wrappers
(`_eigvalsh`/`_svdvals`/`_solve`) and `attention_probe._row_norm` are pinned
with `# fmt: off` / `# fmt: on` — the installed torch stub under-specifies
these functions' return types, and letting black reflow the resulting
`# pyright: ignore` comment strands it on the wrong line.

**`probe_set.py`** — `ProbeBatch(state, choices, mask, family_idx,
own_bird_count)` and `ProbeSet(batches, n_decisions, n_games)`: the fixed
sample of decisions every ablation mode is measured over, grouped by exact
legal-option count (no padding). `from_steps(net, steps, n_games)` builds a
`ProbeSet` from already-collected `training.steps.Step`s (own-bird-count read
from `net.raw_state_stripe_layout().offset_of("card_idx_board")`'s first 15
columns — the POV board slots). `from_self_play(net, run_config, n_games,
seed, device)` plays the games itself via `training.collect.play_game`,
mirroring the run's own seat count and `combine_gain_food` regime.
`subsample_steps(records, max_decisions, rng)` is a standalone reproducible
subsampler over a list of `GameRecord`s (not wired into `from_self_play`).

**`representation.py`** — The main entry. `measure(net, probe_set, *, device,
score_norm, reference_net=None, checkpoint_label="") ->
models.RepresentationReport` runs, under `net.eval()` + `torch.no_grad()`: a
parameter census; one FULL pass that also captures layer activations,
attention statistics, and the trunk's raw input; ZERO/UNIFORM/one-per-head
HEAD_KNOCKOUT ablations (skipped entirely — `attention=None`,
`ablations=family_effects=board_fill_effects=[]` — when the net has no board
attention); the zero/uniform effect sliced by judgment family and by
`models.BOARD_FILL_BANDS`; an optional reference-checkpoint comparison over
the same batches; and, only when `not net.arch.tray_set_embedding and not
net.arch.use_distinct_hand_model`, a trunk input-energy-share breakdown
(continuous remainder / own board / opponent board(s) / tray / hand pool /
one pool per extra card-set stripe, labelled by the stripe it embeds —
`hand_playable_me_pool`, `hand_playable_eggs_me_pool`, `known_hand_opp_pool`
at the live 2-seat layout), else `[]`. Restores `net`'s original train/eval mode
before returning. `summarize_for_loop(report) -> models.RepresentationMetrics`
projects a report down to the Stage-2-sized shape.

**`head_to_head.py`** — `evaluate_substitution(checkpoint_path, mode,
n_pairs, seed, device) -> models.HeadToHeadResult`: loads the checkpoint
twice, installs attention wrappers on the second copy in `mode`, and plays
`training.evaluate.evaluate_vs_opponent`'s paired-deal harness between them —
the *actual game-strength* counterpart to `representation.measure`'s
KL/flip/value-shift dependence metrics.

**`cli.py`** — `main_analysis(argv) -> int` for `wingspan analysis`, one
sub-command: `probe TARGET [--checkpoint-dir DIR] [--games N] [--reference
TARGET] [--head-to-head N] [--seed N] [--device DEV] [--json PATH] [--width
N]`. `TARGET` / `--reference` use `players.spec.parse_player_spec`; `human`
and `random` are rejected (exit 1 — there is no network to probe). Prints one
rich table per report section (same console/UTF-8 conventions as
`reporting.inspect_cli`) and, with `--json`, writes
`report.model_dump_json(indent=2)`. A nonzero `--head-to-head` attaches a
UNIFORM and a ZERO `head_to_head.evaluate_substitution` result to the report
after the main measurement.

## Caveats

- **Dependence, not necessity.** An ablation's near-zero effect means the
  policy/value output barely moved when that block was perturbed on *this*
  probe set — it does not mean the block is safe to delete. The board could
  be carrying the same information redundantly elsewhere (e.g. the
  continuous per-slot scalars alongside the attended card identities), or the
  probe set could simply under-sample the situations where the block matters.
- **Reference comparisons assume the same era.** `reference_net` is scored
  through the *same* probe-set batches (built for the primary net's live
  encoding); this module does not check or reconcile artifact eras
  (`docs/VERSIONING.md`). Comparing across a FRESH encoding change will
  silently feed one net vectors it was not trained to read.
- **Input-share decomposition is regime-limited.** `_trunk_input_shares` only
  understands the default trunk-input layout (pooled hand, no tray-set
  embedding); nets built with `use_distinct_hand_model` or `tray_set_embedding`
  report an empty `trunk_input_shares` list rather than a wrong one.
