# Architecture pair: A (control) vs B (reclaimed)

Two runs, identical except for the main network, to test the probe findings from
the `2026.08.11-v1.5` run (see `docs/TRAINING.md` §6.5): the trunk and choice
encoder tails are near-linear and low-rank, and the 8-head board attention is
indistinguishable from a uniform average. B removes the dead depth, adds
LayerNorm, and keeps one attention head; A is the current architecture.

| | A (control) | B (reclaimed) |
|---|---|---|
| `trunk_layers` | (128, 128, 64, 64) | (128, 64) |
| `choice_layers` | (128, 64, 64) | (128, 64) |
| `trunk_layernorm` / `choice_layernorm` | off | on |
| `board_attention_heads` | 8 | 1 |
| parameters | 659,277 | 630,453 |

Shared protocol (both files):

- 1000 games/iter, `target_iterations` 300, seed 0, cpu collect / cuda train.
- `bootstrap_opponent` = `champion_2026.08.11-v1.5_iter1954.pt` (the finished
  run's `best.pt`), `random_phase_win_rate` 1.0 — the run never graduates, so
  every iteration trains and is scored against the same frozen champion. The
  bootstrap phase pauses the paired greedy eval; strength is the per-iteration
  `collection_win_rate` and `avg_margin` in `metrics.jsonl`.
- DAgger `clone_iters` 25: the first 25 iterations imitate the champion
  (cross-architecture expert), then PPO+GAE continues as before.
- `entropy_coef` 0.01 constant (no anneal — the clone hands over a sharp policy).
- `probe_every` 5, `probe_decisions` 4096.

## Launch (one arm at a time; each is ~27 h on this box)

The files are FLIGHT PLAN defaults files. The configurator seeds from
`./configurator_defaults.json` when the checkpoint directory has no saved run,
so swap the arm's file in, launch, then restore the tracked defaults:

```
cp runs/arch_pair/configurator_defaults.A.json configurator_defaults.json
wingspan dashboard --checkpoint-dir runs/arch_pair/A --run-name arch_pair_A --collect-device cpu --train-device cuda --no-resume
git checkout -- configurator_defaults.json
```

Repeat with `B`. Confirm on the config screen that the run reads
"seeded from user defaults" and that the architecture block matches the
table above before pressing Start.

## Readout when both finish

1. `imitation_loss` at iteration 24, A vs B (capacity to represent the champion).
2. `collection_win_rate` / `avg_margin` vs the champion, both arms overlaid;
   iteration of first 50% crossing and the level at 300.
3. `representation` rows every 5 iterations: `rank95_over_width` on B's
   64-wide trunk output and choice tail (> 0.7 rising = now the constraint;
   < 0.2 = shrink again), dead fraction, attention uniform KL.
4. Final tournament, mirrored deals, at least 200 games per pair:
   `wingspan tournament --no-picker --ai runs/arch_pair/A --ai runs/arch_pair/B --games-per-pair 200`
   plus the champion (`wingspan play` by path) as the fixed yardstick.
5. `games_per_sec`, A vs B.
