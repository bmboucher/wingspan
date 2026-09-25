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

`A.json` / `B.json` are full `RunConfig` files (see
`wingspan.training.config_file`), each with its own `checkpoint_dir`,
`run_name`, `resume=false`, `collect_device=cpu`, `train_device=cuda` already
set — no swapping a defaults file in and out is needed:

```
wingspan dashboard --config runs/arch_pair/A.json --start
```

```
wingspan dashboard --config runs/arch_pair/B.json --start
```

Dropping `--start` opens the FLIGHT PLAN config screen seeded from the file
instead of launching immediately, so you can review the architecture block
against the table above (or tweak a run-identity flag such as
`--checkpoint-dir`) before pressing Start.

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

## Results (both arms finished 2026-09-25)

Runs archived under `runs/arch_pair/{A,B}/archive/arch_pair_{A,B}/` (`last.pt`,
`final_300.pt`, `setup.pt`, `metrics.jsonl`, `run_config_*.json`). Each arm took
~14.7 h at 7.5 games/s (no throughput difference between the two shapes).

**The comparison is confounded by the clone phase.** `clone_iters` had never
been used in a real run before; over 25 iterations the imitation loss stayed
flat (A 1.167 → 1.163, B 1.167 → 1.107), the policy entropy at the first PPO
iteration was 1.17 nats (the random-init value), and the margin against the
champion was unchanged (−56 → −55). Cause: `learner.update` routes the
imitation phase through the single-step path (one optimizer step per
iteration) while PPO takes one step per reuse epoch (4 per iteration); both
accumulate gradients across minibatches rather than stepping per
minibatch, which supervised cloning needs. The value-MSE
was the only effective gradient, and against a fixed strong opponent its
target is nearly constant, so both trunks collapsed toward rank 1 during the
clone phase. B's LayerNorm let it recover; A never did.

| readout | A (control) | B (reclaimed) |
|---|---|---|
| `imitation_loss` @ iter 24 | 1.163 | 1.107 |
| `collection_win_rate` vs champion, last 25 iters | 9.8% | 26.3% |
| `avg_margin` vs champion, last 25 iters | −17.8 | −9.0 |
| first iteration ≥ 50% vs champion | never | never (still rising at 300) |
| trunk rank95 (per layer) at iter 295 | 9/128, 3/128, 2/64, 2/64 | 68/128, 30/64 |
| choice rank95 at iter 295 | 39/128, 2/64, 2/64 | 66/128, 28/64 |
| dead fraction, trunk tail / choice tail | 0.25 / 0.05 | 0.00 / 0.00 |
| attention uniform-KL at iter 295 | 7e-5 | 2e-4 |
| games/s (PPO phase) | 7.49 | 7.55 |

Probe trajectory: A's trunk went 70 → 39 → 9 → 1 (rank95/128 at iters 0, 5,
10, 20) and stayed ≤ 3 in every trunk layer for all 275 PPO iterations, with
the choice tail at 2/64. B's trunk went 71 → 3 by iter 30, then recovered to
54 by iter 100 and 68 by 150, with zero dead units from iter 100 on.

Tournament, greedy play, 200 mirrored games per pair, seed 0 (champion =
`champion_2026.08.11-v1.5_iter1954.pt` with the finished run's `setup.pt`):

| pair | record | win rate (95% CI) | mean margin |
|---|---|---|---|
| champion vs A | 191-8-1 | 95.5% ± 2.9 | +21.5 |
| champion vs B | 158-35-7 | 79.0% ± 5.6 | +12.0 |
| B vs A | 145-48-7 | 72.5% ± 6.2 | +8.6 |

Elo: champion 1806, B 1472, A 1221. Mean scores: champion 83.3, B 72.1, A 63.3.

**What this settles.** LayerNorm on the trunk and choice encoder is a clear
win: it is the only ingredient that let B recover from the collapse, and B ends
with no dead units and full-rank-ish layers where A is a rank-2 bottleneck.
Removing the dead depth and cutting attention to one head cost nothing on
throughput and B's single head is as uniform as A's eight.

**What it does not settle.** Whether the depth removal itself helps, because A
never trained as an intact network. A clean rerun needs `clone_iters` 0 (or a
fixed clone phase that takes multiple epochs of minibatch steps) with the same
pinned champion; A with LayerNorm added would isolate the depth question.
