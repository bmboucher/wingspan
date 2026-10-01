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
never trained as an intact network. A clean rerun needs a clone phase that
actually moves the policy, with the same pinned champion; A with LayerNorm
added would isolate the depth question. Both are set up below.

## Rerun with the fixed clone phase (files ready 2026-09-25)

The clone phase now steps once per shuffled minibatch (`clone_epochs` 4 ×
`clone_minibatch_steps` 2048, roughly 180 optimizer steps per clone iteration
instead of one; see `docs/TRAINING.md` §6.8). In a 24-game smoke run against
the same champion the imitation loss fell from 1.12 to 0.62 in six iterations
where the old path stayed flat at the uniform level 1.16–1.18, and the value
loss went to ~0.01 instead of ~1.4, so the value-only gradient that collapsed
A's trunk no longer dominates. The champion's own entropy on student states is
about 0.21 nats, which is the floor the imitation loss can reach.

Three files, same protocol as A/B (1000 games/iter, 300 iterations, pinned
champion, `clone_iters` 25, seed 0, cpu collect / cuda train), differing only
in the main network and the run identity:

| | A2 | A2ln | B2 |
|---|---|---|---|
| derived from | A | A + LayerNorm | B |
| `trunk_layers` | (128, 128, 64, 64) | (128, 128, 64, 64) | (128, 64) |
| `choice_layers` | (128, 64, 64) | (128, 64, 64) | (128, 64) |
| `trunk_layernorm` / `choice_layernorm` | off | on | on |
| `board_attention_heads` | 8 | 8 | 1 |
| `checkpoint_dir` | `runs/arch_pair/A2` | `runs/arch_pair/A2ln` | `runs/arch_pair/B2` |

A2 vs A2ln isolates LayerNorm at fixed depth; A2ln vs B2 isolates the depth
removal (and the head count) with LayerNorm held constant. A2 is the control
that shows whether the un-normalised architecture survives an intact clone
phase at all; if the budget allows only two arms, run A2ln and B2.

```
wingspan dashboard --config runs/arch_pair/A2.json --start
wingspan dashboard --config runs/arch_pair/A2ln.json --start
wingspan dashboard --config runs/arch_pair/B2.json --start
```

Readout is the same list as above, with two additions: `imitation_loss` at
iteration 24 should now sit well below 1.0 for every arm (it was 1.167 for
both A and B), and the `representation` rows during the clone phase should
show `rank95_over_width` holding up rather than falling to 1/64.

### Rerun results (A2ln finished 2026-09-26, B2 2026-09-29, A2 2026-09-30)

Archived under `runs/arch_pair/{A2,A2ln,B2}/archive/arch_pair_*/`. Same
protocol as the first pair; only the clone phase changed.

**The clone phase works now.** Imitation loss over the 25 clone iterations:

| iter | A2 | A2ln | B2 |
|---|---|---|---|
| 0 | 0.825 | 0.644 | 0.625 |
| 4 | 0.507 | 0.436 | 0.408 |
| 12 | 0.404 | 0.394 | 0.378 |
| 24 | 0.371 | 0.374 | 0.368 |

(first pair: 1.167 → 1.163 / 1.107; champion entropy floor ≈ 0.21; uniform
≈ 1.17). Value loss during cloning stayed at 0.02–0.07 (was 0.7–1.5), the
policy entropy at the first PPO iteration was 0.37 nats (was 1.17), and by
iteration 24 every arm was already at 48–49% collection win rate and ≈ 0
margin against the champion (was 0% and −55). Twenty-five clone iterations,
about 1.5 h, reach parity with a champion that took the original run ~1100
iterations. The clone update takes ~53 s per iteration versus ~9 s before.

A2's un-normalised trunk still dipped during the first clone iterations
(trunk rank95 70 → 24/5/3/3 at iter 5) but recovered by iter 10 once the
cross-entropy gradient dominated; nothing collapsed permanently.

**All three arms are the same strength at 300 iterations.**

| readout | A2 | A2ln | B2 |
|---|---|---|---|
| parameters | 659,277 | 660,557 | 630,453 |
| `imitation_loss` @ iter 24 | 0.371 | 0.374 | 0.368 |
| `collection_win_rate` vs champion, last 25 | 54.3% | 55.2% | 55.1% |
| `avg_margin` vs champion, last 25 | +1.60 | +1.85 | +1.87 |
| first trailing-5 ≥ 50% | iter 168 | iter 148 | iter 26 |
| 25-iter block win rate, iters 25–49 | 44.0% | 44.5% | 47.4% |
| 25-iter block win rate, iters 150–174 | 48.1% | 48.8% | 50.1% |
| trunk rank95 (per layer) at iter 295 | 72/128, 36/128, 8/64, 6/64 | 74/128, 56/128, 24/64, 16/64 | 72/128, 29/64 |
| choice rank95 at iter 295 | 55/128, 4/64, 3/64 | 68/128, 29/64, 23/64 | 67/128, 26/64 |
| dead fraction, trunk tail / choice tail | 0.00 / 0.00 | 0.08 / 0.00 | 0.02 / 0.00 |
| attention uniform-KL at iter 295 | 1.2e-2 | 7.2e-3 | 5.3e-3 |
| collect s / 1000 games (PPO phase) | 161 | 151 | 148 |
| games/s (PPO phase) | 6.26 | 6.61 | 7.01 |
| self-play eval score at 300 | 81.3 | 81.7 | 81.4 |

Paired per-iteration differences over iterations 25–299: B2 − A2 win rate
+2.2 points (sd 2.1), A2ln − A2 +0.6, B2 − A2ln +1.6; over iterations
200–299 the B2 − A2 gap is +1.3 points and A2ln − A2 is +0.2. B2 is slightly
ahead through the middle of the run and the gap closes by the end. All three
are still rising slowly at 300 (block means 0.52 → 0.54 → 0.55). Games/s are
lower than the first pair's 7.5 because the games are longer: 121 decisions
per game against 93, a consequence of the stronger policies, not of the
architectures. Within the rerun, B2 collects ~9% faster than A2.

Tournament, greedy play, 200 mirrored games per pair, seed 0 (the first
pair's B included as a yardstick; A omitted, it lost 95.5% to the champion):

| pair | row win rate (95% CI) | row margin |
|---|---|---|
| champion vs A2 | 57.0% ± 6.9 | +2.3 |
| champion vs A2ln | 51.2% ± 6.9 | +1.6 |
| champion vs B2 | 54.8% ± 6.9 | +2.1 |
| A2 vs A2ln | 45.8% ± 6.9 | −0.3 |
| A2ln vs B2 | 46.2% ± 6.9 | +0.1 |
| A2 vs B2 | 48.8% ± 6.9 | −0.4 |
| A2 / A2ln / B2 vs B (first pair) | 75.5% / 77.0% / 76.5% | +10 |
| champion vs B | 81.8% ± 5.4 | +11.0 |

Elo: A2ln 1560, B2 1551, champion 1537, A2 1537, B 1315. Every pair among
the three new arms is within its confidence interval of 50%. The champion
still edges all three under greedy play even though the sampled collection
win rate is 55%; going first is worth ~8–15 points in these pairs, which is
most of the champion's margin.

**What this settles.**

- The first pair's result was entirely the clone-phase confound. With a
  working clone phase the three architectures are indistinguishable at 300
  iterations, in sampled play, in greedy play and in self-play score.
- The dead depth can be removed at no cost. B2 matches A2 and A2ln with 4.4%
  fewer parameters, collects ~9% faster, and uses what it has: trunk and
  choice tails at 0.45 rank95/width versus A2's 0.09 and 0.05. A2's tails are
  still near-linear pass-throughs (linear R² 0.99), as the original probe
  found.
- LayerNorm is a robustness win, not a strength win at this budget. A2 and
  A2ln end equal, but A2ln cloned faster at the start (0.64 vs 0.82 at iter
  0), never dipped during cloning, and keeps every layer at rank ≥ 16/64
  where A2's deep layers sit at 3–8/64.
- Cloning from the champion is the cheap part of training. Parity in 25
  iterations means a new architecture can be brought to the current best
  policy in ~1.5 h, so architecture comparisons should run from this start
  rather than from scratch.

**What it does not settle.** Whether any of the shapes would pull ahead with
a longer run or a different seed: one seed each, all three still rising at
300, and the champion yardstick stops discriminating once the students pass
it. A longer run would need a stronger or moving opponent (graduate off the
pinned champion) to separate them. Recommended default going forward: B2's
shape ((128, 64) trunk and choice, LayerNorm on, one attention head), on cost
grounds.
