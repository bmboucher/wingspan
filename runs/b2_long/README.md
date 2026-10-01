# B2 long run: 2000 iterations from a champion clone

One run file, `2026.10.01-v1.5-b2.json`, that takes the B2 shape from the
`runs/arch_pair` rerun and trains it for 2000 iterations as the next main run
in `checkpoints/`. The first 25 iterations clone the `2026.08.11-v1.5`
champion exactly as the A2/A2ln/B2 rerun did; after that the run follows the
normal opponent ladder (graduate from the champion, then advancing frozen
self-play generations) instead of the pinned-champion yardstick the pair used.

## What the file is

Derived from `runs/arch_pair/B2.json` by `model_copy`; the `architecture`,
`training` and `engine` sections are byte-identical to B2 (trunk (128, 64),
choice (128, 64), trunk/choice LayerNorm on, one board-attention head,
`entropy_coef` 0.01 constant, PPO+GAE, 2048-step minibatches). Everything
that differs:

| field | B2 | this run | why |
|---|---|---|---|
| `run.target_iterations` | 300 | 2000 | the ask |
| `run.target_eval_games` | 1000 | 1000 | final self-play eval at the target |
| `run.eval_every` / `eval_games` | 5 / 500 | 2 / 500 | the ladder keys off the eval EWMA; every iteration would cost ~23 % of wall time |
| `run.probe_every` / `probe_decisions` | 5 / 4096 | 10 / 4096 | 200 probe rows over the run |
| `run.checkpoint_dir` / `run_name` | `runs/arch_pair/B2` / `arch_pair_B2` | `checkpoints` / `2026.10.01-v1.5-b2` | main-run convention |
| `run.resume` | false | true | the dashboard can resume after interruptions |
| `opponent.bootstrap_opponent` | champion | champion (same file) | clone expert and first opponent |
| `opponent.random_phase_win_rate` | 1.0 (never graduates) | 0.55 | see below |
| `opponent.opponent_reset_win_rate` | 0.65 (unused) | 0.60 | see below |
| `opponent.opponent_max_iterations` | 0 | 250 | see below |
| `opponent.eval_ewma_alpha` | 0.7 | 0.7 | same as the champion run |
| `misc.seed` | 0 | 0 | the clone phase should reproduce B2's numbers |
| `dagger` | clone_iters 25, epochs 4, minibatch 2048 | same | fixed clone phase |

### The ladder thresholds

Two numbers were chosen from data rather than inherited, and both are easy to
change in the file before launch.

**Graduation at 0.55, not 0.65.** In the bootstrap phase the run trains and is
scored against the frozen champion; it graduates when the EWMA (alpha 0.7) of
the sampled `collection_win_rate` clears `random_phase_win_rate`. The three
rerun arms never got near 0.65 against the champion in 300 iterations: their
best EWMA was A2 0.563, A2ln 0.581, B2 0.580. B2's EWMA first crossed 0.50 at
iteration 23, 0.52 at 147, 0.55 at 228 and 0.58 at 270. At 0.55 the run should
graduate around iteration 230 and start self-play as a policy that reliably
beats the champion in sampled play. At 0.65 it would most likely never leave
the bootstrap phase.

**Generation advance at 0.60 with a 250-iteration cap.** The champion run used
0.65 with no cap. Its generations lasted 15, 48, 54, 43, 67, 142, 205 and 500
iterations, and generation 9 then lasted the remaining 892 iterations without
firing: eval win rate against the frozen gen-9 self averaged 0.547, the best
single eval was 0.628 and the best EWMA was 0.615. The last 45 % of that run
trained against one fixed opponent. A 0.60 trigger is still three CIs above
50 % at 500 eval games (CI ±4.3 points), and the 250-iteration cap guarantees
the reference keeps moving even when the trigger stalls.

## Launch

`checkpoints/` currently holds the finished champion run, and headless launch
refuses to overwrite or archive on its own. `prepare_headless_launch` on this
file today returns:

```
the run in checkpoints/ has an incompatible architecture — archive it first via
`wingspan dashboard --checkpoint-dir checkpoints` then the config screen's [A]
archive action
```

So, in order:

1. `wingspan dashboard --checkpoint-dir checkpoints`, press `[A]` to archive
   the `2026.08.11-v1.5` run into `checkpoints/archive/`, quit.
2. `wingspan dashboard --config runs/b2_long/2026.10.01-v1.5-b2.json --start`

The bootstrap expert is `runs/arch_pair/champion_2026.08.11-v1.5_iter1954.pt`
(a copy of the champion's `best.pt`), so archiving `checkpoints/` does not
move the file the run depends on. The same config resolved cleanly through
`prepare_headless_launch` against an empty directory, and
`validate_launchable` reports no problems.

To resume after an interruption, run the same `--config --start` command; the
file carries `resume: true` and the directory will hold a compatible run.

## Cost

B2's PPO iterations took about 200 s (collect 148 s, update 53 s) at 121
decisions per game; a 500-game eval took about 61 s in the champion run, here
every second iteration. That is roughly 230 s per iteration, or 5.5 days for
2000 iterations, before accounting for games getting longer as the policy
improves (the champion's games ran 260 decisions by the end, roughly double).
Plan on one to two weeks.

## Sanity checks in the first hours

With seed 0 and the same expert the clone phase should reproduce B2's numbers
closely:

- `imitation_loss` 0.625 at iteration 0, about 0.37 by iteration 24 (floor
  0.21 = the champion's own entropy).
- `collection_win_rate` 48 to 49 % and `avg_margin` near 0 by iteration 24.
- `representation` rows: no rank collapse during the clone phase (B2 at
  iteration 295 had trunk 72/128, 29/64; choice 67/128, 26/64; dead fraction
  about 0).
- Entropy about 0.37 at the first PPO iteration (25), not 1.17.

Anything far from these means the launch is not what this README describes.

## Readout at the end

- The run's own curve: eval win rate per generation, generation change
  iterations, and whether advances came from the 0.60 trigger or the cap.
- A greedy tournament against the old champion and against the 300-iteration
  B2 arm, 200 mirrored games per pair, seed 0, as in `runs/arch_pair/README.md`:
  competitors are run directories holding `last.pt` + `setup.pt` +
  `run_config_*.json`, and a champion directory can be assembled from
  `best.pt` + `setup.pt` + the latest `run_config`.
- Probe tails at the end against B2's iteration-295 values, to see whether
  2000 iterations use more of the (128, 64) layers than 300 did.
