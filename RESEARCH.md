This document outlines a series of research projects we want to use our RL model for Wingspan on.

# Related to Model Training

## Project: General architecture exploration
For each submodel in our overall architecture (setup, card encoding, state encoding, choice encoding, value head, each decision head), let's define a range of possible numbers and sizes of hidden layers. There's a lot of sample space to explore here, so I think to keep it simple we can start with a "lite" version (1 small hidden layer) and a "heavy" version (3 larger hidden layers) for every submodel - we want to run a series of training tests where we pin every submodel to its "lite" version except one (which we sweep across all the submodels) which uses "heavy".
Metrics to consider: Runtime/throughput (games/sec), time to "converge" (based on some definition), relative performance (play them against each other)
Main questions:
1. What is the optimal size/shape for each of the submodels?
2. Does this answer vary between the different decision heads enough that we need to configure each separately for most runs?

## Project: Impact of RNG
Let's train some small number of models (3-5) with different RNG seeds, starting from different random parameter initializations etc, but with exactly the same config otherwise and then put them in a tournament (we'll call a model "trained" when its EWMA average points/game has been above some configurable threshold for some configurable number of iterations). Main questions:
1. How much does a model's performance vary when the "points scored" metric is similar?
2. Do multiple models trained on different "paths" arrive at the same endpoint or do they have different "playstyles"?
3. What is the impact of training a model against a model that is also optimal but significantly different (rather than self-play)? i.e. if we train 2 different models from different RNG seeds to some minimum threshold, and then start training them against each other, do they end up looking similar or different (i.e. playstyles converge or just adapt to each other)?
4. Does that kind of cross-breeding improve performance? i.e. if I train model A and B, and then I run N more iterations of just A, and just B, and the combined A vs B, do the resulting models look similar or different?

## Project: Impact of Card Embedding dimensions
Let's train some range of models with the card embedding dimension varying to see the overall impact on training time and performance. In particular, I'm curious to see if we can get meaningful results from a dimensions as low as 3, because I think it would be really interesting to visualize the cards as points in space.

## Project: Extra-Long Training Run
We should do this only after we have a model that we really like - let's continue the training for some huge amount of time, maybe increase batch sizes and parallelize more to make it more efficient. What we're looking to see is evidence of "grokking" - does the model continue to learn while out-of-sample performancee remains stagnant, and then all of a sudden jump to a new plateau when it "figures something out"?

# Related to Model Function

## Project: Does the main action model know "which" card it wants to play?
We've split out the choice of main action (play bird, gain food, lay eggs, or draw cards) from the secondary choice of which bird to play. This made sense from a model construction perspective, but I'm curious if the model deciding to play a bird is also calculating the relative value of doing that based on which cards will actually be available to play in the next choice, or if it is just making a high-level call on building the board vs running the engine.
One way to test this is to look at a sample of main action decision points where we had >1 birds available to play, and to look at how the model output changes if we remove each of those birds from the hand. If the score for "play a bird" goes down when we remove one bird, but not when we remove another, that indicates that the main action model is weighting that bird heavily in the decision. We should be able to do a basic statistical analysis to determine how often this happens, and whether the "chosen" bird correlates well with the actual selection from the "play which bird" model.

## Project: Does the model learn not to add eggs when it can't?
One action the game engine will allow that does literally nothing is going to the Grassland when there are no brown power birds in it and no available egg capacity. This option will always be presented to the model for scoring, but we would hope that it learns not to do it from observation. We can test that by running a large number of game and capturing every data point that matches that description (the model is offered a choice to gain no eggs + activate no birds) and seeing what rate the model chooses to lay eggs in this scenario. Our expectation is zero but we should also compare it to the average across all decision points to get a sense of how often this matters in simulated games.

# Using the Model for Card Valuation / Game Analysis

## Project: Setup card stats
There are ~484 billion possible opening hands of 5 bird cards and 2 bonus cards, so we obviously can't run them all through the model. But we can run a very large sample (a few million) and then generate the full list of 504 setup selection options for each, and score all of those in the model to determine what our policy would be for each hand.
Main questions:
1. Which cards in the set have the highest probability of being selected if they are in the opening hand?
2. Which cards in the set have the most impact on the game's eventual outcome?
3. What does our "ideal" opening hand look like?

## Project: General card stats
This one is pretty simple - we just want to lock a model down and play as many games as fast as we can to get a large sample size, capturing metrics we can analyze. Of particular interest:
* The Games In-Hand (GIH) win rate - how often did we win when we drew a given card or had it in our opener?
* The Games On-Board (GOB) win rate - how often did we win when we had a given card on our board? How does that breakdown per slot (i.e. are certain birds much better in certain locations than others)? 

## Project: Are there "low-scoring" and "high-scoring" games?
I often feel when I play in person that some games are "good" for both players and some are "bad" for both players - obviously both players share in the draws off the deck, but there isn't really enough interaction between players to explain this. I suppose it could also be the end-of-round goals, since those sometimes line up well and sometimes don't, so optimizing for them may mean sub-optimal plays in other places. At any rate, the main question is: is there any correlation between P0's score and P1's score? Simple enough statistical metric, we should probably just start including it in the standard analysis for any run.

---

# Addendum: Infrastructure capacity & gap analysis (2026-06-02, revised 2026-06-06)

*Appended assessment — does the current codebase let us run the six projects above, what has to be built first, and how the existing AWS/cloud stack can absorb the heavy work. This section only catalogs gaps; it does not change any proposal above.*

## What we can build on today

The primitives most of these projects need already exist, even though no
analysis harness wires them together:

- **Per-decision scoring.** `training.policy.policy_probs(net, device, state_vec, choice_feats, family_idx)` runs one forward pass and returns the full softmax over a decision's legal candidates; the same forward also yields the value head `V(s)`. This is the atom for any "score this situation / score every option" study (Projects 2, 4).
- **Setup enumeration + scoring.** `setup_model.enumerate_setup_candidates(dealt_cards, dealt_bonus)` already produces the exact **504-option** keep set (same order the engine offers), and `SetupNet` + `setup_model.candidates.select_by_margins` score and pick one. Project 4's "generate the 504 options for a hand and score them" is essentially already implemented — it just needs to be driven over a large sample and tallied.
- **Reproducible play + introspection.** `wingspan play` (`cli.py` + the `players` package) runs any matchup (human / random / checkpoint / path / run dir, either seat, `--greedy`, `--seed`) and annotates every decision in the log with the policy's ranked probability distribution. `mp_collect.ProcessCollector` fans self-play across CPU cores (~60 games/s on ~10 cores; ~200k games/hour — TRAINING.md §4.1).
- **Durable per-game records.** Every finished game writes a `metrics.GameOutcome` row to `games.jsonl`: per-seat **full six-way `ScoreBreakdown`**, winner, decision count, per-family decision counts, and the **`seed`** (so any game is exactly replayable). These already stream to S3 as immutable `games/<session>/chunk_*.jsonl` during cloud runs.
- **Self-describing checkpoints.** `TrainConfig` + `ModelArchitecture` are stored with each checkpoint and reconstituted via `model.PolicyValueNet.from_model_config`, so any saved run can be reloaded for offline scoring/analysis.
- **A working cloud training stack** (`deploy/` + `wingspan.cloud`): containerized headless runner, S3 persistence, Spot-interruptible + resumable, the `wingspan-monitor` multi-run roster, and Terraform for S3 + an ECS cluster with a Fargate Spot capacity provider + IAM task roles.
- **Generic event reporting framework.** `wingspan.instrumentation` exposes 15 typed game events — the full game/round/turn lifecycle (`GAME_START/END`, `ROUND_START/END`, `TURN_START/END`), the decision flow (`MAKING_DECISION`, `MADE_DECISION`), and domain-specific events (`BIRD_PLACED`, `FOOD_GAINED`, `EGGS_LAID`, `CARDS_DRAWN`, `ROUND_GOAL_SCORED`, `PLAYER_FINAL_SCORED`, `SETUP_APPLIED`) — via a `Instrumentation` dispatcher that the engine holds. Handlers are Pydantic models (`CallbackHandler` subclasses) registered by name with `@registry.register`; an `InstrumentationConfig` wires named handler instances to their events, serializes/deserializes through `model_dump`/YAML, and round-trips through checkpoints. Two bundled handlers ship as reference implementations: `DecisionLogger` (one JSONL row per genuine decision) and `CardVisitRecorder` (per-game bird-play tally). An engine with no instrumentation holds the shared empty router — no per-call overhead.
- **Unified players package.** `wingspan.players` (`spec` → `factory` → `loaders`) centralizes the player-spec grammar (`human` / `random` / named checkpoint / run-dir / direct path) and checkpoint loading that was previously spread across `selfplay.py`, `tournament.participants`, and the CLI. Both `wingspan play` and the tournament use it as the single entry point.
- **Artifact versioning (MODEL_VERSION 0.1) + compat shim.** The `wingspan.compat` package with its v0.0 choice-encoding shim is in place, the v0.1 fixture set is committed, and the FRESH/REGIME versioning policy is enforced in code (inference loaders check the artifact version and refuse mismatches cleanly).

## Per-project verdict & gaps

### Project: General architecture exploration — ⚠️ needs sweep orchestration

`ModelArchitecture` exposes independent width lists for every submodel: card
encoder (`card_encoder_layers` → EMBED), state trunk (`trunk_layers`), choice
encoder (`choice_layers`), value head (`value_layers`), plus a separate setup
net (`setup_hidden_layers`). All six submodels are independently sizable, and
per-decision-family head sizing is also fully supported (see below).

- [ ] **Define the lite/heavy presets** per submodel (EMBED / TRUNK / CHOICE / per-head SCORER / VALUE / setup), as a small set of named `ModelArchitecture`/`TrainConfig` presets.
- [ ] **Sweep launcher.** Generate one run-file per (submodel→heavy, rest lite) cell plus an all-lite baseline, and start them as independent runs. The `wingspan-monitor` roster already compares live runs; what's missing is emitting the configs.
- [ ] **Convergence metric.** Define "time/iterations to converge" (e.g. eval-win-rate-vs-fixed-opponent crossing a threshold, or a plateau detector over `metrics.jsonl`) and compute it per run — not currently derived.
- [ ] **Head-to-head tournament + Elo.** `wingspan play` plays A-vs-B by checkpoint path, but there is no round-robin matrix / Elo aggregation across the swept checkpoints to answer "play them against each other."

### Project: Does the main-action model know which card it wants? — ⚠️ primitives exist, harness does not

- [ ] **Counterfactual-scoring harness.** Reach a `macro_action` decision with >1 playable bird (replay from `seed`, or snapshot states during collection), score "play a bird" with `policy_probs`, then re-encode with each candidate bird removed from the hand and re-score; record the per-bird delta. Requires the state-replay/perturbation utility (see cross-cutting gaps).
- [ ] **Cross-head readout.** Capture the `play_bird` head's pick at the same decision point to correlate "the bird the macro head weighted most" with "the bird actually played."
- [ ] **Stats layer.** How often a removal moves the macro score, and the correlation between the weighted bird and the play-bird selection. (Single-machine scale; not infra-blocked.)

### Project: Does the model learn not to add eggs when it can't? — ⚠️ handler + analysis script needed

The instrumentation framework removes the need for a replay/snapshot harness —
a `MadeDecisionHandler` subclass can filter for the exact scenario
(`GainEggsDecision` with zero available egg capacity and no activatable
brown-power bird) and write matching rows to JSONL during live collection. The
engine hooks are already firing; this is now purely a "write the handler"
problem.

- [ ] **Predicate handler.** Implement a `CallbackHandler` (subclassing `MadeDecisionHandler`) that inspects the game state at `MADE_DECISION` time, flags the no-op-eggs scenario, and records the policy's choice to JSONL. Register it in the `instrumentation` handlers package and wire it via `InstrumentationConfig`.
- [ ] **Enable via run config.** Add the handler to a stats-collection run's `InstrumentationConfig`; the config is already serializable so a cloud run file supports it without code changes.
- [ ] **Baseline rate.** Analysis script: read the output JSONL to compute the scenario's rate and compare to the overall decision-point rate.

### Project: Setup card stats — ✅ closest to runnable; ⚠️ "impact on outcome" is the expensive half

Scoring the 504 options for a hand is already implemented (`enumerate_setup_candidates` + `SetupNet`/`select_by_margins`). The missing pieces are the sampler and the aggregation, both embarrassingly parallel.

- [ ] **Mass-sampling harness.** Sample a few million deals (5 birds + 2 bonus per seat from the deck), enumerate + score, take the policy's pick (argmax), and tally per-card "selected-when-present" frequency — Q1, and the "ideal hand" (Q3). Built entirely on existing primitives.
- [ ] **Outcome-impact decision (Q2).** Selection frequency ≠ outcome impact. Either use the predicted-margin / value head as a *proxy* (cheap, available now) or actually **play games to completion** from each setup and measure win rate (expensive — shares the per-card outcome logging of Project 5). Pick one and state it.
- [ ] **Handle both setup paths.** With `use_setup_model=False`, setups are scored by the main net's SETUP head instead of `SetupNet`; the harness should cover both or standardize on the setup model.

### Project: General card stats (GIH / GOB win rate, per-slot) — ⚠️ engine hooks exist, handler + aggregation needed

The instrumentation framework has closed the "no engine hooks" gap from the
original assessment. `BIRD_PLACED` delivers bird, habitat, and slot (via
`played_bird`); `SETUP_APPLIED` delivers the opening keep; `PLAYER_FINAL_SCORED`
delivers the per-seat score breakdown; and `GAME_END` is where the winner is
known. A multi-event `CallbackHandler` spanning these four events can accumulate
per-game card visit data and write a per-game record on `GAME_END`, with
GIH/GOB/per-slot win rate computable purely from the collected JSONL. The
`CardVisitRecorder` bundled handler is a direct template.

- [ ] **Card-stats handler.** Implement a `CallbackHandler` subclassing `BirdPlacedHandler`, `SetupAppliedHandler`, `PlayerFinalScoredHandler`, and `GameEndHandler`, accumulating per-seat bird-play data (with habitat and slot) and opener cards, and writing one JSONL row per game on `game_end` with the winner linked in. Register and wire via `InstrumentationConfig`.
- [ ] **Fixed-model stats-collection run.** Drive `mp_collect` (or a simpler single-process driver) with the locked checkpoint and the card-stats instrumentation config enabled. No new code needed beyond the handler — just a run config.
- [ ] **Aggregation.** GIH / GOB / per-slot win-rate tables with sample sizes and CIs, reading the per-game JSONL output.

### Project: Are there low/high-scoring games? — ✅ runnable now (analysis script only)

`GameOutcome` already stores both seats' full score breakdowns per game in `games.jsonl` (local and offloaded to S3), so no new collection is required.

- [ ] **Correlation script.** Read `games.jsonl` (or the S3 game chunks) and compute corr(P0 total, P1 total) and per-component correlations.
- [ ] **Fold into standard analysis.** Add the P0/P1 score correlation to the per-run metrics/dashboard so it is tracked on every run, as the proposal suggests.

## Cross-cutting gaps (shared by several projects)

- [ ] **An offline-analysis package** (e.g. `wingspan.analysis`) that loads a checkpoint via `from_model_config` and hosts the counterfactual scorer (P2), the setup sampler (P4), and the stats aggregators (P4–P6) on the existing primitives. Handler implementations for P3 and P5 live in `wingspan.instrumentation.handlers` instead.
- [ ] **A games-log reader/aggregator** that consumes `games.jsonl` locally **and** the S3 `games/<session>/chunk_*.jsonl` chunks (P5, P6) — a small extension of `cloud.s3sync` (list + download many shard objects).
- [ ] **A state-replay/perturbation utility** keyed off `seed` (reach decision *k*, snapshot/perturb the `GameState`, re-encode) — needed only for P2 now (P3 uses live instrumentation instead).

## Offloading to AWS (reusing the cloud stack)

The `deploy/` + `wingspan.cloud` stack already solves the hard parts — image,
S3 layout, `s3sync`, Spot interruptibility, an IAM task role scoped to the
bucket, the ECS cluster + Fargate Spot capacity provider, the run-task override
pattern, and the `wingspan-monitor` roster. The catch: **everything cloud runs
today is `loop.TrainingLoop`** (`HeadlessRunner` wraps only the trainer). The
research projects are mostly *fixed-model inference/stat jobs at volume* —
embarrassingly parallel and Spot-friendly — so the cleanest path is a second
**analysis-job** type on the same scaffold, run as map/reduce.

An important simplification from the instrumentation framework: a stats-collection
cloud run is now just a training run with a locked checkpoint (no gradient
updates) and an `InstrumentationConfig` specifying the data-capturing handlers —
no separate job mode needed for P3 and P5. For Projects 1, 2, and 4, a distinct
analysis-job type is still the right model.

- [ ] **Second container entrypoint / job mode.** Alongside the training runner, add an analysis runner that reads a *job-file* from S3 (mirroring `CloudRunFile`), runs a fixed-checkpoint shard, and writes partial results to S3 — keeping the Spot graceful-stop and resuming a shard from its last committed offset.
- [ ] **Sharding + launcher (the "map").** Split work into K shards — deal-seed ranges (P4, P5), decision-sample ranges (P2), or run-configs (P1) — upload the job-file(s), and `aws ecs run-task` K Spot tasks, each writing `analysis/<job>/shard_<k>.(jsonl|parquet)`.
- [ ] **Aggregation step (the "reduce").** A final task (or local) that reads all shards from S3 into the project's tables (per-card GIH/GOB, setup frequencies, score correlation) — and for P1, the head-to-head tournament + Elo across the per-config training runs.
- [ ] **Run analysis over existing training output.** Because ordinary runs already stream `games/<session>/chunk_*.jsonl` to S3, Project 6 (and Project 5, once the card-stats handler is written) can reduce directly over the game chunks of normal training runs — no separate collection job needed.
- [ ] **Right-size the task.** The current task definition is fixed at `cpu=2048 / memory=8192` (2 vCPU). `mp_collect` scales with cores, so large map jobs want either bigger task sizes (more vCPU per task) or many small tasks; parameterize this in the Terraform/launcher.
- [ ] **Reuse the monitor.** Emit a per-shard `status.json` so `wingspan-monitor` (or a small variant) can show analysis-job progress the same way it shows training runs.

**Bottom line.** Project 6 is runnable today with only an analysis script.
Project 4 needs only a sampler harness and an outcome-impact decision.
Project 3 needs a single handler implementation — the engine hooks are live, no
replay machinery required. Project 5 likewise needs a handler + aggregation
layer (the engine hooks exist via `BIRD_PLACED` / `SETUP_APPLIED`); neither 3
nor 5 requires the analysis-job cloud entrypoint. Project 2 still needs the
state-perturbation/replay utility plus the counterfactual harness. Project 1
needs a real architecture change (per-head sizing) plus sweep/compare
orchestration. The cloud stack needs an analysis-job entrypoint for Projects 1,
2, and 4; Projects 3, 5, and 6 can run over normal instrumented collection runs.