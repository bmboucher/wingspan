# TRAINING.md

A proposed training program for the per-family actor-critic described in
[`DECISIONS.md`](DECISIONS.md). This document is the *plan*, not the code:
it explains what to run, how big to run it, how to know whether it is working,
and when (and only when) to make the network bigger.

It is written for a reader who knows Wingspan and the broad strokes of machine
learning but not the day-to-day mechanics of reinforcement-learning (RL)
training. Terms are explained at first use and collected in a glossary at the
end (§11). Where a recommendation is non-obvious, the rationale is spelled out.

Everything quantitative below was **measured on this codebase** (current `main`,
RTX 4080, 12-core CPU, the `model.PolicyValueNet` defaults) rather than guessed.
The measured profile is in §1 and is the foundation for every later decision.

> **Status — as-built (read me first).** This document was written as a forward
> *plan*; most of it now ships. The live trainer is `python -m wingspan.training`
> (the `wingspan.training` package). **Already implemented:**
>
> - **Length-bucketed batches** (§4.2a) — `learner._BUCKET_EDGES =
>   (2,4,8,16,32,64,128,256,512,2048)`; the old single-pad-to-widest tensor is gone.
> - **Parallel collection** (§4.1) — a persistent CPU process pool
>   (`mp_collect.ProcessCollector`) *and* an in-process CUDA batched-inference
>   collector (`batched_collect.BatchedCollector`); per-iteration weight broadcast.
> - **The shared card embedding** (§6.3) — one card table is read everywhere a
>   bird appears (board, tray, hand, choices), via integer index stripes.
> - **A paired-game evaluation harness** (§7) — mirrored deals, 95% CI, mean
>   margin, and an *advancing frozen-opponent ladder* (random → gen1 → gen2 → …).
> - **Resumable checkpoints** (§5.1) — model + optimizer + config + metrics +
>   progress counters + git SHA + encoding-era stamp, written atomically as
>   `last.pt`/`best.pt`/`opponent.pt`, plus `metrics.jsonl` and `games.jsonl`.
> - **Seeding** of Python / NumPy / torch (§5); a **separate setup model** with an
>   opt-in actor-critic mode (§6.4–6.5); and a **`decision_delta` reward mode**
>   alongside the default `terminal_margin` (§2).
>
> **Device.** Both CPU and CUDA are supported (the §1.4/§4 "collect on CPU, update
> on GPU" framing is no longer a hard split): CPU runs collect through the process
> pool; CUDA runs the batched collector and the learner update on the GPU.
>
> **Still open:** a frozen multi-checkpoint **league + Elo** (§5.2/§7 — only
> the single advancing ladder exists today); flatten + segment-softmax (§4.2b);
> and per-card visit-count logging (§8).
>
> **Now shipped:** PPO + reuse epochs and GAE (§3.4) — see the `policy_loss`,
> `ppo_clip_eps`, `ppo_reuse_epochs`, and `gae_lambda` config knobs.
>
> Two caveats on the numbers below. (1) The §1.1 sizes have been **re-measured
> against current `main`**: state vector **795**, choice vector **215**,
> **21** decision classes, **~1.02 M** parameters with the shared embedding — the
> old 532 k / 13-heads-are-80%-of-params framing is obsolete (the heads are now
> ~39%, the trunk ~37%). (2) Checkpoints do **not** store RNG state, so a resumed
> session is *not* bit-for-bit identical — but every collected game is
> seed-reproducible from `config.misc.seed`, so the per-game logs are stable.
> Engine fidelity as of this pass: all core bird powers fire (including the four
> pink "when another player …" reactors) and all 16 round goals score, pinned by
> `tests/test_power_coverage.py` and `tests/test_round_goal_coverage.py`.

---

## 0. TL;DR — the program in one page

The single most important finding (still true): **the GPU is not the bottleneck.
The bottleneck is everything around it** — how data is generated, how it is
batched, how progress is measured. The infrastructure half of that program has
since been built; the algorithm and capacity half remains.

Priorities, in order. Each links to its section; ✅ = shipped, ☐ = still open.

1. ✅ **The three correctness/scaling defects are fixed** (§4.2, §4.3):
   - The choice-tensor is no longer padded to the *widest* decision in the batch;
     it is **length-bucketed** (edges `2…2048`), so the ≤4-option majority pads to
     4, not 504. The old "97.6 % padding, 11.4 GB peak" failure mode is gone.
   - The `MAX_CHOICES_HARD` assert that crashed collection on a wide
     `PlayBirdDecision` is **removed** — replaced by soft logging thresholds plus
     the play-bird cost split (the wide payment enumeration moved to
     `PayBirdFoodDecision`).
   - Collection routes by device: **CPU process pool** or **CUDA batched
     inference** (§4.1) — batch-of-one GPU inference is no longer on the path.

2. ✅ **An honest evaluation harness exists** (§7): mirrored paired games against a
   fixed reference opponent, a 95 % CI, mean margin, and an advancing
   frozen-opponent ladder. (Still open: a multi-checkpoint league + Elo.)

3. ✅ **PPO + GAE shipped** (§3.4). The algorithm upgrade is done: PPO's clipped
   reuse epochs and GAE are opt-in REGIME knobs (`policy_loss`, `reward_mode=gae`,
   `ppo_clip_eps`, `ppo_reuse_epochs`, `gae_lambda`). Defaults reproduce today's
   REINFORCE behavior exactly; minibatching (§3.3/§4.2b) is still deferred.

4. ✅ **Self-play is parallelized** (§4.1) — a persistent CPU worker pool with
   per-iteration weight broadcast, plus an optional CUDA batched collector.

5. ☐ **Only then consider a bigger network** (§6), and only on evidence of
   *underfitting* measured against the §7 yardstick. Note the parameter budget
   has shifted: with the shared embedding and the wider trunk, the per-family
   heads are now ~**39 %** of parameters and the shared trunk ~**37 %** — the
   trunk is no longer the cheap part, but it is still the thing that lifts every
   head at once when widened.

A concrete phased schedule with exit criteria is in §9.

---

## 1. Where the time and memory actually go (the measured profile)

You cannot size a training run sensibly without knowing the shape of the data
it produces. Here is this game, as the encoder and engine actually emit it.

### 1.1 Feature and parameter sizes

| Quantity | Value | Source |
|---|---|---|
| State vector length | **795** | `encode.state_size()` |
| Per-choice feature length | **215** | `encode.choice_feature_dim()` |
| Judgment-family heads (main net) | **12** | `decisions.active_decision_families(False)` |
| Distinct decision classes | **21** | `decisions.ALL_DECISION_CLASSES` |
| Total parameters (default arch) | **1,015,949** | `model.PolicyValueNet()` |

The main net carries **12** family heads, not 13: `setup` is its own model by
default (§6.4), so the `SETUP` head is excluded from `PolicyValueNet`.

Where the ~1 M parameters live has *inverted* since this document was first
written — the shared card embedding and the wider trunk pulled the centre of
mass out of the heads:

| Component | Parameters | Share |
|---|---|---|
| State trunk (shared) | 376,576 | 37.1 % |
| Per-choice encoder (shared) | 173,184 | 17.0 % |
| Card encoder + shared embedding | 37,056 | 3.6 % |
| Hand encoder (shared) | 32,704 | 3.2 % |
| **12 scoring heads** | **396,300** | **39.0 %** |
| Value head (shared) | 129 | 0.0 % |

The heads (39 %) and the shared trunk (37 %) are now comparable; the trunk is no
longer the "cheapest part." This still matters for §6, but the lesson flips: the
shared representation is now a first-class consumer of the parameter budget, so
widening it is a real (not free) capacity decision — just one that lifts every
head and the critic at once.

> **Term — MLP (multi-layer perceptron):** the plainest kind of neural network,
> a stack of `Linear` layers with a nonlinearity (here `ReLU`) between them.
> Each scoring head takes the concatenated trunk+choice context, runs it through
> a small MLP, and emits a single score.

### 1.2 A game's decisions, by judgment family

A self-play game records on average **~140 trainable decisions** (range 106–191
over 20 games; single-option forced moves are not recorded). But those decisions
are wildly unevenly distributed across the heads. The table below is the
original 20-game profile, including the `setup` row; note that `setup` is now
scored by its own separate model (§6.4), so it is no longer one of the main net's
12 family heads — but the data-imbalance lesson is unchanged:

| Family (one head) | Steps/game | Share | |
|---|--:|--:|---|
| `macro_action` | 52.0 | 37.2 % | ████████████ |
| `bird_acquisition` | 26.6 | 19.0 % | ██████ |
| `gain_food` | 19.8 | 14.1 % | █████ |
| `egg_placement` | 9.2 | 6.5 % | ██ |
| `bird_discard` | 8.8 | 6.3 % | ██ |
| `play_bird` | 6.7 | 4.8 % | █ |
| `commit_to_cost` | 5.2 | 3.8 % | █ |
| `egg_removal` | 4.3 | 3.1 % | █ |
| `habitat_placement` | 2.5 | 1.8 % | ▌ |
| `setup` | 2.0 | 1.4 % | ▌ |
| `spend_food` | 1.9 | 1.4 % | ▌ |
| `bonus_valuation` | 0.7 | 0.5 % | ▏ |
| `misc_rare` | 0.1 | 0.1 % | ▏ |

The spread is roughly **370×** between `macro_action` (52/game) and `misc_rare`
(0.1/game — just 2 examples across 20 games). This is the central fact about
training this network: **the heads do not learn at the same rate, because they
do not see the same amount of data.** Consequences threaded through the rest of
the document:

- The high-traffic heads (`macro_action`, `bird_acquisition`, `gain_food`)
  will converge quickly and could justify more capacity.
- The rare heads (`misc_rare`, `bonus_valuation`, `setup`, `spend_food`) are
  starved. `misc_rare` needs ~10,000 games just to accumulate ~1,000 examples.
  Adding neurons to a starved head only helps it *overfit* faster (§6).
- `setup` is a special case: only 2 examples per game (one opening per player),
  yet it is the highest-stakes decision in the game and has a 504-wide menu.
  Rare **and** high-variance **and** high-dimensional — it deserves its own
  treatment (§6.4), exactly what the separate setup model now provides
  (DECISIONS.md §2.13).

### 1.3 Choice-set sizes, and the padding problem

Almost every decision is small, but a few are enormous:

| Choices in the decision | Share of all decisions |
|---|--:|
| ≤ 4 | **89.5 %** |
| 5–20 | 6.2 % |
| 21–100 | 1.5 % |
| 101–504 | 1.9 % |

> **Historical (pre-bucketing).** The measurements in this subsection describe
> the original single-pad-to-widest training step. They are kept because they are
> the *motivation* for the fix — length-bucketing (§4.2a) — which has since
> shipped, so the pathological numbers below no longer occur. The choice-feature
> width is also now **215**, not the 260 used in the worked example below.

89.5 % of decisions offer four options or fewer. But the opening draft offers
**504**, and a food-rich late-game `PlayBirdDecision` can offer **several
hundred** (we observed 376, 370, even 637). The old training step stacked
*all* decisions in a batch into one tensor padded to the widest one:

> **Term — padding & masking:** neural nets want rectangular tensors, but our
> decisions have different numbers of options. The fix is to pad every decision
> out to the largest option-count `K` in the batch with dummy rows, and carry a
> *mask* marking which rows are real so the dummies get zero probability. Cheap
> when option-counts are similar; ruinous when they are not.

Measured on a default 32-game batch (4,428 steps):

- Padded choice tensor shape: **(4428, 504, 260)** → 2,231,712 option-slots.
- Real option-slots: **52,560** → **only 2.4 % of the tensor is real data.**
- Memory for that tensor: **2.32 GB** (float32). Length-bucketed: **~0.055 GB**
  — a **42× reduction**.
- One update on this batch: **1.78 s on the GPU** (32 s on CPU), peaking at
  **11.41 GB** of GPU memory.

On a 16 GB RTX 4080, the *default* configuration used 71 % of the card purely to
store padding, and doubling the games would have run it out of memory. This was
the highest-leverage single fix in the document; it is now done (§4.2a).

### 1.4 Throughput: collection dominates, the update is trivial

| Operation | Time | Notes |
|---|--:|---|
| Self-play, 1 game, CPU | **163 ms** | batch-of-one inference per decision |
| Self-play, 1 game, GPU | **320 ms** | 2× *slower* — transfer/launch latency |
| Gradient update, 4,428 steps, GPU | 1.78 s | per *iteration*, not per game |
| Gradient update, 4,428 steps, CPU | 32 s | — |

The asymmetry is the whole story of §4. Generating data is sequential, Python-
bound, and slow; the gradient update on a ~1 M-parameter net is nearly free on a
modern GPU. **A single sequential collector leaves the GPU mostly idle, waiting
for the next batch of games.** Every throughput recommendation follows from this
— and the two collectors that answer it (CPU process pool, CUDA batched
inference) have since shipped (§4.1). The per-game timings below were measured on
the original sequential path and are kept as the motivating profile.

---

## 2. A five-minute RL vocabulary

The training loop uses a handful of terms repeatedly. Read once; refer back as
needed.

- **Episode / game.** One full Wingspan game, start to final scoring. The unit
  of data generation.
- **Step / transition / decision.** One moment where the agent picked among ≥2
  options. ~140 per game here.
- **Policy (π).** The thing we are learning: a function from "game state +
  list of options" to a probability for each option. In this codebase it is
  `softmax` over the per-family head's scores.
- **Self-play.** Both seats are driven by the *same* network. Symmetry comes
  from encoding the board from the deciding player's point of view (DECISIONS.md
  §0) and giving the two seats opposite-signed rewards.
- **Reward / return (G).** What we want to maximize. Two orthogonal knobs:
  - `reward_mode` controls *how credit spreads* across decisions. Default
    (`terminal_margin`): the end-of-game value is broadcast flat to every step.
    Opt-in (`decision_delta`): each decision's return is the
    `reward_discount` (γ)-discounted sum of the **per-decision value changes**
    from that step onward, so a decision is credited only with what it actually
    moved. γ=0 keeps only the immediate change; γ=1 telescopes to the final
    value minus the value before that decision.
  - `reward_basis` controls *what value* is used. Default (`margin`): own score
    minus opponent's — the two seats get opposite signs, gradient pushes toward
    *winning*. Opt-in (`own_score`): each player's own absolute final score —
    both seats receive positive rewards, gradient pushes toward *scoring more*
    regardless of the opponent. The two axes are fully independent (four
    combinations). All are shape-preserving (REGIME — no checkpoint restart).
- **On-policy.** An algorithm that may only learn from data generated by the
  *current* policy. REINFORCE and PPO are on-policy; this is why data is thrown
  away after each update and why fresh games must be generated constantly.
- **Policy gradient / REINFORCE.** The simplest on-policy method: "nudge the
  network to make the actions it took in winning games *more* likely and the
  actions in losing games *less* likely," scaled by how much it won/lost by.
- **Critic / value function V(s).** A second output that *predicts* the return
  from a given state. Used as a **baseline**.
- **Advantage A = G − V(s).** "Did this game turn out better or worse than the
  position deserved?" Subtracting the critic's prediction does not change *what*
  the policy converges to, but it dramatically shrinks the *noise* in the
  gradient — the single most important variance-reduction trick in policy-
  gradient RL.
- **Entropy bonus.** A nudge that rewards the policy for keeping *some*
  uncertainty, so it keeps exploring instead of collapsing onto one move too
  early.
- **Iteration vs. epoch (these are different here — see §3.2).** An *iteration*
  is one collect-then-update cycle. An *epoch*, in PPO, is one reuse pass over
  the batch you just collected.

---

## 3. The training loop we should run

### 3.1 The shape of one iteration

The synchronous baseline — start here because it is the easiest to get
*correct*, then scale (§4) and strengthen (§3.4):

```
repeat (one ITERATION):
  1. COLLECT a batch of N self-play games with the current weights
     (CPU, in parallel — §4.1). Record every multi-option decision.
  2. COMPUTE each step's return G (final margin, normalized — or per-decision
     discounted margin deltas when `reward_mode = decision_delta`) and, with the
     critic, its advantage A = G − V(s). Normalize advantages (§3.3).
  3. UPDATE the weights on that batch (GPU): policy loss + value loss
     − entropy bonus, minibatched, with PPO clipping (§3.4).
  4. DISCARD the data (it is now off-policy) and loop.
  periodically: EVALUATE vs. fixed opponents (§7) and CHECKPOINT (§5).
```

The `wingspan.training` package implements steps 1–4 today: REINFORCE with a
value baseline and **advantage normalization**, a **length-bucketed** update
(one forward/backward per bucket, losses summed — §4.2a), parallel collection,
periodic paired evaluation, and atomic checkpointing. What it does *not* yet do
is PPO's clipped reuse epochs and GAE (§3.4). The recommendations below evolve
it; they do not throw it away.

### 3.2 How big should an "epoch" be?

This is the question with the most jargon hiding in it, so first untangle two
meanings of "epoch":

- In **supervised learning**, an epoch is one pass over a *fixed* dataset. RL
  has no fixed dataset — the policy generates its own data — so this meaning
  does not apply.
- The RL analog is the **iteration**: collect a fresh batch of games, update,
  throw the data away. "How big is an epoch" really means **how many games per
  iteration** (the *batch size*).
- **PPO adds a third meaning** (§3.4): within one iteration it does several
  *optimization epochs* = reuse passes over the just-collected batch (typically
  3–10), because its clipping makes limited reuse safe. We will call those
  *reuse epochs* to keep them distinct.

**Count games, not steps.** It is tempting to size the batch by the ~140
steps/game. Resist it: because the reward is a *single* end-of-game margin
shared across all 140 of a game's decisions, those 140 steps are heavily
*correlated* — they share one outcome. For the policy-gradient *direction*, one
game is closer to **one** noisy label than to 140 independent ones. So the batch
must contain enough **games** that a few lucky/unlucky outcomes do not dominate
the gradient.

**Recommended starting batch: 64–256 games per iteration** (≈ 9 k–35 k steps).

- Start at **128 games**. It is large enough that the gradient is not at the
  mercy of single-game luck, small enough to iterate quickly, and (after the
  §4.2 padding fix) trivial on GPU memory.
- **How to tune it from there:** watch the evaluation curve (§7). If progress is
  jagged — the policy lurches up and down between iterations — the batch is too
  small (gradient too noisy); double it. If progress is smooth but you are
  burning compute, you can shrink it. The principled tool is the *gradient noise
  scale* (the batch size at which the gradient stops being mostly noise), but in
  practice the eval-smoothness heuristic is enough.
- **The trade-off in one line:** bigger batch → lower-variance gradient → can
  use a higher learning rate and take more confident steps, but fewer updates
  per game generated, so each game of data is used less aggressively.

`setup` decisions (2/game) and `misc_rare` (0.1/game) will accumulate *very*
slowly at any batch size — see §6.4 for what to do about that. Batch size does
not fix data starvation; it only controls gradient noise.

### 3.3 How — and how often — to update the weights

**The optimizer and the loss.** Keep `Adam` (a robust default optimizer that
adapts the step size per-parameter). The loss combines three terms, exactly as
the current code does, and this part is already right:

```
loss = policy_loss  +  VALUE_COEF · value_loss  −  ENTROPY_COEF · entropy
```

- `policy_loss = −(log π(chosen) · advantage).mean()` — push up the probability
  of actions that beat the baseline.
- `value_loss  = MSE(V(s), G)` — train the critic to predict the return.
- `entropy`    — subtracted, i.e. *maximized*, to preserve exploration.

Recommended coefficients (the current defaults are reasonable starting points):
`VALUE_COEF = 0.5`, `ENTROPY_COEF = 0.01` (anneal toward `0.001` late in
training so the policy is allowed to sharpen), gradient clipping at `5.0`
(already present — caps the update size so one freak batch cannot wreck the
weights).

**Both of the changes this section originally called for have shipped:**

1. ✅ **Advantage is normalized per batch** — `learner` centres the advantage and
   divides by its standard deviation (`_ADV_STD_EPS = 1e-6`) before the policy
   loss, so gradient magnitudes are stable from the first iteration to the last
   regardless of how good the critic is. The raw return is scaled by
   `training.score_norm` first; normalization removes any remaining dependence on
   that constant's exact value.

2. ✅ **Epsilon-greedy is gone.** Collection samples purely from the policy
   softmax (the setup model has its own `policy_temperature` knob). This keeps the
   update on-policy without importance weighting; exploration is controlled by the
   entropy bonus, as intended.

**How often to update.** One update *per minibatch*:

> **Term — minibatch:** rather than computing the gradient over all ~18 k steps
> at once (memory-heavy, and only one update per iteration), split the batch into
> chunks of a few thousand steps and take one optimizer step per chunk. More
> updates per iteration, lower peak memory.

Recommended: **minibatch ≈ 4,096 steps**, shuffled. With a 128-game iteration
(~18 k steps) that is ~4–5 minibatches; with PPO's reuse epochs (next), ~3–4
passes over them → ~15 updates per iteration.

**Actor weight freshness.** Because the method is on-policy, the games must be
generated by *current-ish* weights. In the synchronous loop this is automatic
(collect, then immediately update). When you parallelize (§4.1), refresh each
worker's copy of the weights **every iteration**. PPO tolerates the small
staleness that introduces; plain REINFORCE does not, which is another reason to
adopt PPO before going asynchronous.

### 3.4 The algorithm upgrade path

Do not jump straight to the fanciest algorithm; climb this ladder and stop when
the strength curve (§7) plateaus.

1. ✅ **REINFORCE + value baseline + advantage normalization** (shipped). Gets a
   clean baseline strength curve. Enough to crush a random opponent and validate
   the whole pipeline.

2. ✅ **PPO (Proximal Policy Optimization)** — **shipped**. Set via
   `policy_loss = "ppo"` in the training config.
   - **What it adds:** it lets you safely take *several* gradient passes over
     each collected batch (the "reuse epochs" of §3.2) instead of one, by
     *clipping* the update so the new policy can't move too far from the policy
     that generated the data in a single iteration. More learning per game
     collected — directly attacks our collection bottleneck (§4) — and much more
     stable than multi-pass REINFORCE.
   - **Why it matters here:** games are expensive (§1.4) and data is discarded
     after each iteration. PPO extracts 3–10× more signal from each batch before
     throwing it away.
   - **Config knobs** (all REGIME — no checkpoint restart):
     - `policy_loss = "ppo"` — enables clipped surrogate; default `"reinforce"`
       reproduces today's behavior exactly.
     - `ppo_clip_eps = 0.2` — clip radius ε in `clip(ratio, 1±ε)`.
     - `ppo_reuse_epochs = 4` — reuse passes per collected batch; 1 = single pass.
   - **Note:** full-batch reuse (no minibatching). Minibatch shuffling (§4.2b)
     is still deferred as a follow-up.
   - **Diagnostics:** `clip_fraction` and `approx_kl` are logged each iteration
     and appear in `metrics.jsonl` (0.0 on the single-pass / DAgger path).

3. ✅ **GAE (Generalized Advantage Estimation)** — **shipped**. Set via
   `reward_mode = "gae"` alongside `policy_loss = "ppo"` for variance reduction.
   - **The problem it solves:** today every one of a game's ~140 decisions is
     credited with the *same* terminal margin. A brilliant turn in a game you
     lost gets a negative signal; a blunder in a game you won gets a positive
     one. Over many games this averages out, but slowly and noisily.
   - **What GAE does:** uses the critic's value estimates of *intermediate*
     states to assign each decision a more *local* advantage — "how much better
     did the position get right after this move?" — instead of waiting for the
     final score. A knob λ (≈0.95) trades off the high variance of waiting for
     the end against the bias of trusting the critic. With λ=1,γ=1 it reduces
     exactly to the `decision_delta` advantage `G/score_norm − V`; as λ→0 it
     collapses to a one-step TD residual.
   - **Config knob** (REGIME): `gae_lambda = 0.95` (visible in the configurator
     when `reward_mode = "gae"`).
   - **Implementation note:** `behavior_logp` (old log-prob) and `value_pred`
     (critic V(s) in normalized-return units) are captured at collection time and
     stored on each `Step`. The `gae_advantages` kernel in `timestamps.py` runs a
     torch-free backward sweep per player per game.

**Beyond PPO** there is the AlphaZero family (search-guided self-play with MCTS).
It is powerful but a poor early fit for Wingspan: the game has **hidden
information** (opponent hand, deck order) and **chance** (dice, shuffles), which
break vanilla MCTS and require determinization machinery. PPO self-play is the
pragmatic path and aligns better with the project's analytical goal — the
per-family heads and the value head give *direct, interrogable* readouts
(DECISIONS.md §2), whereas a search wrapper hides the policy's reasoning behind
the tree. Revisit AlphaZero only if PPO self-play plateaus below your strength
target.

---

## 4. Making the GPU earn its keep (throughput and memory)

Recall the core asymmetry (§1.4): collecting games is slow and sequential; the
gradient update is nearly free. The job of this section is to keep the GPU fed.

### 4.1 Decouple actors from the learner; parallelize collection

> **Status: shipped.** Both collectors described here exist. On CPU,
> `mp_collect.ProcessCollector` runs a persistent worker pool (reused across
> iterations to amortize Windows spawn cost, capped at 16 workers) with weights
> broadcast each iteration via a versioned on-disk `_mp_weights.pt`. On CUDA,
> `batched_collect.BatchedCollector` runs the "batched-inference actor loop"
> described at the end of this subsection — many games stepped concurrently
> through one batched forward pass. The collector is chosen by `misc.device`.

The standard scalable-RL architecture, scaled to one machine:

- **Actors** — a pool of worker *processes* (not threads — Python's GIL
  serializes the game logic; only `torch`'s matmuls release it), each holding a
  **CPU** copy of the weights, each playing games and shipping finished
  `Trajectory` objects back over a queue. `Trajectory` is a Pydantic model
  holding NumPy arrays, which pickles across the process boundary cleanly.
- **Learner** — the main process: pull trajectories until it has a full
  iteration's worth, run the GPU update (§3), then broadcast the new weights to
  the actors.

**Why CPU actors.** Measured: batch-of-one inference is **2× slower on the GPU**
(320 vs. 163 ms/game) because each of a game's ~140 decisions is a tiny forward
pass dominated by transfer and kernel-launch latency, not arithmetic. Keep
self-play on CPU; reserve the GPU for the one place batching pays off — the
update.

**The throughput math** (at 163 ms/game/core):

| Setup | Games/sec | Games/hour | Time to 1,000,000 games |
|---|--:|--:|--:|
| 1 core (today) | 6.0 | ~22 k | ~46 hours |
| ~10 worker cores | ~60 | ~216 k | **~5 hours** |

The 12-core box can comfortably run ~10 actors and still service the learner.
That is the difference between a multi-day run and an overnight one.

**How much data is "enough"?** For the analytical payoff (per-card power
rankings, bonus-card value, opening theory), the binding
constraint is that each of the 180 birds must be *seen, played, and evaluated*
many times across many contexts. As an order of magnitude, expect to need
**10⁵–10⁶ games** before per-card readouts stabilize, and far more for the
rarest cards. Treat this as a hypothesis to *measure*, not a fixed target:
log a per-card visit count (how often each bird was offered/played) and watch
when the quantities you care about stop moving. The throughput table is why
§4.1 is a priority — without it, 10⁶ games is a weekend; with it, an afternoon.

**The batched-inference alternative (now shipped as `batched_collect`).** Instead
of many processes, a single loop steps many games forward concurrently and
collects their current decisions into one batched forward pass — turning the
batch-of-one GPU penalty into a GPU *win*, and avoiding process overhead. It is
fiddlier because games desynchronize (different games reach decisions at
different times); `BatchedCollector` handles this with a batch-inference server
that blocks until every live game has a pending request, then runs one
padded/masked forward and hands results back. It is the CUDA-path collector;
the CPU path uses the process pool.

### 4.2 Kill the padding (the 42× memory win)

§1.3 measured that 97.6 % of the choice tensor was padding and that the default
update peaked at 11.4 GB. Two fixes, in increasing order of cleanliness:

**(a) Length-bucketing — the quick win. ✅ Shipped.** `learner` groups the batch's
steps into buckets by option-count (`_BUCKET_EDGES = (2, 4, 8, 16, 32, 64, 128,
256, 512, 2048)`, each step assigned to the smallest edge ≥ its option count) and
pads *within* each bucket. The 89.5 % of decisions with ≤4 options pad to 4, not
504. Each bucket gets one forward/backward and the losses are summed over a single
shared backward pass — roughly a 40× memory reduction versus the old flat pad.

**(b) Flatten + segment-softmax — the clean design. ☐ Still open.** Eliminate
padding
entirely. Concatenate every candidate across the whole batch into one
`(ΣK, choice_dim)` matrix (here ΣK ≈ 52 k, not 4428×504 ≈ 2.2 M), run the
choice-encoder once over it, gather each candidate's decision-context, score,
and do a *segmented* softmax — a softmax computed independently per decision
using a per-candidate "which decision do I belong to" segment id (via
`scatter`/`index_add`-style reductions). A 2-option decision and the 504-option
draft are then handled by identical code with zero waste, and even the bucketing
(§4.2a) becomes unnecessary. More work than bucketing; adopt it when the encoder
is touched next.

### 4.3 The choice-count crash is gone ✅

The original `MAX_CHOICES_HARD = 600` `assert` aborted collection if any decision
exceeded it. A food-rich `PlayBirdDecision` enumerated one candidate per
`(bird, habitat, payment)` combination and could exceed 600 — we hit **637** on
the first 32-game attempt with fresh weights, a *trajectory-dependent* crash that
would have ambushed a long unattended run hours in. Two fixes landed and together
remove it:

- **The hard assert is removed.** `encode.layout` now carries only soft logging
  thresholds — `SOFT_CHOICE_WARN_THRESHOLD = 20` (log a warning above this) and
  `RUNAWAY_CHOICE_THRESHOLD = 10000` (log a likely-bug warning) — and the encoder
  never truncates or aborts. Width is handled downstream by length-bucketing
  (§4.2a), whose largest bucket edge is 2,048.
- **The play-bird cost split** shrank `PlayBirdDecision` to one candidate per
  `(bird, habitat)` pair (bounded by hand size × 3). The payment enumeration moved
  to the follow-up `PayBirdFoodDecision`, whose width is the payment count for a
  single bird — so the food-rich wild-payment blow-up, if it recurs, is bounded
  there and absorbed by the buckets, not a crash.

### 4.4 Headroom for later

These matter once the model is bigger (§6); they are noise at 0.5 M params, but
worth knowing they exist:

- **Mixed precision (`torch.autocast` with bf16).** The RTX 4080 (Ada) runs
  bf16 roughly 2× faster than fp32 and halves activation memory. Free speed once
  the update is compute-bound.
- **`torch.compile(net)`** (PyTorch 2.x) fuses kernels for a modest speedup;
  the compile cost only pays off on a larger model.
- **Pinned host memory + `non_blocking=True`** on the host→GPU batch transfer
  overlaps copy with compute.
- **Keep the optimizer state on the GPU**; only the freshly-collected batch
  crosses the bus each iteration.

---

## 5. Checkpointing and reproducibility

This section's program is **largely shipped.** Checkpoints are now full and
resumable, written atomically every iteration; what remains open is the frozen
multi-checkpoint *league* (§5.2) and per-card logging (§8). The original
motivation — the old code saved `{model, args}` once at the very end, inadequate
for hour-long runs and for a project whose *output is analysis* — is preserved
below for context.

### 5.1 What a checkpoint must contain

A checkpoint has to let you (a) resume a crashed run *exactly*, and (b) re-derive
results later. **As shipped**, `loop_checkpoint.atomic_save` writes a single dict
per checkpoint containing: `config` (the full `RunConfig.model_dump()`), `model`
(`state_dict`), `optimizer` (Adam moments), `metrics` (this iteration's
`IterationMetrics`), `progress` (resumable counters — iteration, opponent
generation, phase), `git_sha`, and `version` (the run's *encoding era*, not the
live `MODEL_VERSION` — era-pinned resume, see `docs/VERSIONING.md`). The one thing
it deliberately does **not** carry is RNG state, so resume is not bit-for-bit
identical (each game is still seed-reproducible from `config.misc.seed`).

The illustrative sketch below predates the real `RunConfig` (a nested Pydantic
config — `architecture` / `run` / `training` / `opponent` / `engine` / `misc`),
but conveys the intent of a self-describing checkpoint:

```python
# illustrative — house style: Pydantic config, module-qualified imports
class TrainConfig(pydantic.BaseModel):
    """Every hyperparameter, versioned, so a checkpoint is self-describing."""
    games_per_iter: int = 128
    minibatch_steps: int = 4096
    reuse_epochs: int = 4
    lr: float = 3e-4
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    gae_lambda: float = 0.95
    # architecture descriptor — detect incompatibility on load
    hidden: int = 128
    state_dim: int
    choice_dim: int
    family_order: tuple[str, ...]  # decisions.ALL_DECISION_FAMILIES, as strings

# saved per checkpoint:
#   config          : TrainConfig.model_dump()
#   model           : net.state_dict()
#   optimizer       : optimizer.state_dict()        # Adam's momentum, etc.
#   scheduler       : scheduler.state_dict()        # if an LR schedule is used
#   iteration       : int                           # for resume + LR schedule
#   total_games     : int
#   git_sha         : str                           # which code produced this
#   rng             : python / numpy / torch / cuda RNG states
```

Two details that are easy to skip and painful to omit:

- **The optimizer state.** Adam keeps a running estimate of each parameter's
  gradient statistics; dropping it on resume causes a visible training hiccup.
- **The architecture descriptor + family order.** `ALL_DECISION_FAMILIES` is
  documented as append-only precisely so head→family alignment survives across
  checkpoints (`decisions.py`). Storing the order *in* the checkpoint lets a
  loader *verify* it rather than silently misroute heads if the enum ever
  changes.

### 5.2 Cadence, retention, and safety

- ✅ **Every iteration:** atomically overwrite `last.pt` (`atomic_save` writes a
  temp file then moves it, so a crash mid-write cannot corrupt the checkpoint).
- ✅ **Gated on evaluation (§7):** keep `best.pt`, updated only when eval strength
  improves within a generation. The last checkpoint is not always the best one.
- ✅ **The advancing opponent:** `opponent.pt` is frozen whenever the policy beats
  the current reference, forming the §7 ladder (random → gen1 → gen2 → …).
- ☐ **Still open — the immutable `iter_{n}.pt` league.** Today only the *single*
  advancing opponent is retained, not a set of frozen past selves. A true league
  (many `iter_{n}.pt` snapshots used as evaluation *and* self-play opponents to
  stop the policy chasing its own tail) and the Elo over it (§7.3) remain to do.
- ✅ **Reproducibility hygiene:** Python, NumPy, and torch are all seeded at
  startup; the git SHA and full `RunConfig` travel in every checkpoint;
  per-iteration metrics append to `metrics.jsonl` and per-game outcomes to
  `games.jsonl` beside the checkpoints. A spreadsheet-grade log is enough to
  start; graduate to MLflow or Weights & Biases for hosted dashboards if wanted,
  but do not block training on that infrastructure.

---

## 6. Choosing and growing the architecture

The user's question — *how do we decide the architecture; when do we add layers
or neurons?* — has a disciplined answer that resists the urge to add capacity
first.

### 6.1 The meta-point: capacity is not this project's bottleneck

A ~1 M-parameter network is still *small*; the RTX 4080 could train 10–50 M
parameters for this game without strain. The data generation (§4.1), batching
(§4.2), evaluation (§7), and the shared card embedding (§6.3) are now sound, so
the remaining preconditions before spending capacity are the algorithm upgrade
(§3.4) **and** measured evidence of underfitting. Spend the capacity budget last.

### 6.2 The procedure (model-agnostic, and the answer to "when do I add neurons?")

> **Terms — underfitting vs. overfitting.** *Underfitting:* the model is too
> weak (or undertrained) to capture the pattern — it is mediocre on both the
> data it trained on and on fresh data. The cure is *more capacity or more
> training*. *Overfitting:* the model has memorized quirks of its training data
> and does worse on fresh data than on training data. The cure is *more data or
> more regularization* — **not** more capacity, which makes it worse. The whole
> game of sizing a network is telling these two apart, which requires the
> out-of-sample yardstick of §7.

1. **First prove the small model can learn at all.** It must crush the random
   agent (§7) decisively. If it cannot, the problem is a bug or the algorithm —
   *never* fix that by adding neurons.
2. **Add capacity only on evidence of underfitting:** the training objective
   stalls (policy entropy collapses onto a confident-but-mediocre policy; value
   MSE plateaus high) **and** eval strength plateaus below your target.
3. **Diagnose which way you are failing** using the train-vs-eval gap:
   - Both mediocre → **underfit** → add capacity (§6.3) or train longer.
   - Training strong, eval weaker → **overfit** → more games, more entropy /
     weight decay, smaller heads — not more parameters.
4. **Change one thing at a time**, and compare the two checkpoints by playing
   them **head-to-head** (§7) — a direct A-vs-B match is far more sensitive than
   comparing each one's win rate against random.
5. **Let capacity follow data** (§1.2): grow the heads that see lots of data;
   leave the starved ones small.

### 6.3 If/when you do scale, in priority order for *this* network

- ✅ **Share the card embedding (done).** A bird's identity used to be read by
  *two unrelated* weight matrices — the hand multi-hot through the state trunk,
  the candidate one-hot through the choice encoder — so "American Robin" had two
  separate learned meanings depending on whether it sat in your hand or on the
  table. It is now one shared card table (an `nn.Embedding` over the 180 birds,
  plus bonus cards), read everywhere a bird appears via integer index stripes
  (`bird_id`, `board_idx`; DECISIONS.md §1). The card's representation is
  consistent everywhere, it generalizes better, and — the project's headline goal
  — that single embedding table *is* the per-card power readout.
- **Widen the trunk before the heads.** The shared "read the board" trunk is the
  thing every decision and the critic depend on. It is no longer the cheap part
  of the net — at the default `(128, 128)` it is already ~37 % of parameters
  (§1.1) — so widening it to 256–512 is a real capacity decision, but it lifts
  every head and the critic at once, whereas widening a head helps only the
  families that exercise it.
- **Sit the heads on the data.** Give `macro_action`, `play_bird`,
  `bird_acquisition`, and `gain_food` more width; keep the rare heads small.

### 6.4 The starved and singular heads

`misc_rare` (0.1/game), `bonus_valuation` (0.7), `spend_food` (1.9), and `setup`
(2.0) will barely train no matter the batch size. Options, roughly in order of
effort:

- **Keep them deliberately small** so they cannot overfit their few examples
  (the design already pools three rare, unrelated judgments — which habitat to
  move a bird into, which played bird's power to repeat, and who gains food first
  — into `misc_rare` for exactly this reason — DECISIONS.md §2.10).
- **Lean on the shared trunk and shared card embedding** so a starved head
  inherits a good board representation and good card vectors, and only has to
  learn a thin final mapping.
- **Give them auxiliary supervision.** `bonus_valuation` and `setup` have
  analyzable structure — a bonus card's category and VP thresholds, an opening's
  affordable curve — that can become hand-computed training targets,
  letting these heads learn from heuristics instead of waiting for the rare
  on-policy signal (the bonus head's `bonus_delta` stripe now carries exactly
  these structured terms — DECISIONS.md §2.9).
- **Treat `setup` as its own model** (now the default, DECISIONS.md §2.13): it
  is rare, high-variance, high-dimensional, and game-defining. The separate
  setup net realizes this; it trains on-policy with REINFORCE from iteration 0
  by default, and can optionally use a random-generation bootstrap phase as
  an initializer (legacy warmup knobs, see §6.5).

### 6.5 The setup model's training schedule and actor-critic mode

#### Default: MODEL_DRIVEN from iteration 0

By default (`setup_record_start_iter=0, setup_train_iter=0`) the setup model is
in MODEL_DRIVEN mode from iteration 0 and trains on-policy with REINFORCE
immediately. No burn-in or offline-fit warmup occurs; the value and policy heads
learn purely from the games the setup net itself played. This is the "train from
start" regime and is the right choice when REINFORCE is enabled
(`setup_use_actor_critic=True`) because the policy gradient provides a useful
learning signal even from random-weight initializations.

#### Legacy warmup schedule (opt-in)

Set `setup_record_start_iter > 0` and `setup_train_iter > setup_record_start_iter`
to restore the three-phase warmup:

| Phase | Condition | Behaviour |
|---|---|---|
| RANDOM_NO_RECORD | `iter < setup_record_start_iter` | Random setups, nothing recorded (burn-in) |
| RANDOM_RECORD | `setup_record_start_iter ≤ iter < setup_train_iter` | Random setups recorded to disk |
| MODEL_DRIVEN | `iter ≥ setup_train_iter` | One-time offline MSE fit, then on-policy updates |

This was the original regime, designed for pure MSE regression where the value
head needed a stable bootstrap dataset before driving selection.

#### Actor-critic mode

The default `SetupNet` with only a value head is trained by MSE regression.
The problem is that **the regression target is high-variance and provides no
gradient to the selection mechanism** — the value head learns what a good setup
looks like, but the sampling distribution only improves via greedy re-ranking,
never via direct policy-gradient flow.

Set `setup_use_actor_critic = true` in the training config to enable an opt-in
actor-critic mode that adds a policy head and trains it with REINFORCE.

#### How it works

**Architecture.** When `setup_use_actor_critic=True`, `SetupNet` builds a second
`policy_mlp` head alongside the existing value `mlp`, with identical shape. Both
heads share the frozen card/hand embedder and the learned per-set embedding —
only the final readout layers are separate. Toggle `SetupArchitecture.use_policy_head`
in the shape key, so switching the flag invalidates and resets the setup checkpoint.

**Collection (MODEL_DRIVEN phase).** Instead of ranking candidates by the value
head and picking the top-scoring one, `play_game_with_setup` calls
`SetupNet.policy_and_value()` to get policy logits for all K candidates, then
samples via softmax (`setup_policy_temperature` still applies). The full
`(K, feature_dim)` candidate feature matrix is stored in
`SetupSample.all_candidates` (float16 after IPC compaction) along with
`SetupSample.chosen_idx`. These fields are **in-memory only** and are not
persisted to the JSONL store — the offline bootstrap format is unchanged.

**IPC cost.** Each sample carries a `(K, SETUP_FEATURE_DIM)` float16 array.
With K=504 (bonus included) and `setup_model.SETUP_FEATURE_DIM = 308` (current),
that is ~303 KB/sample, ~0.6 MB/game (two seats), ~155 MB/iteration at 256 games.
That is no longer negligible — re-measure against the IPC budget, and note that
`split_setup_bonus` (§6.6) halves K to 252, roughly halving this cost.

**Training (MODEL_DRIVEN phase).** One on-policy REINFORCE step replaces the
plain MSE `online_update`:

```
advantage = (margin / score_norm) − value_pred.detach()   # actor sees critic only as a baseline
log_probs  = log_softmax(policy_logits)                    # over all K candidates
loss = pg_coef   * (−log_probs[chosen_idx] * advantage)   # REINFORCE
     + value_coef * MSE(value_pred[chosen_idx], target)   # keep the critic honest
     − entropy_coef * H(softmax(policy_logits))            # exploration bonus
```

The advantage is clamped to `[−10, 10]` for stability. Samples without
`all_candidates` (e.g. RANDOM_RECORD samples replayed from the store) are
silently skipped — only the on-policy MODEL_DRIVEN games contribute.

**The offline fit at `setup_train_iter` is unchanged.** It trains the value head
via MSE on the bootstrap RANDOM_RECORD samples as before. The policy head first
trains on-policy once MODEL_DRIVEN collection begins.

**Hyperparameters.** Three coefficients control the loss blend:

| Config field | Default | Role |
|---|---|---|
| `setup_pg_coef` | `1.0` | Weight on the policy-gradient term |
| `setup_value_coef` | `0.5` | Weight on the value MSE term |
| `setup_entropy_coef` | `0.01` | Entropy bonus (keeps exploration alive) |

Start with the defaults. Raise `setup_entropy_coef` (0.05–0.1) if the policy
collapses to deterministic early; lower it once the policy is stable. The
`setup_policy_temperature` parameter (default 0.5) still governs sampling
independently of entropy regularization — they are complementary.

**This is a setup-FRESH change.** Toggling `setup_use_actor_critic` changes
`SetupArchitecture.shape_key` and will invalidate any existing `setup.pt`,
resetting the setup net. The main `PolicyValueNet` is not affected and does
not require a `MODEL_VERSION` bump.

### 6.6 Setup-split knobs: deferring bonus and food picks to in-game heads

Two optional flags move pieces of the opening judgment out of the setup model
and into the regular in-game decision heads. Both are REGIME (shape-preserving,
resumable) — they are gated on `use_setup_model` and never change tensor shapes.

**`split_setup_bonus`** (`bool`, default `False`; gate: `split_setup_bonus_active`):
When on, candidates drop the bonus axis (`bonus_card = None`; the setup
encoder's bonus block stays all-zero). The opening bonus pick is instead asked
as a normal `CHOOSE_BONUS` decision right after the keep (§2.9 of DECISIONS.md).
Concentrates all bonus-valuation signal in one place on-policy. Candidate set
shrinks from 504 to 252 for the standard 5-card / 2-bonus deal.

**`split_setup_food`** (`bool`, default `False`; gate: `split_setup_food_active`):
When on, candidates carry `kept_foods = ()` (food block all-zero;
`SETUP_FEATURE_DIM` unchanged). The opening food pick is asked as sequential
in-game decisions right after the keep, routing through GAIN_FOOD or SPEND_FOOD
depending on birds kept:

| Birds kept | Player starts with | Decisions |
|---|---|---|
| 0 | 5 food | none (all food kept) |
| 1 | 5 food | 1 × `SpendFoodDecision` |
| 2 | 5 food | 2 × `SpendFoodDecision` (no-repeat) |
| 3 | 0 food | 2 × `GainFoodDecision` (no-repeat) |
| 4 | 0 food | 1 × `GainFoodDecision` |
| 5 | 0 food | none (no food kept) |

This adds one on-policy sample per seat per game to the GAIN_FOOD / SPEND_FOOD
heads (§2.4, §2.5 of DECISIONS.md), which are among the most data-starved
heads (§6.4). When active, `setup_food_sets` is ignored — random setup
generation emits `kept_foods = ()` directly, skipping the biased food-sampling
and cross-product assembly that normal random generation performs.

Both flags can be active simultaneously. Neither touches `MODEL_VERSION` or
`setup_architecture_key`.

### 6.7 DAgger behavioral cloning (clone-then-RL)

#### What it does

By default every run starts from random weights. **DAgger** (Dataset Aggregation)
behavioral cloning lets a frozen prior checkpoint (the **expert**) teach a new
student network before the RL loop begins. For the first `clone_iters` iterations:

1. The **student** plays both seats of self-play (on-policy rollouts — DAgger's
   defining property vs vanilla BC: the labeled states are the ones the student
   actually visits, not the expert's).
2. At each decision, the frozen **expert** also receives the same legal options
   and produces its full soft policy distribution (`policy_probs`).
3. The expert's soft distribution is stored as `Step.expert_probs` (IPC-only;
   not persisted to `games.jsonl`).
4. The learner updates via **cross-entropy to the expert's soft targets** (plus
   the usual value-head MSE to warm the critic); the policy-gradient and entropy
   terms are disabled. The loss is:
   `loss = CE(student, expert) + value_coef * value_MSE`

After `clone_iters` iterations the expert is dropped, `Step.expert_probs` is no
longer computed, and the run reverts to the normal actor-critic REINFORCE loop.
A resumed run past `clone_iters` simply finds DAgger inactive — no config change
required.

#### Config

```yaml
# Run B clones from run A's checkpoint, then switches to RL at iteration 10:
train:
  run:
    checkpoint_dir: checkpoints-run-b
  opponent:
    bootstrap_opponent: none   # incompatible with DAgger clone phase (see below)
  dagger:
    expert_checkpoint: checkpoints/last.pt   # frozen expert; any prior .pt
    clone_iters: 10                          # pure imitation for iters 0..9
```

The configurator's EVAL section exposes both fields as a "dagger" group (alongside
the bootstrap group). `dagger_expert_checkpoint` cycles like `bootstrap_opponent`:
`none` / archived run paths / custom path. `clone_iters` is an int field.

#### Constraints

- **device=cpu required.** The expert runs one forward pass per decision in the
  `mp_collect` worker process — the same CPU-only per-game net machinery used by
  the bootstrap opponent. Setting `device=cuda` with an expert path raises a
  `ValueError` at startup. (Collection always uses `mp_collect` on CPU; only the
  learner's backprop step would benefit from CUDA, and it costs <0.2s/iter.)
- **bootstrap_opponent must be 'none' when clone_iters > 0.** The bootstrap phase
  puts the student at seat 0 only (vs random/opponent at seat 1), so the expert
  would label only seat 0's decisions. DAgger requires self-play — both seats are
  the student, so the expert labels both. The validator enforces
  `clone_iters > 0 ⟹ bootstrap_opponent == 'none'`.
- **clone_iters >= 1 when expert is set.** A path with `clone_iters=0` is a silent
  no-op that the validator rejects.

#### Cross-architecture expert

The expert may be a **different architecture or era** than the student. The expert
labels decisions at collection time through its *own* encoder (`expert_net.encode_state` /
`expert_net.encode_choices`), so its feature widths are self-consistent regardless of
the student's geometry. Candidate-index alignment is guaranteed by construction:
both nets receive the same `Decision` object; `encode_choices` iterates
`enumerate(decision.choices)`, so row `i` is candidate `i` for any net.

#### Family-skip guard (SETUP)

`SETUP` is always last in `decisions.ALL_DECISION_FAMILIES`. If the expert was
trained without the SETUP head (`include_setup=False`, i.e. `use_setup_model=False`
in its architecture) and the student has the SETUP head, SETUP decisions are not
labeled (`expert_probs=None`) and are masked out of the imitation mean via the
`has_expert` tensor. With matching architectures this never fires.

#### Collection throughput during the clone phase

The clone phase roughly **doubles the per-decision collection cost**: one student
forward + one expert forward per decision (~130/game), both in the same worker
process on CPU. Expect the dashboard's **g/s** readout to drop ~40–50% during
`clone_iters` iterations and recover fully once DAgger deactivates at iteration
`clone_iters`. The learner update phase is unchanged (the value head runs as
always; cross-entropy replaces the policy-gradient term, which is of similar
compute cost).

#### Dashboard metrics

During the clone phase `metrics.jsonl` carries a non-null `imitation_loss` (the
mask-weighted mean CE to the expert's soft targets). After `clone_iters` it is
`null`. The log events append `· DAgger clone` to the COLLECT line during the
clone phase.

#### Versioning

REGIME (no `MODEL_VERSION` bump, no compat shim). The `DaggerConfig` section
defaults to disabled; pre-DAgger configs validate unchanged. See `docs/VERSIONING.md`.

---

### 6.8 Using a pre-trained checkpoint as the bootstrap opponent

#### The run-A → run-B pattern

By default the bootstrap phase (`opponent.bootstrap_opponent: random`) trains
against the built-in random agent. Set `opponent.bootstrap_opponent` to a `.pt`
checkpoint path (any prior run's `checkpoint_dir/last.pt`, or any archived
checkpoint) to replace the random agent with a frozen greedy copy of that
checkpoint (`none` disables the bootstrap phase entirely):

```yaml
# Run A trains to completion (or any milestone), leaving checkpoints/last.pt.
# Run B opts into the run-A opener via its config — the FLIGHT PLAN configurator
# or a cloud run-file's nested opponent section:
train:
  run:
    checkpoint_dir: checkpoints-run-b
  opponent:
    bootstrap_opponent: checkpoints/last.pt   # frozen greedy copy of run A
```

This is useful when a second run needs a head start against a stronger opener —
the bootstrap opponent is much stronger than the random agent, so the run B policy
must work harder to graduate. The version-routing in `players.loaders.load_policy_net`
handles all prior artifact versions automatically (the COMPAT shim dispatch), so
any previously trained checkpoint works even if it was trained against an older
encoding layout.

#### CPU-only constraint

A checkpoint-path `opponent.bootstrap_opponent` is only valid when `device="cpu"`
is set. The `batched_collect` code path (CUDA, GPU learner)
has no opponent machinery — every vs-random game on the `mp_collect` path spawns
one opponent agent per game, but the batched collector runs all games in lockstep
without per-game opponents. Configuring the bootstrap checkpoint on a CUDA run
raises a `ValueError` at startup so the error is immediate and clear. Setting
`device="cuda"` is only relevant for the learner's backprop step; **collection
always uses `mp_collect` (CPU) while the bootstrap phase is active**, so this
constraint is not a practical limitation.

#### Setup-net limitation

The bootstrap opponent's **setup net is not loaded**. In runs where
`use_setup_model=True` the opponent's opening keep is resolved by the random
generator (same as random-agent bootstrap), even if the checkpoint was trained
with a setup model. Only the opponent's main game policy is loaded and replayed
greedy. This means the bootstrap-opponent is weaker in the opening than it would
be in a self-play game — the in-game play is faithful, but the starting position
may be slightly worse than the checkpoint's actual trained opening.

---

## 7. Evaluating out-of-sample performance

The core harness now exists (`training.evaluate` / `loop_eval`); what remains
open is the multi-checkpoint league and the Elo over it (§7.3). The reasoning
that motivated it is preserved below.

### 7.1 Why a raw self-play win rate is empty

A naive training log that reports "player 0 wins" measures nothing. In self-play
**both seats are the same network**, so that number reflects only the
first-player advantage plus noise — it sits near 50 % no matter how strong or
weak the policy is. It cannot tell you whether yesterday's run beat today's.
**As shipped**, evaluation instead plays the policy against a *fixed* reference
opponent (§7.3), which is the part that carries signal.

### 7.2 What "out-of-sample" means here

> **Term — out-of-sample / held-out.** Performance measured on situations the
> model was *not* trained on. In supervised learning that is a held-out slice of
> a dataset. In self-play RL there is no dataset to hold out, so out-of-sample
> means two concrete things: **(a)** games played from **fresh random seeds**
> (new deals, shuffles, dice) distinct from training seeds, and **(b)** play
> against **opponents the policy did not train against**.

### 7.3 The evaluation harness

> **Status.** `evaluate.evaluate_vs_opponent` plays `n_pairs` *mirrored* deals
> against a fixed reference opponent in greedy mode and returns an `EvalResult`
> with win rate, a 95 % CI, mean margin, and the opponent's generation. The
> reference is the **random agent** during the bootstrap phase, then an
> **advancing frozen self** (`opponent.pt`) once the policy graduates
> (`loop_eval`). ☐ Still open: a fixed heuristic agent, a *multi-checkpoint*
> league (only the single advancing opponent is kept today), and Elo over it.

Play a fixed suite of games against **reference opponents**, with the policy in
**greedy mode** (pick the argmax option, no sampling, no entropy — you are
measuring strength, not exploring):

- **The random agent** — the sanity floor. A learning policy should climb toward
  ~100 % win rate against it within the first hours; if it does not, stop and
  debug before doing anything else.
- **A simple heuristic agent**, if one exists or is cheap to write (e.g. greedy
  points-per-food) — a more demanding, *fixed* bar that does not move as the
  policy improves.
- **Frozen past checkpoints (the league)** — the most informative signal once
  the policy is decent: is iteration 500 actually beating iteration 100?

**Control the variance with paired (mirror) games.** Wingspan has a real
first-player and deal advantage. To stop that from masking the signal, play each
evaluation deal *twice* — once with your policy as player 0, once with seats
swapped on the **same seed** — and average. This cancels the deal/first-player
luck and gives a far tighter estimate from the same number of games.

**Report it honestly.** Win rate is a coin-flip statistic, so attach a
confidence interval: for `n` games the 95 % interval is roughly
`p ± 1.96·√(p(1−p)/n)`. To distinguish a 55 % win rate from 50 % you need on the
order of 400+ paired games; eyeballing 20 games proves nothing. Track, per
evaluation:

- win rate + CI and mean score margin vs. each reference;
- an **Elo rating** computed over the league (the chess-style relative-strength
  number — convenient because it summarizes "how much stronger than my past
  selves" in one monotone curve);
- per-family policy entropy and per-family loss (are the heads still learning,
  or have they collapsed?);
- gradient norm and value-loss (training-health signals).

### 7.4 What overfitting looks like in self-play (and how to catch it)

It is not memorizing a dataset; it is two subtler failure modes:

- **Chasing your own tail.** The policy becomes great at beating its *current*
  self while getting *worse* against older checkpoints — it has specialized to a
  self-play quirk rather than getting genuinely better. Detection: Elo vs. the
  frozen league stalls or drops even as self-play metrics look fine. Mitigation:
  occasionally draw self-play opponents from the league, not just the latest
  weights.
- **A value head that overfits** its returns and stops being a useful baseline.
  Detection: value-loss keeps falling on freshly-collected data but advantages
  stop being informative (policy progress stalls).

The single defense is the §7.3 yardstick: **fixed opponents and held-out seeds**.
When eval strength diverges from training metrics, trust eval.

---

## 8. What to log (the proportionate MLOps slice)

This is a research training loop, not a deployed service, so skip the
production-serving apparatus (containers, canaries, latency SLOs). The parts of
standard ML-ops that *do* apply are **experiment tracking** and **training-health
monitoring**. Most of this list now ships as `IterationMetrics` rows appended to
`metrics.jsonl`, plus per-game `GameOutcome` rows in `games.jsonl`. ✅/☐ marks
shipped vs open:

- ✅ iteration, wall-clock, total games, games/sec (throughput regression alarm);
- ✅ loss components: policy, value, entropy, total; gradient norm;
- ✅ advantage mean/std (sanity on normalization);
- ◑ per-family **step counts** this iteration (`family_counts`) ship; per-family
  **policy entropy and loss** disaggregation does not yet;
- ✅ eval block (every N iterations): win rate + CI and margin vs. the current
  reference — ☐ league Elo still open (§7.3);
- ☐ **per-card visit counts** — how often each of the 180 birds was offered and
  played. Still open, and the single most valuable missing log: it is both a
  data-starvation monitor and the raw material for the card-power analysis the
  project exists to produce.

Alert (even just a printed warning) if: games/sec collapses, any loss goes
non-finite, policy entropy crashes to ~0 early (premature collapse — raise the
entropy bonus), or win-rate-vs-random stops climbing (pipeline is broken).

---

## 9. A phased program with exit criteria

Tie it together. Do not start a phase until the previous one's exit criterion is
met — that discipline is what keeps the analysis trustworthy.

**Phase 0 — Make it correct and safe (hours). ✅ Done.**
The `MAX_CHOICES_HARD` crash is removed (§4.3); the batch is length-bucketed
(§4.2a); torch is seeded and a full resumable checkpoint is written (§5); the
evaluation harness vs. the random agent exists (§7). The exit criteria are met: a
run completes without crashing or OOM-ing, checkpoints resume (weights +
optimizer + counters; games stay seed-reproducible, though resume is not
bit-for-bit — RNG state is not stored), and the loop prints an honest
win-rate-vs-random with a confidence interval.

**Phase 1 — An honest single-machine baseline. ◑ Pipeline built; verify the
curve.**
The loop is REINFORCE + value baseline + advantage normalization with epsilon
dropped (§3.3); the default batch is `run.games_per_iter`. What remains is the
*empirical* exit check on a given run: win rate vs. random must climb decisively
past ~90 % with a smooth strength curve. If a run does not show that, debug here
before proceeding.

**Phase 2 — Scale throughput. ✅ Done.**
Parallel CPU actors (`mp_collect`) with per-iteration weight broadcast, plus the
CUDA batched collector (`batched_collect`) (§4.1). *Exit (per run):* confirm the
collector keeps the learner fed — the GPU should no longer be the idle component.

**Phase 3 — Strengthen the algorithm (days). ◑ Algorithm done; league open.**
PPO + GAE (§3.4) are now shipped REGIME knobs. Still open: the multi-checkpoint
league, league-Elo evaluation, and occasional league opponents (§7). *Exit:*
league Elo rises monotonically over a long run; the policy beats its Phase-1
self head-to-head by a clear margin.

**Phase 4 — Grow capacity only if warranted (ongoing). ◑ Embedding done.**
The shared card embedding (§6.3) is adopted — worthwhile on its own. Then, only on
measured underfitting (§6.2), widen the trunk and the high-traffic heads, and
give the starved heads (§6.4) auxiliary supervision (the separate `setup` model
already exists). *Exit (per change):* the bigger model beats the smaller one
head-to-head by more than the confidence interval. If it does not, revert — you
were not capacity-bound.

Underlying all phases: every result is reproducible (seeded, git-stamped,
config-logged), and the harness — not intuition — decides whether a change
helped.

---

## 10. Summary of concrete recommendations

| # | Recommendation | Section | Status | Why |
|---|---|---|---|---|
| 1 | Length-bucket the batch (segment-softmax later) | §4.2 | ✅ / ☐ | ~40× less memory; old run hit 11.4 GB |
| 2 | Remove `MAX_CHOICES_HARD` | §4.3 | ✅ | 637-option play crashed collection intermittently |
| 3 | Route collection by device (CPU pool / CUDA batched) | §1.4, §4.1 | ✅ | batch-1 GPU inference was 2× slower |
| 4 | Eval harness (fixed opponent, paired games, CIs) | §7 | ✅ | self-play win rate is ~50 % and measures nothing |
| 5 | Batch = 64–256 **games**/iteration | §3.2 | ✅ | count games, not steps — one terminal reward/game |
| 6 | Normalize advantages; drop epsilon-greedy | §3.3 | ✅ | stable gradients; keep the update on-policy |
| 7 | REINFORCE → PPO + GAE | §3.4 | ✅ | more learning per expensive game; lower variance |
| 8 | Parallel CPU actors + CUDA batched collector | §4.1 | ✅ | collection is the wall: ~46 h → ~5 h for 1 M games |
| 9 | Full resumable checkpoints + metrics/games logs | §5, §8 | ✅ | resume, reproduce, evaluate |
| 9b | Multi-checkpoint league + Elo + per-card visit logs | §5.2, §7, §8 | ☐ | catch tail-chasing; card-power readout |
| 10 | Share the card embedding; widen trunk before heads | §6.3 | ✅ / ☐ | consistent per-card readout; shared lift |
| 11 | Size heads to data; special-case `setup`/rare heads | §1.2, §6.4 | ✅ | 370× data imbalance; setup is its own model |
| 12 | Add capacity last, only on measured underfitting | §6 | ☐ | heads ~39 % of params now; algo/data bind first |

---

## 11. Glossary

- **Actor / learner.** In a parallel setup, *actors* generate games (CPU here);
  the *learner* does the gradient updates (GPU here).
- **Advantage.** Return minus the critic's baseline; "better or worse than
  expected." Lowers gradient variance.
- **Baseline / critic / value head.** A predicted return `V(s)`, subtracted from
  the actual return to make the learning signal less noisy.
- **Batch / minibatch.** All the steps in one iteration / a smaller chunk of them
  used for a single optimizer step.
- **Bootstrapping.** Estimating a state's value partly from the *predicted* value
  of later states (what GAE/TD do) instead of waiting for the final outcome.
- **Elo.** A relative-strength rating from head-to-head results; convenient as a
  single monotone "is it getting stronger?" curve.
- **Entropy (of a policy).** How spread-out its probabilities are. A bonus on it
  preserves exploration.
- **Epoch (PPO reuse epoch).** One reuse pass over the just-collected batch. *Not*
  a pass over a fixed dataset (RL has none).
- **GAE.** Generalized Advantage Estimation — a low-variance way to compute
  advantages by blending multi-step value estimates (knob λ).
- **GIL.** Python's Global Interpreter Lock; why CPU-bound game logic must be
  parallelized with *processes*, not threads.
- **Greedy (evaluation).** Always take the highest-scoring option (argmax), no
  sampling — used to measure strength.
- **Iteration.** One collect-then-update cycle. The RL analog of an "epoch."
- **Mixed precision / bf16.** Training in a lower-precision number format for
  speed and memory, with negligible quality loss on modern GPUs.
- **On-policy.** Learns only from data generated by the current policy; data is
  discarded after each update. REINFORCE and PPO are on-policy.
- **Overfitting / underfitting.** Memorizing quirks (cure: data/regularization)
  vs. too weak to learn the pattern (cure: capacity/training). §6.2.
- **Padding / masking.** Filling ragged option-lists out to a common width with
  dummy rows, marked by a mask so they get zero probability.
- **Policy gradient / REINFORCE.** The basic on-policy method: make winning
  actions more likely, losing actions less likely.
- **PPO.** Proximal Policy Optimization — policy gradient with a clip that allows
  safe multi-pass reuse of each batch; the recommended workhorse here.
- **Return / reward.** What we maximize — here the final score margin.
- **Self-play.** Both seats driven by the same network.
- **Segment-softmax.** A softmax computed independently per group ("segment") in
  a flat tensor; lets all decisions share one un-padded forward pass.
