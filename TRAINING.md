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
> *plan*; much of it now ships. The live trainer is `python -m wingspan.training`
> (the `wingspan.training` package), which already implements: length-bucketed
> batches (§4.2a), Python/NumPy/torch seeding (§5), a paired-game evaluation
> harness with a 95% CI and a frozen-opponent ladder (§7), resumable full
> checkpoints — model + optimizer + counters + config + git SHA (§5.1), and
> parallel **CPU** self-play workers with per-iteration weight broadcast (§4.1).
> **Training is CPU-only now** — the CUDA framing below ("collect on CPU, update
> on GPU", §1.4/§4) is historical; the update also runs on CPU.
>
> Still open: PPO + reuse epochs and GAE (§3.4), a frozen `iter_N.pt` league
> (§5.2), the shared card embedding (§6.3), and per-card visit-count logging (§8).
>
> Two caveats on the numbers/claims below: the §1.1 measured sizes predate the
> encoder expansion (as of the landing-slot encoding the state vector is **768**
> and the choice vector **215**, not 282/260 — re-measure before quoting), and
> checkpoints do **not** store RNG state, so a resumed
> session is *not* bit-for-bit identical — but every collected game is
> seed-reproducible from `config.seed`, so the per-game logs are stable. Engine
> fidelity as of this pass: all core bird powers fire (including the four pink
> "when another player …" reactors) and all 16 round goals score, pinned by
> `tests/test_power_coverage.py` and `tests/test_round_goal_coverage.py`.

---

## 0. TL;DR — the program in one page

The single most important finding: **the GPU is not the bottleneck and the
network is not too small. The bottleneck is everything around them** — how data
is generated, how it is batched, how progress is measured. Fix those first;
reach for a bigger model last.

Priorities, in order. Each links to its section.

1. **Fix three correctness/scaling defects before any long run** (§4.2, §4.3):
   - The choice-tensor is padded to the *widest* decision in the batch (the
     504-option opening draft), so **97.6 % of that tensor is padding** and the
     default 32-game update already peaks at **11.4 GB** of GPU memory — a 64-game
     run would run out of memory on a 16 GB card. → length-bucket the batch.
   - A `PlayBirdDecision` can exceed the `MAX_CHOICES_HARD = 600` safety cap
     (we saw **637**) and crash collection. It is trajectory-dependent, so it
     *will* surface during a long unattended run. → raise/remove the cap.
   - Self-play runs network inference on whatever `--device` is set; on `cuda`
     it is **2× slower** than CPU because each decision is a batch-of-one
     forward pass. → collect on CPU, update on GPU.

2. **Build an honest evaluation harness** (§7). Today the only metric is
   "player 0's self-play win rate," which is ~50 % *by symmetry* and measures
   nothing. Without a real yardstick you cannot tell whether any change helped.

3. **Replace plain REINFORCE with PPO + a normalized advantage** (§3.4). The
   current update has very high variance because every one of a game's ~140
   decisions is credited with the *same* end-of-game margin.

4. **Parallelize self-play across CPU workers** (§4.1). Collection is the wall —
   ~163 ms/game on one core. Ten workers turn ~22 k games/hour into ~200 k/hour,
   the difference between a two-day run and an overnight one.

5. **Only then consider a bigger network** (§6), and only on evidence of
   *underfitting* measured against the §7 yardstick. The per-family heads are
   already **80 % of the parameters**; the shared "read the board" trunk is the
   cheap part and is the more likely thing to widen.

A concrete phased schedule with exit criteria is in §9.

---

## 1. Where the time and memory actually go (the measured profile)

You cannot size a training run sensibly without knowing the shape of the data
it produces. Here is this game, as the encoder and engine actually emit it.

### 1.1 Feature and parameter sizes

| Quantity | Value | Source |
|---|---|---|
| State vector length | **768** | `encode.state_size()` |
| Per-choice feature length | **215** | `encode.CHOICE_FEATURE_DIM` |
| Judgment-family heads | **13** | `decisions.ALL_DECISION_FAMILIES` |
| Distinct decision classes | **17** | `decisions.ALL_DECISION_CLASSES` |
| Total parameters (`hidden=128`) | **532,110** | `PolicyValueNet` |

Where those half-million parameters live is the surprise:

| Component | Parameters | Share |
|---|---|---|
| State trunk (shared) | 52,736 | 9.9 % |
| Per-choice encoder (shared) | 49,920 | 9.4 % |
| **13 scoring heads** | **429,325** | **80.7 %** |
| Value head (shared) | 129 | 0.0 % |

Each head is a `256 → 128 → 1` MLP (≈33 k parameters), and there are 13 of them.
The "expensive shared representation" the design prizes (the trunk) is actually
the *cheapest* part of the network. This matters for §6: if you ever add
capacity, the heads are where parameters multiply fastest, and most heads are
data-starved (next).

> **Term — MLP (multi-layer perceptron):** the plainest kind of neural network,
> a stack of `Linear` layers with a nonlinearity (here `ReLU`) between them.
> "`256 → 128 → 1`" means it takes a 256-number input, squeezes it to 128, then
> to a single score.

### 1.2 A game's decisions, by judgment family

A self-play game records on average **~140 trainable decisions** (range 106–191
over 20 games; single-option forced moves are not recorded). But those decisions
are wildly unevenly distributed across the 13 heads:

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

89.5 % of decisions offer four options or fewer. But the opening draft offers
**504**, and a food-rich late-game `PlayBirdDecision` can offer **several
hundred** (we observed 376, 370, even 637). The training step currently stacks
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

On a 16 GB RTX 4080, the *default* configuration already uses 71 % of the card,
purely to store padding. Double the games and it runs out of memory. This is the
highest-leverage single fix in the document (§4.2).

### 1.4 Throughput: collection dominates, the update is trivial

| Operation | Time | Notes |
|---|--:|---|
| Self-play, 1 game, CPU | **163 ms** | batch-of-one inference per decision |
| Self-play, 1 game, GPU | **320 ms** | 2× *slower* — transfer/launch latency |
| Gradient update, 4,428 steps, GPU | 1.78 s | per *iteration*, not per game |
| Gradient update, 4,428 steps, CPU | 32 s | — |

The asymmetry is the whole story of §4. Generating data is sequential, Python-
bound, and slow; the gradient update on a half-million-parameter net is nearly
free on a modern GPU. **A single GPU spends almost all of its time idle, waiting
for the next batch of games.** Every throughput recommendation follows from this.

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
- **Reward / return (G).** What we want to maximize. Here: the **final score
  margin** (your score minus the opponent's), the same for every step you took
  in that game.
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
  2. COMPUTE each step's return G (final margin, normalized) and, with the
     critic, its advantage A = G − V(s). Normalize advantages (§3.3).
  3. UPDATE the weights on that batch (GPU): policy loss + value loss
     − entropy bonus, minibatched, with PPO clipping (§3.4).
  4. DISCARD the data (it is now off-policy) and loop.
  periodically: EVALUATE vs. fixed opponents (§7) and CHECKPOINT (§5).
```

The current `train.py` already implements a primitive version of steps 1–3
(REINFORCE with a value baseline, one full-batch update per iteration). The
recommendations below evolve it; they do not throw it away.

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

**Two changes to make now:**

1. **Normalize the advantage per batch** — subtract its mean, divide by its
   standard deviation, before the policy loss. Right now the advantage scale is
   tied to the arbitrary `SCORE_ADVANTAGE_NORM = 50` constant and shrinks as the
   critic improves, which silently changes the effective learning rate over
   training. Normalizing makes gradient magnitudes stable from the first
   iteration to the last. This is standard practice and essentially free.

2. **Drop epsilon-greedy exploration** (`DEFAULT_EPSILON = 0.05`). Policy
   gradient is *on-policy*: the math assumes the action you learn from was drawn
   from the current policy π. Epsilon-greedy instead takes a uniformly-random
   action 5 % of the time — those steps were *not* drawn from π, so the gradient
   on them is biased (correcting it properly would need importance weighting).
   The softmax already explores; control exploration with the entropy bonus (and
   optionally a sampling temperature), not with epsilon. Cleaner and unbiased.

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

1. **REINFORCE + value baseline + advantage normalization** (today, plus the
   §3.3 fixes). Get a clean baseline strength curve from this. It is enough to
   crush a random opponent and to validate the whole pipeline.

2. **PPO (Proximal Policy Optimization)** — the recommended workhorse.
   - **What it adds:** it lets you safely take *several* gradient passes over
     each collected batch (the "reuse epochs" of §3.2) instead of one, by
     *clipping* the update so the new policy can't move too far from the policy
     that generated the data in a single iteration. More learning per game
     collected — directly attacks our collection bottleneck (§4) — and much more
     stable than multi-pass REINFORCE.
   - **Why it matters here:** games are expensive (§1.4) and data is discarded
     after each iteration. PPO extracts 3–10× more signal from each batch before
     throwing it away.
   - **Settings to start:** clip ε = 0.2, reuse epochs = 4, minibatch ≈ 4,096,
     advantage normalization on.

3. **GAE (Generalized Advantage Estimation)** — adopt alongside PPO for variance
   reduction.
   - **The problem it solves:** today every one of a game's ~140 decisions is
     credited with the *same* terminal margin. A brilliant turn in a game you
     lost gets a negative signal; a blunder in a game you won gets a positive
     one. Over many games this averages out, but slowly and noisily.
   - **What GAE does:** uses the critic's value estimates of *intermediate*
     states to assign each decision a more *local* advantage — "how much better
     did the position get right after this move?" — instead of waiting for the
     final score. A knob λ (≈0.95) trades off the high variance of waiting for
     the end against the bias of trusting the critic. With pure terminal rewards
     and no bootstrapping (today's setup), the advantage is just `G − V(s)`;
     GAE's benefit appears once the critic is good enough to bootstrap from.

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

**A faster alternative to many processes** (optional, more code): a single
*batched-inference* actor loop that steps many games forward in lockstep,
collecting all the games' current decisions into one batched forward pass. This
turns the 2× GPU penalty into a GPU *win* by giving it real batches, and avoids
process overhead — but it is fiddly because games desynchronize (different
games reach decisions at different times). Start with the process pool; consider
batched inference only if collection is still the wall after parallelizing.

### 4.2 Kill the padding (the 42× memory win)

§1.3 measured that 97.6 % of the choice tensor is padding and that the default
update peaks at 11.4 GB. Two fixes, in increasing order of cleanliness:

**(a) Length-bucketing — the quick win.** Group the batch's steps into buckets
by option-count (e.g. `≤4`, `≤8`, `≤16`, `≤64`, `≤512`) and pad *within* each
bucket. The 89.5 % of decisions with ≤4 options then pad to 4, not 504. Run one
forward/backward per bucket and sum the losses. Measured effect: choice-tensor
memory drops from 2.32 GB to ~0.055 GB. A few dozen lines; do this first.

**(b) Flatten + segment-softmax — the clean design.** Eliminate padding
entirely. Concatenate every candidate across the whole batch into one
`(ΣK, choice_dim)` matrix (here ΣK ≈ 52 k, not 4428×504 ≈ 2.2 M), run the
choice-encoder once over it, gather each candidate's decision-context, score,
and do a *segmented* softmax — a softmax computed independently per decision
using a per-candidate "which decision do I belong to" segment id (via
`scatter`/`index_add`-style reductions). A 2-option decision and the 504-option
draft are then handled by identical code with zero waste, and the
`MAX_CHOICES_HARD` cap (§4.3) becomes irrelevant. More work than bucketing;
adopt it when the encoder is touched next.

### 4.3 Remove the choice-count crash

`encode.MAX_CHOICES_HARD = 600` is an `assert` that aborts collection if a
decision exceeds it. A food-rich `PlayBirdDecision` enumerates one candidate per
`(bird, habitat, payment)` combination and **can exceed 600** — we hit **637**
on the first 32-game attempt with fresh weights. It is *trajectory-dependent*,
so it does not happen every run, which makes it worse: it will ambush a long
unattended run hours in. Fixes: raise the cap substantially (e.g. 2,000), or
remove the hard assert and rely on the soft warning plus the bucketing/segment
machinery (which handles any width). Either way, this must be resolved before
the first multi-thousand-game run. Optionally, also bound the payment
enumeration so a single decision cannot generate a pathological number of
near-duplicate wild-payment candidates.

*(Update: the play-bird cost split shrank `PlayBirdDecision` to one candidate
per `(bird, habitat)` pair — bounded by hand size × 3. The payment enumeration
moved to the follow-up `PayBirdFoodDecision`, whose width is the payment count
for a single bird; the food-rich wild-payment blow-up concern now applies
there, not to the play menu.)*

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

The current code saves `{model, args}` once, at the very end. That is
inadequate for runs measured in hours and for a project whose *output is
analysis* (which demands reproducibility).

### 5.1 What a checkpoint must contain

A checkpoint has to let you (a) resume a crashed run *exactly*, and (b) re-derive
results later. Save a single dict per checkpoint:

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

- **Every iteration:** atomically overwrite `last.pt` (write to a temp file,
  then `os.replace` — a crash mid-write must not corrupt the only checkpoint).
- **Gated on evaluation (§7):** keep `best.pt`, updated only when eval strength
  improves. The last checkpoint is not always the best one.
- **Every N iterations:** write an immutable snapshot `iter_{n}.pt`. These form
  the **league** — the frozen past selves used both as evaluation opponents
  (§7) and, optionally, as self-play opponents to stop the policy from chasing
  its own tail.
- **Reproducibility hygiene** (this is a research loop, so this is the
  proportionate slice of "MLOps"): seed Python, NumPy, and torch at startup
  (the current `main` seeds Python's `random` but not torch's weight init);
  record the git SHA and full `TrainConfig`; append per-iteration metrics to a
  `metrics.jsonl` next to the checkpoints. A spreadsheet-grade log is enough to
  start; graduate to MLflow or Weights & Biases if you want hosted dashboards
  and run comparison, but do not block training on that infrastructure.

---

## 6. Choosing and growing the architecture

The user's question — *how do we decide the architecture; when do we add layers
or neurons?* — has a disciplined answer that resists the urge to add capacity
first.

### 6.1 The meta-point: capacity is not this project's bottleneck

A 0.5 M-parameter network is *small*; the RTX 4080 could train 10–50 M parameters
for this game without strain. But more parameters will not help until the data
generation (§4.1), batching (§4.2), algorithm (§3.4), and evaluation (§7) are
sound, **and** there is measured evidence of underfitting. Spend the capacity
budget last.

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

- **Share the card embedding (do this regardless — it is an improvement, not
  just more capacity).** Today a bird's identity is read by *two unrelated*
  weight matrices: the hand multi-hot goes through the state trunk, the
  candidate one-hot through the choice encoder. So "American Robin" has two
  separate learned meanings depending on whether it sits in your hand or on the
  table. Replace both with one shared `nn.Embedding(180, d)` (and `(26, d)` for
  bonus cards). The card's representation becomes consistent everywhere it
  appears, it generalizes better, and — the project's headline goal — that
  single embedding table *is* the per-card power readout (the `bird_id` stripe
  embedded through the shared card table, DECISIONS.md §1).
- **Widen the trunk before the heads.** The shared "read the board" trunk is the
  thing every decision and the critic depend on, and at `hidden=128` it is the
  cheap part of the net (§1.1). Widening it to 256–512 lifts every head and the
  critic at once. Widening the heads helps only the families that exercise them.
- **Sit the heads on the data.** Give `macro_action`, `play_bird`,
  `bird_acquisition`, and `gain_food` more width; keep the rare heads small.

### 6.4 The starved and singular heads

`misc_rare` (0.1/game), `bonus_valuation` (0.7), `spend_food` (1.9), and `setup`
(2.0) will barely train no matter the batch size. Options, roughly in order of
effort:

- **Keep them deliberately small** so they cannot overfit their few examples
  (the design already pools the two rarest decisions into `misc_rare` for
  exactly this reason — DECISIONS.md §2.10).
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
  value-regression setup net realizes this, with its random-generation
  bootstrap phase standing in for the heuristic objective.

---

## 7. Evaluating out-of-sample performance

This is the most important thing the project currently lacks. Without it, none
of the choices above can be judged.

### 7.1 Why the current metric is empty

The training log reports "player 0 wins." In self-play **both seats are the same
network**, so this number measures only the first-player advantage plus noise —
it sits near 50 % no matter how strong or weak the policy is. It cannot tell you
whether yesterday's run was better than today's.

### 7.2 What "out-of-sample" means here

> **Term — out-of-sample / held-out.** Performance measured on situations the
> model was *not* trained on. In supervised learning that is a held-out slice of
> a dataset. In self-play RL there is no dataset to hold out, so out-of-sample
> means two concrete things: **(a)** games played from **fresh random seeds**
> (new deals, shuffles, dice) distinct from training seeds, and **(b)** play
> against **opponents the policy did not train against**.

### 7.3 The evaluation harness

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
monitoring**. Log per iteration, to `metrics.jsonl` and/or a tracker:

- iteration, wall-clock, total games, games/sec (throughput regression alarm);
- loss components: policy, value, entropy, total; gradient norm;
- advantage mean/std (sanity on normalization);
- per-family: step count this iteration, policy entropy, loss;
- eval block (every N iterations): win rate + CI and margin vs. each reference,
  league Elo;
- **per-card visit counts** — how often each of the 180 birds was offered and
  played. This is both a data-starvation monitor and the raw material for the
  card-power analysis the project exists to produce.

Alert (even just a printed warning) if: games/sec collapses, any loss goes
non-finite, policy entropy crashes to ~0 early (premature collapse — raise the
entropy bonus), or win-rate-vs-random stops climbing (pipeline is broken).

---

## 9. A phased program with exit criteria

Tie it together. Do not start a phase until the previous one's exit criterion is
met — that discipline is what keeps the analysis trustworthy.

**Phase 0 — Make it correct and safe (hours).**
Fix the `MAX_CHOICES_HARD` crash (§4.3); length-bucket the batch (§4.2a); seed
torch and write a full resumable checkpoint (§5); add the evaluation harness vs.
the random agent (§7). *Exit:* a 200-game run completes without crashing or
OOM-ing, checkpoints resume cleanly (weights + optimizer + counters; collected
games stay seed-reproducible, though resume is not bit-for-bit — RNG state is not
stored), and you can print an honest win-rate-vs-random with a confidence
interval.

**Phase 1 — An honest single-machine baseline (a day).**
Synchronous loop, REINFORCE + value baseline + advantage normalization, drop
epsilon (§3.3). Batch 128 games/iteration. *Exit:* win rate vs. random climbs
decisively past ~90 % and the strength curve is smooth — proof the pipeline
learns. If it does not, debug here; do not proceed.

**Phase 2 — Scale throughput (a day, then runs get cheap).**
Parallel CPU actors + GPU learner with per-iteration weight broadcast (§4.1).
*Exit:* ≥ ~50 games/sec sustained and the GPU is no longer the idle component.

**Phase 3 — Strengthen the algorithm (days).**
PPO with reuse epochs + GAE (§3.4); add the league and league-Elo evaluation and
occasional league opponents (§7). *Exit:* league Elo rises monotonically over a
long run; the policy beats its Phase-1 self head-to-head by a clear margin.

**Phase 4 — Grow capacity only if warranted (ongoing).**
Adopt the shared card embedding (§6.3) — worthwhile on its own. Then, only on
measured underfitting (§6.2), widen the trunk and the high-traffic heads, and
give the starved heads (§6.4) auxiliary supervision or a separate `setup` model.
*Exit (per change):* the bigger model beats the smaller one head-to-head by more
than the confidence interval. If it does not, revert — you were not capacity-
bound.

Underlying all phases: every result is reproducible (seeded, git-stamped,
config-logged), and the harness — not intuition — decides whether a change
helped.

---

## 10. Summary of concrete recommendations

| # | Recommendation | Section | Why |
|---|---|---|---|
| 1 | Length-bucket (then segment-softmax) the batch | §4.2 | 42× less memory; default run already hits 11.4 GB |
| 2 | Raise/remove `MAX_CHOICES_HARD` | §4.3 | 637-option play crashes collection intermittently |
| 3 | Collect on CPU, update on GPU | §1.4, §4.1 | batch-1 GPU inference is 2× slower |
| 4 | Build an eval harness (fixed opponents, paired games, CIs) | §7 | self-play win rate is ~50 % and measures nothing |
| 5 | Batch = 64–256 **games**/iteration (start 128) | §3.2 | count games, not steps — one terminal reward/game |
| 6 | Normalize advantages; drop epsilon-greedy | §3.3 | stable gradients; keep the update on-policy |
| 7 | REINFORCE → PPO + GAE | §3.4 | more learning per expensive game; lower variance |
| 8 | Parallel CPU actors + GPU learner | §4.1 | collection is the wall: ~46 h → ~5 h for 1 M games |
| 9 | Full resumable checkpoints + league + metrics log | §5, §8 | resume, reproduce, evaluate, and analyze |
| 10 | Share the card embedding; widen trunk before heads | §6.3 | consistent per-card readout; cheap shared lift |
| 11 | Size heads to data; special-case `setup`/rare heads | §1.2, §6.4 | 370× data imbalance across heads |
| 12 | Add capacity last, only on measured underfitting | §6 | the heads are already 80 % of params; data/algo bind first |

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
