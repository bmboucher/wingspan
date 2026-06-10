# Critic Reward Calculation

**Question:** How are critic rewards calculated — terminal margin or per-step deltas?

> **Update:** the per-step margin-delta shaping discussed under
> *Alternatives → Per-step score-delta shaping* is now available as an opt-in
> reward mode (`TrainConfig.reward_mode = decision_delta`, with a
> `reward_discount` γ). The terminal-margin behavior analyzed below remains the
> default. See `docs/TRAINING.md` §2 (Reward / return) and `learner._flatten`.

---

## Current implementation

### Return calculation (`learner.py:132-150`)

Every step in every game receives the **terminal score margin** as its return.
There is no per-step delta, no discounting, and no bootstrapping.

The calculation in `_flatten` (`learner.py:132-150`):

```python
score_0, score_1 = record.breakdowns[0].total, record.breakdowns[1].total
per_pov = (
    (score_0 - score_1) / score_norm,
    (score_1 - score_0) / score_norm,
)
for step in record.steps:
    flat_steps.append(step)
    returns.append(per_pov[step.player_id])
```

Every step in the same game receives the identical scalar return: the final
score margin from that step's player's point of view, divided by `score_norm`.

**`score_norm`** is a configurable hyperparameter (`config.py:63`), defaulting
to `50.0`. It is a fixed divisor that rescales the raw score margin (a typical
Wingspan score is 50–100 points; the margin is often 5–20 points) into
roughly the `[-1, 1]` range. It is *not* computed from the game — it is a
static constant set before training begins and kept fixed throughout a run.

**Return range:** with `score_norm = 50.0` and typical score margins of
±5 to ±40 points, returns fall in roughly `[-0.8, 0.8]` for most games, with
wider tails for blowout wins or losses. The same value is broadcast to every
one of the ~140 trainable steps in the game.

### Value head training (`learner.py:55-126`)

The value head is trained by MSE to the full terminal return (`learner.py:105`):

```python
value_loss = F.mse_loss(value_all, return_all)
```

There is **no Bellman target** and **no bootstrapping**: the target is the
Monte Carlo terminal return, not a one-step TD estimate using the value of the
next state. This is pure Monte Carlo REINFORCE with a value baseline.

The advantage used in the policy gradient loss is (`learner.py:99`):

```python
advantage = return_all - value_all.detach()
```

That is, `A = G - V(s)` where `G` is the terminal return and `V(s)` is the
critic's prediction for the current state. The `.detach()` ensures gradients
flow only through the policy, not back through the critic via the advantage.
The advantage is then normalized per batch (`learner.py:100-102`):

```python
adv_mean = advantage.mean()
adv_std  = advantage.std()
norm_advantage = (advantage - adv_mean) / (adv_std + _ADV_STD_EPS)
```

This keeps gradient magnitudes stable regardless of how accurate the critic
has become (TRAINING.md §3.3).

The full loss combines all three terms (`learner.py:104-107`):

```python
policy_loss = -(logp_all * norm_advantage).mean()
value_loss  = F.mse_loss(value_all, return_all)
entropy     = entropy_all.mean()
loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
```

Default coefficients: `value_coef = 0.5`, `entropy_coef = 0.01`.

### Value head architecture (`model/core.py:374-384`)

The value head is a **single shared MLP** across all 13 judgment families
(PlaceBird, LayEgg, GainFood, etc.). It is built by `_build_value_head`
(`core.py:374-384`):

```python
self.value_head = mlp.build_readout(
    arch.trunk_embed_width,
    arch.value_layers,
    activation=arch.activation,
    dropout=arch.dropout,
)
```

`mlp.build_readout` (`mlp.py:66-86`) builds a stack of
`Linear → activation → (optional Dropout)` hidden layers, followed by a bare
`Linear(·, 1)` with no activation. The output is squeezed to a scalar
(`core.py:196`):

```python
value = self.value_head(state_ctx).squeeze(-1)  # (B,)
```

**Default architecture (`architecture.py:79`):** `value_layers = ()` — an empty
tuple — which means **no hidden layers**: the value head is a single linear
projection from `trunk_embed_width` (default `128`) directly to a scalar. The
measured parameter count confirms this: the value head contributes only
**129 parameters** out of 532,110 total (TRAINING.md §1.1), which is
`128 * 1 + 1 = 129` (weight + bias for one linear layer).

The value head reads `state_ctx` — the trunk's embedding of the current board
state — which is the same input used by all 13 family scoring heads before they
concatenate the choice embedding. The value head therefore sees the same
compact board representation regardless of which family's decision is being
made.

**Shared head across families:** all 13 judgment families (macro action, play
bird, lay egg, gain food, etc.) feed through the same value head. The value
function `V(s)` must therefore generalize: it is predicting the terminal
outcome from a board state irrespective of whether the current decision is
"which habitat to activate" or "which egg to remove." The trunk embedding is
the bridge — it is supposed to capture everything relevant about the board,
so the value head only needs to learn a linear combination of its features.

---

## Implications

### Credit assignment noise

With a terminal return broadcast to all steps, every decision in the game
receives the same credit signal. A brilliant setup that enabled a chain of
high-scoring turns gets exactly the same reward as the routine final egg-lay
that happened to pad the score. A defensive pick that stopped the opponent
from completing a bonus card appears as an undifferentiated part of the same
margin.

The value baseline `V(s)` partially compensates: the advantage `G - V(s)`
is positive only if the game went *better than the state implied*, and
negative if it went *worse*. If the critic learns to predict early-game
trajectories well (a strong assumption), the advantage for a brilliant early
setup move will be large and positive, because from that early state the
expected return was low and the result was high. But this hinges on the
quality of the critic, which is trained on only 129 parameters in its default
form.

The fundamental credit assignment problem is that the policy gradient cannot
distinguish "I won because of decision `t`" from "I won despite decision `t`."
It can only learn the aggregate signal over many games.

### Variance analysis

A Wingspan game has ~140 trainable decisions (TRAINING.md §1.2). With terminal-
return REINFORCE, every one of those ~140 steps acts as an independent noisy
estimate of the final outcome. The variance of the gradient estimator grows
with game length — the classic "reward attribution problem" of policy gradient
methods in long-horizon environments.

TRAINING.md §3.2 identifies this explicitly: "one game is closer to **one**
noisy label than to 140 independent ones." The batch-size recommendation
(64–256 games/iteration, start at 128) is calibrated to average over enough
game outcomes that a few lucky wins do not dominate the gradient.

The value head reduces this: the normalized advantage `(G - V(s)) / std(A)`
has variance reduced by however well `V(s)` predicts the return. But the
default value head is a single linear layer (129 params), trained on a sparse
signal (the same terminal return), so its variance-reduction capability is
limited in early training when the critic is weakly fitted.

TRAINING.md §3 notes this as the justification for the PPO + GAE upgrade
(§3.4): "the current update has very high variance because every one of a
game's ~140 decisions is credited with the *same* end-of-game margin."

---

## Alternatives

### Per-step score-delta shaping

Compute a per-step reward by diffing the score margin across adjacent decisions:

```
r_t = (score_me_after_t - score_opp_after_t) / score_norm
    - (score_me_before_t - score_opp_before_t) / score_norm
```

**Pros:**
- Directly credits the decisions that score points. A bird play that
  immediately adds eggs gets a positive delta; a turn that wastes food gets
  near-zero or negative.
- Dramatically reduces variance for short-horizon scoring decisions.

**Cons specific to Wingspan:**
- Many points are deferred. End-game bonuses (tucked cards, eggs in nests,
  the five bird bonus categories) and round-goal bonuses materialize at the
  end of the round or game, not at the decision point. A setup move that
  positions the engine to score a 14-point bonus next round appears as `r_t ≈ 0`
  for the 20 turns before the bonus fires.
- Setup decisions (habitat selection, initial hand) and many bird powers
  (pink "when another player…" effects) have deferred value that score-delta
  shaping cannot capture.
- Requires the engine to expose a queryable "current score" after every step,
  which is not part of the current `GameRecord` structure (`collect.py:59-80`).
  Adding it would require storing per-step snapshots, increasing IPC cost.

**Overall:** better signal for immediate-scoring families (`egg_placement`,
`bird_acquisition` when birds score eggs on play), misleading signal for
strategic setup families. The most impactful decisions in Wingspan are often
the ones that score 0 points now but shape the late-game engine.

### Discounted return

Apply a discount factor `gamma < 1.0` so earlier decisions receive credit
for a larger fraction of the eventual outcome:

```
G_t = sum_{k=0}^{T-t} gamma^k * r_{t+k}
```

With a per-step reward of zero and a terminal reward at `T`, this collapses
to `G_t = gamma^(T-t) * final_margin` — the terminal margin discounted by
distance from the end.

**Pros:** standard in the RL literature; easy to implement.

**Cons for board games:**
- In Wingspan, a decision made on turn 3 is no less important than one made
  on turn 40 — both feed into the final score. Discounting artificially
  de-emphasizes early turns.
- `gamma` near 1.0 (e.g. `0.99`) recovers the current undiscounted behavior
  for games of ~140 steps (`0.99^140 ≈ 0.25`), so the de-emphasis is mild
  but real. `gamma = 1.0` is the current approach.
- No theoretical benefit over `gamma = 1.0` for a fully-observed finite-
  horizon game with a terminal payoff and no intermediate rewards.

**Overall:** not recommended for this game. The undiscounted return (`gamma = 1`)
is the correct objective for a finite-horizon board game.

### Per-step potential shaping (reward shaping)

Use the current score margin as a potential function `phi(s)` and define:

```
r_t = gamma * phi(s_{t+1}) - phi(s_t)
    = gamma * (score_margin_after / score_norm)
            - (score_margin_before / score_norm)
```

The total shaped return telescopes:

```
G_shaped = gamma^T * phi(s_T) - phi(s_0)
         = final_margin * gamma^T / score_norm - initial_margin / score_norm
```

With `gamma = 1.0` this simplifies to `final_margin - initial_margin`, which
for a Wingspan game starting from an empty board reduces to just the
final score margin — identical to the current approach.

For `gamma < 1.0` the shaping introduces the same discounting bias discussed
above. The Ng et al. (1999) reward shaping theorem guarantees policy invariance
for any fixed `phi` with `gamma < 1`, but the theorem's equivalence only holds
when `phi` is exactly the potential of the true value function — using the
current score margin as `phi` introduces a bias term proportional to
`phi(s_0)` that is non-zero whenever games start at different opening states
(which they always do in Wingspan due to random setups).

**Pros:** dense signal from per-step score changes; theoretically grounded.

**Cons:** the engine must expose per-step score snapshots (same caveat as
per-step delta shaping above); `gamma = 1` reduces to the current approach;
`gamma < 1` introduces the discounting bias and startup bias simultaneously.

**Overall:** the theoretically interesting case is `gamma = 1` with a learned
potential — i.e. using `V(s)` as the potential, which is exactly what the
advantage `G - V(s)` already computes. This is the current approach, not an
alternative to it.

---

## Recommendation

The current approach — terminal Monte Carlo return, single shared linear value
head, normalized advantages — is **not the binding bottleneck at this stage of
training.** The binding bottlenecks are throughput (collection speed, TRAINING.md
§4), the algorithm upgrade from REINFORCE to PPO (TRAINING.md §3.4), and the
lack of a real evaluation harness (TRAINING.md §7). Both of those must be
resolved before credit assignment variance becomes the diagnosable problem.

**The value head capacity is worth noting as a future concern.** At 129
parameters it can only learn a linear function of the trunk embedding. If the
trunk embedding is rich and the board state is well-linearized therein, this
suffices. If convergence stalls and the value MSE plateaus high (indicating the
critic cannot fit the returns), adding one hidden layer to `value_layers`
(e.g. `value_layers = (128,)`) adds only ~16,512 parameters and should be the
first thing tried before blaming the return signal itself.

**The credit assignment problem is real but expected.** Terminal-return
REINFORCE is the standard starting point for finite-horizon games.
TRAINING.md §3.4 already identifies GAE as the next step: it uses the critic's
intermediate value estimates to produce per-decision advantages that are more
local than the terminal return, without requiring per-step score snapshots
from the engine. GAE is the recommended upgrade, and it is already planned for
Phase 3 of the training program — nothing needs to change in the current code
until then.

**Per-step score-delta shaping is not recommended** for this game. The highest-
value decisions in Wingspan (habitat selection, bird engine construction) score
zero points at the moment they are made and accumulate value across many
subsequent turns. Score-delta shaping would systematically under-credit these
decisions while over-crediting routine egg-laying. It also requires non-trivial
engine changes to expose per-step score snapshots.
