# Reward modes: how actor and critic losses are computed

This explains the math behind `cfg.reward_mode` (`learner._flatten`,
`learner._terminal_margin_returns`, `learner._decision_delta_returns`,
`learner._gae_flatten`) in four cases:

1. **`terminal_margin`** — the old method: every decision gets the terminal
   point margin.
2. **`decision_delta` with γ = 0** — per-decision margin differences, no
   lookahead.
3. **`decision_delta` with γ = 1** — per-decision margin differences, full
   lookahead.
4. **`gae`** — Generalized Advantage Estimation: a backward TD-residual sweep
   over captured value estimates, with its own `(γλ)^Δt` decay, producing the
   advantage and value target directly rather than an intermediate return
   `G_k`.

Cases 1-3 assume `reward_basis = MARGIN`, the default; see **Reward basis**
below for how `OWN_SCORE` changes the picture.

## Notation and setup common to all four cases

Fix one player. Let $S_1, S_2, \dots, S_K$ be the states at that player's
successive decisions, and let $M$ be the **final** point margin from their POV.

**Reward basis assumption.** Everything below through Case 4 assumes the
default `reward_basis = MARGIN` — $P(S)$ and $M$ are margins (own score minus
the best other seat's score), so the two seats' values are opposite-signed.
`RewardBasis.OWN_SCORE` swaps every margin quantity below for the player's own
absolute score instead (both seats then get positive values); see **Reward
basis: MARGIN vs OWN_SCORE** near the end of this document.

- $P(S)$ — the running point margin of state $S$ (own score − opponent's, as if
  the game ended now). The collector snapshots this as `margin_before` right
  before each decision (`collect.running_margin`).
- $V(S)$ — the critic's value estimate for state $S$.
- $P(S, A)$ — the actor's logit (pre-softmax score) for action $A$ in state $S$.
- `end_game_bonus` (config default `0.0`) — added to $M$ for the winning seat
  and subtracted for every other seat before it enters any of the return/
  advantage computations below (`returns.terminal_values`), so it discounts
  back through prior decisions exactly like any other part of the terminal
  margin. `OWN_SCORE` basis adds it only to the winner's value, leaving other
  seats' terminal values unchanged.

Note that $S_{k+1}$ is the state at your **next** decision, so
$P(S_{k+1}) - P(S_k)$ includes both your action's effect **and** whatever the
opponent did in between (margin is own − opponent).

Cases 1-3 each just produce a per-step **return** $G_k$, from which the
identical advantage/loss machinery below derives $A_k = G_k - V(S_k)$. Case 4
(`gae`) instead produces $A_k$ (and the critic's value target) directly,
skipping the intermediate $G_k$ — see Case 4 below — but feeds into the same
normalization and loss formulas.

**Policy probability.** The probability of the chosen action $A^{(k)}$ is the
masked softmax over the legal options:

$$
\log \pi\big(A^{(k)} \mid S_k\big)
  = P\big(S_k, A^{(k)}\big) - \log \sum_{j} e^{P(S_k, A_j)}
$$

**Advantage** (critic as baseline, detached so the policy loss doesn't push
gradients into the critic):

$$
A_k = G_k - V(S_k),
\qquad
\hat A_k = \frac{A_k - \mathrm{mean}(A)}{\mathrm{std}(A) + \varepsilon}
$$

where the mean/std are taken over the whole batch (all steps of all games, both
seats).

**Losses:**

$$
\mathcal{L}_{\text{actor}}
  = -\frac{1}{N}\sum_k \log \pi\big(A^{(k)} \mid S_k\big)\, \hat A_k
\qquad
\mathcal{L}_{\text{critic}}
  = \frac{1}{N}\sum_k \big(V(S_k) - G_k\big)^2
$$

combined as
$\mathcal{L} = \mathcal{L}_{\text{actor}} + c_v \mathcal{L}_{\text{critic}} - c_e H$
(entropy bonus $H$). All returns are divided by `score_norm` (50); that scaling
is omitted below.

So the only question is: **what is $G_k$?**

## Case 1 — `terminal_margin` (old method)

$$
G_k = M \quad \text{for every } k
$$

Every decision in the game gets the same return: the final margin. The critic
learns $V(S) \approx \mathbb{E}[M \mid S]$ — "given this state, what will the
final margin be?" The actor's advantage $M - V(S_k)$ says "did the game end
better than the critic expected from here?" — so a brilliant turn-2 play and a
blunder on turn 20 in the same won game both get credited with the same $M$.
Credit assignment happens *only* through the critic's baseline; the return
itself carries no information about which decision earned the points.

## Case 2 — `decision_delta`, γ = 0

The per-step reward is the margin change between consecutive decisions, with
the terminal margin appended as the last checkpoint:

$$
r_k = P(S_{k+1}) - P(S_k),
\qquad
r_K = M - P(S_K)
$$

The return is $G_k = r_k + \gamma\, G_{k+1}$, so with $\gamma = 0$:

$$
G_k = P(S_{k+1}) - P(S_k)
$$

Purely myopic: each decision is credited only with the point swing realized
before your *next* decision. The critic learns
$V(S) \approx \mathbb{E}[P(S') - P(S)]$ — the expected immediate margin delta.
The problem is that Wingspan is full of deferred payoffs: an engine bird, a
bonus-card pickup, or egg capacity built for a future round produce **zero**
immediate margin change, so with γ = 0 those decisions get no credit at all —
the points show up as someone else's reward many steps later (bonus cards
literally land on the final step's reward only, since they only score at game
end).

## Case 3 — `decision_delta`, γ = 1

With $\gamma = 1$ the backward sum telescopes:

$$
G_k = \sum_{j=k}^{K} r_j
    = \big(P(S_{k+1}) - P(S_k)\big)
    + \big(P(S_{k+2}) - P(S_{k+1})\big)
    + \cdots
    + \big(M - P(S_K)\big)
    = M - P(S_k)
$$

i.e. **final margin minus the margin already on the board when you decided** —
"how much *future* swing happened from here on." (The docstring on
`_decision_delta_returns` calls this out explicitly.)

Compare with case 1:

$$
G_k^{(\gamma=1)} = G_k^{(\text{old})} - P(S_k)
$$

The two differ only by subtracting the *currently observable* margin. For the
actor this matters less than it looks — a state-dependent offset is exactly
what a baseline absorbs — but for the **critic** it's a real change in job
description: in case 1, $V(S)$ must predict current margin *plus* future swing;
in case 3 it only predicts the future swing, since the banked $P(S_k)$ is
subtracted out of the target. Points you'd already scored before a decision can
no longer inflate or deflate that decision's credit, which removes a large,
easily-observable variance component from both the return and the critic's
regression target.

## Case 4 — `gae`

GAE (`timestamps.gae_advantages`, driven from `learner._gae_flatten`) does not
route through a single per-step return $G_k$ the way cases 1-3 do. It instead
walks the same per-player checkpoint sequence — $P(S_1), \dots, P(S_K), M$ —
backward and produces the advantage $\hat A_k$ (pre-normalization: $A_k$) and
the critic's value target directly, using the *captured* value estimates
$V(S_k)$ from collection time (`Step.value_pred`) rather than treating $V$ as
something to be fit against a Monte Carlo return after the fact:

$$
r_k = P(S_{k+1}) - P(S_k), \qquad r_K = M - P(S_K)
$$

$$
\delta_k = r_k + \gamma^{\Delta t_k}\, V(S_{k+1}) - V(S_k)
\qquad\text{(} V(S_{K+1}) := 0 \text{, the terminal has no further value)}
$$

$$
A_k = \delta_k + (\gamma\lambda)^{\Delta t_k}\, A_{k+1}
\qquad\text{(} A_{K+1} := 0 \text{)}
\qquad\qquad
\text{target}_k = A_k + V(S_k)
$$

where $\Delta t_k = t_{k+1} - t_k$ is the game-clock gap (see **The game
clock** below) and `gae_lambda` ($\lambda$, default $0.95$) is a second decay
applied on top of $\gamma$ — it controls how much the backward sweep trusts
the critic's own estimates $V(S_{k+1})$ versus the realized rewards $r_k$ at
each step, independent of $\gamma$'s discounting of *how far ahead* a reward
is credited. $A_k$ is what feeds the advantage-normalization step above (in
place of $G_k - V(S_k)$); $\text{target}_k$ replaces $G_k$ as the critic's
regression target.

This is a strict generalization of cases 2 and 3: at $\lambda = 1$ the
$(\gamma\lambda)^{\Delta t}$ decay reduces to $\gamma^{\Delta t}$ and the
backward sum telescopes to exactly the MC return $G_k$ from the intermediate
γ formula above, so $A_k = G_k/\text{score\_norm} - V(S_k)$ and
$\text{target}_k = G_k/\text{score\_norm}$ — the case-1/3 values, scaled
(`timestamps.gae_advantages`'s docstring calls this out as the correctness
check). At $\lambda = 0$, $A_k = \delta_k$ collapses to one-step TD: only the
very next checkpoint and the critic's own next-step estimate matter, the most
biased/lowest-variance end of the spectrum. GAE requires `behavior_logp` and
`value_pred` captured at collection time, since both the TD residual and the
advantage depend on estimates that must come from the *acting* policy, not a
value refit after the fact.

## Summary

| Case | Return / advantage | Critic target $V(S)$ learns |
|---|---|---|
| (1) `terminal_margin` | $G_k = M$ | expected final margin |
| (2) `decision_delta`, γ = 0 | $G_k = P(S_{k+1}) - P(S_k)$ | expected one-step margin delta |
| (3) `decision_delta`, γ = 1 | $G_k = M - P(S_k)$ | expected *remaining* margin gain |
| (4) `gae` | $A_k$ (TD-residual sweep, above) | $A_k + V(S_k)$, blending realized reward with the critic's own estimate |

Intermediate γ (the config default is `reward_discount = 1.0`, mode default
still `TERMINAL_MARGIN`) interpolates:

$$
G_k = \sum_{j \ge k} \gamma^{\,t_j - t_k}\, r_j
$$

exponentially down-weighting point swings the further in the future they land —
a soft credit horizon between the myopic extreme (2) and the full-horizon
extreme (3). The exponent is measured on the **game clock** (next section), not
in decision steps.

## The game clock: discounting in game time, not decision steps

Decision steps are wildly uneven in game time — a bare lay-eggs turn is one
recorded decision while a play-bird turn with chained powers can be six — so a
fixed per-step γ would discount the future faster through decision-dense turns.
Instead every recorded decision carries a timestamp $t$ (`Step.timestamp`) and
the discount between consecutive checkpoints is $\gamma^{\Delta t}$:

$$
G_k = r_k + \gamma^{\,t_{k+1} - t_k}\, G_{k+1}
$$

The clock (`wingspan.training.timestamps`):

- **Setup window** (before any turn): the hand keep at $0$, the deferred bonus
  pick at $\tfrac13$, the deferred food picks at $\tfrac23$ — the same values
  for both seats, modeled as simultaneous. Multiple food picks share $\tfrac23$
  ($\Delta t = 0$, and $\gamma^0 = 1$ even at $\gamma = 0$, so credit passes
  through zero-time links undecayed).
- **Main actions**: the $n$-th turn of the game (counting both seats' turns in
  order, $2 \times (8+7+6+5) = 52$ total) has its main-action decision at
  exactly $n$ — consecutive integers alternate players.
- **Mid-turn decisions** (everything recorded inside turn $T$'s window after
  its main action, including the *opponent's* reaction decisions): linearly
  interpolated, the $j$-th of $k$ at $T + \tfrac{j}{k+1}$. The interpolation is
  resolved after the game (`finalize_timestamps`), since a turn's decision
  count is only known once it ends.
- **Terminal checkpoint**: the final margin sits at the end of the last turn's
  window, $t = 53$ for a full game (`GameRecord.final_timestamp`), shared by
  both seats.

The extremes are unchanged in spirit: $\gamma = 1$ still telescopes exactly to
case 3 ($\Delta t$ never matters when $\gamma^{\Delta t} = 1$), and $\gamma = 0$
is still the myopic extreme (every positive-$\Delta t$ link cuts the future
off). What changes is everything in between: an engine bird whose payoff lands
five of *your* turns later is discounted by $\gamma^{10}$ (your consecutive
main actions are two timestamp units apart, since the opponent's turn sits
between them) regardless of how many decisions anyone made in between.

One more implementation nuance worth knowing: under the default `MARGIN`
basis, the two seats' steps live in the same batch with opposite-signed
margins (zero-sum self-play), and because rewards are differenced per player
using only *that player's* decision checkpoints, opponent moves between your
decisions fold into your $r_k$ — the reward measures "how did the margin move
between my decisions," not "what did my action alone score." Under
`OWN_SCORE` basis (below) this is no longer zero-sum — both seats' checkpoint
sequences are their own non-negative running score, not opposite-signed
margins — but the per-player checkpoint routing and the game-clock $\Delta t$
discounting are otherwise unchanged.

## Reward basis: MARGIN vs OWN_SCORE

Everything above is written for `reward_basis = MARGIN` (`RewardBasis.MARGIN`,
the config default), where $P(S)$ is the running margin and $M$ the final
margin — own score minus the best other seat's, so a 2-player game's two
seats always get exactly opposite values.

`RewardBasis.OWN_SCORE` swaps the checkpoint quantity for each player's own
absolute score instead:

- $P(S)$ becomes `score_before` (the collector's per-step running own-score
  snapshot) rather than `margin_before`.
- $M$ becomes the player's own final score rather than the margin against the
  best opponent.

Concretely, `learner._terminal_margin_returns`, `_decision_delta_returns`, and
`_gae_flatten` all branch on `cfg.training.reward_basis`: with `OWN_SCORE`
they read `step.score_before` where the formulas above read `step.margin_before`,
and `returns.terminal_values` computes each seat's own final score (with
`end_game_bonus` added only for the winner) rather than a signed margin. Every
formula in this document — cases 1-4, the game clock, batch-wide advantage
normalization — carries over unchanged with this substitution; only the sign
relationship between seats changes. Because both seats' values are now
positive rather than opposite-signed, the gradient pushes each seat toward
maximizing its own raw score regardless of what the opponent scores, rather
than toward beating the opponent — this is a genuinely different training
objective, not just a rescaling.
