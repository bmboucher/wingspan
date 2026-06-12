# Reward modes: how actor and critic losses are computed

This explains the math behind `cfg.reward_mode` (`learner._flatten`,
`learner._terminal_margin_returns`, `learner._decision_delta_returns`) in three
cases:

1. **`terminal_margin`** — the old method: every decision gets the terminal
   point margin.
2. **`decision_delta` with γ = 0** — per-decision margin differences, no
   lookahead.
3. **`decision_delta` with γ = 1** — per-decision margin differences, full
   lookahead.

## Notation and setup common to all three cases

Fix one player. Let $S_1, S_2, \dots, S_K$ be the states at that player's
successive decisions, and let $M$ be the **final** point margin from their POV.

- $P(S)$ — the running point margin of state $S$ (own score − opponent's, as if
  the game ended now). The collector snapshots this as `margin_before` right
  before each decision (`collect.running_margin`).
- $V(S)$ — the critic's value estimate for state $S$.
- $P(S, A)$ — the actor's logit (pre-softmax score) for action $A$ in state $S$.

Note that $S_{k+1}$ is the state at your **next** decision, so
$P(S_{k+1}) - P(S_k)$ includes both your action's effect **and** whatever the
opponent did in between (margin is own − opponent).

Every reward mode just produces a per-step **return** $G_k$; the loss machinery
downstream is identical in all three cases.

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

## Summary

| Case | Return $G_k$ | Critic target $V(S)$ learns |
|---|---|---|
| (1) `terminal_margin` | $M$ | expected final margin |
| (2) `decision_delta`, γ = 0 | $P(S_{k+1}) - P(S_k)$ | expected one-step margin delta |
| (3) `decision_delta`, γ = 1 | $M - P(S_k)$ | expected *remaining* margin gain |

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

One more implementation nuance worth knowing: the two seats' steps live in the
same batch with opposite-signed margins (zero-sum self-play), and because
rewards are differenced per player using only *that player's* decision
checkpoints, opponent moves between your decisions fold into your $r_k$ — the
reward measures "how did the margin move between my decisions," not "what did
my action alone score."
