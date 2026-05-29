# DECISIONS.md

A review of every discrete decision the Wingspan engine exposes to an agent,
the modelling considerations for each, and an assessment of how appropriate
the current decision boundaries are for the long-term plan of training **one
model per kind of choice**.

This document is descriptive of the code as it stands (`src/wingspan/`) and
prescriptive about where the taxonomy should move. It is written for the RL
side of the project, but the decision taxonomy it describes is the same one
the manual CLI and the engine itself use, so it doubles as a map of the
game's branching factor.

---

## 0. The one thing to fix first: today there is *not* one model per decision

Your stated design — "each discrete *type* of choice a player could make will
have its own model" — is **not** what the code does today. It is worth being
precise about the gap before discussing individual decisions, because it
changes how to read every "appropriateness" note below.

What exists today (`src/wingspan/model.py`, `src/wingspan/encode.py`):

- **One** `PolicyValueNet`. A two-layer state trunk, a two-layer per-choice
  encoder, **one** scoring MLP, **one** value head.
- The network is told *which* decision it is answering only by a **19-wide
  one-hot stripe** appended to the state vector
  (`encode._encode_decision_type`, indexed by `decisions.ALL_DECISION_CLASSES`).
- Every decision — setup, main action, "which egg to discard", "which die to
  take", "which bonus to keep" — flows through that same network. The policy
  is a *pointer* policy: it scores each legal candidate from its per-choice
  feature row and softmaxes (`model.PolicyValueNet.forward`).

So the decision *taxonomy* (the 19 `Decision` subclasses in
`src/wingspan/decisions.py`) is real and well-factored; the *model* that
consumes it is monolithic. The taxonomy is the right backbone for your
per-decision-model plan — but realizing that plan means turning the single
scorer into **per-family scoring heads** (or, for a couple of decisions, fully
separate networks), keyed off the decision class instead of leaning on a
one-hot.

The design space is a spectrum:

| Approach | What it is | Pros | Cons |
|---|---|---|---|
| **Monolithic (today)** | 1 trunk, 1 scorer, decision-type one-hot | Max parameter sharing; uniform variable-action handling; one training loop | Gradient interference across very different judgments; rare decisions starve; the net must learn to *route* on the one-hot |
| **Shared trunk + per-family heads** (recommended) | 1 trunk + 1 choice-encoder, *K* scoring heads selected by decision family, shared value | Specialization where it matters; keeps the shared "read the board" representation; one training loop; analyzable per family | Slightly more plumbing; must define the family→head map |
| **Fully separate models** (your literal phrasing) | *N* independent networks | Cleanest credit assignment; each trainable/analyzable in isolation | *N×* parameters; no shared board representation; the rare decision types see almost no data; harder joint training under REINFORCE |

The recommendation that falls out of the per-decision analysis below: **fully
separate models for the two structurally unique, high-stakes decisions
(`SetupDecision`, `MainActionDecision`); a small set of shared judgment-family
heads for everything else; one shared value/critic for all of them** (position
value is a property of the board, not of the decision being asked).

---

## 1. How a decision flows through the engine

Every decision point is one call to `Engine.ask(agent, decision)`
(`src/wingspan/engine/core.py:131`). The contract:

- A `Decision[C]` carries `player_id`, a human `prompt`, and a non-empty
  `choices: list[C]`. `C` is a `Choice` subclass (or a union including
  `SkipChoice` / `PayCostChoice`).
- The agent returns one of the offered choices; `ask` validates membership by
  Pydantic field-equality and rejects anything illegal.
- **Forced moves are not modelled.** When a decision has exactly one legal
  choice, the engine helpers usually resolve it inline (`_pick_habitat`,
  `take_one_from_feeder`, `_pick_food_payment` all short-circuit on a
  singleton), and the training agent explicitly skips recording single-choice
  decisions (`train.make_policy_agent`: `if n == 1: return …` without
  appending a `Step`). So the trainable surface is only decisions with **≥2
  real options**.
- The value head reads **only** state (POV-aware), so it is naturally
  decision-type-agnostic and can stay shared no matter how the policy is split.

`ALL_DECISION_CLASSES` (`src/wingspan/decisions.py:321`) pins the stable order
of the decision-type one-hot. Any refactor into per-family heads must keep that
class→index map stable (or re-train) so existing checkpoints stay aligned — the
family→head map should be a pure function of the decision class.

---

## 2. Catalog of the 19 decision types as built

`Choice` shape, where each fires, whether it can be declined, and how many
*distinct call sites* construct it (the multiplexing count — high counts are
where one decision type is doing several jobs).

| # | Decision class | Choice type(s) | Fires when… | Skip? | Call sites |
|---|---|---|---|---|---|
| 1 | `MainActionDecision` | `MainActionChoice \| PlayBirdChoice` | top of every turn | no | 1 |
| 2 | `SetupDecision` | `SetupChoice` | once per player, pre-round-1 | no | 1 |
| 3 | `PlayBirdPickCardDecision` | `BirdChoice` | extra-play path only (main path folds this into the `PlayBirdChoice`) | no | 1 |
| 4 | `PlayBirdPickHabitatDecision` | `HabitatChoice` | playing a 2-habitat bird via the extra-play path | no | 1 |
| 5 | `PlayBirdPickFoodPaymentDecision` | `FoodPaymentChoice` | ≥2 ways to pay a bird's food cost (extra-play path) | no | 1 |
| 6 | `PlayBirdPickEggToPayDecision` | `BoardTargetChoice \| SkipChoice` | spending an egg | varies | **2** (play-bird/convert egg cost; discard-egg-for-wild) |
| 7 | `GainFoodPickDieDecision` | `FoodChoice` | taking a birdfeeder die | no | **2** (main Gain Food; each-player-gains-die power) |
| 8 | `LayEggPickBirdDecision` | `BoardTargetChoice \| SkipChoice` | placing an egg | varies | **3** (main lay/convert/lay-any; pink reactor; lay-on-nest) |
| 9 | `DrawCardsPickSourceDecision` | `DrawSourceChoice` | drawing a card (tray slot vs blind deck) | no | 1 |
| 10 | `BirdPowerPickFoodDecision` | `FoodChoice \| SkipChoice \| PayCostChoice` | many power food picks | varies | **6** (see §4.1) |
| 11 | `BirdPowerPickBirdFromHandDecision` | `BirdChoice` | drafting one of several drawn birds (Oystercatcher) | no | 1 |
| 12 | `BirdPowerPickPlayedBirdDecision` | `PlayedBirdChoice` | choosing which bird's power to repeat | no | 1 |
| 13 | `BirdPowerPickBonusCardDecision` | `BonusCardChoice` | keeping one of several drawn bonus cards | no | 1 |
| 14 | `BirdPowerTuckFromHandDecision` | `BirdChoice \| SkipChoice` | tucking a card from hand | yes | 1 |
| 15 | `BirdPowerPickStartingPlayerDecision` | `PlayerIdChoice` | choosing who starts an each-player feeder gain | no | 1 |
| 16 | `BirdPowerPickHabitatDecision` | `HabitatChoice` | move-bird-if-rightmost destination | no | 1 |
| 17 | `GainFoodConvertDecision` | `BirdChoice \| SkipChoice` | Forest trade space: discard a card for +1 food | yes | 1 |
| 18 | `LayEggsConvertDecision` | `FoodChoice \| SkipChoice` | Grassland trade space: spend a food for +1 egg | yes | 1 |
| 19 | `DrawCardsConvertDecision` | `PayCostChoice \| SkipChoice` | Wetland trade space: discard an egg for +1 card | yes | 1 |

Two structural facts to carry forward:

- **The same `Choice` shape serves opposite judgments.** `BoardTargetChoice`
  is used both to *place* an egg (#8) and to *remove* one (#6), and the
  encoder featurizes both identically (`encode._featurize_board_target`:
  habitat one-hot, slot, eggs, capacity-remaining, cached, tucked). The only
  thing telling the network "more eggs here is good" (placement) from "more
  eggs here is bad to lose" (removal) is the decision-type one-hot.
- **"Skip?" is per-call-site, not per-type.** #6, #8 and #10 attach a
  `SkipChoice` in some contexts and not others (e.g. the main Lay Eggs action
  forces placement — `actions.lay_one_egg` appends no skip — while the pink
  reactor and lay-on-nest powers make it optional). The *type* permits skip;
  the *call site* decides. A per-type model therefore sees a moving target for
  "can I decline?", which it can only read from the presence/absence of a skip
  candidate, not from the decision identity.

---

## 3. Per-decision analysis, grouped by the judgment it exercises

The 19 classes are cut by **where in the turn they fire** (their trigger), not
strictly by **what judgment they require**. Grouping them by judgment is the
useful lens for "one model per type", because the judgment is what a model
would specialize in. Each group below gives the decision(s), the real
modelling considerations (what an expert weighs / what the net must learn), and
an appropriateness verdict.

### 3.1 Opening draft — `SetupDecision` (#2)

**What it is.** Once per player, before round 1: keep a subset of the 5 dealt
birds (each kept bird costs 1 food off a one-of-each starting stash), keep the
complementary number of foods, and keep exactly one of two dealt bonus cards.
Exposed as a **single** decision enumerating every legal combination — 504 for
the standard 5/2 deal (`engine/core._build_setup_choices`).

**Modelling considerations.** Hand/food affordability (can the kept foods
actually play the kept birds early?), curve (cheap birds first), bonus-card
alignment with the kept birds, habitat spread, and engine pieces (brown
"activate" powers). This is a *joint* optimization: food value depends on which
birds you kept, and bonus value depends on both — which is exactly why bundling
into one combined action space is correct.

**Appropriateness — strongly its own model. ✅** Different cadence (once per
game), different action shape (a fixed combinatorial menu, not a board
operation), and no board state to read yet. This is the cleanest candidate in
the whole game for a fully separate network. **But the featurization is the
real bottleneck, not the boundary**: `encode._featurize_setup` reduces a
candidate to *aggregate* kept-set stats (sum of points/cost/egg-limit, count),
a spent-food multi-hot, and `bonus_card.id % 16`. It cannot see *which specific
birds* were kept, so it cannot learn card-specific opening synergies — the very
thing the project wants to answer ("opening-hand selection"). A per-setup model
will be capped by this until the candidate encodes the actual kept cards.

### 3.2 Macro action — `MainActionDecision` (#1)

**What it is.** The strategic spine: at the top of each turn choose among Gain
Food, Lay Eggs, Draw Cards, or **a specific** `(bird, habitat, payment)` play.
Playing a bird is *not* a fourth enum value — each legal play is its own
`PlayBirdChoice` in the menu (`engine/core._main_action_decision`), so the
habitat and food-payment sub-picks are folded in here and only the egg cost
remains a follow-up.

**Modelling considerations.** Engine-building vs. immediate points; the
action-reward track (more birds in a row ⇒ a stronger action — so where you
play matters beyond this turn); tempo and action-cube scarcity (8→7→6→5 per
round); the current round goal; food/egg/card economy; opponent denial.

**Appropriateness — strongly its own model. ✅** Highest-leverage decision and
structurally unique. Note a real asymmetry the scorer must bridge: the three
habitat actions are encoded as a nearly-featureless token (one scalar action
index in the SPECIAL stripe, `encode._featurize_main_action`), while each play
candidate is feature-rich (full bird stripe + habitat + payment). So "is gain
food better than playing this bird?" is a comparison where one side's value
lives almost entirely in the **state**, not the choice features. A dedicated
head (and arguably a dedicated network) is well justified.

### 3.3 Bird valuation — #3, #9, #11, #14, #17 (and `SetupChoice.kept_cards`)

**The shared judgment.** "How valuable is *this bird* to me — to keep/play vs.
to give up?" Five decision types ask a flavor of it:

- `PlayBirdPickCardDecision` (#3) — which hand bird to play (extra-play path).
- `DrawCardsPickSourceDecision` (#9) — take a *named* tray bird or draw blind
  from the deck. (Adds a value-of-information twist the others lack: the deck
  option is unseen.)
- `BirdPowerPickBirdFromHandDecision` (#11) — keep one of several freshly drawn
  birds (American Oystercatcher draft).
- `BirdPowerTuckFromHandDecision` (#14) — give a hand card up to tuck it (it
  becomes 1 VP + a card-count for goals) — or skip.
- `GainFoodConvertDecision` (#17) — give a hand card up to gain 1 food — or
  skip.

**Modelling considerations.** Points, cost vs. current food, habitat fit and
open slots, power synergy with the board, bonus-card progress, and — for the
"give up" variants — whether the card is worth more in hand than as a
tuck/food/discard.

**Appropriateness — keep as distinct decision points, route to one shared
"bird-value" head.** The trigger and the action set differ (skip vs. no skip;
play vs. tuck vs. discard), so they are legitimately separate *decision
points*. But the core judgment is one skill, and the per-choice encoder already
featurizes a bird identically everywhere (`encode._fill_bird`). A single
bird-valuation head, conditioned on the decision-type one-hot to know the
*context* (what happens to the bird), is the right granularity. Giving each its
own fully independent model would fragment a single skill across five
data-starved learners.

### 3.4 Food acquisition — `GainFoodPickDieDecision` (#7) + part of #10

**The shared judgment.** "Which food advances my plans?" — choosing a face
from the birdfeeder, or a food from the supply.

**Modelling considerations.** The food costs of birds you intend to play; wild
flexibility; scarcity of a face still showing in the feeder; future curve.

**Appropriateness — currently split by plumbing; should be unified. ⚠️** The
*identical* act "take a die from the feeder" is asked through **two different
decision types** depending on who triggers it: the main Gain Food action and
the each-player power use `GainFoodPickDieDecision` (#7;
`actions._take_one_die_active`, `powers._h_each_player_gains_die_choose_order`),
but every power that calls `actions.take_one_from_feeder` asks it through
`BirdPowerPickFoodDecision` (#10) instead. That is an artifact of two code
paths, not two judgments. For a per-decision-model plan, **unify the
feeder/supply "gain a food" judgment under one head** regardless of trigger.

### 3.5 Food expenditure — `PlayBirdPickFoodPaymentDecision` (#5), `LayEggsConvertDecision` (#18), + part of #10

**The shared judgment.** The *inverse* of acquisition: "which food can I most
afford to part with / which payment preserves the most flexibility?"

- `PlayBirdPickFoodPaymentDecision` (#5) — among legal payment combinations for
  a bird (matters when wild or substitutable costs create choices).
- `LayEggsConvertDecision` (#18) — spend one food to lay one extra egg.
- `BirdPowerPickFoodDecision` used by Green Heron (`powers._h_trade_wild_food`,
  first half) — which food to discard to the supply before re-drawing a
  different one.

**Modelling considerations.** Hold wild for flexible future costs; don't strand
a bird you intend to play by spending the food it needs; value the marginal
egg/trade against the food given up.

**Appropriateness — spend is its own judgment and is currently entangled with
gain. ⚠️** The cleanest issue in the taxonomy: **food *gains* and food
*give-aways* share decision type #10** (`BirdPowerPickFoodDecision`). A model
keyed on that type must learn both "pick the food I most want" and "pick the
food I least want" from one head, told apart only by state. Recommend a
distinct "spend-food" head and stop overloading #10 (see §4.1).

### 3.6 Egg placement — `LayEggPickBirdDecision` (#8)

**What it is.** "Which of my birds gets the egg?" Used everywhere an egg is
*added*: the main Lay Eggs action, the Grassland conversion, `LAY_EGG_ANY`
powers, the all-players/lay-on-nest powers, and the pink lay-egg reactors
(3 call sites, the most-reused placement decision).

**Modelling considerations.** Round-goal nests/habitats; bonus cards keyed on
eggs; prefer high-`egg_limit` birds (more future capacity) unless a goal rewards
spreading; birds you will keep to game end; capacity remaining
(`encode._fill_board_target` surfaces eggs + capacity-remaining).

**Appropriateness — well-unified; deserves its own head. ✅** Frequent, and the
judgment is genuinely the same across every trigger, so the heavy reuse is
correct. One caveat shared with §3.7: because placement and removal use the
same `BoardTargetChoice` features, the head leans on the decision-type one-hot
to know the *sign* of "more eggs here."

### 3.7 Egg removal — `PlayBirdPickEggToPayDecision` (#6)

**What it is.** "Which egg do I spend?" Used wherever an egg is *removed*: the
play-bird egg cost and the Wetland conversion (both via
`actions.discard_an_egg`), and the discard-egg-for-wild power
(`powers._h_discard_egg_for_wild`).

**This is exactly your worked example — and the code already half-realizes
it.** You said the *which egg to discard* choice "should have similar
considerations across different contexts" and be one model, separate from the
*decision to pay*. The engine already routes all three egg-removal contexts
through this single decision type, with identical `BoardTargetChoice`
featurization. So the "which egg" sub-skill **is** unified today. ✅

Three caveats keep it from being clean:

1. **Naming.** `PlayBirdPickEggToPayDecision` is generic in use but named for
   one caller. Rename to something like `RemoveEggDecision`.
2. **The reason is hidden.** All three contexts emit the same decision-type
   one-hot, so the model cannot tell "I'm paying to play a bird" from "I'm
   trading an egg for wild food." Willingness and which-egg differ by reason
   (e.g. a wild-food trade may be skippable and lower-stakes). The "value at
   stake" / reason should be an explicit feature if this head is to behave
   differently per context.
3. **Skip varies by context** (mandatory for the play-bird cost, optional for
   the wild trade) — see §2.

**Appropriateness — the right boundary, with a rename and a reason feature.**

### 3.8 Commit-to-cost (yes/no exchange) — #19, part of #10, and the folded-in cases

**The shared judgment.** "Is this exchange worth it, given my position and the
round goal?" — independent of *which* resource is then chosen. This is the
**other half** of your worked example: "the decision to pay … should be
separated from the decision of which [resource] to take."

Today it is modelled **three inconsistent ways**:

- **Standalone yes/no** — `DrawCardsConvertDecision` (#19, egg→card) and
  tuck-from-deck-paid (`powers._h_tuck_from_deck_paid`, food→tuck) offer
  `PayCostChoice \| SkipChoice`. The `PayCostChoice` carries **no fields** — its
  *type* is the entire signal (`decisions.PayCostChoice`), so the commit head
  leans entirely on state + decision-type, with nothing describing *what is
  gained for what is paid*.
- **Folded into the resource pick's skip** — the Forest (#17, card→food) and
  Grassland (#18, food→egg) conversions roll "should I?" into the same decision
  as "which card/food?", via a `SkipChoice` among the resource candidates.
- **Implicit / forced** — the play-bird egg cost has *no* yes/no at all: once a
  `PlayBirdChoice` is chosen, the egg payment is mandatory (`do_play_bird` just
  calls `discard_an_egg` `egg_cost` times). The "decision to pay eggs to play a
  bird" you named is, in the code, **subsumed into the macro action pick** (#1)
  rather than being a separate decision.

**Appropriateness — this is the biggest mismatch with your plan. ⚠️** You want
"should I pay?" to be its own consistently-separated model; the code expresses
it as a standalone choice, a skip option, or nothing, depending on the
trigger. Recommendation: define one **"accept exchange?" decision/head**, give
its `PayCostChoice` real features (resource gained, resource paid, quantities),
and split the downstream "which resource" pick into the appropriate gain/spend
head. The play-bird egg cost can stay folded into #1 (it is genuinely part of
choosing the play), but it should then *not* be described as a separate pay
decision.

### 3.9 Bonus-card valuation — `BirdPowerPickBonusCardDecision` (#13) + `SetupChoice.bonus_card`

**The shared judgment.** "Which bonus fits the board/plan I'm building?" — how
many qualifying birds you have or can still get, and whether a VP threshold is
reachable.

**Appropriateness — same judgment, two homes, weak features. ⚠️** Setup bundles
the bonus pick into #2; mid-game keep-a-bonus is #13. More importantly, the
encoder represents a bonus card only as `bonus_card.id % 16`
(`encode._featurize_bonus_card`, and the same hash inside `_featurize_setup`) —
an identity hash, not a description of what the bonus rewards. The network
cannot actually *evaluate* a bonus. Since "bonus-card value" is one of the
project's headline analytical questions, this is a featurization gap that a
dedicated bonus head would expose immediately. Recommend a real bonus encoding
(category, thresholds, current qualifying-bird count) feeding a shared
bonus-valuation head used by both #2 and #13.

### 3.10 Habitat placement — `PlayBirdPickHabitatDecision` (#4) + `BirdPowerPickHabitatDecision` (#16)

**The shared judgment.** "Which row benefits most?" — choosing a habitat for a
two-habitat bird (#4) or a destination for a moved bird (#16,
move-if-rightmost).

**Modelling considerations.** Which action track you want to strengthen
(tempo); the egg-cost ladder (each row's `next_egg_cost` depends on its
length); row-power synergy; the round goal's habitat.

**Appropriateness — one judgment, could share a head. ✅(minor)** Low frequency;
fold both into a small "habitat-placement" head. No need for separate models.

### 3.11 Rare / structural — `BirdPowerPickStartingPlayerDecision` (#15), `BirdPowerPickPlayedBirdDecision` (#12)

- #15 — pick who starts an each-player feeder gain (turn-order manipulation).
- #12 — pick which adjacent brown / predator power to **repeat**
  (`PlayedBirdChoice`).

**Appropriateness — do *not* give these their own models. ⚠️** They fire on a
handful of cards, so a dedicated network would be perpetually data-starved.
#12 in particular is a *high-value* judgment ("which power is best to copy?")
but very rare — a good argument for folding it into the shared bird/power heads
(it is really "value this bird's power again") rather than isolating it. #15 is
low-stakes; a heuristic or the shared misc head is plenty.

---

## 4. Cross-cutting findings

### 4.1 `BirdPowerPickFoodDecision` (#10) is overloaded across four judgments

It is the most-multiplexed type (6 call sites) and spans judgments that a
per-decision model should keep apart:

| Call site | Real judgment |
|---|---|
| `actions.take_one_from_feeder` (feeder die for powers/pink) | **gain** food (feeder) |
| `powers._h_discard_egg_for_wild` (wild pick) | **gain** food (supply) |
| `powers._h_fewest_forest_gains_die` | **gain** food (feeder) |
| `powers._h_trade_wild_food` — discard half (Green Heron) | **spend / give away** food |
| `powers._h_trade_wild_food` — gain half | **gain** food (supply) |
| `powers._h_tuck_from_deck_paid` | **commit** to a fixed cost (yes/no) |

One head being asked to "pick the food I most want", "pick the food I least
want", and "decide whether to pay at all" is the clearest case where the
current type boundary is too coarse for your plan. Split into the gain-food
(§3.4), spend-food (§3.5), and commit (§3.8) heads.

### 4.2 The taxonomy is cut by *trigger*, not by *judgment*

Three symptoms, all already noted above: the same judgment split across types
(take-a-die: #7 vs #10), opposite judgments merged into one type (gain vs spend
in #10; place vs remove sharing `BoardTargetChoice` features), and "should I
pay?" expressed three different ways (§3.8). For "one model per type" to mean
"one model per skill", re-cut along judgment lines.

### 4.3 Featurization caps several heads before the boundary matters

Independently of how the policy is split, three candidate encodings throw away
the signal the corresponding head would need:

- **Setup** candidates expose only aggregate kept-set stats, not the specific
  cards (§3.1).
- **Bonus** candidates are an id hash, not a description (§3.9).
- **`PayCostChoice`** candidates carry no features at all (§3.8).

For the analytical goals (card power, bonus value, opening selection) these
are the limiting factor, not the decision granularity.

### 4.4 One critic is fine; the policy is what should specialize

REINFORCE today assigns every step the same terminal score-advantage return
and subtracts one shared value baseline (`train.train_step`). Because position
value is decision-agnostic, **keep the value head shared** across whatever
policy split you adopt — per-family or per-decision policy *heads* on top of a
shared trunk and shared critic is the change that buys specialization without
multiplying the critic or starving it of data.

---

## 5. Recommended judgment-family taxonomy

A concrete target for "per-decision models", mapping the 19 as-built classes
onto the skill each exercises. Each family = one scoring head on a shared trunk
(except where noted as a candidate for a fully separate network).

| Family (head) | As-built decision classes | Note |
|---|---|---|
| **Setup draft** | #2 | Separate network; fix card-level features |
| **Macro action** | #1 | Separate network; the strategic spine |
| **Bird valuation** | #3, #9, #11, #14, #17 (+ setup's kept-cards) | One head, context via one-hot |
| **Gain food** | #7, gain-parts of #10 | Unify the two take-a-die paths |
| **Spend food** | #5, #18, Green-Heron-discard part of #10 | Stop sharing with gain |
| **Egg placement** | #8 | Own head; frequent |
| **Egg removal** | #6 | Own head; rename; add a "reason" feature |
| **Commit-to-cost** | #19, tuck-paid part of #10 (+ the yes/no inside #17/#18) | Consistent yes/no; feature the exchange |
| **Bonus valuation** | #13 (+ setup's bonus pick) | Needs real bonus features |
| **Habitat placement** | #4, #16 | Small shared head |
| **Misc / rare power** | #12, #15 | Do not isolate; data-starved |

---

## 6. Recommended next steps (in order)

1. **Introduce a `decision → family` map** (pure function of the decision
   class) and give `PolicyValueNet` one scoring head per family while keeping
   the shared trunk, shared choice-encoder, and shared value head. This is the
   minimal change that turns "one model" into "one model per kind of choice"
   without exploding parameter count or starving rare decisions.
2. **Split `BirdPowerPickFoodDecision`** (§4.1) into gain-food / spend-food /
   commit decisions so the family map is clean and the conflated judgments stop
   sharing gradients.
3. **Unify the two "take a die" paths** (#7 vs #10) onto the gain-food family
   regardless of trigger (§3.4).
4. **Make "should I pay?" a first-class, consistently-separated decision**
   (§3.8) with features on `PayCostChoice` describing the exchange; leave the
   mandatory play-bird egg cost folded into the macro action and document it as
   such.
5. **Fix the three featurization gaps** (§4.3) — specific setup cards, real
   bonus-card features, exchange features on `PayCostChoice` — before reading
   anything analytical off the trained heads.
6. **Rename `PlayBirdPickEggToPayDecision` → a generic egg-removal name** and
   add a "reason / value-at-stake" feature so the shared removal head can act
   differently per context (§3.7).
7. **Keep `SetupDecision` and `MainActionDecision` as the two candidates for
   fully separate networks** if/when you want to train or analyze them in
   isolation; everything else stays head-on-shared-trunk.
