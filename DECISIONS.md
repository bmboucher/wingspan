# DECISIONS.md

How the Wingspan simulator turns a game of Wingspan into a set of *decisions*,
and how the model we plan to train mirrors the structure of those decisions.

Wingspan is a game about a relatively small number of repeated judgments —
*which bird is worth playing? which food do I want? where does this egg go? is
this trade worth it?* — applied over and over in different situations. A good
player isn't someone who has memorized a separate rule for every card; it's
someone who has gotten good at each of those underlying judgments and knows
which one a given moment is asking for.

This document explains how the codebase encodes that idea. It is written for a
reader who understands Wingspan and the broad strokes of machine learning, but
not necessarily this code. It proceeds in three steps:

1. **The model architecture** — the shape of the network we train, and why it
   has the shape it does.
2. **The three vocabularies** — *choices*, *decisions*, and *judgment
   families* — and a full catalog of each.
3. **A tour of the judgments** — what an expert weighs at each kind of
   decision, and how the design captures that.

The same decision taxonomy drives the human CLI, the engine's turn loop, and
the RL pipeline, so this doubles as a map of where the game branches.

---

## 1. The model architecture

### 1.1 The core idea: score candidates, don't enumerate actions

Most board-game RL networks emit a fixed-size policy vector — one output per
possible action. That is awkward for Wingspan, where the number of legal
actions swings wildly from turn to turn: you might have two legal bird plays or
twenty, three foods to choose from or one, 504 opening hands to weigh at setup.
A fixed output vector would have to reserve a slot for every action that could
*ever* be legal and mask out the rest.

Instead the network is a **pointer-style** policy. At every decision it is
handed the current game state plus *a list of the legal options, each described
by its own feature vector*, and it produces **one score per option**. A softmax
over those scores is the policy. Nothing about the network's shape depends on
how many options there are — two or two hundred, the machinery is identical.
This is the natural fit for a game whose branching factor is so uneven.

### 1.2 Shared body, specialized heads

The deeper design choice is *what gets shared and what gets specialized*.

Two things are true at once in Wingspan. First, almost every decision is made
by **reading the same board**: your food, your birds and their eggs, the round
goal, the feeder, what the opponent is doing. That "read the position" skill is
common to everything. Second, the *judgments themselves are different skills*:
deciding which food to grab is nothing like deciding which egg to give up,
which is nothing like deciding whether a one-for-one trade is worth taking.

So the network is built as **a shared body with specialized heads**:

- A **state trunk** reads the board once and produces a context vector. Shared
  by everything.
- A **per-choice encoder** reads each candidate option's features into an
  embedding. Shared by everything.
- A **bank of scoring heads**, one per *judgment family* (see §2.3). The
  context vector and a candidate's embedding are concatenated and passed
  through **the head for whatever judgment this decision exercises**. Only this
  final scorer is specialized.
- A single **value head** (the critic) hangs off the trunk and estimates how
  good the current position is.

The phrase to keep in mind is **one model per *kind* of choice**. We do not
train a separate network for the Belted Kingfisher and another for the Common
Loon; we train one *food-gain* judgment, one *egg-placement* judgment, one
*is-this-trade-worth-it* judgment, and so on, each as a head that sees every
situation in the game that exercises that skill — regardless of which card or
rule triggered it.

### 1.3 The box diagram

```
        Game state (from the deciding                Each legal option, as its own
        player's point of view)                      feature vector  (K of them)
                    │                                          │
                    ▼                                          ▼
          ┌───────────────────┐                    ┌───────────────────────┐
          │    State trunk     │                    │   Per-choice encoder   │
          │   (2-layer MLP)    │   shared           │     (2-layer MLP)      │  shared
          └─────────┬─────────┘                    └───────────┬───────────┘
                    │ state context                            │ one embedding
        ┌───────────┴───────────┐                              │ per candidate
        ▼                        ▼                             │
 ┌──────────────┐      concat(state context, ◄─────────────────┘
 │  Value head  │             candidate embedding)
 │  (shared     │                    │
 │   critic)    │                    ▼
 └──────┬───────┘     ┌───────────────────────────────────────┐
        │             │   Scoring head for THIS decision's     │   ← chosen by
        ▼             │   judgment family                      │     judgment
  position value      │   (1 of K specialized heads)           │     family
                      └────────────────────┬───────────────────┘
                                           ▼
                                  one score per candidate
                                           │
                                           ▼
                            softmax over the legal candidates
                                  = the policy for this decision
```

Read left to right: the board is read once into a shared context; each option
is read into a shared embedding; the two are joined and scored by the *one* head
that matches the judgment being asked; the scores become a probability
distribution over the legal options. The critic reads only the board, so it is
shared no matter how the policy heads are organized.

### 1.4 How a single decision flows through the engine

The engine never asks an agent for "an action number." Every decision point is
one call of the form `Engine.ask(agent, decision)`, where the `decision`
carries:

- whose decision it is (the point-of-view player),
- a human-readable prompt (used by the CLI and the game log), and
- a **non-empty list of legal choices**, each a typed object.

The agent returns one of the offered choices, and `ask` verifies it was
actually on the menu. For the model, the flow is: encode the state from the
deciding player's point of view, encode each choice into a feature row, look up
which judgment family this decision belongs to, score the candidates with that
family's head, and sample.

**Forced moves are not really decisions.** When only one option is legal, the
engine usually resolves it without consulting the policy, and the training agent
declines to record single-option decisions. The trainable surface is therefore
just the moments with a genuine fork — two or more real options.

### 1.5 Why this middle ground

There is a spectrum here. At one extreme, a single monolithic scorer handles
every decision and has to learn to behave completely differently depending on a
"what kind of decision is this?" flag — different judgments fight over the same
weights, and the rare ones get drowned out. At the other extreme, a fully
separate network per decision gives the cleanest specialization but throws away
the shared "read the board" skill, multiplies the parameter count, and starves
the infrequent decisions of data.

The shared-trunk / per-family-head design is the productive middle: it keeps the
one expensive thing worth sharing (a learned representation of the position) and
specializes the one thing worth specializing (the judgment). It also keeps a
single value head, which is correct on principle — *how good is my position* is
a property of the board, not of which question you happen to be answering right
now.

The two genuinely singular decisions — the **opening draft** and the
**top-of-turn action pick** — are good candidates to eventually promote to fully
separate networks, because they have a unique cadence and unusually high stakes.
That is a possible future refinement, not a present limitation; today they are
heads on the shared trunk like the rest.

---

## 2. The three vocabularies

The code separates three ideas that are easy to conflate. Keeping them distinct
is what makes the taxonomy clean.

### 2.1 Choices — the *shape of the data* an option carries

A `Choice` is one selectable option, and each subclass exists to model one
**data shape**. There is no generic "payload"; every option exposes its data
through named, typed fields. The full set:

| Choice class | Carries | Used for |
|---|---|---|
| `SkipChoice` | (nothing) | declining an optional decision |
| `MainActionChoice` | the action type | picking Gain / Lay / Draw / Play Bird |
| `BirdChoice` | a bird | picking a bird from a hand or drawn pile |
| `PlayBirdChoice` | bird + habitat | a committed bird play (its costs are follow-ups) |
| `FoodPaymentChoice` | a complete payment multiset | paying a committed play's printed food cost |
| `PlayedBirdChoice` | a bird already in play | powers that target a bird on the board |
| `HabitatChoice` | a habitat | designating a row |
| `FoodChoice` | a food token | gaining or spending one food |
| `BoardTargetChoice` | a (habitat, slot) cell | adding/removing an egg on one of your birds |
| `BonusCardChoice` | a bonus card | keeping one of several bonus cards |
| `DrawSourceChoice` | tray slot *or* deck | where to draw a card from |
| `PlayerIdChoice` | a player | turn-order powers |
| `SetupChoice` | kept birds + kept foods + bonus | one whole opening keep |
| `PayCostChoice` | the terms of a fixed trade | taking a yes/no optional exchange |
| `ResetBirdfeederChoice` | (nothing) | affirming the optional feeder reroll |

The same data shape is deliberately reused across unrelated situations:
`BoardTargetChoice` describes a bird-with-eggs whether you are *placing* an egg
or *removing* one. The two situations look identical as data — what differs is
the *judgment*, and that difference lives in the scoring head, not the choice.

### 2.2 Decisions — *a specific fork* in the game

A `Decision` is a single branch point: a prompt plus a list of choices of one
shape. It is generic over the choice type it offers, so a decision that may be
declined offers, say, `BoardTargetChoice | SkipChoice` and the consumer can tell
the two apart by type. There are **19** decision classes — one per genuinely
distinct fork the engine can present.

### 2.3 Judgment families — *the skill* a decision exercises

A `DecisionFamily` is the underlying skill a decision tests, and it is the unit
the model specializes on: **one scoring head per family**. Several decision
classes collapse onto one family when they ask the same question for different
reasons. There are **13** families. The mapping from a decision class to its
family is a pure function of the class — a decision always routes to the same
head.

The 19 decisions and the 13 families:

| Family (one scoring head) | Decision class(es) that route to it | The skill |
|---|---|---|
| `SETUP` | `SetupDecision` | choosing a whole opening |
| `MAIN_ACTION` | `MainActionDecision` | which of the four actions this turn |
| `PLAY_BIRD` | `PlayBirdDecision` | which bird to play, and where |
| `DRAW_BIRD` | `DrawCardsPickSourceDecision`, `BirdPowerPickBirdFromHandDecision` | which bird to *take* |
| `DISCARD_BIRD` | `BirdPowerTuckFromHandDecision`, `DiscardBirdForFoodDecision` | which bird to *give up* |
| `GAIN_FOOD` | `GainFoodDecision` | which food to gain |
| `SPEND_FOOD` | `SpendFoodDecision`, `SpendFoodForEggDecision`, `PayBirdFoodDecision` | which food to give up |
| `LAY_EGG` | `LayEggDecision` | which bird gets the egg |
| `PAY_EGG` | `RemoveEggDecision` | which bird loses an egg |
| `SKIP_OPTIONAL` | `AcceptExchangeDecision` | is taking this optional exchange worth it? |
| `CHOOSE_BONUS` | `BirdPowerPickBonusCardDecision` | which bonus card fits my plan |
| `MISC_RARE` | `BirdPowerPickPlayedBirdDecision`, `BirdPowerPickGainOrderDecision`, `BirdPowerPickHabitatDecision` | rare structural picks |
| `RESET_BIRDFEEDER` | `ResetBirdfeederDecision` | is a fresh feeder roll worth more than what's showing? |

Two structural facts follow from this split and are worth holding onto:

- **One data shape can serve opposite skills, and the head is what tells them
  apart.** Placing an egg and removing an egg both present a `BoardTargetChoice`
  with identical features (habitat, slot, current eggs, capacity remaining,
  cached food, tucked cards). What makes "more eggs already here" *attractive*
  for placement but *costly* for removal is that the two route to different
  heads — `LAY_EGG` versus `PAY_EGG`.
- **"Can I decline?" is a property of the moment, not the decision type.** The
  main Lay Eggs action forces you to place the egg somewhere; a pink between-turn
  power lets you decline the very same kind of placement. So a `SkipChoice` is
  present or absent depending on the call site, and a head reads "am I allowed to
  pass?" from whether a skip option is on the menu.

The decision-type identity is *also* fed to the network as a small one-hot
appended to the state. With the judgment now carried by the head, that one-hot
does a narrower job: it gives a head that serves more than one decision class
(e.g. `DRAW_BIRD`, which sees both the draw-source pick and the
Oystercatcher draft) enough context to tell its own call sites apart.

---

## 3. A tour of the judgments

This section walks the 13 families in roughly the order a turn encounters them,
explaining what an expert actually weighs and how the design lets the model
learn it. The throughline: each family is *one skill*, exercised wherever the
game asks for it.

### 3.1 `SETUP` — the opening draft

**What happens.** Once per player before round 1: you are dealt 5 birds and 2
bonus cards, and you start with one food of each type. You keep some subset of
the birds (each kept bird costs one of your starting foods), keep the foods you
didn't spend, and keep exactly one bonus card. The engine presents this as a
*single* decision enumerating every legal combination — 504 of them for the
standard deal — so the model faces one clean, fixed-shape choice rather than a
sequence of interacting sub-picks.

**What an expert weighs.** This is a joint optimization: the value of a food
depends on which birds you kept (can you actually afford to play them early?),
and the value of a bonus card depends on both. You're balancing an affordable
opening curve, habitat spread, engine pieces (brown "when activated" powers),
and bonus-card alignment all at once. Bundling everything into one combined
choice is exactly right, because these pieces cannot be valued independently.

**Why its own head.** The opening has a unique cadence (once per game), a unique
shape (a fixed combinatorial menu, not a board operation), and no board to read
yet. It is the single cleanest candidate in the game for a fully separate
network later. Crucially, each candidate exposes *which specific birds* it keeps
(see §4 on card identity), so this head can learn genuine card-by-card opening
synergies — which is precisely the kind of question ("what makes a good opening
hand?") the whole project is trying to answer.

### 3.2 `MAIN_ACTION` + `PLAY_BIRD` — the strategic spine, in two steps

**What happens.** At the top of each turn you choose *which* of four actions to
take: Gain Food, Lay Eggs, Draw Cards, or Play a Bird. This is the
`MAIN_ACTION` decision, and it picks the action *type* only. Play-a-bird is
offered only when you actually have a legal play. If you choose to play a bird,
a *follow-up* `PLAY_BIRD` decision picks which bird and in which habitat — one
candidate per legal (bird, habitat) pair, offered only when the pair's costs
are completable. The costs themselves then resolve as further follow-ups, eggs
then food (§3.7 and §3.5).

**What an expert weighs.** Engine-building versus immediate points; the
action-reward track (more birds in a habitat row makes that action stronger, so
where you build matters beyond this turn); tempo and the shrinking cube budget
(8 → 7 → 6 → 5 actions across the rounds); the current round goal; the food/egg/
card economy; and denying the opponent.

**Why two heads.** "Which action is worth a cube this turn?" and "which bird is
worth playing, where?" are different questions, so they get different heads.
This is the well-known *action-type-then-arguments* factorization, and it
buys a clean, legible signal: the `MAIN_ACTION` head's scores read directly as
"how often is playing a bird worth more than an engine action?" The four
action-type options are intentionally featureless tokens — their value lives in
the *board state*, not in the option itself — while the (bird, habitat) detail
lives on the `PLAY_BIRD` candidates. The same `PLAY_BIRD` head also handles the
extra plays some powers grant, because "which bird is worth playing?" is the
same skill whether it's your main action or a bonus — though an extra play,
being optional, first passes through a `SKIP_OPTIONAL` accept (§3.8).

Both portions of a bird's cost are handled as follow-ups rather than folded
into the play candidate, resolving in the printed order: the egg cost via
`RemoveEggDecision` (§3.7), then the food payment via `PayBirdFoodDecision`
(§3.5). The strategic pick — *this bird is worth a cube, here* — is thereby
kept separate from the spend logistics of paying for it, and each cost trains
the generic judgment it actually exercises ("which egg / which tokens can I
most afford to lose?") alongside every other egg and food spend in the game.

### 3.3 `DRAW_BIRD` vs `DISCARD_BIRD` — valuing birds, in both directions

"How valuable is this bird?" is really two opposite skills, and they get two
heads.

**Acquisition** — *given birds I don't yet hold, which do I take?*
- `DrawCardsPickSourceDecision` — take a *named* face-up tray bird, or draw
  blind from the deck. (This one carries a value-of-information wrinkle the
  others don't: the deck option is unseen.)
- `BirdPowerPickBirdFromHandDecision` — keep one of several freshly drawn birds
  (the American Oystercatcher draft).

**Discard** — *given birds I hold, which do I give up?*
- `BirdPowerTuckFromHandDecision` — give a hand card up to tuck it behind a bird
  (it becomes a point plus progress toward tuck-count goals).
- `DiscardBirdForFoodDecision` — give a hand card up to gain one extra food (the
  Forest trade space, step 2 after committing via ``AcceptExchangeDecision``).

**What an expert weighs.** Points, cost versus current food, habitat fit and open
slots, power synergy with what's already on the board, bonus-card progress —
and, on the discard side, whether a card is worth more held than spent.

**Why split by direction.** A bird's features are encoded identically wherever it
appears, so each head specializes purely in *direction*. Acquisition and discard
pull in opposite directions on the very same features — a card you'd eagerly
draft is a card you'd be reluctant to toss — which is exactly why a single
"bird value" head would be the wrong grain. Note that choosing which hand bird
to *play* is not in this group; that lives in `PLAY_BIRD` (§3.2), because a play
is also a question of habitat and timing, not just card value.

### 3.4 `GAIN_FOOD` — acquiring food

**The skill.** "Which food advances my plans?" — choosing a die face from the
birdfeeder, or a token from the supply.

**What an expert weighs.** The food costs of the birds you intend to play; the
flexibility of wild food; the scarcity of a face still showing in the feeder;
the shape of your future curve.

**Why unified.** Gaining food is *one* judgment no matter what triggered it, so
every food gain in the game routes through this single family: the main Gain Food
action, the each-player feeder gains, and every bird power that hands you a food
(a specific die, any die, a predator's feeder grab, the fewest-forest gain, the
wild half of a discard-for-wild power, and so on). A skip option appears only
when the gain is genuinely optional. Unifying these means the model sees a large,
varied stream of "pick a food" situations and gets good at the skill, instead of
re-learning it separately for each card.

### 3.5 `SPEND_FOOD` — giving food up

**The skill.** The inverse of §3.4: "which food can I most afford to part with,
and which payment keeps me most flexible?"

- `PayBirdFoodDecision` — pay a committed bird play's printed cost, choosing
  among the legal payment multisets (1-for-1 matching, 2-for-1 substitution,
  wild fills). This is the dominant food-spending event in the game and the
  bulk of this head's data.
- `SpendFoodDecision` — hand a food back to the supply (e.g. the lose-half of a
  trade-a-wild power).
- `SpendFoodForEggDecision` — spend a food to lay one extra egg (the Grassland
  trade space, step 2 after committing via `AcceptExchangeDecision`).

**What an expert weighs.** Hold wild food for flexible future costs; don't strand
a bird you mean to play by spending the food it needs; weigh the marginal egg or
trade against what you give up.

**Why separate from gaining.** Gaining and spending food are opposite skills, so
they get opposite heads. A power like the wild-food trade is modeled as a
*chain* — gain a food (a `GAIN_FOOD` step), then give one back (a `SPEND_FOOD`
step) — so the two opposite judgments never share weights. The bird food cost
follows the same logic from the other side: *whether* the bird is worth playing
is the `PLAY_BIRD` pick, while *how to pay* is settled here afterwards, so the
payment judgment trains on every food spend in the game rather than being
locked inside the play candidates. When only one payment is legal there is
nothing to decide and the engine resolves it without consulting the policy.

### 3.6 `LAY_EGG` — where the egg goes

**The skill.** "Which of my birds gets this egg?" Used everywhere an egg is
*added*: the main Lay Eggs action, the Grassland conversion, lay-any-egg powers,
the all-players and lay-on-a-nest-type powers, and the pink between-turn
reactors. It is the most heavily reused placement skill in the game.

**What an expert weighs.** Round-goal nests and habitats; bonus cards keyed on
eggs; favoring high-capacity birds (more room for future eggs) unless a goal
rewards spreading; favoring birds you'll keep to game end; and how much capacity
each bird has left.

**Why one head.** The judgment really is the same across every trigger, so the
heavy reuse is correct — and concentrating all egg-placement experience into one
head is what lets it get good.

### 3.7 `PAY_EGG` — which egg to spend

**The skill.** "Which egg can I best afford to lose?" Used wherever an egg is
*removed*: paying a bird's egg cost, the Wetland egg-for-card conversion, and the
discard-egg-for-wild power.

This is a clean illustration of the whole design philosophy. The *which egg*
question is the same skill regardless of why you're spending the egg, so all
three contexts route to one head with one consistent feature shape.

And, importantly, **the reason for spending the egg is deliberately not shown to
this head — because it doesn't matter.** Whether to pay at all, and how many eggs
it costs, is settled *upstream* by a different decision (the `SKIP_OPTIONAL`
head for the trades, or the `PLAY_BIRD` pick for a bird's cost). By the time this
head runs, that commitment is already made; "which of my birds gives up the egg?"
is then orthogonal to why — you take it off your least-valuable spot either way.
So this head correctly sees only the egg-source options. Whether you're allowed
to decline varies by context (mandatory for a bird's cost, optional for some
trades), surfaced as the presence or absence of a skip option.

### 3.8 `SKIP_OPTIONAL` — is this optional exchange worth taking?

**The skill.** "Is taking this worth it, given my position and the round goal?"
— the yes/no half of any fully-determined optional offer, independent of *which*
resource gets used (that's a separate decision). This is the natural partner to
§3.7: the *decision to commit* is its own skill, separate from *which resource
to pay with*. (Formerly named `COMMIT_TO_COST`; renamed when the cost-free
extra-play accept joined the family — the common thread is skipping or taking
an optional offer, not necessarily paying.)

It handles the fully-determined offers: the Wetland egg-for-card conversion
(the bird the egg comes off is the separate `PAY_EGG` follow-up), the
discard-food-to-tuck powers (where the food and the number of tucks are fixed by
the card), and the power-granted extra bird play (accept opens the `PLAY_BIRD`
menu; skip forfeits the credit). The accept option carries the **terms of the
offer as typed fields** — food paid, eggs paid, cards gained, cards tucked, bird
plays unlocked — so the head can literally weigh what's gained against what's
paid, rather than scoring a blank "accept" token.

Two cases are intentionally *not* routed here, both for good reason:

- **When the yes/no is inseparable from a resource pick**, it stays with that
  pick. The Forest (card-for-food) and Grassland (food-for-egg) trade spaces fold
  "should I?" into the same decision as "which card/food?" via a skip option,
  because you decide whether to trade *as* you decide which resource to give up.
- **When there's no real yes/no**, there's no commit decision. Once you've
  chosen a bird to play, paying its egg and food costs is mandatory — so that
  "decision" is already part of the action pick (§3.2), and the follow-ups only
  choose *which* egg and *which* tokens.

### 3.9 `CHOOSE_BONUS` — which bonus card fits the plan

**The skill.** "Which bonus card matches the board I'm building?" — how many
qualifying birds you have or can still get, and whether a VP threshold is in
reach. This judgment appears in two homes: bundled into the opening keep
(§3.1) and, mid-game, the keep-one-of-several bonus power.

Each bonus card is identified to the network individually (§4), so the head can
learn a per-card value — a direct line to the project's "how valuable is each
bonus card?" question. A natural future enrichment is to *also* hand the head the
bonus card's structured terms (its category, its VP thresholds, your current
qualifying-bird count), so it can generalize across bonus cards rather than
learning each one in isolation. The per-card identity is the backbone that makes
that learnable.

### 3.10 `MISC_RARE` — the rare structural picks

Three decisions fire on only a handful of cards:

- `BirdPowerPickGainOrderDecision` — designate who gains first in an each-player
  feeder gain (the Anna's / Ruby-throated Hummingbird order pick).
- `BirdPowerPickPlayedBirdDecision` — choose which adjacent power to *repeat*.
- `BirdPowerPickHabitatDecision` — designate the destination row for a *moved*
  bird (the move-if-rightmost powers). Choosing a habitat for a two-habitat
  bird you're *playing* is part of the `PLAY_BIRD` candidate (§3.2), so this
  covers only the move powers. (This briefly had its own `MOVE_HABITAT` family;
  it fires far too rarely to feed a dedicated head and was folded in here.)

These are pooled into one shared head on purpose. A dedicated head for something
that fires a few times a game would be perpetually starved of training data;
pooling keeps it learning from a steady (if small) stream. Repeat-a-power is
genuinely a high-value judgment — "which power is best to copy?" — but it's far
too rare to isolate; a richer future treatment could score it *through* the head
of whatever power it copies.

### 3.11 `RESET_BIRDFEEDER` — is a fresh roll worth more?

**The skill.** Wingspan lets a player reroll the whole feeder before gaining
food whenever every die shows the same face. The judgment — "is a fresh roll
worth more than what's showing?" — is offered at every feeder gain, so it gets
its own small head rather than riding along with the food pick itself.

---

## 4. How the board and the cards are represented

The policy is only as good as what it can see. A few representation choices do a
lot of the work.

**Point of view.** The state is always encoded from the perspective of the player
who is about to decide — including when an opponent is prompted mid-power. So
"my food," "my board," "my eggs" always mean the decider's. This is what lets a
single network play both seats in self-play: symmetry comes for free from the
POV encoding plus per-player returns.

**Card identity, not just card stats.** Every bird-carrying option includes a
one-hot over all 180 core-set birds, concatenated with its numeric attributes
(points, costs, egg limit, wingspan, color, nest, per-food cost). Bonus cards get
their own one-hot over the 26 bonus cards. The deciding player's *hand* is
encoded as a multi-hot over birds, and the *opening keep* exposes its kept birds
as a multi-hot too. The first layer over these identity stripes is, in effect, a
**learned per-card embedding** — and that embedding *is* the card-power signal
the project ultimately wants to read out. (As built there are actually *two* such
tables — the hand / opening multi-hot is read by the state trunk while a
candidate's one-hot is read by the choice encoder — so a card has two learned
vectors today; unifying them into one shared `nn.Embedding` is the TRAINING.md
§6.3 refinement, and is what would make the readout a single per-card table.)
Two birds with identical printed stats
are still distinguishable, and the setup head can see exactly which cards an
opening keeps. (Opponent hands stay hidden, as they should — only the size is
revealed.)

**Trade terms, not blank tokens.** A yes/no exchange (§3.8) carries its terms as
features — a symmetric pay→gain ledger over cards, food, eggs, and bird plays
(what the player gives up and receives), plus the gains a shared-benefit power
grants the opponent — so the skip-optional head weighs the actual deal instead of
a featureless "accept" button.

**Uniform candidate features.** Every option is described in one shared feature
layout with type-specific stripes; a given option fills only the stripes that
apply to it. This is what lets the single per-choice encoder read any candidate
from any decision, and the per-family head specialize on top.

---

## 5. The critic, and how training closes the loop

There is exactly one value head, and it reads only the board (from the deciding
player's POV). That's deliberate: *how good is my position* is a fact about the
board, not about which question is being asked. So no matter how the policy heads
are organized, one shared critic is correct — and keeping it shared means it
trains on every step in the game rather than being split thin.

Training is self-play with a straightforward REINFORCE-with-baseline update.
Both seats are driven by the same network; each recorded decision is tagged with
the family head it used and the player who made it. At game's end, every step is
credited with that player's final score margin (so the two seats receive
opposite-signed signals in a decisive game), the shared critic provides the
baseline, and a small entropy bonus keeps exploration alive. The infrastructure
is intentionally minimal but scales cleanly toward stronger algorithms later.

---

## 6. Why this mirrors the game

Step back and the design is a fairly literal model of how the game is actually
played:

- A skilled player has a handful of **transferable judgments** — value a bird,
  value a food, place an egg, weigh a trade — that they apply across many
  different cards and situations. The model has exactly those judgments, one per
  head, each trained on every situation that exercises it.
- The same player **reads one board** and reuses that read for whatever they're
  deciding. The model shares one trunk and one critic for exactly that reason.
- Big, structurally distinct moments — **the opening** and **the choice of
  action each turn** — feel different from the moment-to-moment picks, and the
  design singles them out (their own heads now, candidates for their own networks
  later).
- The questions the project most wants to answer — *which cards are strong, which
  bonus cards are worth it, what makes a good opening* — map directly onto things
  the model is built to learn: per-card embeddings under the relevant heads, a
  setup head that sees specific cards, and a clean action-type head whose scores
  read as the value of playing birds versus building the engine.

The result is a network whose internal structure is a map of the game's decision
structure — which is what makes it not just a player, but something you can
interrogate.
