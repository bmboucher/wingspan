# DECISIONS.md — the decision-model families

A reference report on each judgment family the RL model trains a scoring head
for: where in the game engine each family's decisions arise, what the choice
vector carries when the model scores them, and how much the decisions inside a
family vary. It is written against the code as of June 2026 and is meant to be
kept in sync with it — see the final section, **Maintaining this document**,
for the update protocol.

Source-of-truth files this report describes:

| Topic | File |
|---|---|
| Decision / Choice classes, family mapping | `src/wingspan/decisions.py` |
| Choice-vector stripe layout & offsets | `src/wingspan/encode/layout.py` |
| Per-choice featurizers | `src/wingspan/encode/choice_encode.py` |
| Engine call sites | `src/wingspan/engine/` (core, actions, powers/, reactors) |
| Per-family heads | `src/wingspan/model.py` |
| Setup model | `src/wingspan/setup_model/`, `src/wingspan/training/setup_net.py` |
| Setup / bonus config axes | `src/wingspan/training/config.py` |

---

## 0. How a decision reaches a head

Every decision point in the engine is one `Engine.ask(agent, decision)` call.
The decision carries the deciding player, a prompt, and a non-empty list of
typed `Choice` objects. For the model the flow is:

1. `encode_state` builds the state vector from the **deciding player's point of
   view** (their food/board/hand are "mine" even when an opponent is prompted
   mid-power), ending in a one-hot over the concrete `Decision` class
   (`ALL_DECISION_CLASSES` order).
2. `encode_choices` builds one feature row per legal choice — a shared layout
   of type-specific stripes; each `Choice` subclass's featurizer fills only the
   stripes that apply to it (everything else stays zero).
3. The trunk reads the state into an M-wide context; the choice encoder reads
   each row into an N-wide embedding; the concatenation is scored by **the one
   head belonging to this decision's judgment family**
   (`decisions.family_index_for`, `ALL_DECISION_FAMILIES` order), and a softmax
   over the scores is the policy.

Two standing facts shape everything below:

- **Forced moves never reach a head.** `Engine.ask` auto-resolves any decision
  with exactly one legal choice, and the collector records only genuine forks.
  So every data point a head trains on has ≥ 2 real options.
- **The decision-type one-hot lives on the *state* vector, not the choice
  rows.** Inside a family that serves more than one decision class, that
  one-hot is the only signal telling the head *which* of its call-site shapes
  it is currently scoring.

One config axis changes the shapes: `EncodingSpec.include_setup`
(= `not TrainConfig.use_setup_model`). When the separate setup model owns the
opening (the default), the main net drops the `SETUP` head, the
`SetupDecision` decision-type column, and the trailing `setup_agg` choice
stripe. All three are kept *last* in their stable orders so the exclusion is a
clean truncation.

The family → decision-class map (head order = `ALL_DECISION_FAMILIES`):

| # | Family (head) | Decision class(es) routed to it |
|---|---|---|
| 1 | `MAIN_ACTION` | `MainActionDecision` |
| 2 | `DRAW_BIRD` | `DrawCardsPickSourceDecision`, `BirdPowerPickBirdFromHandDecision` |
| 3 | `DISCARD_BIRD` | `BirdPowerTuckFromHandDecision`, `DiscardBirdForFoodDecision` |
| 4 | `GAIN_FOOD` | `GainFoodDecision` |
| 5 | `SPEND_FOOD` | `SpendFoodDecision`, `SpendFoodForEggDecision`, `PayBirdFoodDecision` |
| 6 | `LAY_EGG` | `LayEggDecision` |
| 7 | `PAY_EGG` | `RemoveEggDecision` |
| 8 | `SKIP_OPTIONAL` | `AcceptExchangeDecision`, `ActivateTuckDecision` |
| 9 | `CHOOSE_BONUS` | `BirdPowerPickBonusCardDecision` |
| 10 | `MISC_RARE` | `BirdPowerPickHabitatDecision`, `BirdPowerPickPlayedBirdDecision`, `BirdPowerPickGainOrderDecision` |
| 11 | `PLAY_BIRD` | `PlayBirdDecision` |
| 12 | `RESET_BIRDFEEDER` | `ResetBirdfeederDecision` |
| 13 | `SETUP` (last; config-excluded) | `SetupDecision` |

---

## 1. The choice vector at a glance

One uniform row per candidate (`encode/layout.py`); base width 396, plus the
trailing 4-dim `setup_agg` stripe when `include_setup` (400). The stripes, in
offset order:

| Stripe | Width | Contents | Filled for |
|---|---|---|---|
| `kind` | 6 | one-hot: bird / food / habitat / payment / board_target / special | every row |
| `gain_food` | 7 | 5 plain food faces + choice-die-as-invertebrate + choice-die-as-seed | food picks (gains *and* single-token spends) |
| `habitat` | 3 | habitat one-hot | play-bird rows, habitat picks, payment context |
| `pay_food` | 5 | per-food payment counts (÷4) | payment multisets, named exchange costs, setup foods-spent |
| `board_target` | 120 | 15 slots × 8 scalars: lay-flag, pay-flag, cached food ×5, tucked | egg add/remove targets; played-bird picks (context, no flag) |
| `main_action` | 4 | one-hot over Gain Food / Lay Eggs / Draw Cards / Play Bird | main-action rows |
| `special` | 2 | `is_skip`, `is_self` | skip rows; player-id rows |
| `exchange` | 12 | pay→gain ledger (÷3): cards/food/eggs paid; food/eggs/cards/tucks/plays gained; 4 reserved opponent-gain terms | accept-exchange rows |
| `board_idx` | 15 | integer card index per board slot, embedded through the shared card table | wherever `board_target` is filled |
| `bird_id` | 180 | bird identity one-hot (multi-hot for setup keeps), embedded through the shared card table — so the candidate's static attributes *and* its learned per-card vector arrive together | every bird-carrying row |
| `bonus_id` | 26 | bonus-card identity one-hot | bonus picks, setup keeps |
| `bonus_delta` | 3 | how this bird advances the decider's **held** bonus cards: qualifying-card count + summed stepped-VP and linear-VP marginals at count+1 | bird keep/play/tray-draw rows |
| `goal_delta` | 8 | how this bird advances each of the 4 round goals: per goal slot, count delta + marginal placement-VP swing | bird keep/play/tray-draw rows |
| `bonus_value` | 5 | what this **offered bonus card** is worth to the decider: board qualifying count, the stepped and linear VP that count pays, and qualifying-bird counts in hand (kept subset at setup) and tray | bonus picks; setup keeps carrying a bonus |
| `setup_agg` | 4 | kept-subset aggregates: Σpoints, Σfood-cost, Σegg-limit, kept count | setup keeps only (`include_setup`) |

Worth remembering when reading the family sections: per-slot **egg counts and
remaining capacity are not in the choice rows** — they ride the state vector's
board stripes, as do the birdfeeder contents, the round goals, bonus progress,
and the hand. A choice row carries only what distinguishes *this* candidate.

---

## 2. The families

Sections mirror the head order above.

### 2.1 `MAIN_ACTION` — which action gets this turn's cube

**Where the engine asks it.** Exactly once at the top of every turn, before
anything else: Gain Food, Lay Eggs, and Draw Cards are always offered (even
when the row is empty and the action would be weak), and Play a Bird is added
only when the player has at least one completable `(bird, habitat)` play right
now. Choosing Play a Bird opens the follow-up `PLAY_BIRD` menu; the other
three run their habitat-row action directly.

**What the choice rows carry.** Almost nothing, deliberately: the special-kind
bit plus the 4-wide `main_action` one-hot. The four options are featureless
tokens — the value of "lay eggs this turn" is a fact about the *board*, so the
entire judgment is read from the state context (food, eggs-capacity, hand,
row counts, cubes left, round goal, opponent posture).

**Variation within the family.** Minimal — one decision class, one shape. The
only structural variation is menu width (3 vs 4 options, depending on whether
a bird play is legal). This is the most homogeneous family.

### 2.2 `DRAW_BIRD` — which bird to take

**Where the engine asks it.**

- *Every single-card draw* (`DrawCardsPickSourceDecision`): each card of the
  main Draw Cards action; the extra card from the Wetland trade-space
  conversion; every "draw a card / draw N cards" bird power, including the
  all-players-draw powers (each seat picks its own source) and the draw step
  of tuck-then-draw powers. Each draw is a separate decision over the current
  face-up tray plus the deck; tray slots emptied mid-turn stay empty until the
  end-of-turn refill, so consecutive draws see a shrinking tray.
- *`BirdPowerPickBirdFromHandDecision`*: currently has no active call sites
  in the engine (retained in `ALL_DECISION_CLASSES` for checkpoint
  compatibility — do not remove or reorder it).

(Brant's take-the-whole-tray and predator tuck-from-deck involve no pick and
never reach this head.)

**What the choice rows carry.** A tray card is a full bird-kind row: bird
identity (→ shared card table: points, costs, nest, habitats, wingspan, color,
bonus categories, learned vector) plus the `bonus_delta` stripe pricing what
acquiring it would do for the decider's held bonus cards and the `goal_delta`
stripe pricing its immediate effect on the four round goals. The **deck option
is a bare special-kind token** — no identity, no stats — which is exactly the
value-of-information shape: the head must weigh named cards against the
expected value of a blind draw.

**Variation within the family.** The blank deck row vs. fully-featured card
rows is the biggest intra-row contrast in any family. Only
`DrawCardsPickSourceDecision` is active; only it ever offers a deck row or a
skip. `BirdPowerPickBirdFromHandDecision` is retained but currently unused.

### 2.3 `DISCARD_BIRD` — which bird to give up

**Where the engine asks it.**

- *Tuck-from-hand powers* (`BirdPowerTuckFromHandDecision`): every "tuck a
  card from your hand behind this bird" power — the plain tuck, tuck-then-draw,
  tuck-then-lay-on-this, tuck-then-lay-any, and tuck-then-gain-food variants —
  one ask per tucked card. **Mandatory once reached**: the activate/skip
  judgment happens in the preceding `ActivateTuckDecision` (see §2.8).
- *The pink tuck reaction* (same class): Horned Lark's "when another player
  plays a bird in **their** [grassland], tuck a card" — fires between turns for
  the reacting player, gated by an `ActivateTuckDecision` (optional; the player
  may decline). If activated, the card selection is mandatory.
- *The Forest conversion discard* (`DiscardBirdForFoodDecision`): step 2 of
  the Forest trade space — which hand card to discard for the extra food die,
  after the trade was committed via a `SKIP_OPTIONAL` accept. Mandatory; the
  yes/no already happened upstream.
- *The Oystercatcher pass-and-return draft* (`BirdPowerDiscardFromHandDecision`):
  American Oystercatcher draws 3 cards into the active player's hand. The
  active player makes **2** mandatory discard decisions to pass cards to the
  opponent; the opponent then makes **1** mandatory discard decision to return
  a card to the active player. No skip on any of these three asks — the
  commitment happened upstream at the power's `AcceptExchangeDecision`.

**What the choice rows carry.** Bird-kind rows: identity (→ card table) plus
`bonus_delta` and `goal_delta`. Note the direction inversion: both stripes
price what the bird would contribute *if it reached the board* — for a
discard, that is the value being forfeited. No skip row ever appears: every
`DISCARD_BIRD` decision is mandatory once it is reached; the optionality lives
upstream in `SKIP_OPTIONAL` (`ActivateTuckDecision` for tucks,
`AcceptExchangeDecision` for the Forest conversion and the Oystercatcher draft).

**Variation within the family.** The card rows are identical in shape across
all sites; what differs is what the give-up buys — a tuck point, a draw, an
egg, a food, or information about which cards the opponent values
(Oystercatcher) — which is **not** in the rows at all. The head reads the
compensation only from the decision-type one-hot plus state context, which is
the main representational thinness of this family.

### 2.4 `GAIN_FOOD` — which food advances the plan

**Where the engine asks it.** The single most-exercised food judgment, unified
across every trigger:

- each die taken during the main Gain Food action;
- the extra die after the Forest conversion (commit and discard settled
  upstream);
- bird powers that pull a die of the player's choice from the feeder: "gain a
  die of your choice from the birdfeeder", and "gain [A] or [B] from the
  birdfeeder" (menu limited to whichever of the two named faces is showing);
- each-player feeder gains: the Anna's / Ruby-throated Hummingbird power
  (every seat, in the chosen order, picks its own die — so the opponent
  answers this too), and "the player(s) with the fewest forest birds gains a
  die" (auto-skipped entirely when activating would only feed the opponent);
- the pink predator reaction: when an opponent's predator hunt succeeds, the
  reacting player's pink bird pulls a die of their choice from the feeder;
- supply picks: choosing which wild food to take from the supply, e.g. the
  gain half of the discard-egg-for-wild powers;
- the gain half of Green Heron's wild-food trade — the one **optional** gain
  (a skip is offered; declining cancels the whole trade).

Powers that grant a *named* food (from supply or feeder) take it without a
decision and never reach this head.

**What the choice rows carry.** Food-kind rows filling the 7-slot `gain_food`
stripe: the five plain die faces, plus two dedicated slots for taking the
invertebrate/seed *choice die* as invertebrate or as seed. The choice-die
slots only light up at feeder gains where the combo face is showing — so the
head scores "burn the flexible die" separately from "spend a rigid single
face" (e.g. to deny the opponent the flexible die). Supply picks use the plain
slots only. What is *in* the feeder rides the state vector.

**Variation within the family.** One decision class, but two sources (feeder
vs. supply — distinguishable by whether choice-die slots can appear), one
optional site (Green Heron) against otherwise mandatory picks, and a decider
who is frequently the non-active player (each-player powers, pink reaction) —
made uniform by the POV state encoding.

### 2.5 `SPEND_FOOD` — which food to part with

**Where the engine asks it.**

- *The bird-play food payment* (`PayBirdFoodDecision`): after a play is
  committed and its egg cost paid, choose among the legal payment multisets
  for the bird's printed cost (1-for-1 matching, 2-for-1 substitution, wild
  fills). The dominant food-spending event and the bulk of this head's data;
  auto-resolved when only one payment is legal.
- *The Grassland conversion spend* (`SpendFoodForEggDecision`): step 2 of the
  Grassland trade space — which single food to spend for the extra egg, after
  the upstream commit.
- *The trade give-back* (`SpendFoodDecision`): the lose half of Green Heron's
  trade — which food goes back to the supply.

All three are mandatory; the yes/no, where one exists, lives upstream in
`SKIP_OPTIONAL`.

**What the choice rows carry.** Two genuinely different shapes:

- **Payment rows** (payment-kind): the candidate multiset's per-food counts on
  the `pay_food` stripe, *plus decision-level context shared by every row* —
  the committed bird's identity (→ card table) and the destination habitat —
  so the head sees what the tokens are buying, not just the tokens leaving.
- **Single-token rows** (food-kind): a one-hot on the `gain_food` stripe — the
  *same* stripe gains use; nothing marks the row as a spend except the
  decision-type one-hot and the head itself.

**Variation within the family.** The starkest structural split of any family:
payment rows and single-token rows populate disjoint stripes (`pay_food` +
bird + habitat vs. `gain_food`), and payment rows are the only place in the
game where identical context features ride along on every candidate. A head
serving this family is really learning two sub-skills — "which multiset
preserves my flexibility" and "which loose token do I miss least" — tied
together by the shared notion of food value.

### 2.6 `LAY_EGG` — which bird gets the egg

**Where the engine asks it.** Everywhere an egg is *added*, one ask per egg:

- each egg of the main Lay Eggs action (mandatory — the egg must go
  somewhere);
- the extra egg after the Grassland conversion;
- "lay an egg on any bird" powers (mandatory);
- "all players lay an egg on a [nest-type] bird" powers: every seat answers
  over its matching birds (mandatory); the active player's optional
  *additional* egg(s) carry a skip;
- the lay halves of tuck-then-lay powers: lay-on-this-bird (a single target
  plus skip — usually auto-resolved away once one side is forced... it is a
  genuine 2-option fork: target or skip) and lay-on-any (mandatory);
- the pink lay reaction: "when another player takes Lay Eggs, lay an egg on a
  [nest] bird" — optional, with skip, answered between turns by the reacting
  player.

**What the choice rows carry.** Board-target rows: the full 15-slot board
block from the decider's own board — per slot the cached-food counts (×5
foods) and tucked-card count, plus the parallel 15 card indices embedding
every occupant through the shared card table — with exactly one slot flagged
`lay_eggs = 1`. Candidates differ **only in which slot carries the flag**; the
rest of the block is identical context. Current egg counts and remaining
capacity per slot are read from the state vector's board stripes, not the row.

**Variation within the family.** One decision class, one row shape. Variation
is in (a) skip presence (mandatory main-action lays vs. optional pink /
additional lays), (b) the eligible-target filter (nest-type restrictions are
expressed purely by which slots appear as candidates — the restriction itself
is not a feature), and (c) the decider sometimes being the non-active player.

### 2.7 `PAY_EGG` — which egg to lose

**Where the engine asks it.** Everywhere an egg is *removed*:

- the play-bird egg cost — one decision per egg owed by the destination
  column (mandatory);
- the Wetland conversion's egg discard, after the upstream commit (mandatory);
- the discard-an-egg-from-**another**-bird-for-wild-food powers (optional,
  with skip; the power's own bird is excluded from the targets).

**What the choice rows carry.** Exactly the `LAY_EGG` data shape — the full
board block + card indices — but the targeted slot is flagged `pay_eggs = 1`
instead. The opposite-direction judgment ("more eggs here makes this target
*better* to tap" vs. "*worse* to lose") lives entirely in the head and the
flag.

**Variation within the family.** Mandatory (costs, where the commitment was
the upstream play/trade pick) vs. optional (the wild-food power). The *reason*
the egg is being spent is deliberately not encoded — by the time this head
runs, the commitment is settled, and "which egg do I miss least?" is the same
question regardless.

### 2.8 `SKIP_OPTIONAL` — is this fixed exchange worth taking?

**Where the engine asks it.** Every fully-determined, optional,
take-it-or-leave-it offer:

- the three player-mat trade spaces, whenever the action cube lands on a
  conversion column: Forest (discard 1 card → +1 die), Grassland (spend 1
  food → +1 egg), Wetland (discard 1 egg → +1 card). Each is the step-1
  commit (`AcceptExchangeDecision`); *which* card/food/egg is a follow-up in
  the matching family;
- the discard-1-[food]-to-tuck-N-cards-from-deck powers (Sandhill Crane et
  al.), whose terms are fixed by the card (`AcceptExchangeDecision`);
- each power-granted **extra bird play** (`AcceptExchangeDecision`): accept
  (opens the `PLAY_BIRD` menu; House Wren's grant restricts it to one habitat)
  or forfeit the credit;
- every **tuck-from-hand activation** (`ActivateTuckDecision`): does the
  player want to tuck a card from hand right now? Offered before every
  `BirdPowerTuckFromHandDecision` — both white/brown on-play tuck powers and
  Horned Lark's pink reaction ("when another player plays a bird in their
  [grassland], tuck a card"). Accepting leads to the `DISCARD_BIRD` card
  selection; declining skips the tuck entirely.

**What the choice rows carry.** Always exactly two rows. For
`AcceptExchangeDecision` the accept row is a special-kind token carrying the
**exchange ledger**: twelve normalized terms — cards/food/eggs to pay,
food/eggs/cards/tucks/plays to gain, plus four reserved opponent-gain terms
(currently always zero). When the paid food is a named type it also rides the
`pay_food` stripe. For `ActivateTuckDecision` the accept row is a special-kind
token with the `cards_to_tuck` count in the `EXCHANGE.cards_to_tuck` field —
so the head reads how many cards the player is committing to tuck. In both
cases the skip row is a special-kind token with `is_skip`.

**Variation within the family.** Structurally none — two rows every time.
All variation is in the ledger values (`AcceptExchangeDecision`) or the tuck
count (`ActivateTuckDecision`): the extra-play accept is the degenerate
all-gain point (`gained_play_count = 1`, nothing paid up front, the play's own
costs resolving downstream), while the conversions and tuck trades each light
different pay/gain cells.

### 2.9 `CHOOSE_BONUS` — which bonus card fits the plan

**Where the engine asks it.**

- the "draw N bonus cards, keep K" powers — one decision per kept card, over
  the shrinking drawn pile;
- under the `split_setup_bonus` regime (see §2.13): the **opening bonus
  pick** — asked immediately after the setup keep is applied, over the two
  dealt bonus cards, against a minimal start-of-round-1 snapshot (empty board,
  full cubes, real tray/feeder/goals). It routes through `Engine.ask` like any
  in-game decision, so it adds one on-policy `CHOOSE_BONUS` sample per net
  seat per game.

**What the choice rows carry.** A special-kind token, the 26-wide bonus
identity one-hot, and the 5-dim `bonus_value` stripe pricing the candidate
against the decider's position: the board's current qualifying count, the
stepped and linear VP the card pays at that count, and the qualifying-bird
counts still in hand and on the tray. At the opening pick (empty board) the
board trio is zero and the hand/tray potentials are the live signal. The
card's raw printed terms (thresholds, per-bird rate) still arrive only
through identity — the row carries the *computed standing value*, not the
formula.

**Variation within the family.** The mid-game power pick and the opening pick
share the decision class, so even the decision-type one-hot cannot tell them
apart — the distinguishing signal is the state itself (an empty round-1 board
vs. a developed one). Row shape is otherwise constant; no skip ever.

### 2.10 `MISC_RARE` — the pooled rare structural picks

**Where the engine asks it.** Three unrelated, individually-rare judgments
share this head so none starves:

- *which habitat to move a bird into*
  (`BirdPowerPickHabitatDecision`): the "if this bird is to the right of all
  other birds in its habitat, move it to another habitat" powers — staying put
  is offered as a choice, so declining the move is just picking the current
  habitat;
- *which bird's power to repeat* (`BirdPowerPickPlayedBirdDecision`): the
  repeat-a-brown-power and repeat-a-predator-power birds, choosing among the
  eligible neighbours in the activated row;
- *who gains food first* (`BirdPowerPickGainOrderDecision`): the Anna's /
  Ruby-throated Hummingbird "each player gains a die, starting with the player
  of your choice" order pick.

**What the choice rows carry.** Three disjoint shapes:

- habitat picks: the 3-wide habitat one-hot;
- played-bird picks: the candidate's bird identity (→ card table) plus the
  full board block *as context, with no target flag* — and since the board
  block is the decider's whole board, it is identical on every row; the rows
  differ only by candidate identity;
- gain-order picks: a special-kind token whose `is_self` flag marks the row
  that is the deciding player (going first is usually right, and this makes
  that learnable trivially).

**Variation within the family.** Maximal — by construction. The three classes
populate entirely different stripes, and the state's decision-type one-hot is
the only thing telling the head which sub-judgment it is scoring. This family
is the deliberate trade of specialization for data volume; "repeat which
power?" in particular is a high-value judgment scored through a deliberately
coarse shared head.

### 2.11 `PLAY_BIRD` — which bird, into which habitat

**Where the engine asks it.**

- after `MAIN_ACTION` resolves to Play a Bird: one `PlayBirdChoice` per legal
  `(bird, habitat)` pair the player can complete right now (legal = open
  slot + affordable egg cost + at least one legal food payment);
- for each power-granted **extra play**, after the player accepts it via the
  `SKIP_OPTIONAL` gate — the same menu, filtered to the granting power's
  habitat when one applies (House Wren).

The chosen play's costs are follow-ups in other families, eggs then food:
`PAY_EGG` per egg owed, then `SPEND_FOOD` for the payment multiset. The
strategic pick is kept clean of spend logistics.

**What the choice rows carry.** Bird-kind rows: the bird's identity (→ card
table), the destination habitat one-hot, the `bonus_delta` stripe pricing
the play's marginal contribution to the held bonus cards, and the
`goal_delta` stripe pricing its marginal count/VP swing on each of the four
round goals. No cost features — costs resolve downstream, and only
completable pairs are offered.

**Variation within the family.** One class, one shape. A bird playable in two
habitats produces two rows differing only in the habitat stripe — that is the
intra-decision texture this head specializes in. Notably, a main-action play
and an extra play are **completely indistinguishable** to the model: same
decision class (so the same decision-type one-hot), same row shape, and no
extra-play-credit scalar in the state vector. The model cannot currently
condition on "this is a bonus play"; whether that matters is an open
modelling question.

### 2.12 `RESET_BIRDFEEDER` — is a fresh roll worth more than what's showing?

**Where the engine asks it.** Offered immediately before a feeder gain
whenever every die in the feeder shows the same face (all one food, or all on
the invertebrate/seed choice face):

- before each die of the main Gain Food action, and before the Forest
  conversion's extra die;
- before the feeder-pulling powers: the named-food feeder gains, the
  either-of-two-foods picks, the any-die picks, the gain-all-of-a-food powers,
  each seat's turn in the each-player gains, and the fewest-forest gain.

The *empty*-feeder reroll is automatic and never a decision. One gap to be
aware of: the pink predator-success reaction currently pulls its die without
passing through the reset offer.

**What the choice rows carry.** Nearly nothing, by design: the affirmative
("reroll everything") is a bare special-kind token; the decline is the same
plus `is_skip`. The entire judgment — what is showing, how many dice remain,
what the player needs — is read from the state vector (the 6-dim feeder
stripe: five face counts plus the choice-die count) through the trunk.

**Variation within the family.** None structurally; the same two rows every
time. The contexts (main action vs. the various powers, active vs. reacting
seat) differ only through the state.

### 2.13 `SETUP` — the opening keep, and where the bonus pick lives

The opening (keep some of 5 dealt birds, retain the complementary foods, take
a bonus card) is scored by one of two owners, on the `use_setup_model` config
axis:

- **`use_setup_model = True` (default):** a separate value-regression bandit
  (`wingspan.setup_model` + `training.setup_net`) scores the enumerated
  candidates and the main net drops everything setup-shaped (head, decision
  column, `setup_agg` stripe). Its input vector is, in brief: a kept-cards
  multi-hot, a kept-foods multi-hot, and a kept-bonus one-hot, plus per-deal
  context (the three tray card indices, the six birdfeeder face counts, and
  the four round-goal one-hots) — with the card-identity blocks embedded
  through frozen copies of the main net's shared card encoders.
- **`use_setup_model = False`:** the main net keeps a `SETUP` head and scores
  the same candidates as ordinary choice rows (kept-bird identity multi-hot,
  foods-*spent* on the `pay_food` stripe, the `setup_agg` aggregates, and the
  kept bonus identity — plus, when the candidate carries a bonus, the
  `bonus_value` stripe with its hand potential counted over the **kept**
  subset, not the full dealt hand).

**Whether the bonus pick is folded in is itself a config choice.**
`TrainConfig.split_setup_bonus` (effective only alongside the setup model —
the gate is `split_setup_bonus_active = split_setup_bonus and
use_setup_model`):

- **Off (folded, the default):** the bonus is one axis of the combined keep —
  candidates are every (kept-cards × kept-foods × dealt-bonus) combination,
  504 for the standard 5-card / 2-bonus deal — so whichever model owns setup
  implicitly owns the opening-bonus judgment too, jointly with the cards and
  food (which is the argument *for* folding: the three cannot be valued
  independently).
- **On (split):** the candidate set drops the bonus axis (every candidate
  carries `bonus_card = None`; 252 keeps; the setup encoder's bonus block
  stays all-zero) and the opening bonus is instead asked as a normal in-game
  `CHOOSE_BONUS` decision right after the keep is applied (§2.9). The
  argument *for* splitting: the bonus judgment then trains on the in-game
  head with on-policy credit, concentrating all bonus-valuation experience in
  one place. This knob is shape-preserving (REGIME, resumable), whereas
  `use_setup_model` changes tensor shapes (FRESH).

---

## 3. Maintaining this document

*Instructions for Claude (or any maintainer): this file is a report on live
code and goes stale silently. Update it in the same change as the modelling
edit it describes.*

**When to update — and what to touch:**

| Change in the code | Sections to update here |
|---|---|
| New `Decision` / `Choice` subclass, or a change to `_DECISION_FAMILY` in `decisions.py` | the §0 mapping table + the affected family section(s); a new *family* gets a new section appended in `ALL_DECISION_FAMILIES` order (keep `SETUP` last) |
| New or changed stripe / offset / scale in `encode/layout.py` | the §1 stripe table (widths, totals) + each family section whose rows use the stripe |
| A featurizer in `encode/choice_encode.py` fills different stripes (or fills them differently) | the "What the choice rows carry" paragraph of every family that uses that `Choice` class |
| A new engine call site asks an existing decision (new power handler, new reactor, new conversion) | the "Where the engine asks it" list of the matching family |
| A change to `EncodingSpec`, `use_setup_model`, `split_setup_bonus`, or the setup candidate enumeration | §0 (the config-axis paragraph) and §2.13 |
| New state-vector signal that materially changes what a head can see (e.g. an extra-play scalar) | the family section(s) that called out the gap — §2.11 currently documents the extra-play blind spot |

**How to verify rather than trust memory:**

- Enumerate a decision class's call sites with
  `grep -rn "decisions.<ClassName>(" src/wingspan/` — every constructor call
  is an ask site (plus `Engine.ask`'s single-choice auto-resolve caveat).
- Re-derive the §1 table from the `_OFF_*` chain and `_*_DIM` constants in
  `encode/layout.py`; re-derive the row contents from the `_featurize_*`
  functions and `_CHOICE_FEATURIZERS` in `encode/choice_encode.py`.
- Re-derive the §0 mapping table from `_DECISION_FAMILY` and
  `ALL_DECISION_FAMILIES` in `decisions.py`. `tests/test_decision_families.py`
  pins the invariants.

**Conventions to preserve:**

- Family sections stay in `ALL_DECISION_FAMILIES` order (append-only, `SETUP`
  last) — mirroring the checkpoint-stable head order makes drift easy to spot.
- Call-site lists use natural language ("after playing House Wren", "when the
  cube lands on a trade column"), naming specific birds where the code does.
- Keep noting what is *not* in the choice rows when it is the judgment's key
  signal (feeder contents, egg counts, the discard's compensation) — those
  absences are modelling decisions and future-work candidates.
- Code and docs elsewhere cite this file by section number (`decisions.py`,
  `model.py`, `engine/core.py`, `TRAINING.md`, tests). If you renumber or
  retitle sections, grep for `DECISIONS.md` across the repo and refresh the
  citations in the same change.
