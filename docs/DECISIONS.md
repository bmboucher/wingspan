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
| Per-family heads | `src/wingspan/model/core.py` |
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
| 3 | `DISCARD_BIRD` | `BirdPowerTuckFromHandDecision`, `BirdPowerDiscardFromHandDecision`, `DiscardBirdForFoodDecision` |
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

One uniform row per candidate (`encode/layout.py`); base width **509**, plus the
trailing 4-dim `setup_agg` and 180-dim `kept_multihot` stripes when
`include_setup` (**693**). The live stripes, in offset order:

| Stripe | Width | Contents | Filled for |
|---|---|---|---|
| `kind` | 6 | one-hot: bird / food / habitat / payment / board_target / special | every row |
| `gain_food` | 7 | 5 plain food faces + choice-die-as-invertebrate + choice-die-as-seed | food-type identifier: die gains, supply gains, and single-token spends — the *type* of the token, not the direction of flow. A **one-hot** for a single `FoodChoice`; a **count vector** for a combined `FoodSubsetChoice` (the `combine_gain_food` regime), filled by `_fill_gain_food_vector` — same 7 slots, raw counts, so a single-unit subset is byte-identical to the one-hot |
| `pay_food` | 5 | per-food payment counts (÷4) | payment multisets, named exchange costs, setup foods-spent |
| `board_target` | 60 | 15 slots × 4 scalars: lay-flag, pay-flag, cached-total (÷max), tucked | egg add/remove targets; played-bird picks (context, no flag) |
| `main_action` | 4 | one-hot over Gain Food / Lay Eggs / Draw Cards / Play Bird | main-action rows |
| `special` | 2 | `is_skip`, `is_self` | skip rows; player-id rows |
| `exchange` | 13 | pay→gain ledger (÷3): cards/food/eggs paid; food/eggs/cards/tucks/plays/cache gained; opponent-gain terms: `opp_food`, `opp_egg`, `opp_card`, `opp_tuck` — what a shared-benefit power additionally grants the opponent | accept-exchange rows |
| `board_hab` | 3 | habitat one-hot for the single slot relevant to this choice (landing slot on placement rows; target slot on board-target rows; the candidate's current slot on played-bird rows) | wherever `board_target` is filled; play-bird rows; payment context; move-habitat rows; played-bird rows |
| `board_col` | 5 | column one-hot for the same slot as `board_hab` | same as `board_hab` |
| `bird_id` | 1 | single integer index column (`bird_index + 1`, 0 = no bird), embedded through the shared card table. **Candidate identity** on placement/food/main-action/draw rows. **Board-target occupant** on `BoardTargetChoice` rows (the bird being laid on / removed from), embedded to bring that bird's attributes into the row | every bird-carrying row; board-target rows (occupant) |
| `bonus_id` | 26 | bonus-card identity one-hot | bonus picks, setup keeps |
| `bonus_delta` | 3 | how this choice moves the decider's **held** bonus cards (static categories *and* the dynamic egg / hand-size / habitat-spread cards): affected-card count + summed stepped-VP and linear-VP marginals (signed) | bird keep/play/tray-draw rows; egg lay/remove targets; move-habitat rows; draw-source deck row, accept rows and the DRAW_CARDS main action (net hand change) |
| `goal_delta` | 8 | how this choice moves each of the 4 round goals: per goal slot, count delta + marginal placement-VP swing (signed; **zero once that round is scored** — payouts freeze) | bird keep/play/tray-draw rows; egg lay/remove targets; move-habitat rows; lay/remove commitment rows (accept trades, LAY_EGGS main action — capacity-capped optimistic bound) |
| `bonus_value` | 5 | what this **offered bonus card** is worth to the decider: board qualifying count, the stepped and linear VP that count pays, and qualifying-bird counts in hand (kept subset at setup) and tray | bonus picks; setup keeps carrying a bonus |
| `becomes_playable` | 180 | hand birds that transition from not-playable to playable as a direct result of the food or eggs this choice grants. **Food-gain path (v0.8+):** baseline is `playable_now ∪ playable_if_eggs`; `_bird_playable` is called with `ignore_eggs=True` — open slot + food-affordable is enough, egg cost is not checked. **Egg-gain path:** unchanged — baseline is `playable_now`, full `_bird_playable`. Exact on `FoodChoice` (`GainFoodDecision`); optimistic best-case on `PayCostChoice` skip_optional and on `GAIN_FOOD`/`LAY_EGGS` `MainActionChoice` rows. Zero on `BoardTargetChoice` (`LayEggDecision`) and all non-gain rows. Embedded through the shared card table (same as `hand_multihot`). | gain-bearing rows: food picks, accept-exchange rows with `gained_food_count > 0` or `gained_egg_count > 0`, `GAIN_FOOD` and `LAY_EGGS` main-action rows |
| `becomes_unplayable` | 180 | currently-playable hand birds that lose playability as a direct result of the food, eggs, or board slot this choice spends. Symmetric counterpart to `becomes_playable`; uses `playable_now` as the baseline (birds fully playable before the choice). **Optimistic** for under-specified removals: a bird survives iff at least one way to remove the tokens still leaves it food-affordable. Exact on `FoodPaymentChoice` (payment multiset known) and `FoodChoice` (`SpendFoodDecision`/`SpendFoodForEggDecision`, −1 token). **Optimistic** on `PayCostChoice` with `paid_food_count > 0` (type unknown). **Full-play** on `PlayBirdChoice` (−1 slot in played habitat, −`next_egg_cost` eggs, −food payment — optimistic over payment alternatives). **Egg-loss** on `BoardTargetChoice` (`RemoveEggDecision`, `is_pay`): −1 egg to `total_eggs`. Zero on `MainActionChoice`, `SetupChoice`, gain-only rows, and every row type not listed. Never populated for the bird being played/paid-for (excluded from its own row's baseline). Embedded through the shared card table (one `card_embed_dim` embedding, summed). Added in **v1.1**; v1.0 artifacts lack this stripe — the `wingspan.compat.v1_0` shim strips it. | spend-bearing rows: `PlayBirdChoice`, `FoodPaymentChoice`, `FoodChoice` (spend context), `BoardTargetChoice` (`RemoveEggDecision`), `PayCostChoice` with food or egg payment |
| `resets_feeder` | 1 | 1 if this combined food-gain option rerolls the birdfeeder — a partial take (fewer than `n` dice → committed reset + re-pick) or a full take that empties the feeder — so the model can tell a smaller-but-rerolls gain apart from a plain smaller gain (the `gain_food` count vector alone cannot). The last *base* stripe (after `becomes_unplayable`, before the trailing setup stripes). Added in **v1.4**; v1.0–1.3 artifacts lack it — the `wingspan.compat.v1_3` shim strips it, and `v1_0` inherits that strip. | `combine_gain_food` `FoodSubsetChoice` rows only |
| `setup_agg` | 4 | kept-subset aggregates: Σpoints, Σfood-cost, Σegg-limit, kept count | setup keeps only (`include_setup`) |
| `kept_multihot` | 180 | multi-hot of the specific kept birds, summed through the shared card table (the kept set is unordered — the single-candidate `bird_id` column stays zero on setup rows) | setup keeps only (`include_setup`) |

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

**What the choice rows carry.** The special-kind bit, the 4-wide
`main_action` one-hot, and consequence pricing on the two options whose
commitment has a determinate resource effect: the **Draw Cards** row carries
the `bonus_delta` of growing the hand by the wetland track count (what the
hand-counting bonus card pays on), and the **Lay Eggs** row carries a
`goal_delta` *bound* — per unscored goal, the capacity-capped best case the
grassland track's eggs could realize (the exact per-target deltas land on the
follow-up `LAY_EGG` rows; for the `birds_no_eggs` anti-goal the best case is
the forced overflow past the already-egged birds' spare room, a non-positive
bound). Gain Food and Play a Bird stay featureless tokens —
their value is a fact about the *board*, read from the state context (food,
eggs-capacity, hand, row counts, cubes left, round goal, opponent posture).

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
acquiring it would do for the decider's held bonus cards (including the
hand-counting dynamic card — any draw is +1 hand) and the `goal_delta`
stripe pricing its immediate effect on the unscored round goals. The **deck
option stays identity-free** — no card, no stats, the value-of-information
shape: the head must weigh named cards against the expected value of a blind
draw — but it carries the same +1-hand `bonus_delta` term the tray rows do,
since a draw from any source grows the hand equally (leaving it off only the
deck row would distort the within-decision comparison).

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
- *End-of-turn discard obligations* (`BirdPowerDiscardFromHandDecision`):
  "Draw N [card]. If you do, discard 1 [card] from your hand at the end of
  your turn." Eight birds (Black Tern, Clark's Grebe, Forster's Tern, Common
  Yellowthroat, Pied-Billed Grebe, Red-Breasted Merganser, Ruddy Duck, Wood
  Duck) accumulate one discard obligation via `turn_end_discards` on each
  activation; the Engine resolves all obligations at turn end, after extra
  plays, in order. Mandatory; the draw is unconditional (no opt-out).

**What the choice rows carry.** Bird-kind rows: identity (→ card table) plus
`bonus_delta` and `goal_delta`. Note the direction inversion: both stripes
price what the bird would contribute *if it reached the board* — for a
discard, that is the value being forfeited (the hand-counting dynamic card's
+1-hand term rides every bird row under the same convention; the
decision-type one-hot carries the direction). No skip row ever appears: every
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
  answers this too; the order pick itself is only asked when the feeder shows
  exactly 2 distinct faces — with >2 or ≤1 faces the active player auto-starts),
  and "the player(s) with the fewest forest birds gains a die" (auto-skipped
  entirely when activating would only feed the opponent);
- the pink predator reaction: when an opponent's predator hunt succeeds, the
  reacting player's pink bird pulls a die of their choice from the feeder;
- supply picks: choosing which wild food to take from the supply, e.g. the
  gain half of the discard-egg-for-wild powers and step 3 of Green Heron's
  wild-food trade (mandatory — activation gate is upstream in SKIP_OPTIONAL);
- Pygmy Nuthatch's tuck reward: "gain 1 [invertebrate] or [seed] from the
  supply" — a mandatory two-option supply pick offered after the tuck is
  accepted (menu limited to whichever of the two named foods is available);
- under the `split_setup_food` regime (see §2.13): the **opening food gain**
  — asked immediately after the setup keep is applied, against a
  start-of-round-1 snapshot (empty board, full cubes, real tray/feeder/goals).
  2 asks for 3 birds kept (no-repeat menu shrinks from 5 to 4), 1 ask for 4
  birds kept (full 5-item menu). Players start with 0 food in this branch;
  these decisions are mandatory and build the opening stock one token at a time.

Powers that grant a *named* food (from supply or feeder) take it without a
decision and never reach this head.

**What the choice rows carry.** Food-kind rows filling the 7-slot `gain_food`
stripe: the five plain die faces, plus two dedicated slots for taking the
invertebrate/seed *choice die* as invertebrate or as seed. The choice-die
slots only light up at feeder gains where the combo face is showing — so the
head scores "burn the flexible die" separately from "spend a rigid single
face" (e.g. to deny the opponent the flexible die). Supply picks use the plain
slots only. What is *in* the feeder rides the state vector.

**The `combine_gain_food` regime (config; default off).** When on, a run of
single-food gains is collapsed into one `GainFoodDecision` whose options are
multi-food *subsets* (`decisions.FoodSubsetChoice`): the same decision class,
decision-type one-hot, and GAIN_FOOD head, but the `gain_food` stripe carries a
**count vector** (raw counts per slot) rather than a one-hot, and
`becomes_playable` reflects the *whole* combined gain (via
`playability.newly_playable_after_foods`) — so a bird needing two foods together
lights up. It collapses three call sites: the Forest base-dice gain
(`actions.combined_feeder_gain`, with the feeder reset / reroll folded in —
partial subset → committed reroll → recurse; `n == 1` delegates to the single-die
path so it stays byte-identical), the ravens' two-wild supply gain, and the
`split_setup_food` opening keep (`actions.combined_supply_gain`). Each feeder
subset that will reroll — a partial take, or a full take that empties the feeder —
is flagged via `FoodSubsetChoice.resets_birdfeeder` and lights the dedicated 1-dim
`resets_feeder` choice stripe, so the model reads the fresh re-pick rather than
only a lower food count.

Toggling the regime is itself shape-preserving and config-carried (REGIME): it
lives on `EngineConfig.combine_gain_food`, stays out of `architecture_key`, and the
`resets_feeder` stripe is always present regardless of the toggle. Adding that
stripe, however, *was* a FRESH change — the v1.4 `MODEL_VERSION` bump with the
`wingspan.compat.v1_3` shim. The Forest *conversion* extra die and the each-player
hummingbird gain stay on the single-die path. See `docs/VERSIONING.md`.

**Variation within the family.** One decision class, but two sources (feeder
vs. supply — distinguishable by whether choice-die slots can appear), a decider
who is frequently the non-active player (each-player powers, pink reaction) —
made uniform by the POV state encoding — and, under `combine_gain_food`, two
option shapes (single `FoodChoice` vs. multi-food `FoodSubsetChoice`). All picks
at this head are mandatory; the yes/no for optional effects resolves upstream in
SKIP_OPTIONAL.

**`becomes_playable` semantics on food-gain rows (v0.8+).** The baseline
excludes `playable_now ∪ playable_if_eggs` (birds already food-affordable with
an open slot, regardless of eggs). The check itself uses `ignore_eggs=True` —
a bird lights up when gaining the offered food meets its food cost AND an open
slot exists; the egg cost is not considered. This means a forest-only bird
costing [seed] lights up on a [seed] gain and stays dark on a [fish] gain, even
when the player has no eggs for the forest slot. The egg-gain path (`LAY_EGGS`,
egg-bearing exchanges) keeps the original full `_bird_playable` predicate.

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
- *The setup food discard* (`SpendFoodDecision`): under the `split_setup_food`
  regime (see §2.13) — asked immediately after the setup keep is applied,
  against a start-of-round-1 snapshot. Players start with 5 food in this
  branch; 1 ask for 1 bird kept (discard 1), 2 asks for 2 birds kept (discard
  2 — no-repeat menu shrinks from 5 to 4 after the first). Mandatory.

All four are mandatory; the yes/no, where one exists, lives upstream in
`SKIP_OPTIONAL`.

**What the choice rows carry.** Two genuinely different shapes, visible in the
`kind` one-hot:

- **Payment rows** (`payment`-kind): the candidate multiset's per-food counts on
  the `pay_food` stripe, *plus decision-level context shared by every row* —
  the committed bird's identity in `bird_id` (→ card table) and its landing
  slot marked by `board_hab` + `board_col` (the payment is asked before the
  bird is placed, so the chosen habitat's next free slot is where it will
  land) — so the head sees what the tokens are buying, not just the tokens
  leaving.
- **Single-token rows** (`food`-kind): a one-hot on the `gain_food` stripe —
  the same stripe used for food gains, because `gain_food` is a food-type
  identifier, not a "gains" stripe. For a spend decision, the hot slot marks
  *which token is being given up*; the direction (spend vs. gain) is encoded
  by the decision-type one-hot, not by the row itself.

**Variation within the family.** Two row shapes inside one head: payment-kind
rows (multisets on `pay_food` + committed bird + landing slot) and food-kind
rows (a single food type on `gain_food`, no bird/slot context). The model is
effectively learning two sub-skills — "which multiset preserves my
flexibility?" and "which loose token do I value least?" — tied together by
the shared notion of food value. Payment rows are the only place in the game
where decision-level context (the committed bird's identity and landing slot)
rides identically on every candidate row.

### 2.6 `LAY_EGG` — which bird gets the egg

**Where the engine asks it.** Everywhere an egg is *added*, one ask per egg:

- each egg of the main Lay Eggs action (mandatory — the egg must go
  somewhere);
- the extra egg after the Grassland conversion;
- "lay an egg on any bird" powers — conditionally optional: when the active
  round goal is `birds_no_eggs` (rewards birds-without-eggs), an
  `AcceptExchangeDecision` precedes each lay so the player can skip rather
  than reduce their no-egg count; otherwise mandatory;
- "all players lay an egg on a [nest-type] bird" powers — four-step sequence:
  (1) active player P0 gets `AcceptExchangeDecision` to veto the whole power
  (exchange ledger carries `gained_egg_count` for P0 and `opp_gained_egg_count`
  for eligible non-active players); if P0 skips, the power does nothing for
  anyone; (2) non-active players, in turn order: if `birds_no_eggs` goal is
  active they each get their own `AcceptExchangeDecision` (otherwise auto-yes);
  eligible accepting players then answer `LayEggDecision`; (3) P0's mandatory
  `LayEggDecision` for their base egg; (4) P0's optional additional egg(s) from
  the power's second sentence, one `LayEggDecision` with skip each;
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
`lay_eggs = 1`. The flagged target also prices its consequences: the
`goal_delta` stripe carries the exact per-goal count/VP swing of this egg
(habitat totals, nest totals, the has-eggs crossing — inverted for the
`birds_no_eggs` anti-goal — the egg-set minimum; only on unscored goals), and
the `bonus_delta` stripe the egg-counting dynamic
bonus thresholds it crosses (Oologist at the first egg, Breeding Manager at
the fourth). Candidates otherwise differ **only in which slot carries the
flag**; the rest of the block is identical context. Current egg counts and
remaining capacity per slot are read from the state vector's board stripes,
not the row.

**Variation within the family.** One decision class, one row shape. Variation
is in (a) skip presence — mandatory main-action lays, conditionally optional
"lay any bird" powers (skip only when `birds_no_eggs` goal is active), and
always-optional pink / additional lays; (b) the eligible-target filter
(nest-type restrictions expressed purely by which slots appear as candidates
— the restriction itself is not a feature; star nests count as every nest);
and (c) the decider sometimes being the non-active player.

### 2.7 `PAY_EGG` — which egg to lose

**Where the engine asks it.** Everywhere an egg is *removed*:

- the play-bird egg cost — one decision per egg owed by the destination
  column (mandatory);
- the Wetland conversion's egg discard, after the upstream commit (mandatory);
- the discard-an-egg-from-**another**-bird-for-wild-food powers — mandatory
  (the power's own bird is excluded from targets); the yes/no trade decision
  lives upstream in an `AcceptExchangeDecision` routed to `SKIP_OPTIONAL`; by
  the time this head runs, the commitment is settled and the question is only
  which egg.

**What the choice rows carry.** Exactly the `LAY_EGG` data shape — the full
board block + card indices — but the targeted slot is flagged `pay_eggs = 1`
instead, and its `goal_delta` / `bonus_delta` stripes price the *removal*
(signed: losing the egg that breaks an egg-set or drops a bird below an
egg-counting bonus threshold shows up as a negative VP swing). The
opposite-direction judgment ("more eggs here makes this target *better* to
tap" vs. "*worse* to lose") lives in the head, the flag, and the signed
deltas.

**Variation within the family.** All call sites are mandatory — the upstream
decision (the bird play, the trade commit, or the `AcceptExchangeDecision` for
the wild-food power) already settled the commitment. The *reason* the egg is
being spent is deliberately not encoded: "which egg do I miss least?" is the
same question regardless of what the egg is buying.

### 2.8 `SKIP_OPTIONAL` — should I do this optional thing, or skip it?

**Where the engine asks it.** Every optional action where the player must
decide "should I commit to this, given these costs and benefits?" The
downstream details — *which* food to spend, *which* egg to give up, *which*
bird to play — are not yet resolved; the head's job is the commit/skip, and
follow-up decisions resolve only if the player commits.

- the three player-mat trade spaces, whenever the action cube lands on a
  conversion column: Forest (discard 1 card → +1 die), Grassland (spend 1
  food → +1 egg), Wetland (discard 1 egg → +1 card). Each is the step-1
  commit (`AcceptExchangeDecision`); *which* card/food/egg is a follow-up in
  the matching family;
- the discard-1-[food]-to-tuck-N-cards-from-deck powers (Sandhill Crane et
  al.), whose terms are fixed by the card (`AcceptExchangeDecision`);
- Green Heron's wild-food trade (discard 1 food → gain 1 food from supply):
  step-1 commit; *which* food to discard (SPEND_FOOD) and *which* to gain
  (GAIN_FOOD) are follow-up mandatory steps;
- each power-granted **extra bird play** (`AcceptExchangeDecision`): accept
  (opens the `PLAY_BIRD` menu; House Wren's grant restricts it to one habitat)
  or forfeit the credit;
- every **tuck-from-hand activation** (`ActivateTuckDecision`): does the
  player want to tuck a card from hand right now? Offered before every
  `BirdPowerTuckFromHandDecision` — both white/brown on-play tuck powers and
  Horned Lark's pink reaction ("when another player plays a bird in their
  [grassland], tuck a card"). Accepting leads to the `DISCARD_BIRD` card
  selection; declining skips the tuck entirely;
- the **discard-1-egg-for-N-wild-food** powers (`AcceptExchangeDecision`):
  accept commits to the trade; *which* egg is the follow-up `PAY_EGG`
  decision (mandatory once committed);
- the **"all players lay an egg on a [nest-type] bird"** powers: P0's
  activation veto (`AcceptExchangeDecision`) — accepting fires the power for
  everyone; skipping cancels it entirely; the exchange ledger includes
  `opp_gained_egg_count` for eligible opponents;
- (conditional on `birds_no_eggs` round goal) each non-active player's per-seat
  accept/skip for the above "all players lay" power (`AcceptExchangeDecision`);
- (conditional on `birds_no_eggs` round goal) each "lay an egg on any bird"
  power, per egg (`AcceptExchangeDecision`) — offered only when at least one
  owned bird has egg room (a full board skips the power without asking);
- the **cache-vs-keep decision** on the six seed-from-feeder birds (Acorn
  Woodpecker, Blue Jay, Clark's Nutcracker, Red-Bellied / Red-Headed
  Woodpecker, Steller's Jay): after the seed is taken from the birdfeeder,
  `AcceptExchangeDecision` offers "cache it on this bird" (accept: the token
  moves from player supply to `pb.cached_food`; ledger: `paid_food=seed,
  paid_food_count=1, gained_cache_count=1`) or "keep it in supply" (skip).

**What the choice rows carry.** Always exactly two rows. For
`AcceptExchangeDecision` the accept row is a special-kind token carrying the
**exchange ledger**: thirteen normalized terms — cards/food/eggs to pay;
food/eggs/cards/tucks/plays/cache to gain; `opp_food`, `opp_egg`, `opp_card`,
`opp_tuck` — what a shared-benefit power additionally grants the opponent
(the Oystercatcher draft sets `opp_gained_card_count`, the all-players-lay
powers `opp_gained_egg_count`). The `gained_cache_count` slot (slot 12, added
in parse-gaps Stage A) carries the seed-caching trade's value — the head can
weigh caching against keeping the token spendable. The head does not pre-judge
which terms are costs and which are benefits; it learns the sign from context.
The terms distinguish quantitatively different trades (e.g. "1 egg for 1 wild"
vs. "1 egg for 2 wild" — both exist). The ledger is also translated into
consequences on the same row: a net hand-card flow (cards drawn minus cards
discarded) fills the `bonus_delta` stripe for the hand-counting bonus card,
and a committed egg gain / egg payment fills the `goal_delta` stripe with the
capacity-capped optimistic bound (the exact delta lands on the follow-up
`LAY_EGG` / `PAY_EGG` target row). When the paid food is a named type it also
rides the `pay_food` stripe. For `ActivateTuckDecision` the accept row is a
special-kind token with the `cards_to_tuck` count in the
`EXCHANGE.cards_to_tuck` field — so the head reads how many cards the player
is committing to tuck. In both cases the skip row is a special-kind token
with `is_skip`.

**Variation within the family.** Structurally minimal — two rows every time.
All variation is in the ledger values. The extra-play accept is the degenerate
all-gain case (`gained_play_count = 1`, nothing paid). The trade-space commits
and tuck trades each light different pay/gain cells. The shared-benefit powers
also populate `opp_gained_egg_count`. The conditional call sites (only when
`birds_no_eggs` goal is active) appear rarely, but when they do the full
ledger context is available to the head.

**Strictly-free exchange rule.** `offer_exchange_or_auto_accept` in
`engine/powers/dispatch.py` skips the agent entirely when the accept row's
`PayCostChoice` ledger is strictly free: zero payment (no `paid_food`,
`paid_food_count`, `paid_card_count`, or `paid_egg_count`) and zero opponent
gain (all `opp_gained_*` fields zero) and at least one positive own gain.
These exchanges are auto-applied and logged with
`auto-accept: <label> (no cost)`. Only exchanges with real tradeoffs — any
payment or any opponent gain — reach the model via `offer_activation_veto`.
Note: the `birds_no_eggs` anti-goal gates (egg-laying when the round goal
rewards empty birds) use `offer_activation_veto` directly; although the ledger
shows only a gain, the round-goal opportunity cost is implicit and the veto is
meaningful.

**Veto label transparency.** When a veto IS offered with an opponent benefit,
the accept choice label must name the benefit so the SKIP_OPTIONAL head can
weigh it. Example: `ROLL_NOT_IN_FEEDER_CACHE` uses
`"roll dice (opponent may gain food x{n_feeders})"` when opposing
`PINK_PREDATOR_FEEDER` birds are present.

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
against the decider's position: the current qualifying count (live game state
for the four dynamic cards — eggs on birds, hand size, habitat spread — and
board tags otherwise), the stepped and linear VP the card pays at that count,
and the qualifying-bird counts still in hand and on the tray (every hand card
counts for the hand-size card). At the opening pick (empty board) the
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
  of your choice" order pick — only presented when the birdfeeder shows exactly
  2 distinct faces; the active player auto-starts otherwise.

**What the choice rows carry.** Three disjoint shapes:

- habitat picks: each destination row carries the moving bird's identity
  (→ card table, the decision's `moving_bird` / `from_habitat` context), its
  **landing slot** marked by `board_hab` + `board_col` — the exact slot the
  bird would occupy (the destination row's next free slot; the "stay" row
  marks the bird's current slot), so the model reads the resulting location
  instead of inferring it from a habitat flag — plus `goal_delta` /
  `bonus_delta` pricing the relocation: habitat bird counts, the egg block
  riding along (including the egg-set minimum), and the habitat-spread bonus
  card; the "stay" row's deltas are naturally all-zero;
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

**What the choice rows carry.** Bird-kind rows: the bird's identity in
`bird_id` (→ card table), its **landing slot** marked by `board_hab` +
`board_col` (the chosen habitat's next free slot — the exact slot the bird
would occupy, so the model reads the resulting location directly), the
`bonus_delta` stripe pricing the play's marginal contribution to the held
bonus cards, and the `goal_delta` stripe pricing its marginal count/VP swing
on each of the four round goals. No cost features — costs resolve downstream,
and only completable pairs are offered.

**Variation within the family.** One class, one shape. A bird playable in two
habitats produces two rows differing only in the marked landing slot — that is
the intra-decision texture this head specializes in. Notably, a main-action play
and an extra play are **completely indistinguishable** to the model: same
decision class (so the same decision-type one-hot), same row shape, and no
extra-play-credit scalar in the state vector. The model cannot currently
condition on "this is a bonus play"; whether that matters is an open
modelling question.

### 2.12 `RESET_BIRDFEEDER` — is a fresh roll worth more than what's showing?

**Where the engine asks it.** Offered immediately before a feeder gain
whenever every die in the feeder shows the same face (all one food, or all on
the invertebrate/seed choice face — `Birdfeeder.reset_available()`):

- before each die of the main Gain Food action, and before the Forest
  conversion's extra die;
- before the feeder-pulling powers: the named-food feeder gains, the
  either-of-two-foods picks, the any-die picks, the gain-all-of-a-food powers,
  each seat's turn in the each-player gains, and the fewest-forest gain;
- before the pink predator-success reaction's pull, offered to the *reacting*
  player.

The *empty*-feeder reroll is automatic and never a decision. Structurally, the
offer cannot be bypassed: every feeder gain routes through one of the two
entry points in `engine/actions.py` (`take_one_from_feeder` for a die of the
player's choice, `take_all_of_food` for the no-choice gain-all powers), and
both run the offer internally before building their menu / count from the
post-reset feeder.

**What the choice rows carry.** Nearly nothing, by design: the affirmative
("reroll everything") is a bare special-kind token; the decline is the same
plus `is_skip`. The entire judgment — what is showing, how many dice remain,
what the player needs — is read from the state vector (the 7-dim feeder
stripe: five face counts, the choice-die count, and a 0/1 reset-availability
flag mirroring the offer condition) through the trunk. The flag is derivable
from the counts, but it is surfaced explicitly so this head — and every other
head deciding around a pending feeder gain — reads it directly.

**Variation within the family.** None structurally; the same two rows every
time. The contexts (main action vs. the various powers, active vs. reacting
seat) differ only through the state.

### 2.13 `SETUP` — the opening keep, and where the bonus pick lives

The opening (keep some of 5 dealt birds, retain the complementary foods, take
a bonus card) is scored by one of two owners, on the `use_setup_model` config
axis:

- **`use_setup_model = True` (default):** a separate actor-critic bandit
  (`wingspan.setup_model` + `training.setup_net`) scores the enumerated
  candidates and the main net drops everything setup-shaped (head, decision
  column, `setup_agg` stripe). Its per-candidate input vector is, in brief: a
  kept-cards multi-hot, a kept-foods multi-hot, and a kept-bonus one-hot, plus
  per-deal context (the three tray card indices, the six birdfeeder face counts,
  and the four round-goal one-hots), plus two candidate-pricing blocks — the kept
  bonus valued against the keep (kept qualifiers, the stepped / linear VP
  they would pay, tray potential) and a per-goal kept-card affinity (how many
  kept cards would advance each goal's category if played) — with the
  card-identity blocks embedded through frozen copies of the main net's
  shared card encoders. The **policy head** reads this whole fused vector to
  rank candidates; the **value head** reads only the action-independent *context*
  stripes (tray / birdfeeder / round goals, plus the bonus-cards-on-offer multi-hot
  in split-bonus mode), so it is the critic `V(s)` and the advantage `target − V(s)`
  no longer self-cancels (v1.2; see `docs/TRAINING.md §6.5`, `docs/VERSIONING.md`).
- **`use_setup_model = False`:** the main net keeps a `SETUP` head and scores
  the same candidates as ordinary choice rows (the kept birds as a multi-hot
  on the dedicated trailing `kept_multihot` stripe — the single-candidate
  `bird_id` column stays zero — foods-*spent* on the `pay_food` stripe, the
  `setup_agg` aggregates, and the kept bonus identity — plus, when the
  candidate carries a bonus, the `bonus_value` stripe with its hand potential
  counted over the **kept** subset, not the full dealt hand).

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

**Whether the opening food pick is folded in is a separate config choice.**
`TrainConfig.split_setup_food` (effective only alongside the setup model —
the gate is `split_setup_food_active = split_setup_food and use_setup_model`):

- **Off (folded, the default):** food is one axis of the combined keep —
  candidates are every (kept-cards × kept-foods × dealt-bonus) combination.
  The setup model jointly values the opening card-food-bonus bundle.
- **On (split):** every candidate carries `kept_foods = ()` (the setup
  encoder's food block is all-zero; `SETUP_FEATURE_DIM` is unchanged — REGIME,
  resumable). The opening food pick is instead asked as sequential in-game
  decisions right after the keep is applied, routing through the GAIN_FOOD or
  SPEND_FOOD head depending on how many birds were kept:

  | Birds kept | Player starts with | Decisions asked |
  |---|---|---|
  | 0 | 5 food | none |
  | 1 | 5 food | 1 × `SpendFoodDecision` (discard 1) |
  | 2 | 5 food | 2 × `SpendFoodDecision` (no-repeat) |
  | 3 | 0 food | 2 × `GainFoodDecision` (no-repeat) |
  | 4 | 0 food | 1 × `GainFoodDecision` |
  | 5 | 0 food | none |

  The argument *for* splitting: the food judgment trains on the dedicated
  GAIN_FOOD / SPEND_FOOD heads with on-policy credit (§2.4, §2.5), adding one
  sample per seat per game to these otherwise lightly-exercised heads. When
  `split_setup_food` is active, `setup_food_sets` is ignored (random setup
  generation emits `kept_foods = ()` directly without food sampling).

**Inspecting the setup encoding in `wingspan play --html`.**  When
`use_setup_model = True`, clicking a keep option in the HTML game log opens the
encoding-viewer modal with the setup net's input vector decoded into two panels:
"Game State" shows the shared deal context (tray birds, birdfeeder counts) and
"This Choice" shows the per-candidate blocks (kept cards, foods, bonus, pricing).
This mirrors the main-net modal's state/choice split.  `complex`-encoded stripes
(round goals) are hidden, identical to the main-net view.  The raw vectors are
captured in `players.decision_probe.PolicyAnnotation.setup_feats` and decoded by
`reporting.encode_viewer.extract_setup_{context,candidate}_stripes` using
`setup_model.setup_stripe_layout(encoding)`.

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
| A change to `EncodingSpec`, `use_setup_model`, `split_setup_bonus`, `split_setup_food`, or the setup candidate enumeration | §0 (the config-axis paragraph) and §2.13 |
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
