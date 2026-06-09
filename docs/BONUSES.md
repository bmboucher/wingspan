# BONUSES.md — Bonus cards and end-of-round goals

This document covers the 26 core-set bonus cards and 16 core-set end-of-round goals in the
simulator: how each is scored, what live-game information the model can observe about its
progress, and which moves advance it.

Bonus cards and round goals are the two largest secondary VP sources after birds. Together
they reward strategic focus — building a board tuned to a held bonus card, or timing eggs and
plays to win the round goal race. Both feed into the same final score formula:
birds + bonus + eggs + tucked + cached + round_goal_points (see `engine.scoring.final_scoring`).

---

## Bonus Cards

### Acquiring bonus cards

Every player draws 2 bonus cards and keeps 1 as their first act during setup (a `SetupDecision`
handled by the separate setup model). During the game, 15 birds carry a white power that reads
"Draw 2 bonus cards, keep 1"; those birds trigger a `choose_bonus` decision presenting 2 drawn
cards as `BonusCardChoice` rows. The keeper is appended to `Player.bonus_cards`; the other is
discarded face-down. A player can therefore hold 1 to 5+ bonus cards depending on how many of
those birds they play.

Opponent bonus card *identities* are hidden information. The model observes only
`len(opp.bonus_cards)` (one normalized scalar in the state vector).

### Scoring

Bonus cards are scored at game end only, via `engine.scoring.final_scoring`. Two payout
structures exist:

- **Tiered** — the card has a printed threshold table (e.g. "5–7 birds: 3 pt; 8+ birds: 7 pt").
  The engine picks the highest threshold met via `bonus_score_for_count`.
- **Per-bird** — the card pays a fixed amount per qualifying bird (e.g. "2 pt per bird").
  The engine multiplies `per_bird_vp × count`.

`BonusCard.thresholds` stores the tiered pairs; `BonusCard.per_bird_vp` is set for per-bird
cards and `None` for tiered ones.

### State encoding — bonus progress stripe

The POV player's bonus card status occupies `4 × _BONUS_ID_DIM` (= 4 × 26 = 104) dimensions
in the state vector, keyed by `cards.bonus_index()`:

| Sub-stripe | Content |
|---|---|
| `held` | Multi-hot: 1.0 for each card the player holds, 0 for cards not held |
| `count` | Normalized qualifying-bird count per held card (÷ `_BONUS_COUNT_SCALE = 5`) |
| `stepped` | Normalized stepped VP at current count per held card (÷ `_BONUS_VALUE_SCALE = 7`) |
| `linear` | Normalized piecewise-linear value per held card (÷ `_BONUS_VALUE_SCALE = 7`) |

Cards not held have all four entries at 0; a held card at 0 qualifying birds has `held=1` but
`count=stepped=linear=0`. This lets the value head plan toward a newly-held card before any
birds qualify. The `linear` sub-stripe rewards partial progress toward the next stepped tier
and provides gradient signal below the first threshold.

The opponent's bonus-card count is a single additional scalar (÷ `_BONUS_COUNT_SCALE`),
immediately following the four sub-stripes.

### Choice encoding — bonus delta

Every candidate bird in a `play_bird`, `draw_bird`, or similar decision carries a
**bonus delta** block (3 dims) in its choice feature row:

| Index | Meaning |
|---|---|
| `_BONUS_DELTA_QUAL` | How many of the player's held bonus cards this bird qualifies for (÷ 5) |
| `_BONUS_DELTA_STEPPED` | Sum of stepped-VP deltas (count → count+1) across all held qualifying cards (÷ 7) |
| `_BONUS_DELTA_LINEAR` | Same sum using the linear payoff (÷ 7) |

This is the marginal price of playing this bird against your current bonus portfolio. A bird
qualifying for two held cards at a combined stepped delta of +4 VP will show a higher delta
than one qualifying for one card at +1 VP.

**What the delta covers**: static qualifying conditions only — food diet, nest, habitat,
wingspan, name-based, and power-mechanic categories. The four dynamic cards (Breeding Manager,
Ecologist, Oologist, Visionary Leader) are priced only where their condition can be evaluated
at play time: Ecologist is priced via `bonus_count_delta_for_play_habitat`; the egg-based and
hand-based cards are not priced in the play delta (a freshly played bird has no eggs and does
not change hand size).

### Choice encoding — bonus card identity

When a `BonusCardChoice` row is presented (choosing which card to keep), a separate
**bonus value** block (5 dims) prices the offered card against the player's current board:

| Index | Meaning |
|---|---|
| `_BONUS_VALUE_QUAL` | Board birds currently qualifying for this card (÷ 5) |
| `_BONUS_VALUE_STEPPED` | Stepped VP at that count (÷ 7) |
| `_BONUS_VALUE_LINEAR` | Linear value at that count (÷ 7) |
| `_BONUS_VALUE_HAND` | Hand birds that would qualify if played (÷ 5) |
| `_BONUS_VALUE_TRAY` | Tray birds that would qualify if played (÷ 5) |

The card's identity is also encoded as a one-hot over all 26 bonus cards
(`CHOICE_BONUS_ID_OFFSET`, 26 dims) so the model can learn card-specific priors.

### Bird attribute encoding for bonus categories

Seven "intrinsic" bonus categories are embedded directly into every bird's static attribute
vector as a 7-dim multi-hot (the `_BONUS_CATS_DIM` block):

> **Anatomist, Backyard Birder, Cartographer, Historian, Large Bird Specialist,
> Passerine Specialist, Photographer**

These are categories whose signal is *not* already captured by another part of the attribute
vector. The remaining 19 categories are redundant with other bird attributes and are excluded
from this block to avoid duplication:

- Food diet (6 cards): already captured by the food-cost 6-vector.
- Nest type (4 cards): already captured by the nest 4-one-hot.
- Habitat specialty / Bird Bander: already captured by the habitat multi-hot.
- Bird Counter / Falconer: already captured by the flocking and predator flags.
- Dynamic cards (4): not a static bird attribute at all.

---

## Bonus card groups

### Group 1 — Food diet (6 cards)

These cards reward specializing in birds that eat a particular food type. The qualifying
condition is a printed food symbol in the bird's food cost.

| Card | Condition | Payout |
|---|---|---|
| **Bird Feeder** | Birds that eat `[seed]` | 5–7 birds: 3 pt; 8+ birds: 7 pt |
| **Fishery Manager** | Birds that eat `[fish]` | 2–3 birds: 3 pt; 4+ birds: 8 pt |
| **Food Web Expert** | Birds that eat *only* `[invertebrate]` | 2 pt per bird |
| **Omnivore Specialist** | Birds that eat `[wild]` | 2 pt per bird |
| **Rodentologist** | Birds that eat `[rodent]` | 2 pt per bird |
| **Viticulturalist** | Birds that eat `[fruit]` | 2–3 birds: 3 pt; 4+ birds: 7 pt |

**Scoring**: Static. `Bird.bonus_categories` tags every qualifying bird at load time by
reading the per-bird column in `master.json`; `bonus_qualifying_count` counts tagged birds in
play. Food Web Expert requires the bird's *entire* cost to be invertebrate — birds with any
other food in the cost do not qualify.

**Model visibility**: The food-cost 6-vector in every bird's attribute block (5 specific foods
+ wild) lets the model infer food-diet eligibility for any candidate bird. The bonus progress
stripes carry the live count/stepped/linear values for held food-diet cards.

**Advances by**: Playing any qualifying bird. The bonus delta in the choice feature row prices
the marginal play. Habitat selection does not affect qualification; these are food-only.

---

### Group 2 — Nest type (4 cards)

These cards reward filling the board with a specific nest type. Star (`[star]`) nests are
wildcards and count toward every concrete-nest bonus.

| Card | Condition | Payout |
|---|---|---|
| **Enclosure Builder** | Birds with `[ground]` or `[star]` nests | 4–5 birds: 4 pt; 6+ birds: 7 pt |
| **Nest Box Builder** | Birds with `[cavity]` or `[star]` nests | 4–5 birds: 4 pt; 6+ birds: 7 pt |
| **Platform Builder** | Birds with `[platform]` or `[star]` nests | 4–5 birds: 4 pt; 6+ birds: 7 pt |
| **Wildlife Gardener** | Birds with `[bowl]` or `[star]` nests | 4–5 birds: 4 pt; 6+ birds: 7 pt |

All four have identical payout curves — they differ only in which nest they track.

**Scoring**: Static. `cards.nest_matches(bird.nest, target_nest)` is true for matching nests
and for star nests; this is baked into `Bird.bonus_categories` at load time.

**Model visibility**: The nest 4-one-hot in every bird's attribute block (bowl/cavity/ground/
platform, with star encoded as all-ones) lets the model read a candidate's nest type directly.
The bonus progress stripes provide current count and VP for held nest-type cards.

**Advances by**: Playing any bird with a matching (or star) nest. Habitat does not matter —
a cavity bird played into forest advances Nest Box Builder identically to one played into
wetland.

---

### Group 3 — Habitat specialty (3 cards)

These cards reward concentrating birds in a single habitat. The condition is that a bird can
live *only* in the named habitat — multi-habitat birds do not qualify.

| Card | Condition | Payout |
|---|---|---|
| **Forester** | Birds that can only live in `[forest]` | 3–4 birds: 4 pt; 5 birds: 8 pt |
| **Prairie Manager** | Birds that can only live in `[grassland]` | 2–3 birds: 3 pt; 4+ birds: 8 pt |
| **Wetland Scientist** | Birds that can only live in `[wetland]` | 3–4 birds: 3 pt; 5 birds: 7 pt |

**Scoring**: Static. Single-habitat birds are tagged at load time; multi-habitat birds are
excluded.

**Model visibility**: The habitat 3-multi-hot in the bird attribute directly encodes which
habitats each bird can live in. A bird with exactly one bit set is a habitat specialist; the
model can infer single-habitat status from that bit pattern. The bonus progress stripes carry
live counts. Note: a habitat-specialist bird must still be *played into its only legal
habitat*, so the habitat choice at play time is fully determined.

**Advances by**: Playing a single-habitat bird of the correct type. Because the bird can only
go to one habitat, there is no placement decision to optimize for these cards.

---

### Group 4 — Wingspan class (2 cards)

These cards reward playing birds at the physical extremes of the wingspan range.

| Card | Condition | Payout |
|---|---|---|
| **Large Bird Specialist** | Birds with wingspan > 65 cm | 4–5 birds: 3 pt; 6+ birds: 6 pt |
| **Passerine Specialist** | Birds with wingspan ≤ 30 cm | 4–5 birds: 3 pt; 6+ birds: 6 pt |

Note the asymmetry: Large Bird Specialist excludes exactly-65 cm birds; Passerine Specialist
includes exactly-30 cm birds (≤ not <).

**Scoring**: Static. Wingspan is read from `Bird.wingspan_cm` at load time. Birds with
`wingspan_cm == 0` (no wingspan data) are excluded from both.

**Model visibility**: `Bird.wingspan_cm` is encoded as a normalized scalar (`÷ _WINGSPAN_SCALE
= 200`) in every bird attribute vector. Both cards appear in the 7-dim curated bonus-category
multi-hot (`Large Bird Specialist` and `Passerine Specialist`), giving the model an explicit
qualifying flag per candidate bird.

**Advances by**: Playing any qualifying bird. These cards interact with
Forester/Prairie/Wetland Scientist if a habitat-only bird also falls in the wingspan range.

---

### Group 5 — Power type (2 cards)

These cards reward playing birds with a specific gameplay mechanic encoded as a power symbol.

| Card | Condition | Payout |
|---|---|---|
| **Bird Counter** | Birds with a `[flocking]` power | 2 pt per bird |
| **Falconer** | Birds with a `[predator]` power | 2 pt per bird |

Both qualify regardless of power color — a pink flocking power counts for Bird Counter just as
a brown one does.

**Scoring**: Static. `Bird.flocking` and `Bird.predator` boolean fields are set at load time.

**Model visibility**: Separate `flocking` and `predator` flags are encoded in the bird
attribute vector (`_OFF_ATTR_FLOCK` and `_OFF_ATTR_PRED`). The model reads these directly
without needing the curated bonus-category multi-hot.

**Advances by**: Playing any bird with the matching mechanic symbol. The bonus delta at
`play_bird` time prices the marginal contribution.

---

### Group 6 — Name-based (4 cards)

These cards qualify birds based on keywords in their common name. They are the most
"lookup-intensive" category at load time — each bird's name is scanned for the matching
word list, and the result is baked into `Bird.bonus_categories`.

| Card | Condition | Payout |
|---|---|---|
| **Anatomist** | Birds with body-part words in their name (beak, belly, bill, breast, cap, chin, collar, crest, crown, eye, face, head, neck, rump, shoulder, tail, throat, wing) | 2–3 birds: 3 pt; 4+ birds: 7 pt |
| **Cartographer** | Birds with geography words in their name (American, Atlantic, Baltimore, California, Canada, Carolina, Chihuahua, Eastern, Inca, Mississippi, Mountain, Northern, Prairie, Sandhill, Savannah, Western) | 2–3 birds: 3 pt; 4+ birds: 7 pt |
| **Historian** | Birds named after a person — operationalized as any bird with `'s` in its name | 2 pt per bird |
| **Photographer** | Birds with color words in their name (ash, black, blue, bronze, brown, cerulean, chestnut, ferruginous, gold, gray, green, indigo, lazuli, purple, red, rose, roseate, ruby, ruddy, rufous, snowy, violet, white, yellow) | 4–5 birds: 3 pt; 6+ birds: 7 pt |

**Scoring**: Static. Determined purely by bird name matching at load time.

**Model visibility**: All four appear in the 7-dim curated bonus-category multi-hot in every
bird's attribute vector. The model gets an explicit bit for each category per candidate bird,
without needing to parse the name itself.

**Advances by**: Playing a qualifying bird. These categories have no habitat or egg
dependency — once a bird is played, its contribution is fixed.

---

### Group 7 — Printed stats (2 cards)

These cards qualify birds based on a numeric attribute printed on the card (point value or
food cost total), not a categorical tag.

| Card | Condition | Payout |
|---|---|---|
| **Backyard Birder** | Birds worth < 4 points (i.e. 0, 1, 2, or 3 VP) | 5–6 birds: 3 pt; 6+ birds: 6 pt |
| **Diet Specialist** | Birds with a food cost of exactly 3 food tokens | 2–3 birds: 3 pt; 4+ birds: 6 pt |

**Scoring**: Static. `Bird.points` and the total of `Bird.food_cost.counts` are read at load
time and the qualifying flag is baked into `Bird.bonus_categories`.

**Model visibility**: Backyard Birder appears in the 7-dim curated bonus-category multi-hot.
Diet Specialist is not in the multi-hot (its signal is covered by the food-cost 6-vector;
a bird with food_cost summing to 3 qualifies). Both still get the full held/count/stepped/
linear bonus progress treatment in the state vector, and the bonus delta in choice vectors
correctly prices them.

**Advances by**: Playing a qualifying bird. There is no habitat, egg, or timing dependency.

---

### Group 8 — Dynamic cards (4 cards)

Unlike all other bonus cards, these four evaluate live game state at scoring time rather than
a fixed per-bird tag. Their qualifying counts are computed via `_DYNAMIC_BONUS_COUNTERS`
rather than `Bird.bonus_categories`. They also require updating incrementally during play so
the bonus delta signal is accurate during decision-making.

#### Breeding Manager

"Birds that have at least 4 eggs laid on them" — **1 pt per bird**.

The qualifying count is the number of played birds currently carrying ≥ 4 eggs.
`bonus_count_delta_for_egg` fires on each egg-lay event: when laying or removing eggs on a
bird, if the egg count crosses the 4-egg threshold in either direction, the count changes by
±1. A bird at exactly 3 eggs gaining 1 egg adds 1 to the count; losing an egg from 4 removes 1.

Egg limit sets a hard ceiling on how many birds can ever qualify: only birds with egg_limit ≥ 4
can reach 4 eggs. Cavity birds (egg_limit 4–5) are the most natural targets.

**Model visibility**: Breeding Manager holds `held/count/stepped/linear` bonus progress stripes
in the state vector, and egg-lay `BirdTargetChoice` rows include the bonus delta contribution
from crossing the 4-egg threshold.

**Advances by**: Laying eggs on birds that are at or approaching 4 eggs. The egg-lay action
(main action) and any bird power that lays eggs can improve this count. Playing a new bird
with a high egg limit creates future capacity.

#### Ecologist

"Birds in your habitat with the fewest birds — ties count" — **2 pt per bird**.

The qualifying count is `min(len(board[habitat]) for habitat in ALL_HABITATS)` — the length
of the shortest row. Ties share the minimum: if all three rows have 3 birds, the count is 3.

`bonus_count_delta_for_play_habitat` computes the marginal change when playing into a given
habitat: +1 only when the target habitat is the *unique* shortest row (so playing into it
raises the minimum). If two rows are tied at the minimum, playing into one of them does not
change the minimum and the delta is 0.

`bonus_count_delta_for_move` handles the rarer case of a bird migrating between habitats.

**Model visibility**: The board summary's row-length scalars (one per habitat, `÷
_ROW_SLOTS_SCALE`) let the model track relative row lengths. The Ecologist bonus progress
stripes encode the current minimum and its VP. The bonus delta for `play_bird` choices shows
which habitat target would raise the minimum row (those choices get a nonzero delta).

**Advances by**: Playing into the shortest row. The optimal strategy is to grow all three rows
in parallel rather than specializing, which puts Ecologist in tension with habitat-specialist
bonus cards (Forester, Prairie Manager, Wetland Scientist) and single-habitat eggs for round
goals.

#### Oologist

"Birds that have at least 1 egg laid on them" — VP: 7–8 birds: 3 pt; 9+ birds: 6 pt.

The qualifying count is the number of played birds that have ≥ 1 egg. `bonus_count_delta_for_egg`
fires at the 1-egg threshold: laying the first egg on any eggless bird adds 1 to the count;
removing the last egg subtracts 1. Once a bird has its first egg, additional eggs don't move
the count.

**Model visibility**: The bonus progress stripes carry the live count. Egg-lay target choice
rows encode the delta for crossing the 1-egg threshold on each candidate slot.

**Advances by**: Getting the first egg on each bird (breadth, not depth). The Lay Eggs main
action and any bird power that lays eggs advance Oologist. There is an important tension:
laying eggs is also the primary driver of several round goals, so the model must allocate
eggs between breadth (Oologist) and depth (egg-count goals like `eggs_in_habitat`).

#### Visionary Leader

"Bird cards in hand at end of game" — VP: 5–7 birds: 4 pt; 8+ birds: 7 pt.

The qualifying count is simply `len(player.hand)` at game end. `bonus_count_delta_for_hand`
returns `delta_cards` for any hand-size change (draw or discard).

**Model visibility**: The hand summary stripe encodes hand size (normalized). The bonus
progress stripes carry the live count and VP at current hand size.

**Advances by**: Drawing bird cards and *not* playing them. The Draw Cards main action, bird
powers that draw cards, and declining to play drawn birds all increase the count. Playing
birds or discarding cards decrease it. Visionary Leader creates the principal tension between
drawing and playing birds.

**⚠ Gap**: The bonus delta in `play_bird` choice rows reflects the lost card from the hand,
but the *future* draw opportunities foregone by spending this turn on play are not priced
directly.

---

## End-of-Round Goals

### Setup and tile structure

The 16 core-set goals come from 8 double-sided physical tiles (tile_ids 0–7 in `goals.json`).
Each tile's two sides are opposites on the same theme (e.g. "birds in forest" vs. "eggs in
forest"). At game start, 4 of the 8 tiles are randomly selected and oriented (side chosen
randomly), giving 4 goals — one per round. The set of 4 goals is public information from turn 1.

Each goal has a `category` string that the scoring engine dispatches on (20 canonical categories
are defined in `layout._GOAL_CATEGORIES`). The engine returns 0 for any unknown category, so
unsupported goals degrade gracefully.

### Payout structure

At the end of each round, `engine.scoring.score_round_goal` counts each player's category value
and applies the 2P placement rule:

| Round | 1st place | 2nd place |
|---|---|---|
| 1 | 4 VP | 1 VP |
| 2 | 5 VP | 2 VP |
| 3 | 6 VP | 3 VP |
| 4 | 7 VP | 4 VP |

Tie rule: tied players share 1st and 2nd, each earning `floor((first + second) / 2)`. A player
with a count of exactly 0 does not place and scores 0 regardless of the opponent's count.

### State encoding — round goal stripe

All four round goal slots are encoded in the state vector
(`_ROUND_GOALS_STRIPE_DIM = 4 × 23 = 92` dims):

| Dims | Content |
|---|---|
| [0..19] | 20-dim category one-hot (indexed by `layout._GOAL_CATEGORIES` order) |
| [20] | POV player's current category count (÷ `_GOAL_COUNT_SCALE = 5`) |
| [21] | Opponent's current category count (÷ 5) |
| [22] | VP the POV player would earn if the round scored now (÷ `_ROUND_GOAL_POINTS_SCALE = 10`) |

Encoding all four rounds (not just the current one) lets the model plan toward later-round
goals it is already accumulating toward. Already-scored rounds hold their frozen at-scoring
standings — those stripes never change again even as the boards evolve.

### Choice encoding — goal delta

Candidate bird choices in `play_bird` decisions carry a **goal delta** block (8 dims = 4
rounds × 2) in the choice feature row:

| Dims | Content |
|---|---|
| [2k] | Count delta from playing this bird against round k's goal |
| [2k+1] | VP delta from that count change (at current opponent standing) |

For the egg-lay action, the accept row of the `AcceptExchangeDecision` carries an *optimistic
best-case* count/vp delta (`goal_best_case_for_eggs`), and each `BirdTargetChoice` row carries
the exact delta for laying on that specific slot (`goal_count_delta_for_egg`).

Bird-move events (from bird powers that migrate a bird) compute deltas via
`goal_count_delta_for_move`.

### Interaction with the `birds_no_eggs` anti-goal

**`birds_no_eggs`** counts birds with 0 eggs — so a *higher* count is *bad* (you want fewer
eggless birds). When this goal is active, the egg-lay accept rows correctly reflect this: each
egg laid on an eggless bird reduces the count by 1. Brown egg-laying powers that are normally
"always beneficial" become gated behind `skip_optional` when this goal is active, because
declining to lay eggs can be the right move if it would hurt your standing. See BIRDS.md for
details on this gate.

---

## End-of-round goal groups

### Group A — Birds in habitat (3 goals, tile_ids 0–2 front side)

| Goal | Category | Tile |
|---|---|---|
| Birds in `[forest]` | `birds_forest` | 0 |
| Birds in `[grassland]` | `birds_grassland` | 1 |
| Birds in `[wetland]` | `birds_wetland` | 2 |

**Scored as**: Count of birds played into the named habitat on the POV player's board.

**Advances by**: Playing any bird (with the matching habitat in its habitat set) into that
habitat. Each `play_bird` candidate contributes a delta of 1 if the bird can live in the goal
habitat. The habitat placement choice is explicit for multi-habitat birds — a dual-habitat bird
placed in the goal habitat adds 1; placed elsewhere, it adds 0.

**Encoder category list index**: 0 (`birds_forest`), 1 (`birds_grassland`), 2 (`birds_wetland`).

**Tension**: Each bird placed in one habitat foregoes progress in the other two bird-count
goals (if they were in play). Forest/grassland/wetland-specialist bonus cards (Forester, Prairie
Manager, Wetland Scientist) align naturally with their matching birds-in-habitat goal.

---

### Group B — Eggs in habitat (3 goals, tile_ids 0–2 reverse side)

| Goal | Category | Tile |
|---|---|---|
| Eggs in `[forest]` | `eggs_forest` | 0 |
| Eggs in `[grassland]` | `eggs_grassland` | 1 |
| Eggs in `[wetland]` | `eggs_wetland` | 2 |

**Scored as**: Total egg tokens on birds in the named habitat row.

**Advances by**: Laying eggs on any bird in the goal habitat — whether via the Lay Eggs main
action or a bird power. Each egg contributes +1 to the count. Advancing an eggs-in-habitat
goal requires both (a) having birds in that row and (b) having egg capacity on those birds.

**Delta computation**: `goal_count_delta_for_egg` returns `delta_eggs` when the egg event is
in the matching habitat; 0 otherwise. Bird moves (`goal_count_delta_for_move`) can transfer
the egg block between habitats.

**Encoder category list index**: 3 (`eggs_forest`), 4 (`eggs_grassland`), 5 (`eggs_wetland`).

---

### Group C — Eggs on nest type (4 goals, tile_ids 3–6 reverse side)

| Goal | Category | Tile |
|---|---|---|
| Eggs in `[bowl]` nests | `eggs_bowl` | 3 |
| Eggs in `[cavity]` nests | `eggs_cavity` | 4 |
| Eggs in `[ground]` nests | `eggs_ground` | 5 |
| Eggs in `[platform]` nests | `eggs_platform` | 6 |

**Scored as**: Total eggs on all birds with the matching nest type (any habitat). Star nests
are wildcards: a star-nest bird's eggs count toward *all four* of these goals.

**Advances by**: Laying eggs on birds with the matching (or star) nest. Nest type is immutable
once a bird is played, so the relevant advance has two components: (a) playing birds with the
target nest, and (b) laying eggs on them. The nest-type bonus cards (Enclosure Builder, Nest
Box Builder, Platform Builder, Wildlife Gardener) align naturally with the matching nest goal.

**Delta computation**: `goal_count_delta_for_egg` checks `cards.nest_matches(played_bird.bird.nest,
goal_nest)` — star nests pass every nest check.

**Encoder category list index**: 6 (`eggs_bowl`), 7 (`eggs_cavity`), 8 (`eggs_ground`),
9 (`eggs_platform`).

---

### Group D — Birds with eggs by nest type (4 goals, tile_ids 3–6 front side)

| Goal | Category | Tile |
|---|---|---|
| `[bowl]` birds with ≥ 1 egg | `bowl_birds_with_eggs` | 3 |
| `[cavity]` birds with ≥ 1 egg | `cavity_birds_with_eggs` | 4 |
| `[ground]` birds with ≥ 1 egg | `ground_birds_with_eggs` | 5 |
| `[platform]` birds with ≥ 1 egg | `platform_birds_with_eggs` | 6 |

**Scored as**: Count of birds with the matching nest type that have at least 1 egg. Star nests
count toward all four. This is a *breadth* goal (each bird contributes at most 1 regardless of
how many eggs it carries), unlike the depth-oriented eggs-on-nest group above.

**Advances by**: Laying the first egg on each matching-nest bird. Subsequent eggs on the same
bird do not advance the count. The advance structure mirrors Oologist — one count point per
bird that transitions from eggless to egg-carrying.

**Delta computation**: `goal_count_delta_for_egg` tracks the has-eggs threshold crossing:
`int(has_eggs_after) - int(had_eggs)`. Returns +1 on first egg, 0 for subsequent eggs, -1
when the last egg is removed.

**Encoder category list index**: 10 (`bowl_birds_with_eggs`), 11 (`cavity_birds_with_eggs`),
12 (`ground_birds_with_eggs`), 13 (`platform_birds_with_eggs`).

---

### Group E — Composite goals (2 goals, tile_id 7)

#### Total birds (category `total_birds`)

**Scored as**: Total birds in play across all three habitats.

**Advances by**: Playing any bird anywhere, by 1. This is the simplest round goal — every play
advances it, so advantage is earned by playing faster or more birds than the opponent.

**Delta computation**: `goal_count_delta_for_bird` always returns 1.

**Encoder category list index**: 17.

#### Egg sets (3 habitats) (category `egg_sets_3habitats`)

**Scored as**: `min(eggs_in_forest, eggs_in_grassland, eggs_in_wetland)` — the number of
complete (1 forest, 1 grassland, 1 wetland) egg sets.

This is the only goal with a min-across-habitats structure. The count only increases when an
egg is placed into the habitat that currently has the fewest eggs (raising the floor). An egg
into an already-leading habitat adds 0.

**Advances by**: Laying eggs in the habitat with the lowest egg count. Optimal play water-fills
the three habitats, keeping counts as balanced as possible. Playing a bird into a new habitat
creates capacity; the Lay Eggs action delivers eggs toward the floor habitat.

**Delta computation**: `goal_count_delta_for_egg` recomputes `min(egg_sums)` before and after
each egg event. Bird moves can shift the egg block between habitats and recalculate the minimum.
The best-case bound for an egg-lay commitment (`goal_best_case_for_eggs`) uses greedy water-fill:
each egg goes to the lowest habitat with remaining capacity.

**Encoder category list index**: 18.

---

## Cross-cutting notes

### Tile pairing and goal selection

The 8 core tiles pair a "shallow breadth" side with a "deep depth" side:
- Tiles 0–2: birds-in-habitat ↔ eggs-in-habitat (count of birds vs. total eggs)
- Tiles 3–6: birds-with-eggs ↔ eggs-on-nest (count of qualifying birds vs. total eggs)
- Tile 7: total-birds ↔ egg-sets (simple total vs. cross-habitat minimum)

A game with tile 3 showing "cavity birds with eggs" rewards spreading eggs across many cavity
birds; the same tile flipped to "eggs in cavity" rewards concentrating eggs on a few
high-capacity birds. The model sees the category one-hot in the state vector and must adapt.

### Bonus–goal alignment

Several bonus card and round goal pairs reward the same underlying board state:

| Bonus | Aligned round goal |
|---|---|
| Forester / Prairie Manager / Wetland Scientist | birds_forest / birds_grassland / birds_wetland |
| Enclosure Builder | eggs_ground or ground_birds_with_eggs |
| Nest Box Builder | eggs_cavity or cavity_birds_with_eggs |
| Platform Builder | eggs_platform or platform_birds_with_eggs |
| Wildlife Gardener | eggs_bowl or bowl_birds_with_eggs |
| Oologist | Any `*_birds_with_eggs` goal |
| Breeding Manager | Any eggs-in-habitat or eggs-on-nest goal (indirectly, via egg depth) |

When a held bonus card aligns with the active round goal, the choice delta signals reinforce
each other: a play that advances both gets a high `_BONUS_DELTA_STEPPED` *and* a positive
`_GOAL_DELTA_VP`.

### Gap — scoring-only categories

The 20 goal categories encoded in `_GOAL_CATEGORIES` cover all 16 core-set goals. The European
and Oceania expansion goals (tucked_cards, wingspan_under_30, wingspan_over_65, birds_no_eggs)
occupy the remaining 4 category slots and are evaluable by the engine, but the physical tiles
that use them are not drawn in a core-only game.

`tucked_cards` is used by the European "birds with tucked cards" goal and aligns with the
Citizen Scientist bonus card (also European; not in the 26 core bonus cards). `wingspan_under_30`
and `wingspan_over_65` align with the Passerine Specialist and Large Bird Specialist bonus cards
respectively.
