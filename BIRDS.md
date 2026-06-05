# BIRDS.md — How every bird power is implemented

This document walks through all 180 birds in the simulated core set and explains, in
plain language, how each printed power actually behaves in the engine. Birds are
grouped by mechanic, but **every bird's exact printed power text appears verbatim**
(in quotes) so the parsing can be checked against the real cards. Bracketed tags like
`[seed]` / `[egg]` / `[card]` are the inline icons from the card text.

Decision points are referred to by the decision-model IDs the RL policy trains
against: `main_action`, `play_bird`, `draw_bird`, `discard_bird`, `gain_food`,
`spend_food`, `lay_egg`, `pay_egg`, `skip_optional`, `choose_bonus`, `misc_rare`,
`reset_birdfeeder`.

Anything that deviates from the printed rules is flagged inline with **⚠ Gap** and
collected again in the [Gaps and deviations](#gaps-and-deviations) section at the end.

## Conventions that apply everywhere

- **When powers fire.**
  - *White* ("when played") powers fire exactly once, immediately after the bird's
    egg and food costs are paid and the bird is placed.
  - *Brown* ("when activated") powers fire every time the owner takes the main
    action of the habitat row the bird sits in; the row's brown powers run
    right-to-left (newest bird first). They can also fire an extra time via the two
    "repeat a brown power" birds.
  - *Pink* ("once between turns") powers fire in reaction to an opponent's action;
    the specific trigger is given per bird below. The "once between turns" cap is
    enforced: `PlayedBird.pink_fired` is set when a pink power commits a reaction,
    preventing it from firing again in the same between-turns window. The flag is
    cleared at the start of the owner's next turn. A decline or no-eligible-target
    does **not** consume the use — only a committing fire does.
  - Birds with no power do nothing (last section before the gaps).
- **Forced choices resolve silently.** Any decision that ends up with exactly one
  legal option is auto-resolved without consulting the player/model (it is logged as
  a skipped decision). So when a description below says "the player picks…", that
  pick only reaches the model when there are ≥ 2 options.
- **The birdfeeder reset rules.** The two printed reset rules fire at different
  moments. **Empty → reroll is instant:** the moment a take removes the last die,
  the feeder is rerolled on the spot (every die that leaves the feeder goes through
  one shared routine that does this), so the feeder is never observed empty at any
  later decision point — players always see real dice when weighing unrelated
  choices (which bird to draw, what to play, etc.). **Single face → reroll is
  optional and waits for the next gain:** just before any feeder gain — from main
  actions and bird powers alike — if every die shows the same face, the player
  about to take is offered a `reset_birdfeeder` decision (reroll first, or take
  as-is). This optional offer is implied and not repeated in every entry below.
  (The pre-gain routine also carries a defensive empty-feeder reroll, but the
  instant reroll above makes it unreachable in normal play.)
- **The supply is infinite.** The printed rules treat the general food supply as
  unlimited; the engine matches this exactly — there is no counter, no depletion
  check, and gaining from the supply can never fail.
- **"Always beneficial" powers run with no opt-out.** The real game makes every
  power optional; the engine instead hard-codes a power as mandatory when declining
  could never be the better move — the underlying assumption being that *more of
  your own resources is always at least as good* (the one known exception, eggs
  while the birds-without-eggs round goal is active, is handled by conditional
  `skip_optional` gates on the egg powers). Each such group below explains the
  reasoning. Note this assumption is about *free* gains only — plenty of powers pay
  costs for benefits, and those always go through an accept/decline.
- **"All players…" powers offer the active player a veto.** When a power also hands
  the opponent resources, "free gain for me" is no longer sufficient reason to force
  activation: denying the opponent an egg/card/die can be worth more than the active
  player's own gain. All such powers now open with a `skip_optional` veto whose
  accept row carries both the active player's gain and the `opp_gained_*` ledger.
  Declining cancels the whole power for everyone.
- **Optional powers go through `skip_optional`.** Whenever a power involves paying
  a cost (or could plausibly be declined), the first thing presented is a
  `skip_optional` decision whose accept row carries the full ledger of what would be
  paid and gained; declining ends the power immediately. After accepting, the
  follow-up choices are mandatory — the commitment is settled up front.

---

## 1. Birds with no power (6 birds)

**American Woodcock, Blue-Winged Warbler, Hooded Warbler, Prothonotary Warbler,
Trumpeter Swan, Wild Turkey.**

Printed power text: *(none — the card has no power box)*.

These birds never activate and present no decisions. They contribute only their
printed points, eggs laid on them, and bonus/goal eligibility.

---

## 2. Gain food from the supply (7 birds)

| Bird | Exact printed text |
|---|---|
| American Goldfinch (white) | "Gain 3 [seed] from the supply." |
| Brown Pelican (white) | "Gain 3 [fish] from the supply." |
| Blue-Gray Gnatcatcher (brown) | "Gain 1 [invertebrate] from the supply." |
| Painted Whitestart (brown) | "Gain 1 [invertebrate] from the supply." |
| Yellow-Bellied Sapsucker (brown) | "Gain 1 [invertebrate] from the supply." |
| Northern Cardinal (brown) | "Gain 1 [fruit] from the supply." |
| Spotted Towhee (brown) | "Gain 1 [seed] from the supply." |

- **When:** the white ones when played; the brown ones on each activation of their row.
- **Option to activate?** No — fully automatic, no decision of any kind is presented.
- **Why always beneficial:** the named food is added to the player's stash for free;
  there is no cost, no cap on personal food, and the supply is infinite.
- **Subsequent choices:** none. The food type and amount are fixed by the card.

---

## 3. "All players gain 1 [food] from the supply" (6 birds)

| Bird | Exact printed text |
|---|---|
| Baltimore Oriole (brown) | "All players gain 1 [fruit] from the supply." |
| Black-Chinned Hummingbird (brown) | "All players gain 1 [fruit] from the supply." |
| Eastern Phoebe (brown) | "All players gain 1 [invertebrate] from the supply." |
| Scissor-Tailed Flycatcher (brown) | "All players gain 1 [invertebrate] from the supply." |
| Osprey (brown) | "All players gain 1 [fish] from the supply." |
| Red Crossbill (brown) | "All players gain 1 [seed] from the supply." |

- **When:** on each activation of the bird's row.
- **Option to activate?** **Yes** — because the opponent also gains, the active
  player first gets a `skip_optional` veto. The accept row carries both their own
  gain (`gained_food_count=1`) and the opponent's (`opp_gained_food_count=1`).
  Declining ends the power with nobody gaining. If no opponents exist, the power
  runs unconditionally.
- **Subsequent choices (on accept):** none — everyone receives the fixed food
  automatically.

---

## 4. Gain a specific food from the birdfeeder (7 birds)

| Bird | Exact printed text |
|---|---|
| Acorn Woodpecker (brown) | "Gain 1 [seed] from the birdfeeder, if available. You may cache it on this bird." |
| Blue Jay (brown) | "Gain 1 [seed] from the birdfeeder, if available. You may cache it on this bird." |
| Clark's Nutcracker (brown) | "Gain 1 [seed] from the birdfeeder, if available. You may cache it on this bird." |
| Red-Bellied Woodpecker (brown) | "Gain 1 [seed] from the birdfeeder, if available. You may cache it on this bird." |
| Red-Headed Woodpecker (brown) | "Gain 1 [seed] from the birdfeeder, if available. You may cache it on this bird." |
| Steller's Jay (brown) | "Gain 1 [seed] from the birdfeeder, if available. You may cache it on this bird." |
| Great Crested Flycatcher (brown) | "Gain 1 [invertebrate] from the birdfeeder, if available." |

- **When:** on each activation of the bird's row.
- **Option to activate?** No — the take itself is automatic. After the standard reset
  check, the engine takes up to one die of the named food from the feeder. If none
  is showing, the power silently does nothing ("if available").
- **Why always beneficial:** a free food with no cost; the worst case is "nothing
  available", which is a no-op either way.
- **Subsequent choices:** Great Crested Flycatcher — none; the take is automatic.
  The six seed birds: after taking the seed (which lands in the player's supply), a
  `skip_optional` decision is offered — "cache this seed on the bird" or "keep it
  in supply". The accept row carries `paid_food=seed / paid_food_count=1 /
  gained_cache_count=1` (the food moves from supply to the bird's cached-food total,
  worth a point at game end); the skip row leaves the seed spendable. If the feeder
  had no seed, the cache decision is never reached.

---

## 5. Gain one of two foods from the birdfeeder (3 birds)

| Bird | Exact printed text |
|---|---|
| Indigo Bunting (brown) | "Gain 1 [invertebrate] or [fruit] from the birdfeeder, if available." |
| Western Tanager (brown) | "Gain 1 [invertebrate] or [fruit] from the birdfeeder, if available." |
| Rose-Breasted Grosbeak (brown) | "Gain 1 [seed] or [fruit] from the birdfeeder, if available." |

- **When:** on each activation of the bird's row.
- **Option to activate?** No — the gain itself is mandatory (free food, as above),
  but *which* food is a real choice.
- **Subsequent choices:** after the reset check, the player gets a `gain_food`
  decision listing every way the feeder can currently yield either of the two named
  foods (plain dice and, for invertebrate/seed, the invertebrate-or-seed choice
  die). If neither food is showing, the power is skipped ("if available"). If only
  one way exists, it auto-resolves.

---

## 6. Gain 1 die of your choice (1 bird)

**American Redstart (brown):** "Gain 1 [die] from the birdfeeder."

- **When:** on each activation of its row.
- **Option to activate?** No — mandatory; a free die can never hurt.
- **Subsequent choices:** the reset check, then a `gain_food` decision over every
  face currently showing in the feeder (the feeder is never empty at this point, so
  there is always something to take).

---

## 7. Gain ALL of a food from the birdfeeder (2 birds)

| Bird | Exact printed text |
|---|---|
| Bald Eagle (white) | "Gain all [fish] that are in the birdfeeder." |
| Northern Flicker (white) | "Gain all [invertebrate] that are in the birdfeeder." |

- **When:** once, when played.
- **Option to activate?** No — automatic. After the reset check, every die that can
  yield the named food (including choice dice for invertebrate) moves to the
  player's stash, one at a time. Zero showing → power does nothing.
- **Why always beneficial:** strictly free food; nothing is paid and there is no
  upper limit on personal food.
- **Subsequent choices:** none beyond the possible reset offer.

---

## 8. Each player gains a die, you pick who starts (2 birds)

**Anna's Hummingbird (brown), Ruby-Throated Hummingbird (brown)** — identical text:

> "Each player gains 1 [die] from the birdfeeder, starting with the player of your choice."

- **When:** on each activation of the bird's row.
- **Option to activate?** **Yes** — because the opponent also gains a die, the
  active player first gets a `skip_optional` veto. The accept row carries
  `gained_food_count=1` (own die) and `opp_gained_food_count=1` (opponent's die).
  Declining skips the whole power.
- **Subsequent choices (on accept), in order:**
  1. The active player gets a `misc_rare` decision choosing which player gains
     first (in a 2-player game: me or the opponent).
  2. In that order, each player — using *their own* agent — goes through the reset
     check and then a `gain_food` pick of one die from the feeder. Going first
     matters when the feeder holds contested faces.

---

## 9. "Player(s) with the fewest…" (3 birds)

| Bird | Exact printed text |
|---|---|
| Hermit Thrush (brown) | "Player(s) with the fewest birds in their [forest] gain 1 [die] from birdfeeder." |
| American Bittern (brown) | "Player(s) with the fewest birds in their [wetland] draw 1 [card]." |
| Common Loon (brown) | "Player(s) with the fewest birds in their [wetland] draw 1 [card]." |

These powers compare a count across players and reward whoever is at the minimum
(ties included). The rational-play analysis splits three ways: if the active player
has **strictly fewer** than the opponent, only they benefit, so activating is always
right (mandatory is fine); if they have **strictly more**, only the opponent would
benefit, so declining is always right (auto-skip is fine); if the counts are
**tied**, the power is effectively "each player gains…" and the veto applies.

**Hermit Thrush** (forest, gains a die):

- **When:** on each activation of its (forest) row.
- **Option to activate?** Partially shortcut: if the active player has strictly
  more forest birds than the minimum, the power is *auto-skipped* without asking
  anyone. In the tied case (all players share the minimum), a `skip_optional` veto
  is offered — the accept row carries `gained_food_count=1` (own die) and
  `opp_gained_food_count=N` for each other tied player. In the strictly-fewer case,
  the power runs forced.
- **Subsequent choices:** every player whose forest count equals the minimum —
  which always includes the active player when the power runs — takes one die using
  *their own agent*: reset check, then their own `gain_food` pick from the feeder.

**American Bittern / Common Loon** (wetland, draws a card):

- **When:** on each activation of the bird's (wetland) row.
- **Option to activate?** Partially shortcut: if the active player has strictly
  more wetland birds than the minimum, the power is *auto-skipped*. In the
  strictly-fewer *or tied* case, the power runs: every player at the minimum draws
  one card via their own agent (one `draw_bird` pick each).
- **⚠ Gap (no veto in tied case):** in the tied case both players draw, so a veto
  should be offered as Hermit Thrush does; instead the draw runs unconditionally.

---

## 10. Cache a food on this bird (18 birds)

Cached food stays on the bird and is worth a point at game end; in this engine it
can never be spent.

### 10a. Unconditional cachers (5 birds)

**Carolina Chickadee, Juniper Titmouse, Mountain Chickadee, Red-Breasted Nuthatch,
White-Breasted Nuthatch** (all brown) — identical text:

> "Cache 1 [seed] from the supply on this bird."

- **When:** on each activation of the bird's row.
- **Option to activate?** No — automatic, no decisions.
- **Why always beneficial:** a free point on the bird; nothing is paid.
- **Subsequent choices:** none.

### 10b. Dice-rolling predators (13 birds)

Rodent hunters (brown, predator) — **American Kestrel, Barn Owl, Broad-Winged Hawk,
Burrowing Owl, Eastern Screech-Owl, Ferruginous Hawk, Mississippi Kite** — identical
text:

> "Roll all dice not in birdfeeder. If any are [rodent], cache 1 [rodent] from the supply on this bird."

Fish hunters (brown, predator) — **Anhinga, Black Skimmer, Common Merganser, Snowy
Egret, White-Faced Ibis, Willet** — identical text:

> "Roll all dice not in birdfeeder. If any are [fish], cache 1 [fish] from the supply on this bird."

- **When:** on each activation of the bird's row.
- **Option to activate?** Normally no — a free cache is always beneficial. Exception:
  when the opponent has one or more not-yet-fired `PINK_PREDATOR_FEEDER` birds (§27b)
  in play, a `skip_optional` veto is offered first. The accept row carries
  `gained_cache_count=1` and `opp_gained_food_count=N` (one per qualifying reactor).
  A decline avoids triggering those reactors. The veto appears only when the opponent
  can actually benefit, so the model is never trained on trivially-obvious rows.
- **How the roll works:** the engine counts `BIRDFEEDER_DICE − feeder.total()` —
  the dice currently *outside* the feeder. Each such die is rolled independently;
  each face is equiprobable across all 5 foods and one "choice" face. If any roll
  shows the named food, the cache happens and `trigger_pink_predator_success` fires
  (§27b reactors gain a die each). If all dice are in the feeder (no dice outside),
  the roll is skipped entirely with "no dice outside feeder; skipped".
- **Subsequent choices:** none. The outcome is determined by the random roll.
- **Repeatability:** Hooded Merganser (§23b) can repeat these dice-roll predators
  alongside deck-hunt predators.

---

## 11. Lay 1 egg on this bird (4 birds)

**California Quail, Mourning Dove, Northern Bobwhite, Scaled Quail** (all brown) —
identical text:

> "Lay 1 [egg] on this bird."

- **When:** on each activation of the bird's row.
- **Option to activate?** Usually no — automatic; the egg appears on the bird
  (capped at its printed egg limit; at the limit the power does nothing). Exception:
  when the active round goal rewards birds *without* eggs, a `skip_optional` gate is
  offered first (`gained_egg_count=1`). Accepting commits to laying; declining
  preserves the bird's empty state for the goal.
- **Why always beneficial (outside goal):** an egg is a guaranteed point at no cost.
- **Subsequent choices:** none — the target is the bird itself.

---

## 12. Lay 1 egg on any bird (4 birds)

**Baird's Sparrow, Cassin's Sparrow, Chipping Sparrow, Grasshopper Sparrow** (all
brown) — identical text:

> "Lay 1 [egg] on any bird."

- **When:** on each activation of the bird's row.
- **Option to activate?** Usually no. The egg-laying is treated as always beneficial
  (a free point) **except** when the active round goal is the one that rewards birds
  *without* eggs — only then is laying potentially harmful, so the player first gets
  a `skip_optional` decision ("lay 1 egg" vs skip). Outside that goal the power runs
  as mandatory so the model isn't trained on a trivially obvious yes.
- **Subsequent choices:** a `lay_egg` decision over every owned bird with egg room
  (auto-resolves when only one bird has room; silently skipped when no bird does).
  If the goal-conditioned `skip_optional` was offered and declined, nothing happens.

---

## 13. Lay 1 egg on each bird with a given nest (4 birds)

| Bird | Exact printed text |
|---|---|
| Ash-Throated Flycatcher (white) | "Lay 1 [egg] on each of your birds with a [cavity] nest." |
| Bobolink (white) | "Lay 1 [egg] on each of your birds with a [ground] nest." |
| Inca Dove (white) | "Lay 1 [egg] on each of your birds with a [platform] nest." |
| Say's Phoebe (white) | "Lay 1 [egg] on each of your birds with a [bowl] nest." |

- **When:** once, when played.
- **Option to activate?** Usually no — automatic; eggs appear on every matching bird
  with room (star/wildcard nests count as matching). Exception: when the active round
  goal rewards birds without eggs, a `skip_optional` gate is offered with the real
  egg count that would be laid in the ledger. Outside that goal the power is forced.
- **Subsequent choices:** none — every eligible bird receives one automatically.

---

## 14. All players lay an egg on a nest type (3 birds)

| Bird | Exact printed text |
|---|---|
| Lazuli Bunting (brown) | "All players lay 1 [egg] on any 1 [bowl] bird. You may lay 1 [egg] on 1 additional [bowl] bird." |
| Pileated Woodpecker (brown) | "All players lay 1 [egg] on any 1 [cavity] bird. You may lay 1 [egg] on 1 additional [cavity] bird." |
| Western Meadowlark (brown) | "All players lay 1 [egg] on any 1 [ground] bird. You may lay 1 [egg] on 1 additional [ground] bird." |

- **When:** on each activation of the bird's row.
- **Option to activate?** **Yes.** Because the power helps the opponent too, the
  active player first gets a `skip_optional` veto. The accept row's ledger shows
  `gained_egg_count = min(2, own_eligible_count)` (the most the active player could
  gain counting the extra egg) and `opp_gained_egg_count = opp_eligible_count`
  (number of opponents who have an eligible bird). Declining cancels the whole power
  for everyone. If *nobody* has an eligible bird the power is skipped without asking.
  Star (wildcard) nests are counted correctly via `cards.nest_matches`.
- **Subsequent choices, in order (on accept):**
  1. Each *opponent* in turn order who has a matching bird with room — normally
     forced (free egg); under the birds-without-eggs round goal each opponent first
     gets their own `skip_optional` accept/decline via their own agent.
  2. The active player places their mandatory base egg (`lay_egg` over matching
     birds; skipped if they have none).
  3. The active player may place the "1 additional" egg: forced commit to the
     `lay_egg` pick (the extra was counted in the accept-row ledger, so it's
     committed on accept), but the anti-egg-goal gate is repeated here too when
     that goal is active. The menu *excludes* the bird that received the base egg,
     so the same bird cannot get both eggs; if that leaves no eligible bird the
     extra is silently skipped.

---

## 15. Draw cards (13 birds)

All three sub-groups share one implementation: the player draws N cards, one at a
time, each via a `draw_bird` decision choosing a face-up tray card or the top of the
deck (tray slots emptied this turn stay empty until end of turn). (The two
fewest-in-wetland drawers, American Bittern and Common Loon, are covered with
Hermit Thrush in §9.)

### 15a. Plain draws (3 birds)

| Bird | Exact printed text |
|---|---|
| Mallard (brown) | "Draw 1 [card]." |
| Black-Necked Stilt (white) | "Draw 2 [card]." |
| Carolina Wren (white) | "Draw 2 [card]." |

- **When:** Mallard on row activation; the white two when played.
- **Option to activate?** No — drawing is free and there is no hand limit, so it is
  treated as always beneficial.
- **Subsequent choices:** one `draw_bird` pick per card.

### 15b. Draw, then discard at end of turn (8 birds)

**Black Tern, Clark's Grebe, Forster's Tern** (brown) — identical text:

> "Draw 1 [card]. If you do, discard 1 [card] from your hand at the end of your turn."

**Common Yellowthroat, Pied-Billed Grebe, Red-Breasted Merganser, Ruddy Duck, Wood
Duck** (brown) — identical text:

> "Draw 2 [card]. If you do, discard 1 [card] from your hand at the end of your turn."

- **When:** on each activation of the bird's row.
- **Option to activate?** No — runs as a mandatory free draw.
- **Subsequent choices, in order:**
  1. One `draw_bird` pick per card (N cards drawn from the tray-or-deck menu).
  2. If at least one card was actually drawn (hand grew), one end-of-turn discard
     obligation is registered. At the end of the turn, after all effects and extra
     plays settle, the engine presents one `discard_bird` decision per owed discard.
     The player picks which hand card to discard; on an empty hand the obligation
     fizzles. The net effect for the draw-1 group is 0 new cards; for the draw-2
     group it is +1.

### 15c. Pay an egg to draw (2 birds)

**Franklin's Gull, Killdeer** (brown) — identical text:

> "Discard 1 [egg] to draw 2 [card]."

- **When:** on each activation of the bird's row.
- **Option to activate?** **Yes** — a `skip_optional` decision with the ledger
  "discard 1 egg → draw 2 cards" (`paid_egg_count=1 / gained_card_count=2`). If no
  eggs exist anywhere on the board, the power is silently skipped.
- **Subsequent choices (on accept), in order:**
  1. A `pay_egg` decision choosing which bird's egg to discard (any owned bird
     with an egg, including the bird itself).
  2. Two `draw_bird` picks.

---

## 16. All players draw (5 birds)

**Canvasback, Northern Shoveler, Purple Gallinule, Spotted Sandpiper,
Wilson's Snipe** (all brown) — identical text:

> "All players draw 1 [card] from the deck."

- **When:** on each activation of the bird's row.
- **Option to activate?** **Yes** — because the opponent also draws, the active
  player first gets a `skip_optional` veto. The accept row carries
  `gained_card_count=1` (own draw) and `opp_gained_card_count=1` (opponent's draw).
  Declining skips the whole power.
- **Subsequent choices (on accept):** each player in turn draws one card straight
  off the deck (no tray menu — the printed "from the deck" is honored) silently via
  their own agent. The draw is automatic; no `draw_bird` decision is presented.

---

## 17. Brant (1 bird)

**Brant (white):** "Draw the 3 face-up [card] in the bird tray."

- **When:** once, when played.
- **Option to activate?** No — three free cards, strictly beneficial.
- **Subsequent choices:** none — all face-up tray cards (up to 3) go to hand at
  once. The tray slots are left empty and refilled at the normal end-of-turn refill;
  there is no immediate refill, so subsequent draws during the same turn see those
  slots as empty.

---

## 18. American Oystercatcher (1 bird)

**American Oystercatcher (white):**

> "Draw [card] equal to the number of players +1. Starting with you and proceeding clockwise, each player selects 1 of those cards and places it in their hand. You keep the extra card."

- **When:** once, when played.
- **Option to activate?** **Yes** — a `skip_optional` decision first, with the
  ledger "draw 3 cards, pass 2 to opponent, receive 1 back" (net: you +2 cards,
  opponent +1). Declining skips the whole power.
- **Subsequent choices (2-player), in order:**
  1. Three cards are drawn from the deck into your hand.
  2. You pass cards to the opponent until only one of the drawn cards remains with
     you: two `discard_bird` picks choosing which drawn card to give away each time.
     (Equivalently: you keep 1 of the 3.)
  3. The opponent, holding the 2 passed cards, returns all but one: one
     `discard_bird` pick (their agent) choosing which card to send back to you.
  4. The returned card re-enters your hand. Net result: you kept your favorite of
     the 3 plus whichever of the other two the opponent liked less; the opponent
     kept one.
  (The deck never runs short — when it empties, the discard pile is shuffled back
  in automatically. The handler's fewer-cards fallback exists only as a defensive
  guard for total exhaustion, all 180 cards simultaneously held/played/tucked,
  which does not occur in practice.)

---

## 19. Draw 2 bonus cards, keep 1 (15 birds)

**Atlantic Puffin, Bell's Vireo, California Condor, Cassin's Finch,
Cerulean Warbler, Chestnut-Collared Longspur, Greater Prairie-Chicken, King Rail,
Painted Bunting, Red-Cockaded Woodpecker, Roseate Spoonbill, Spotted Owl,
Sprague's Pipit, Whooping Crane, Wood Stork** (all white) — identical text:

> "Draw 2 new bonus cards and keep 1."

- **When:** once, when played.
- **Option to activate?** No — mandatory. Reasoning: a bonus card can only ever add
  points (its worst case is scoring zero), so gaining one is never harmful.
- **Subsequent choices:** 2 bonus cards come off the bonus deck and the player makes
  one `choose_bonus` pick; the other card is discarded. (If the bonus deck has only
  1 card the pick is forced; if it is empty the power is skipped.)

---

## 20. Tuck a card from hand (22 birds)

All hand-tucking powers share the same two-step shape: first a `skip_optional` gate
("tuck 1 card behind this bird?" — the engine never forces you to give up a card in
hand), then, on accept, a mandatory `discard_bird` pick of *which* hand card to tuck.
The tucked card is worth a point on the bird at game end. With an empty hand the
power is skipped without asking.

### 20a. Tuck → gain choice of two supply foods (1 bird)

**Pygmy Nuthatch (brown):**

> "Tuck 1 [card] from your hand behind this bird. If you do, gain 1 [invertebrate] or [seed] from the supply."

- **When:** on each activation of its row.
- **Subsequent choices, in order:** the `skip_optional` gate → on accept, the
  `discard_bird` pick → then a mandatory `gain_food` decision presenting exactly
  two choices: invertebrate or seed from the supply. Declining the gate yields
  nothing.

### 20b. Tuck → draw 1 (9 birds)

**American Coot, American Robin, Barn Swallow, House Finch, Purple Martin,
Ring-Billed Gull, Tree Swallow, Violet-Green Swallow, Yellow-Rumped Warbler** (all
brown) — identical text:

> "Tuck 1 [card] from your hand behind this bird. If you do, draw 1 [card]."

- **When:** on each activation of the bird's row.
- **Subsequent choices, in order:** `skip_optional` gate → on accept, `discard_bird`
  pick of the card to tuck → then a mandatory `draw_bird` pick (tray or deck) for
  the replacement card. Declining the gate ends the power with nothing tucked or
  drawn.

### 20c. Tuck → lay an egg on this bird (6 birds)

**Brewer's Blackbird, Bushtit, Common Grackle, Dickcissel, Red-Winged Blackbird,
Yellow-Headed Blackbird** (all brown) — identical text:

> "Tuck 1 [card] from your hand behind this bird. If you do, you may also lay 1 [egg] on this bird."

- **When:** on each activation of the bird's row.
- **Subsequent choices, in order:** `skip_optional` gate → on accept, `discard_bird`
  pick → then, if the bird has egg room, a forced `lay_egg` decision targeting this
  bird (auto-resolves — only one target). The "you may also" is handled via a
  `skip_optional` gate *only when* the birds-without-eggs round goal is active
  (so the model sees a genuine whether-question only when declining is rational).
  Outside that goal the egg is laid unconditionally. If the bird is at its egg limit
  the egg step is silently skipped.

### 20d. Tuck → lay an egg on any bird (1 bird)

**White-Throated Swift (brown):**

> "Tuck 1 [card] from your hand behind this bird. If you do, lay 1 [egg] on any bird."

- **When:** on each activation of its row.
- **Subsequent choices, in order:** `skip_optional` gate → on accept, `discard_bird`
  pick → then a mandatory `lay_egg` decision over every owned bird with egg room
  (no skip — the commitment was the gate). With no room anywhere the egg fizzles.

### 20e. Tuck → gain a fixed food from the supply (4 birds)

| Bird | Exact printed text |
|---|---|
| Cedar Waxwing (brown) | "Tuck 1 [card] from your hand behind this bird. If you do, gain 1 [fruit] from the supply." |
| Dark-Eyed Junco (brown) | "Tuck 1 [card] from your hand behind this bird. If you do, gain 1 [seed] from the supply." |
| Pine Siskin (brown) | "Tuck 1 [card] from your hand behind this bird. If you do, gain 1 [seed] from the supply." |
| Vaux's Swift (brown) | "Tuck 1 [card] from your hand behind this bird. If you do, gain 1 [invertebrate] from the supply." |

- **When:** on each activation of the bird's row.
- **Subsequent choices, in order:** `skip_optional` gate → on accept, `discard_bird`
  pick → the named food is then added automatically (no further choice). Declining
  the gate yields nothing.

### 20f. Horned Lark — see §27 (its tuck is a pink reaction).

---

## 21. Spend a food to tuck 2 from the deck (5 birds)

| Bird | Exact printed text |
|---|---|
| American White Pelican (brown) | "Discard 1 [fish] to tuck 2 [card] from the deck behind this bird." |
| Double-Crested Cormorant (brown) | "Discard 1 [fish] to tuck 2 [card] from the deck behind this bird." |
| Black-Bellied Whistling-Duck (brown) | "Discard 1 [seed] to tuck 2 [card] from the deck behind this bird." |
| Canada Goose (brown) | "Discard 1 [seed] to tuck 2 [card] from the deck behind this bird." |
| Sandhill Crane (brown) | "Discard 1 [seed] to tuck 2 [card] from the deck behind this bird." |

- **When:** on each activation of the bird's row.
- **Option to activate?** **Yes** — a `skip_optional` decision whose accept row
  carries the ledger "pay 1 fish/seed → tuck 2". If the player has none of the named
  food the power is skipped without asking.
- **Subsequent choices (on accept):** none — the food is deducted and 2 cards come
  blind off the top of the deck straight onto the bird (worth 2 points). The player
  never sees or picks the tucked cards. Declining keeps the food and tucks nothing.

---

## 22. Predator hunts — look at a deck card (10 birds)

**Greater Roadrunner (brown, predator):**

> "Look at a [card] from the deck. If less than 50cm, tuck it behind this bird. If not, discard it."

**Barred Owl, Cooper's Hawk, Northern Harrier, Red-Shouldered Hawk, Red-Tailed Hawk,
Swainson's Hawk** (all brown, predator) — identical text:

> "Look at a [card] from the deck. If less than 75cm, tuck it behind this bird. If not, discard it."

**Golden Eagle, Great Horned Owl, Peregrine Falcon** (all brown, predator) —
identical text:

> "Look at a [card] from the deck. If less than 100cm, tuck it behind this bird. If not, discard it."

- **When:** on each activation of the bird's row (and via Hooded Merganser, §23b).
- **Option to activate?** Normally no — the player risks nothing on a miss (the
  discarded card costs them nothing), and a success is a free tuck. Exception: when
  one or more opposing not-yet-fired `PINK_PREDATOR_FEEDER` birds (§27b) are in
  play, a `skip_optional` veto is offered. The accept row carries `gained_tuck_count=1`
  (success-case gain) and `opp_gained_food_count=N` (one per qualifying reactor).
  The gate appears only when the opponent can actually benefit, so the model is never
  trained on trivially-obvious always-accept rows.
- **Subsequent choices:** none. The top deck card's wingspan is compared to the
  printed threshold: strictly less → tucked on the predator, then
  `trigger_pink_predator_success` fires (§27b reactors gain a die each); otherwise
  it goes to the discard pile. The player never chooses anything; the reveal is pure
  chance. Hunts repeated via Hooded Merganser (§23b) run through the same handler
  and inherit the same gate.

---

## 23. Repeat another bird's power (3 birds)

### 23a. Repeat a brown power

**Gray Catbird, Northern Mockingbird** (both brown) — identical text:

> "Repeat a brown power on another bird in this habitat."

- **When:** on each activation of the bird's row.
- **Option to activate?** No explicit gate. If no other brown-powered bird in the
  same row has a modeled, repeatable power, it is skipped silently. Otherwise the
  player must pick a target — but many repeated powers contain their own
  `skip_optional` gates, so a player who wants "nothing" can often pick such a bird
  and then decline inside it. Repeating a strictly-beneficial power is itself
  strictly beneficial, which is the rationale for not adding an outer gate.
- **Subsequent choices:** a `misc_rare` decision choosing which bird's brown power
  to repeat (auto-resolves with a single candidate; the two repeat birds themselves
  and unmodeled powers are never offered). The chosen bird's power then runs
  exactly as in its own section, with all of its usual decisions; anything it tucks
  or caches lands on the *repeated* bird.

### 23b. Repeat a predator power

**Hooded Merganser (brown):**

> "Repeat 1 [predator] power in this habitat."

- **When:** on each activation of its (wetland) row.
- **Subsequent choices:** a `misc_rare` pick among other predators in the same row
  whose power is a §22-style deck hunt (`PREDATOR_HUNT`) **or** a dice-roll cache
  (`ROLL_NOT_IN_FEEDER_CACHE` — the §10b birds). Either kind qualifies. The chosen
  predator's power runs in full, including the conditional veto if opposing §27b
  reactors are in play. Skipped silently when no qualifying predator is present.

---

## 24. Move if rightmost (8 birds)

**Bewick's Wren, Blue Grosbeak, Chimney Swift, Common Nighthawk, Lincoln's Sparrow,
Song Sparrow, White-Crowned Sparrow, Yellow-Breasted Chat** (all brown) — identical
text:

> "If this bird is to the right of all other birds in its habitat, move it to another habitat."

- **When:** on each activation of the bird's row — but only does anything when the
  bird is currently the rightmost (most recently placed) bird in its row.
- **Option to activate?** **Yes, built into the menu:** the decision always includes
  an explicit "stay in <current habitat>" row, so the player can decline the move.
- **Subsequent choices:** a `misc_rare` decision listing "stay" plus each *other*
  habitat that still has an open slot. Choosing a destination pops the bird off its
  row and appends it as the rightmost bird of the destination row (no cost is paid).
  If the bird isn't rightmost, or no other habitat has space, the power is skipped
  silently.

---

## 25. Play an additional bird (10 birds)

### 25a. Named habitat (9 birds)

Forest — **Downy Woodpecker, Red-Eyed Vireo, Ruby-Crowned Kinglet, Tufted Titmouse**
(all white), identical text:

> "Play an additional bird in your [forest]. Pay its normal cost."

Grassland — **Eastern Bluebird, Mountain Bluebird, Savannah Sparrow** (all white),
identical text:

> "Play an additional bird in your [grassland]. Pay its normal cost."

Wetland — **Great Blue Heron, Great Egret** (both white), identical text:

> "Play an additional bird in your [wetland]. Pay its normal cost."

- **When:** once, when played. The power doesn't play the bird immediately — it
  banks one extra-play credit restricted to the printed habitat, redeemed after the
  current play finishes (chains of these birds accumulate credits).
- **Option to activate?** **Yes.** When the credit is redeemed, if at least one
  legal play exists in that habitat the player gets a `skip_optional` decision: take
  the extra play or forfeit the credit. With no legal play in that habitat the credit
  is silently wasted.
- **Subsequent choices (on accept), in order:**
  1. A `play_bird` decision over every legal (bird-in-hand) pair within the named
     habitat — "legal" means an open slot, affordable egg cost, and at least one
     food payment.
  2. The play's costs resolve as usual: one `pay_egg` pick per egg the destination
     column demands, then a `spend_food` decision among the legal payment
     combinations (auto-resolved when only one works).
  3. The played bird's own white power (if any) then fires, possibly banking
     further extra plays.

### 25b. House Wren

**House Wren (white):** "Play an additional bird in this bird's habitat. Pay its normal cost."

Identical flow to §25a, except the banked credit remembers the habitat House Wren
was played into, and both the `skip_optional` offer and the `play_bird` menu are
restricted to that habitat.

---

## 26. Trade and convert powers (6 birds)

### 26a. Discard an egg for wild food (5 birds)

**American Crow, Black-Crowned Night-Heron, Fish Crow** (all brown) — identical
text:

> "Discard 1 [egg] from any of your other birds to gain 1 [wild] from the supply."

**Chihuahuan Raven, Common Raven** (both brown) — identical text:

> "Discard 1 [egg] from any of your other birds to gain 2 [wild] from the supply."

- **When:** on each activation of the bird's row.
- **Option to activate?** **Yes** — a `skip_optional` decision with the ledger
  "discard 1 egg → gain 1 (or 2) food". If no *other* bird has an egg, the power is
  skipped without asking ("any of your **other** birds" is honored — eggs on the
  crow/raven itself don't count).
- **Subsequent choices (on accept), in order:**
  1. A `pay_egg` decision: which other bird loses the egg.
  2. One `gain_food` decision per wild food (1 or 2 picks), each over all five food
     types in the supply. Declining at the gate keeps the egg and gains nothing.

### 26b. Green Heron

**Green Heron (brown):** "Trade 1 [wild] for any other type from the supply."

- **When:** on each activation of its row.
- **Option to activate?** No — the trade is forced. Silently skipped if the player
  holds no food at all.
- **Subsequent choices (on activation), in order:**
  1. A `spend_food` decision: which food to give back to the supply (mandatory,
     over every food the player holds).
  2. A `gain_food` decision: which food to take from the supply (mandatory; the
     just-returned food is back in the supply by then, so trading a food for itself
     is legal and a natural way to decline the power's economic effect without
     needing a gate).

---

## 27. Pink powers — reactions to the opponent (12 birds)

Pink birds never act on their owner's turn. They sit on the board and react when the
*other* player does the trigger action. The once-between-turns cap is enforced: each
pink bird fires at most once per between-turns window. A decline or no-eligible-target
does **not** consume the use; only a committing fire does (see the global convention).

### 27a. When the opponent lays eggs (5 birds)

| Bird | Exact printed text |
|---|---|
| American Avocet | "When another player takes the "lay eggs" action, lay 1 [egg] on another bird with a [ground] nest." |
| Barrow's Goldeneye | "When another player takes the "lay eggs" action, lay 1 [egg] on another bird with a [cavity] nest." |
| Bronzed Cowbird | "When another player takes the "lay eggs" action, lay 1 [egg] on a bird with a [bowl] nest." |
| Brown-Headed Cowbird | "When another player takes the "lay eggs" action, lay 1 [egg] on a bird with a [bowl] nest." |
| Yellow-Billed Cuckoo | "When another player takes the "lay eggs" action, lay 1 [egg] on a bird with a [bowl] nest." |

- **When:** immediately after the opponent completes a Lay Eggs main action.
- **Option to activate?** Usually no — the owner gets a forced `lay_egg` decision
  listing every owned bird whose nest matches (star nests count as wild) and has
  egg room. A lone eligible bird auto-resolves. Exception: when the birds-without-eggs
  round goal is active, a `skip_optional` gate is offered first; declining preserves
  empty birds for the goal.
- **"Another bird" vs "a bird":** the pink bird itself is always excluded as a
  target (honoring the "another bird" wording on American Avocet and Barrow's
  Goldeneye). For the three "a bird" cards (Bronzed Cowbird, Brown-Headed Cowbird,
  Yellow-Billed Cuckoo), `exclude_self=False` is parsed from the text, so self-
  targeting is technically allowed — but those birds' own nests (none or platform)
  can never match the bowl requirement, so it makes no practical difference.
- With no eligible bird the power is skipped silently (does not consume the
  once-between-turns use).

### 27b. When the opponent's predator succeeds (3 birds)

**Black Vulture, Black-Billed Magpie, Turkey Vulture** — identical text:

> "When another player's [predator] succeeds, gain 1 [die] from the birdfeeder."

- **When:** immediately after an opponent's predator power succeeds — either a §22
  deck-hunt that tucks its prey *or* a §10b dice-roll that caches the named food.
  Both call `trigger_pink_predator_success`, which fires these reactors.
- **Option to activate?** No — a free die is always taken. The owner does choose
  *which* die: the standard reset check, then a `gain_food` pick from the feeder.

### 27c. When the opponent plays a bird in a habitat → gain food (2 birds)

| Bird | Exact printed text |
|---|---|
| Belted Kingfisher | "When another player plays a bird in their [wetland], gain 1 [fish] from the supply." |
| Eastern Kingbird | "When another player plays a bird in their [forest], gain 1 [invertebrate] from the supply." |

- **When:** immediately after the opponent places a bird into the named habitat
  (including extra plays — each qualifying play triggers it, subject to the
  once-between-turns cap).
- **Option to activate?** No — automatic; the fixed food is added with no decision.
  Free food, so always beneficial.

### 27d. Horned Lark

**Horned Lark:** "When another player plays a bird in their [grassland], tuck 1 [card] from your hand behind this bird."

- **When:** immediately after the opponent places a bird into their grassland.
- **Option to activate?** **Yes** — giving up a hand card is never forced: the owner
  gets a `skip_optional` gate ("tuck 1 card behind Horned Lark?"); on accept, a
  `discard_bird` pick of which hand card to tuck (+1 point on the lark). Declining
  (or an empty hand) ends the reaction. A tuck is a committing fire; a decline is
  not.

### 27e. Loggerhead Shrike

**Loggerhead Shrike:** "When another player takes the "gain food" action, if they gain any number of [rodent], cache 1 [rodent] from the supply on this bird."

- **When:** immediately after the opponent completes a Gain Food main action during
  which their rodent count increased (rodents gained by that action's dice and row
  powers all count).
- **Option to activate?** No — automatic: 1 rodent moves from the supply onto the
  shrike (+1 point). Free point, no decision.

---

## Gaps and deviations

Everything flagged above, gathered for review.

**Residual modeling choices (not bugs — deliberate decisions):**

1. **Once-between-turns semantics — decline does not consume the use.** Only a
   committing fire sets `pink_fired`; a decline or no-eligible-target leaves the
   flag clear. The printed rules are ambiguous about this; the engine's semantics
   (decline is free) match the most common table interpretation and are less
   punishing to the learning agent.
2. **The unparsed-power fallback is currently unused.** All 180 core birds match a
   pattern, so the `UNIMPLEMENTED` run-as-no-op fallback exists only as
   future-proofing for expansion cards whose text doesn't match any known pattern.

All residual gaps above are closed. The modeling choices in items 1 and 2
reflect deliberate design decisions, not bugs.
