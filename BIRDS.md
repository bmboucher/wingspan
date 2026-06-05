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
    the specific trigger is given per bird below. **⚠ Gap:** the "once between
    turns" cap is not enforced — a pink power fires on *every* qualifying trigger.
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
- **The supply should be unlimited — the engine tracks it anyway.** The printed
  rules treat the general food supply as infinite. The engine instead tracks a
  counter of 99 per type, and every supply-gain first checks the counter and
  **silently grants nothing** if it can't cover the full amount. Food paid by
  players is (with one trade exception) never returned to the counter, so it only
  drains. With 99 per type the check is practically unreachable in a 2-player
  game, but it is a deviation from the rules — see the gaps section.
- **"Always beneficial" powers run with no opt-out.** The real game makes every
  power optional; the engine instead hard-codes a power as mandatory when declining
  could never be the better move — the underlying assumption being that *more of
  your own resources is always at least as good* (the one known exception, eggs
  while the birds-without-eggs round goal is active, is handled by conditional
  `skip_optional` gates on the egg powers). Each such group below explains the
  reasoning. Note this assumption is about *free* gains only — plenty of powers pay
  costs for benefits, and those always go through an accept/decline.
- **"All players…" powers should offer the active player a veto — most don't.**
  When a power also hands the opponent resources, "free gain for me" is no longer
  sufficient reason to force activation: denying the opponent an egg/card/die can
  be worth more than the active player's own gain. The accept-row ledger has
  dedicated `opp_gained_*` slots so the model can weigh exactly this trade-off, and
  the all-players *lay-egg* power (§14) uses them — the active player gets a
  `skip_optional` veto whose accept row shows their own gain *and* the opponents'.
  The other three symmetric powers (§3 all gain food, §8 each player gains a die,
  §16 all players draw) currently run with **no veto** — flagged in the gaps
  section.
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
  there is no cost, no cap on personal food, and the supply never runs dry, so
  declining could never help.
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
- **Option to activate?** No — automatic, no decisions for anyone.
- **Subsequent choices:** none — everyone receives the fixed food automatically.
- **⚠ Gap (no veto):** because the opponent also gains, this should open with a
  `skip_optional` veto for the active player whose accept row carries both their
  own gain and the `opp_gained_*` ledger (as the all-players egg power §14 does);
  instead it runs unconditionally.
- **⚠ Gap (double gain):** the text matches *two* parsing patterns — the generic
  "gain 1 [food] from the supply" *and* the "all players gain…" pattern — and both
  resulting effects run. The active player therefore gains **2** of the food and the
  opponent gains 1, instead of 1 each as printed.

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
- **Option to activate?** No — automatic. After the standard reset check, the engine
  takes up to one die of the named food from the feeder (a choice die counts for
  seed/invertebrate). If none is showing, the power silently does nothing — this is
  the "if available" clause.
- **Why always beneficial:** a free food with no cost; the worst case is "nothing
  available", which is a no-op either way.
- **Subsequent choices:** none. The take is automatic, not a `gain_food` pick.
- **⚠ Gap (cache clause):** "You may cache it on this bird" is not implemented —
  the food always goes to the player's personal stash and can never be cached for
  points on the six seed birds.

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
- **Option to activate?** No — the power always runs.
- **Subsequent choices, in order:**
  1. The active player gets a `misc_rare` decision choosing which player gains
     first (in a 2-player game: me or the opponent).
  2. In that order, each player — using *their own* agent — goes through the reset
     check and then a `gain_food` pick of one die from the feeder. Going first
     matters when the feeder holds contested faces.
- **⚠ Gap (no veto):** the opponent gains a die too, so the active player should
  first get a `skip_optional` veto with the `opp_gained_*` ledger (as §14 does);
  instead activation is forced.

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
**tied**, the power is effectively "each player gains…" and should be optional with
an `opp_gained_*` ledger, like every other symmetric power.

**Hermit Thrush** has a dedicated implementation:

- **When:** on each activation of its (forest) row.
- **Option to activate?** Partially shortcut: if the active player has strictly
  more forest birds than the minimum, the power is *auto-skipped* without asking
  anyone (correct — it would only feed the opponent). Otherwise — strictly fewer
  **or tied** — it runs with no opt-out.
- **Subsequent choices:** every player whose forest count equals the minimum —
  which always includes the active player when the power runs — takes one die:
  reset check, then their own `gain_food` pick from the feeder.
- **⚠ Gap (no veto when tied):** in the tied case the opponent gains a die too, so
  the active player should first get a `skip_optional` veto with the `opp_gained_*`
  ledger; instead the power is forced. Only the strictly-fewer case is a true
  non-decision.

**American Bittern / Common Loon** have no dedicated implementation:

- **⚠ Gap (condition dropped):** the text falls through to the generic draw
  pattern, so only "draw 1 [card]" is parsed. The fewest-birds-in-wetland
  comparison is never made: the *active player* always draws 1 card (one
  `draw_bird` pick) and opponents never draw, regardless of who has the fewest
  wetland birds. The faithful fix is Hermit Thrush's three-way logic with a draw
  instead of a die.

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
- **Option to activate?** No — automatic.
- **Subsequent choices:** none.
- **⚠ Gap (no roll):** the dice roll is not simulated. The power simply caches 1 of
  the named food from the supply every time — a 100% success rate instead of the
  real probability (which depends on how many dice are outside the feeder, and is
  0% when all five dice are in it).
- **⚠ Gap (no predator trigger):** these are predator powers, but a success here
  does **not** trigger opponents' pink "when another player's [predator] succeeds"
  birds — only the deck-hunting predators of §22 do.

---

## 11. Lay 1 egg on this bird (4 birds)

**California Quail, Mourning Dove, Northern Bobwhite, Scaled Quail** (all brown) —
identical text:

> "Lay 1 [egg] on this bird."

- **When:** on each activation of the bird's row.
- **Option to activate?** No — automatic; the egg appears on the bird (capped at its
  printed egg limit; at the limit the power does nothing).
- **Why always beneficial:** an egg is a guaranteed point at no cost, and eggs also
  fund future bird plays.
- **Subsequent choices:** none — the target is the bird itself.
- **⚠ Gap:** unlike "lay an egg on any bird" (§12), this power is *not* gated when
  the current round goal rewards birds with no eggs — the egg is forced even when it
  could cost goal points.

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
- **Option to activate?** No — automatic, no decisions. Star (wildcard) nests count
  as matching; birds at their egg limit are skipped.
- **Why always beneficial:** every egg is a free point; there is nothing to choose
  because every matching bird with room receives one.
- **Subsequent choices:** none.
- **⚠ Gap:** like §11, this is not gated under the birds-without-eggs round goal.

---

## 14. All players lay an egg on a nest type (3 birds)

| Bird | Exact printed text |
|---|---|
| Lazuli Bunting (brown) | "All players lay 1 [egg] on any 1 [bowl] bird. You may lay 1 [egg] on 1 additional [bowl] bird." |
| Pileated Woodpecker (brown) | "All players lay 1 [egg] on any 1 [cavity] bird. You may lay 1 [egg] on 1 additional [cavity] bird." |
| Western Meadowlark (brown) | "All players lay 1 [egg] on any 1 [ground] bird. You may lay 1 [egg] on 1 additional [ground] bird." |

- **When:** on each activation of the bird's row.
- **Option to activate?** **Yes.** Because the power helps the opponent too, the
  active player first gets a `skip_optional` veto: the accept row's ledger shows how
  many eggs the active player would gain (1 if they have an eligible bird, else 0)
  and how many eligible opponents would gain one. Declining cancels the whole power
  for everyone. (If *nobody* has an eligible bird the power is skipped without
  asking.)
- **Subsequent choices, in order (on accept):**
  1. Each *opponent* in turn order who has a matching bird with egg room places
     their egg via their own `lay_egg` decision. Normally this is automatic-yes for
     them (a free egg); only when the birds-without-eggs round goal is active does
     each opponent first get their own `skip_optional` accept/decline.
  2. The active player places their mandatory base egg (`lay_egg` over matching
     birds; skipped if they have none).
  3. The active player may place the "1 additional" egg: a `lay_egg` decision whose
     menu includes an explicit skip row — accept places the egg, skip forfeits it.
- **⚠ Gap (extra egg may land on the same bird):** "1 additional [nest] bird" means
  a *different* bird from the one that received the base egg, but the extra-egg menu
  is rebuilt fresh from every matching bird with room — the base-egg bird is offered
  again whenever it still has room. The menu must exclude the bird the base egg went
  on; with exactly one eligible bird the extra egg is simply unavailable.
- **⚠ Gap (veto ledger omits the extra egg):** the accept row always advertises 1
  gained egg (or 0), never counting the extra — with two or more eligible birds the
  real exchange is "gain 2 eggs, opponents gain N" but the model is shown "gain 1,
  opponents gain N". The ledger should be 2 when the active player has at least two
  eligible birds and 1 when exactly one (per the same-bird rule above).
- **⚠ Gap (extra egg's skip is unconditional):** the skip row on the extra-egg
  `lay_egg` menu is offered whether or not the birds-without-eggs round goal is
  active. Once the ledger advertises the extra egg, accepting should commit to it
  (optional-then-commit), keeping the skip only under that goal — the one case where
  declining a free egg is rational — mirroring how the opponents' eggs are gated.
- **⚠ Gap (star nests):** the *eligibility* test used for the veto ledger and for
  deciding which opponents participate requires the exact nest type, ignoring star
  (wildcard) nests — but the actual egg-placement menus *do* accept star nests. A
  player whose only matching bird has a star nest is wrongly skipped.

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
- **Subsequent choices:** one `draw_bird` pick per card.
- **⚠ Gap (no discard):** the "discard 1 [card] from your hand at the end of your
  turn" clause is not implemented at all. These birds are pure card gain — the
  draw-2 group nets +2 instead of +1, and the draw-1 group nets +1 instead of 0
  (net card filtering). Because the cost is missing, no opt-out is offered either.

### 15c. Pay an egg to draw (2 birds)

**Franklin's Gull, Killdeer** (brown) — identical text:

> "Discard 1 [egg] to draw 2 [card]."

- **When:** on each activation of the bird's row.
- **⚠ Gap (cost dropped):** only the "draw 2 [card]" half is parsed. The egg is
  never discarded, and consequently no `skip_optional` trade decision is offered —
  the player simply draws 2 free cards via two `draw_bird` picks, even with zero
  eggs on the board.

---

## 16. All players draw (5 birds)

**Canvasback, Northern Shoveler, Purple Gallinule, Spotted Sandpiper,
Wilson's Snipe** (all brown) — identical text:

> "All players draw 1 [card] from the deck."

- **When:** on each activation of the bird's row.
- **Option to activate?** No — mandatory.
- **Subsequent choices:** every player, active player first, makes a `draw_bird`
  pick.
- **⚠ Gap (no veto):** the opponent also draws, so the active player should first
  get a `skip_optional` veto with the `opp_gained_*` ledger (as §14 does); instead
  activation is forced.
- **⚠ Gap (draw source):** the printed text says "from the deck", but each player is
  offered the normal tray-or-deck menu, so face-up tray cards can be taken.
- **⚠ Gap (wrong chooser plumbing):** the opponent's pick is requested through the
  *active player's* agent object rather than being routed to the opponent's own
  agent (the decision is still labeled with the opponent's player id, and in
  self-play both seats share one model, so this only matters for mixed
  human/bot or cross-model games).

---

## 17. Brant (1 bird)

**Brant (white):** "Draw the 3 face-up [card] in the bird tray."

- **When:** once, when played.
- **Option to activate?** No — three free cards, strictly beneficial.
- **Subsequent choices:** none — all face-up tray cards (up to 3) go to hand at
  once.
- **⚠ Gap (early refill):** the tray is refilled immediately after the take, rather
  than at end of turn as the real rules do — a subsequent draw this turn therefore
  sees fresh face-up cards it shouldn't have access to yet.

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

### 20a. Tuck → nothing extra (1 bird)

**Pygmy Nuthatch (brown):**

> "Tuck 1 [card] from your hand behind this bird. If you do, gain 1 [invertebrate] or [seed] from the supply."

- **When:** on each activation of its row.
- **Subsequent choices:** the `skip_optional` gate, then the `discard_bird` pick.
- **⚠ Gap (reward dropped):** the "gain 1 [invertebrate] or [seed] from the supply"
  consequence is not parsed (the either-or supply gain has no pattern), so accepting
  the tuck yields only the tucked-card point and **no food**. The decline option
  still protects the player from a bad trade.

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

### 20c. Tuck → may lay an egg on this bird (6 birds)

**Brewer's Blackbird, Bushtit, Common Grackle, Dickcissel, Red-Winged Blackbird,
Yellow-Headed Blackbird** (all brown) — identical text:

> "Tuck 1 [card] from your hand behind this bird. If you do, you may also lay 1 [egg] on this bird."

- **When:** on each activation of the bird's row.
- **Subsequent choices, in order:** `skip_optional` gate → on accept, `discard_bird`
  pick → then, if the bird has egg room, a `lay_egg` decision offering exactly two
  rows: this bird, or skip (the printed "you may also"). Accepting places the egg;
  skipping ends the power after the tuck. If the bird is at its egg limit the
  egg step is silently skipped.
- **⚠ Gap:** the skip row sits in the `lay_egg` menu, so the lay-egg head — not the
  `skip_optional` head — is asked a whether-question. The egg is a free gain: it
  should be forced (the single target auto-resolves), with a `skip_optional` gate
  offered first only while the birds-without-eggs round goal is active. See gaps
  #19.

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

- **When:** on each activation of the bird's row (and via Hooded Merganser, §23).
- **Option to activate?** No — the hunt runs automatically.
- **Why always beneficial (only without opposing reactors):** the player risks
  nothing of their own — success tucks a card (+1 point), failure just discards a
  deck card. But that holds *only* while the opponent has no pink "predator
  succeeds" bird (§27b) on their board; forcing the hunt is correct in that case
  alone.
- **⚠ Gap (no veto when a success feeds the opponent):** with one or more opposing
  §27b birds in play, every successful hunt also hands the opponent a free
  birdfeeder die per reactor — the tucked point may be worth less than that gift,
  so activating the hunt should be the player's choice. The fix follows the
  conditional-optionality pattern: offer a `skip_optional` gate *only* when an
  opposing §27b reactor is in play (so the model never trains on trivially-obvious
  always-accept rows), the accept row advertising the success-case exchange — 1
  tucked card gained, opponent food gained equal to the reactor count. (The hunt
  can still miss; the ledger states what a success would commit the player to.) A
  hunt repeated via Hooded Merganser (§23b) runs through the same handler and
  inherits the same gate.
- **Subsequent choices:** none. The top deck card's wingspan is compared to the
  printed threshold: strictly less → tucked on the predator (and opponents' pink
  "predator succeeds" birds then react, §27b); otherwise it goes to the discard
  pile. The player never chooses anything; the reveal is pure chance.

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
  whose power is a §22-style deck hunt, then that hunt runs (no further choices).
  Skipped silently when there is no such predator.
- **⚠ Gap:** only deck-hunting predators qualify as repeat targets. The dice-rolling
  cache predators of §10b (Snowy Egret, Willet, etc. — common wetland-mates) are
  *not* offered, because their parsed form is a plain cache with no predator
  semantics.

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
  banks one extra-play credit that is redeemed after the current play finishes (so
  chains of these birds accumulate credits).
- **Option to activate?** **Yes.** When the credit is redeemed, if at least one
  legal play exists the player gets a `skip_optional` decision: take the extra play
  or forfeit the credit. With no legal play the credit is silently wasted.
- **Subsequent choices (on accept), in order:**
  1. A `play_bird` decision over every legal (bird-in-hand, habitat) pair —
     "legal" means an open slot, affordable egg cost, and at least one food payment.
  2. The play's costs resolve as usual: one `pay_egg` pick per egg the destination
     column demands, then a `spend_food` decision among the legal payment
     combinations (auto-resolved when only one works).
  3. The played bird's own white power (if any) then fires, possibly banking
     further extra plays.
- **⚠ Gap (habitat not enforced):** the extra play is **not** restricted to the
  habitat printed on the card — the `play_bird` menu spans all three habitats. Only
  House Wren (below) carries a real restriction.

### 25b. House Wren

**House Wren (white):** "Play an additional bird in this bird's habitat. Pay its normal cost."

Identical flow to §25a, except the banked credit remembers the habitat House Wren
was played into, and both the `skip_optional` offer and the `play_bird` menu are
restricted to that habitat. (The restriction applies to that single credit only.)

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
- **Option to activate?** Currently **yes** — a `skip_optional` gate ("trade 1 food
  → 1 food"). Skipped without asking if the player has no food at all.
- **Subsequent choices (on accept), in order:**
  1. A `spend_food` decision: which food to give back to the supply.
  2. A `gain_food` decision: which food to take from the supply. (The just-returned
     food is back in the supply by then, so trading a food for itself is legal.)
- **⚠ Gap (the gate is redundant — the trade should be mandatory):** because the
  discard resolves before the gain, the identity trade (discard a food, take the
  same food back) is always available and is exactly equivalent to declining. The
  `skip_optional` gate therefore adds nothing the follow-up menus can't express —
  declining just means the player values their worst food more than the best food
  on offer, which the forced `spend_food`/`gain_food` pair expresses as a no-op
  trade. Removing the gate takes a trivially-learnable row off the overloaded
  `skip_optional` head (the inverse of the conditional-optionality fixes above:
  there a missing gate hides a real decision; here a present gate duplicates one).

---

## 27. Pink powers — reactions to the opponent (12 birds)

Pink birds never act on their owner's turn. They sit on the board and react when the
*other* player does the trigger action. (Reminder of the global gap: the real
once-between-turns cap is not enforced — these fire on every qualifying trigger.)

### 27a. When the opponent lays eggs (5 birds)

| Bird | Exact printed text |
|---|---|
| American Avocet | "When another player takes the “lay eggs” action, lay 1 [egg] on another bird with a [ground] nest." |
| Barrow's Goldeneye | "When another player takes the “lay eggs” action, lay 1 [egg] on another bird with a [cavity] nest." |
| Bronzed Cowbird | "When another player takes the “lay eggs” action, lay 1 [egg] on a bird with a [bowl] nest." |
| Brown-Headed Cowbird | "When another player takes the “lay eggs” action, lay 1 [egg] on a bird with a [bowl] nest." |
| Yellow-Billed Cuckoo | "When another player takes the “lay eggs” action, lay 1 [egg] on a bird with a [bowl] nest." |

- **When:** immediately after the opponent completes a Lay Eggs main action.
- **Option to activate?** Currently **yes, built into the menu:** the owner gets a
  `lay_egg` decision listing every *other* owned bird whose nest matches (star
  nests count) and has egg room, **plus an explicit skip row**. With no eligible
  bird the power is skipped silently.
- **⚠ Gap (the skip row is misplaced — the power should be mandatory):** only the
  `skip_optional` head should ever see a skip option; every other decision family
  answers *which*, never *whether*. A free egg is always beneficial except while
  the birds-without-eggs round goal is active, so the right shape is the standard
  conditional-optionality one: run the `lay_egg` pick forced (a lone eligible bird
  auto-resolves), and only under that goal precede it with a `skip_optional` gate.
  The same misplaced skip row appears on §20c's lay-on-self step and §14's extra
  egg — see gaps #19.
- All five are implemented identically — the pink bird itself is always excluded as
  a target ("another bird"), which is harmless for the three "a bird" cards because
  their own nests can never match bowl anyway.

### 27b. When the opponent's predator succeeds (3 birds)

**Black Vulture, Black-Billed Magpie, Turkey Vulture** — identical text:

> "When another player's [predator] succeeds, gain 1 [die] from the birdfeeder."

- **When:** immediately after an opponent's §22 deck-hunt predator tucks its prey.
  (**⚠ Gap:** the §10b dice-roll predators never trigger this.)
- **Option to activate?** No — a free die is always taken. The owner does choose
  *which* die: the standard reset check, then a `gain_food` pick from the feeder.

### 27c. When the opponent plays a bird in a habitat → gain food (2 birds)

| Bird | Exact printed text |
|---|---|
| Belted Kingfisher | "When another player plays a bird in their [wetland], gain 1 [fish] from the supply." |
| Eastern Kingbird | "When another player plays a bird in their [forest], gain 1 [invertebrate] from the supply." |

- **When:** immediately after the opponent places a bird into the named habitat
  (including extra plays — each qualifying play triggers it again).
- **Option to activate?** No — automatic; the fixed food is added with no decision.
  Free food, so always beneficial.

### 27d. Horned Lark

**Horned Lark:** "When another player plays a bird in their [grassland], tuck 1 [card] from your hand behind this bird."

- **When:** immediately after the opponent places a bird into their grassland.
- **Option to activate?** **Yes** — giving up a hand card is never forced: the owner
  gets a `skip_optional` gate ("tuck 1 card behind Horned Lark?"); on accept, a
  `discard_bird` pick of which hand card to tuck (+1 point on the lark). Declining
  (or an empty hand) ends the reaction.

### 27e. Loggerhead Shrike

**Loggerhead Shrike:** "When another player takes the “gain food” action, if they gain any number of [rodent], cache 1 [rodent] from the supply on this bird."

- **When:** immediately after the opponent completes a Gain Food main action during
  which their rodent count increased (rodents gained by that action's dice and row
  powers all count).
- **Option to activate?** No — automatic: 1 rodent moves from the supply onto the
  shrike (+1 point). Free point, no decision.

---

## Gaps and deviations

Everything flagged above, gathered for review. "Engine files" are given as
orientation pointers only.

**Parsing gaps — a printed clause is lost:**

1. **Active player double-gains on "All players gain 1 [food]"** (Baltimore Oriole,
   Black-Chinned Hummingbird, Eastern Phoebe, Scissor-Tailed Flycatcher, Osprey, Red
   Crossbill). The text matches both the personal-gain and the all-players patterns
   and both effects run: active player +2, opponent +1, instead of +1 each.
   (`cards/parse/matchers.py` — the personal-gain matcher has no "All players"
   exclusion, unlike the draw matcher.)
2. **"Player(s) with the fewest birds in their [wetland] draw 1 [card]"** (American
   Bittern, Common Loon — §9) loses its condition entirely: the active player
   always draws 1; opponents never draw even when they qualify. The forest twin
   (Hermit Thrush) has a dedicated implementation to copy; the wetland version fell
   through to the generic draw pattern.
3. **"Discard 1 [egg] to draw 2 [card]"** (Franklin's Gull, Killdeer): the egg cost
   is never charged and no accept/decline is offered — 2 free cards.
4. **"If you do, discard 1 [card] from your hand at the end of your turn"** (Black
   Tern, Clark's Grebe, Forster's Tern, Common Yellowthroat, Pied-Billed Grebe,
   Red-Breasted Merganser, Ruddy Duck, Wood Duck): the end-of-turn discard never
   happens; these are pure +1/+2 card engines.
5. **Pygmy Nuthatch's tuck reward** ("gain 1 [invertebrate] or [seed] from the
   supply") is dropped — accepting the tuck yields only the tucked-card point.
6. **"You may cache it on this bird"** on the six seed-from-feeder birds (Acorn /
   Red-Bellied / Red-Headed Woodpecker, Blue Jay, Clark's Nutcracker, Steller's
   Jay): the caching option does not exist; the seed always goes to the player.

**Simplifications in modeled powers:**

7. **Dice-roll predators always succeed** (the 13 birds of §10b): "Roll all dice not
   in birdfeeder" is not simulated — the cache happens unconditionally, instead of
   with the real (sometimes zero) probability.
8. **Dice-roll predator successes don't count as predator successes:** the pink
   vultures/magpie (§27b) react only to deck-hunt predators (§22), and Hooded
   Merganser (§23b) can only repeat deck-hunt predators — the §10b birds are
   invisible to both, despite being printed predator powers.
9. **"Play an additional bird in your [habitat]" is not habitat-restricted** for the
   nine §25a birds — the extra play may be used in any habitat. Only House Wren's
   restriction is enforced. (Side note: the grant silently fizzles if the granting
   bird were somehow played into a habitat other than the one named on it —
   impossible for these nine single-habitat birds as the set stands.)
10. **Pink powers have no once-between-turns cap** — they fire on every qualifying
    trigger (e.g. Belted Kingfisher triggers once per wetland bird the opponent
    plays in a single turn via extra plays; the vultures trigger once per successful
    hunt in a single row activation).
11. **"All players draw 1 [card] from the deck"** (§16): players may take face-up
    tray cards, not just the deck; and the opponent's tray-vs-deck pick is requested
    through the active player's agent object rather than the opponent's own
    (irrelevant in self-play where both seats share a model, wrong for mixed
    agents). (`engine/powers/grants.py`, all-players-draw handler.)
12. **Brant refills the tray immediately** instead of at end of turn, so later draws
    in the same turn see fresh face-up cards early.
13. **The "1 additional [nest] bird" extra egg is mis-modeled** on the §14
    all-players egg powers, in three coupled ways: (a) the extra-egg menu does not
    exclude the bird that received the base egg, so both eggs can illegally land on
    the same bird; (b) the active player's veto ledger always advertises 1 gained
    egg, never the extra — it should show 2 when at least two birds are eligible and
    1 when exactly one (where the extra is unavailable once same-bird is excluded);
    (c) the extra egg's skip row is offered unconditionally rather than only under
    the birds-without-eggs round goal — once the ledger counts the extra, accepting
    should commit to it except under that goal (and per #19 any such skip belongs
    on a `skip_optional` gate, not as a row in the `lay_egg` menu). (`engine/powers/multi_actor.py`
    all-players-lay-egg handler; `engine/powers/dispatch.py` `lay_one_egg_on_nest`
    rebuilds the menu with no exclusions.)

**Internal inconsistencies:**

14. **Star-nest eligibility mismatch in "All players lay 1 [egg]…"** (§14): the
    who-participates / veto-ledger check requires the exact nest type while the
    actual egg-placement menus accept star (wildcard) nests — a player whose only
    matching bird has a star nest is wrongly skipped, and the accept-row ledger can
    undercount. (`engine/powers/multi_actor.py` vs the shared nest-matching rule.)
15. **The birds-without-eggs round-goal gate is inconsistent across egg powers:**
    "lay on any bird" (§12) and the all-players egg powers (§14) offer a
    `skip_optional` under that goal, but "lay 1 [egg] on this bird" (§11) and "lay 1
    [egg] on each of your birds with a [nest]" (§13) remain forced, so those eggs
    can be pushed onto a player against their goal interest.
16. **Three of the four "all players…" powers lack the active-player veto:** when a
    power also grants the opponent resources, the active player should be offered a
    `skip_optional` veto whose accept row carries the `opp_gained_*` ledger — the
    model may value denying the opponent a gain above taking its own. The
    all-players lay-egg power (§14) implements exactly this; "All players gain 1
    [food]" (§3), "Each player gains 1 [die]" (§8), and "All players draw 1 [card]"
    (§16) run with no veto at all. Hermit Thrush (§9) belongs here too in its
    *tied* case — when the forest counts are equal it is effectively an each-player
    power, but it runs forced (only its strictly-fewer case is a true
    non-decision). The `opp_gained_food_count` / `opp_gained_card_count` ledger
    slots already exist in the choice vector and go unused by these powers.
    (`engine/powers/grants.py` all-players handlers, `engine/powers/multi_actor.py`
    die-draft handler, `engine/powers/tray_trade.py` fewest-forest handler.)
17. **Predator hunts run forced even when success feeds the opponent** (§22): with
    an opposing pink "predator succeeds" bird (§27b) in play, every successful hunt
    gifts the opponent 1 birdfeeder die per reactor, yet the hunt offers no
    `skip_optional` veto — the same veto principle as the all-players powers above,
    here conditioned on the opponent's board. The gate should appear only when at
    least one opposing §27b reactor is in play, with the accept row carrying 1
    gained tucked card and opponent-gained food equal to the reactor count (the
    success-case exchange; the hunt may still miss). Hunts repeated via Hooded
    Merganser (§23b) share the handler and would inherit the gate; the §10b dice
    predators would need the same treatment if their predator-success gap is ever
    fixed. (`engine/powers/predator_repeat.py` predator-hunt handler.)
18. **Green Heron's trade gate is redundant and should be removed** (§26b): the
    discard resolves before the gain, so the identity trade (discard a food, take
    the same food back) is always available and is state-identical to declining —
    the `skip_optional` gate duplicates a decision the mandatory
    `spend_food`/`gain_food` pair can already express as a no-op. The trade should
    run forced (when the player has any food), shrinking the overloaded
    `skip_optional` head's training surface. The mirror image of #16/#17: those add
    gates where a real decision is hidden; this removes one that hides nothing.
    (`engine/powers/tray_trade.py` wild-food-trade handler.)
19. **Skip rows leak into the `lay_egg` menu — only `skip_optional` should ever see
    a skip:** three call sites append an explicit skip row to a `lay_egg` decision,
    putting a whether-question in front of a head that should only answer *which*:
    the §27a pink lay-egg reactors (unconditional), §20c's lay-on-self step
    (unconditional — the printed "you may also"), and §14's extra egg
    (unconditional; part of #13). In all three the egg is a free gain, so the fix
    is the standard conditional-optionality shape: run the `lay_egg` pick forced
    (lone targets auto-resolve), and offer a `skip_optional` gate first only while
    the birds-without-eggs round goal is active. (`engine/reactors.py` pink
    lay-egg firing; `engine/powers/grants.py` tuck-then-lay-self handler;
    `engine/powers/dispatch.py` `lay_one_egg_on_nest` optional path.)

**Minor / cosmetic:**

20. **The general supply is tracked and checkable when it should be infinite:** by
    the printed rules the supply is unlimited — gaining from it can never fail and
    there is nothing to track. The engine instead keeps a 99-per-type counter that
    every supply-gain checks first, **silently granting nothing** (not even a
    partial amount) if the counter can't cover the full gain; player-paid food is
    (one trade aside) never returned, so the counter only drains. Practically
    unreachable at 99 per type in a 2-player game, but the correct model is no
    counter and no checks at all. (`state.py` `food_supply`; guards in
    `engine/powers/grants.py`, `engine/reactors.py`, `engine/powers/egg_trade.py`,
    `engine/powers/tray_trade.py`.)
21. **Pink egg-layers always exclude themselves** as a target even where the card
    says "a bird" rather than "another bird" (Bronzed/Brown-Headed Cowbird,
    Yellow-Billed Cuckoo) — no behavioral difference in the core set because those
    birds' own nests (none/platform) never match the bowl requirement.
22. **The unparsed-power fallback is currently unused:** all 180 core birds match a
    pattern, so the run-as-no-op fallback path exists only as future-proofing — no
    bird in this document relies on it.
