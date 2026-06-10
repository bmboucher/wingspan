# Caching Bird Audit

**Question:** Is low cached-food scoring a visibility gap or base-set economics?

---

## Caching birds and their EffectKind

25 of 180 core-set birds have a caching power. All are listed below in the order
they appear in `master.json`.

| Bird | EffectKind | Handler |
|------|------------|---------|
| Acorn Woodpecker | `GAIN_FOOD_FEEDER_MAY_CACHE` | `grants.py:793` |
| American Kestrel | `ROLL_NOT_IN_FEEDER_CACHE` | `grants.py:239` |
| Anhinga | `ROLL_NOT_IN_FEEDER_CACHE` | `grants.py:239` |
| Barn Owl | `ROLL_NOT_IN_FEEDER_CACHE` | `grants.py:239` |
| Black Skimmer | `ROLL_NOT_IN_FEEDER_CACHE` | `grants.py:239` |
| Blue Jay | `GAIN_FOOD_FEEDER_MAY_CACHE` | `grants.py:793` |
| Broad-Winged Hawk | `ROLL_NOT_IN_FEEDER_CACHE` | `grants.py:239` |
| Burrowing Owl | `ROLL_NOT_IN_FEEDER_CACHE` | `grants.py:239` |
| Carolina Chickadee | `CACHE_FOOD` | `grants.py:223` |
| Clark's Nutcracker | `GAIN_FOOD_FEEDER_MAY_CACHE` | `grants.py:793` |
| Common Merganser | `ROLL_NOT_IN_FEEDER_CACHE` | `grants.py:239` |
| Eastern Screech-Owl | `ROLL_NOT_IN_FEEDER_CACHE` | `grants.py:239` |
| Ferruginous Hawk | `ROLL_NOT_IN_FEEDER_CACHE` | `grants.py:239` |
| Juniper Titmouse | `CACHE_FOOD` | `grants.py:223` |
| Loggerhead Shrike | `PINK_GAIN_FOOD_CACHE` | `reactors.py:248` |
| Mississippi Kite | `ROLL_NOT_IN_FEEDER_CACHE` | `grants.py:239` |
| Mountain Chickadee | `CACHE_FOOD` | `grants.py:223` |
| Red-Bellied Woodpecker | `GAIN_FOOD_FEEDER_MAY_CACHE` | `grants.py:793` |
| Red-Breasted Nuthatch | `CACHE_FOOD` | `grants.py:223` |
| Red-Headed Woodpecker | `GAIN_FOOD_FEEDER_MAY_CACHE` | `grants.py:793` |
| Snowy Egret | `ROLL_NOT_IN_FEEDER_CACHE` | `grants.py:239` |
| Steller's Jay | `GAIN_FOOD_FEEDER_MAY_CACHE` | `grants.py:793` |
| White-Breasted Nuthatch | `CACHE_FOOD` | `grants.py:223` |
| White-Faced Ibis | `ROLL_NOT_IN_FEEDER_CACHE` | `grants.py:239` |
| Willet | `ROLL_NOT_IN_FEEDER_CACHE` | `grants.py:239` |

**Distribution:** 5 birds use `CACHE_FOOD` (chickadees/nuthatches, cache from supply),
11 use `ROLL_NOT_IN_FEEDER_CACHE` (raptors/fish-birds, conditional dice roll), 6 use
`GAIN_FOOD_FEEDER_MAY_CACHE` (jays/woodpeckers, feeder gain with optional cache decision),
and 1 uses `PINK_GAIN_FOOD_CACHE` (Loggerhead Shrike, reactive pink).

All caching birds cache exactly 1 token of a specific food type (seed or rodent or fish).

---

## Handler verification

### `CACHE_FOOD` — `grants.py:223–236`

```
pb.cached_food[eff.food] += eff.amount
```

Power text: "Cache 1 [seed] from the supply on this bird." The handler directly
increments `pb.cached_food` without drawing from supply. This is correct: the supply
is effectively infinite; there is no need to deduct from anything. No gaps.

### `ROLL_NOT_IN_FEEDER_CACHE` — `grants.py:239–301`

Rolls a number of dice equal to the dice currently outside the birdfeeder. If the
target food appears among the rolls, `pb.cached_food[eff.food] += eff.amount` is
applied (`grants.py:296`). On success, `reactors.trigger_pink_predator_success` fires
(`grants.py:299`) so Loggerhead Shrike's opponent can gain from the feeder. The veto
gate (`dispatch.offer_activation_veto`) is presented when opposing `PINK_PREDATOR_FEEDER`
birds are in play (`grants.py:259–274`). No gaps in the caching path itself.

### `GAIN_FOOD_FEEDER_MAY_CACHE` — `grants.py:793–847`

Two-step handler:
1. `actions.take_all_of_food(...)` takes up to 1 seed from the birdfeeder, crediting
   `player.food` (`grants.py:812`).
2. An `AcceptExchangeDecision` offers the player a cache-or-keep choice. On "cache",
   the food is moved from `player.food` to `pb.cached_food` (`grants.py:845–846`). On
   "keep" (SkipChoice), the food stays in `player.food`.

**Known gap — Loggerhead Shrike interaction:** `trigger_pink_gain_food_reactors` is
called in `actions.do_gain_food` (`actions.py:230`) AFTER all row powers complete. It
computes `gained_foods` as the net change in `player.food` since before the action
started (`actions.py:225–229`). If an opponent's Loggerhead Shrike reacts to a
`GAIN_FOOD_FEEDER_MAY_CACHE` bird where the active player chooses to **cache** (not
keep), the seed enters and then exits `player.food` before the diff is taken — net
change is zero, so Loggerhead Shrike does not fire. This is a minor rules inaccuracy
(caching counts as gaining for the purposes of triggering pink reactions in the physical
game), but it affects only the 2-player configuration where the opponent happens to have
Loggerhead Shrike and the active player chooses to cache rather than keep.

### `PINK_GAIN_FOOD_CACHE` — `reactors.py:204–230`, `reactors.py:248–254`

`trigger_pink_gain_food_reactors` (`reactors.py:204`) iterates over all opposing players'
boards, finds birds with `PINK_GAIN_FOOD_CACHE` whose `pink_fired` flag is clear, and
calls `_react_cache_from_supply` (`reactors.py:248`) which does:

```
pb.cached_food[eff.food] += eff.amount
```

The reaction fires only on `other_player` boards (`range(1, num_players)` at
`reactors.py:215`) — correctly modeling "when **another** player gains [food]". No gaps
in the core path.

**Secondary gap (same as above):** the reactor does not fire when an opponent's forest
bird power gains food via `GAIN_FOOD_FEEDER_MAY_CACHE` and then caches it, since the
net change to `player.food` is zero. Loggerhead Shrike should theoretically fire here;
it currently does not.

---

## Model visibility

### Per-slot cached-food stripe — `state_encode.py:291–305`

`_write_slot_continuous` writes a 5-wide block per board slot (one element per food
type in `cards.ALL_FOODS` order) starting at `layout._SLOT_MUT_CACHED` (`layout.py:370`):

```python
for i, food in enumerate(cards.ALL_FOODS):
    vec[mut + layout._SLOT_MUT_CACHED + i] = pb.cached_food[food] / layout._CACHED_FOOD_SCALE
```

- Covers all 5 food types individually.
- Normalized by `_CACHED_FOOD_SCALE = 6.0` (`layout.py:92`), treating 6+ cached tokens
  as the saturation point. In practice caching birds put at most 1–2 tokens on a bird
  per activation, so the signal is well within range.
- This is the most granular visibility the model gets: per-slot, per-food-type counts
  for every bird on both boards.

**Verdict: complete.** The model sees the exact cached-food composition of every slot.

### Board summary cached food — `state_encode.py:85–105`

`_summary_board` writes one scalar per habitat row aggregating the whole row's cached
food:

```python
sum(pb.cached_food.total() for pb in row) / layout._CACHED_FOOD_SCALE
```

(`state_encode.py:97–98`.) This is an aggregate total across food types, used in the
high-level board summary stripe. It is a coarser but redundant signal — the per-slot
stripe above is strictly more detailed.

**Verdict: present but coarse.** The model can see per-habitat cached totals as a
summary feature.

### `caches_food` card feature flag — `state_encode.py:449–461`

`_CACHE_EFFECT_KINDS` (`state_encode.py:450`) contains all four caching EffectKinds:

```python
_CACHE_EFFECT_KINDS: frozenset[cards.EffectKind] = frozenset([
    cards.EffectKind.CACHE_FOOD,
    cards.EffectKind.GAIN_FOOD_FEEDER_MAY_CACHE,
    cards.EffectKind.ROLL_NOT_IN_FEEDER_CACHE,
    cards.EffectKind.PINK_GAIN_FOOD_CACHE,
])
```

`_is_caching_bird` returns True for any bird with one of these effects
(`state_encode.py:460–461`), and the result is written to `_OFF_ATTR_CACHES_FOOD` in
the card feature matrix (`state_encode.py:254`). All 25 caching birds will have this
flag set.

In addition, `_accumulate_effect_exchange` in `state_encode.py:507–514` maps all four
caching kinds to `layout._EXCHANGE_CACHE_TO_GAIN` in the power-exchange vector, so the
card table also signals the expected cache output (1 token per activation).

**Verdict: complete.** All four caching kinds are covered in both the binary flag and
the exchange vector.

### End-game scoring — `scoring.py:391–420`

`final_scoring` (`scoring.py:391`) includes `player.total_cached` in the point total:

```python
total = bird_pts + bonus_pts + eggs + tucked + cached + round_goal
```

`Player.total_cached` (`state.py:279–280`) sums `pb.cached_food.total()` over every
bird on every row — 1 VP per cached food token.

**Bonus cards:** none of the 26 core-set bonus cards rewards cached food (confirmed
by inspecting all 26 `BonusCard.condition` strings; none references caching). The
bonus-card scoring path (`scoring.bonus_qualifying_count`) has no branch for cached food.

**Round goals:** none of the 16 core-set round-goal categories involves cached food
(confirmed by inspecting all 16 `EndRoundGoal.category` strings).

**Verdict:** cached food scores exactly 1 VP per token at end game, with no multiplier
from bonus cards or round goals. There is no bonus card or round-goal reward for
caching in the core set.

### Choice-row exchange encoding — `choice_encode.py:195`

The `AcceptExchangeDecision` for `GAIN_FOOD_FEEDER_MAY_CACHE` presents:
- Accept row: `PayCostChoice(paid_food=seed, paid_food_count=1, gained_cache_count=1)`
- Skip row: `SkipChoice`

In `_featurize_exchange`, the term `layout._EXCHANGE_CACHE_TO_GAIN: choice.gained_cache_count`
maps `gained_cache_count=1` into exchange slot 12 (`layout.py:250`) for the accept row.
The model therefore sees a non-zero `_EXCHANGE_CACHE_TO_GAIN` signal on the accept row
and zero on the skip row.

However, the consequence-pricing block in `_featurize_exchange` (`choice_encode.py:199–215`)
only prices card-count deltas against bonus-card counts and egg-count deltas against
round-goal VP. **There is no consequence pricing for `gained_cache_count`** — no bonus
delta and no round-goal delta attach to a caching decision. The model's only downstream
signal for the value of caching is the `total_cached` term inside `running_score`
(`scoring.py:423–439`), which the value head reads from game-state features, not from
the choice row.

**Verdict:** the exchange stripe signals "this choice caches 1 food" (`gained_cache_count=1`
on the accept row), but there is no per-choice bonus-delta or goal-delta pricing for
caching. The value head must learn from the `total_cached` feature in the state vector
that caching tokens translates to 1 VP each at game end.

---

## Conclusion

**Low cached-food scoring is almost entirely a base-set economics issue, not a
visibility gap.**

The model has full visibility at every relevant surface:

| Surface | Status |
|---------|--------|
| Per-slot cached-food stripe (5 types × all slots) | Complete — `state_encode.py:300–303` |
| Board summary cached-food aggregate per habitat | Present (coarser) — `state_encode.py:97–98` |
| `caches_food` card-attribute flag | Complete, all 4 kinds covered — `state_encode.py:450–454` |
| Power-exchange vector `_EXCHANGE_CACHE_TO_GAIN` | Complete, all 4 kinds — `state_encode.py:507–514` |
| Choice-row exchange `gained_cache_count` slot | Present on AcceptExchange rows — `choice_encode.py:195` |
| End-game scoring (1 VP per token) | Correct — `scoring.py:400–403` |
| Consequence pricing on choice rows | None — no bonus delta / goal delta for caching |

The base-set has no bonus cards or round goals that reward cached food. Every cached
token is worth exactly 1 VP at game end — the same as a tucked card, but harder to
reliably accumulate (the ROLL_NOT_IN_FEEDER_CACHE birds are conditional on dice outcomes,
the GAIN_FOOD_FEEDER_MAY_CACHE birds require giving up feeder food that could instead
stay in the player's supply). A rational agent should play caching birds primarily for
their point values and habitat access, treating the cache accumulation as a modest
end-game bonus rather than an optimization target.

**One minor implementation gap exists:** when a `GAIN_FOOD_FEEDER_MAY_CACHE` bird power
activates and the player chooses to cache (not keep), Loggerhead Shrike (opponent's
`PINK_GAIN_FOOD_CACHE` pink bird) does not fire. The net change to `player.food` is
zero (food flows through `player.food` momentarily then moves to `pb.cached_food`), so
the `gained_foods` diff in `actions.do_gain_food:225–229` sees no gain and
`trigger_pink_gain_food_reactors` does not trigger Loggerhead Shrike. This is a rules
inaccuracy but is unlikely to have a material effect on game outcomes (requires the
opponent to hold Loggerhead Shrike AND the active player to choose cache-not-keep).

**Summary:** implement nothing — the model already sees cached food completely. If
cached-food scores appear low in self-play games, that reflects the economic reality
of the base-set card set: caching is situationally valuable but not a primary strategy
axis, and no bonus multiplier amplifies it.
