"""Bird, bonus-card, and end-of-round-goal definitions loaded from wingsearch data.

We restrict to the core set (180 birds, 26 bonus cards, 16 goals). Each bird's
power text is parsed into a structured ``Power`` once at load time; the engine
later dispatches on those structures rather than re-parsing text.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from importlib import resources
from typing import Optional

from pydantic import BaseModel, ConfigDict

# ---------------------------------------------------------------------------
# Enums

class Habitat(str, Enum):
    FOREST = "forest"
    GRASSLAND = "grassland"
    WETLAND = "wetland"

ALL_HABITATS = [Habitat.FOREST, Habitat.GRASSLAND, Habitat.WETLAND]


class Food(str, Enum):
    INVERTEBRATE = "invertebrate"
    SEED = "seed"
    FISH = "fish"
    FRUIT = "fruit"
    RODENT = "rodent"

ALL_FOODS = [Food.INVERTEBRATE, Food.SEED, Food.FISH, Food.FRUIT, Food.RODENT]


class NestType(str, Enum):
    BOWL = "bowl"
    CAVITY = "cavity"
    GROUND = "ground"
    PLATFORM = "platform"
    STAR = "star"           # wildcard
    NONE = "none"           # birds with no nest icon (rare in core)


class PowerColor(str, Enum):
    BROWN = "brown"   # When activated (column power)
    WHITE = "white"   # When played (one-shot)
    PINK = "pink"     # Once between turns
    YELLOW = "yellow" # End of round (not in core set per data sample)
    NONE = "none"


# ---------------------------------------------------------------------------
# Power IR

class EffectKind(str, Enum):
    """A small library of generic power patterns. Any bird whose printed
    text does not match a known pattern gets ``UNIMPLEMENTED`` and is run
    as a no-op (with a logged warning the first time it triggers)."""
    GAIN_FOOD_SUPPLY = "gain_food_supply"
    GAIN_FOOD_BIRDFEEDER = "gain_food_birdfeeder"
    GAIN_FOOD_FROM_FEEDER_CHOICE = "gain_food_from_feeder_choice"
    GAIN_DIE_ANY = "gain_die_any"
    LAY_EGG_ON_THIS = "lay_egg_on_this"
    LAY_EGG_ANY = "lay_egg_any"
    DRAW_CARDS = "draw_cards"
    CACHE_FOOD = "cache_food"
    TUCK_FROM_HAND = "tuck_from_hand"
    TUCK_FROM_DECK = "tuck_from_deck"
    PLAY_ADDITIONAL_BIRD = "play_additional_bird"
    ALL_PLAYERS_GAIN_FOOD = "all_players_gain_food"
    ALL_PLAYERS_DRAW = "all_players_draw"
    DRAW_BONUS = "draw_bonus"
    DISCARD_EGG_FOR_WILD = "discard_egg_for_wild"
    UNIMPLEMENTED = "unimplemented"


class Effect(BaseModel):
    """Structured representation of a single power effect.

    Carriers are named, typed fields rather than a positional ``extra`` tuple.
    Each ``EffectKind`` documents which fields it consumes; unused fields stay
    ``None``.
    """

    model_config = ConfigDict(frozen=True)

    kind: EffectKind
    amount: int = 0
    food: Optional[Food] = None
    habitat: Optional[Habitat] = None
    raw_text: str = ""

    # --- typed carriers (replace the old untyped ``extra: tuple``) ----------
    keep_count: Optional[int] = None        # DRAW_BONUS_KEEP: # to keep
    max_wingspan_cm: Optional[int] = None   # PREDATOR_TUCK: hunt threshold
    nest: Optional[NestType] = None         # LAY_EGG_ALL_NEST, ALL_PLAYERS_LAY_ON_NEST
    food_a: Optional[Food] = None           # GAIN_FOOD_BIRDFEEDER_CHOICE: first option
    food_b: Optional[Food] = None           # GAIN_FOOD_BIRDFEEDER_CHOICE: second option


@dataclass
class Power:
    color: PowerColor
    effects: list[Effect] = field(default_factory=list)
    raw_text: str = ""


# ---------------------------------------------------------------------------
# Card dataclasses

@dataclass
class Bird:
    id: int
    name: str
    scientific_name: str
    color: PowerColor              # power color (a.k.a. trigger time)
    points: int
    nest: NestType
    egg_limit: int
    wingspan_cm: int
    habitats: tuple[Habitat, ...]  # one or more habitats the bird may live in
    food_cost: dict[Food, int]     # specific food required
    wild_food_cost: int            # number of "wild" (any food) icons
    total_food_cost: int           # sum (specific + wild)
    flocking: bool                 # food may be paid with any subset including wild
    predator: bool
    is_swift_start: bool
    raw_power_text: str
    power: Power
    # which bonus-card categories include this bird (set of bonus IDs)
    bonus_categories: tuple[str, ...] = ()

    @property
    def primary_habitat(self) -> Habitat:
        return self.habitats[0]


@dataclass
class BonusCard:
    id: int
    name: str
    condition: str
    explanatory: str
    vp_text: str
    # mapping: number_of_qualifying_birds -> vp; expanded from the printed text
    thresholds: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class EndRoundGoal:
    id: int
    description: str  # e.g. "[bird] in [forest]"
    # canonical category enum-string for easy dispatch
    category: str
    # placement payouts: 1st/2nd places (2P game)
    payouts_2p: tuple[int, int]


# ---------------------------------------------------------------------------
# Power-text parser. Tag conventions used by wingsearch:
#   [seed]       seed icon
#   [invertebrate] [fish] [fruit] [rodent] [nectar] [wild]
#   [card]       bird card
#   [egg]        egg
#   [die]        birdfeeder die
#   [forest] [grassland] [wetland]
#   [bird] [bowl] [cavity] [ground] [platform]

FOOD_TAGS = {
    "[invertebrate]": Food.INVERTEBRATE,
    "[seed]": Food.SEED,
    "[fish]": Food.FISH,
    "[fruit]": Food.FRUIT,
    "[rodent]": Food.RODENT,
}
HABITAT_TAGS = {
    "[forest]": Habitat.FOREST,
    "[grassland]": Habitat.GRASSLAND,
    "[wetland]": Habitat.WETLAND,
}

NUM_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5}


def _to_int(tok: str) -> Optional[int]:
    if tok.isdigit():
        return int(tok)
    return NUM_WORDS.get(tok.lower())


def _parse_food_token(tok: str) -> Optional[Food]:
    for tag, food in FOOD_TAGS.items():
        if tag == tok:
            return food
    return None


def parse_power(color: PowerColor, text: str) -> Power:
    """Best-effort parser. Recognises a small set of common patterns and
    returns ``UNIMPLEMENTED`` for everything else. Idempotent and safe to
    call once per bird at load time."""
    text = (text or "").strip()
    if not text:
        return Power(color=color, effects=[], raw_text="")

    effects: list[Effect] = []
    t = text.replace("—", "-").replace("“", '"').replace("”", '"')

    # Pattern 1: "Gain N [food] from the supply"
    m = re.search(r"Gain\s+(\d+|a|an|one|two|three)\s+(\[\w+\])\s+from the supply", t, re.I)
    if m and m.group(2) in FOOD_TAGS:
        n = _to_int(m.group(1)) or 1
        effects.append(Effect(kind=EffectKind.GAIN_FOOD_SUPPLY, amount=n, food=FOOD_TAGS[m.group(2)], raw_text=m.group(0)))

    # Pattern 1b: "Gain 1 [foodA] or [foodB] from the birdfeeder" -- e.g.
    # Indigo Bunting, Rose-Breasted Grosbeak, Western Tanager. Matched before
    # the more permissive Pattern 2 so the disjunction wording wins.
    m = re.search(
        r"Gain 1\s+(\[\w+\])\s+or\s+(\[\w+\])\s+from the birdfeeder",
        t, re.I,
    )
    if m and m.group(1) in FOOD_TAGS and m.group(2) in FOOD_TAGS:
        food_a = FOOD_TAGS[m.group(1)]
        food_b = FOOD_TAGS[m.group(2)]
        effects.append(Effect(
            kind=EffectKind.GAIN_FOOD_FROM_FEEDER_CHOICE,
            amount=1,
            food_a=food_a,
            food_b=food_b,
            raw_text=m.group(0),
        ))

    # Pattern 1c: "Gain 1 [die] from the birdfeeder" -- American Redstart.
    # Anchored so it only matches the unqualified [die] wording.
    m = re.match(r"^Gain 1 \[die\] from the birdfeeder\.?$", t, re.I)
    if m:
        effects.append(Effect(kind=EffectKind.GAIN_DIE_ANY, amount=1, raw_text=m.group(0)))

    # Pattern 2: "Gain N [food] from the birdfeeder"
    m = re.search(r"Gain\s+(\d+|a|an|one|two|three)\s+(\[\w+\])\s+from the birdfeeder", t, re.I)
    if m and m.group(2) in FOOD_TAGS:
        n = _to_int(m.group(1)) or 1
        effects.append(Effect(kind=EffectKind.GAIN_FOOD_BIRDFEEDER, amount=n, food=FOOD_TAGS[m.group(2)], raw_text=m.group(0)))

    # Pattern 3a: "Lay N [egg] on this bird"
    m = re.search(r"Lay\s+(\d+|a|an|one|two|three)\s+\[egg\] on this bird", t, re.I)
    if m:
        n = _to_int(m.group(1)) or 1
        effects.append(Effect(kind=EffectKind.LAY_EGG_ON_THIS, amount=n, raw_text=m.group(0)))

    # Pattern 3b: "Lay N [egg] on any bird"
    m = re.search(r"Lay\s+(\d+|a|an|one|two|three)\s+\[egg\] on any bird", t, re.I)
    if m:
        n = _to_int(m.group(1)) or 1
        effects.append(Effect(kind=EffectKind.LAY_EGG_ANY, amount=n, raw_text=m.group(0)))

    # Pattern 4: "Draw N [card]" (but not "All players draw")
    if not re.search(r"All players draw", t, re.I):
        m = re.search(r"Draw\s+(\d+|a|an|one|two|three)\s+\[card\]", t, re.I)
        if m:
            n = _to_int(m.group(1)) or 1
            effects.append(Effect(kind=EffectKind.DRAW_CARDS, amount=n, raw_text=m.group(0)))

    # Pattern 5: "Cache N [food] from the supply on this bird"
    m = re.search(r"Cache\s+(\d+|a|an|one|two|three)\s+(\[\w+\])\s+from the supply on this bird", t, re.I)
    if m and m.group(2) in FOOD_TAGS:
        n = _to_int(m.group(1)) or 1
        effects.append(Effect(kind=EffectKind.CACHE_FOOD, amount=n, food=FOOD_TAGS[m.group(2)], raw_text=m.group(0)))

    # Pattern 6: "Tuck 1 [card] from your hand behind this bird"
    m = re.search(r"Tuck\s+(\d+|a|an|one|two|three)\s+\[card\] from your hand behind this", t, re.I)
    if m:
        n = _to_int(m.group(1)) or 1
        effects.append(Effect(kind=EffectKind.TUCK_FROM_HAND, amount=n, raw_text=m.group(0)))

    # Pattern 7: "Play an additional bird in your [habitat]"
    m = re.search(r"Play an additional bird in your (\[\w+\])", t, re.I)
    if m and m.group(1) in HABITAT_TAGS:
        effects.append(Effect(kind=EffectKind.PLAY_ADDITIONAL_BIRD, habitat=HABITAT_TAGS[m.group(1)], raw_text=m.group(0)))

    # Pattern 8: "All players gain 1 [food] from the supply"
    m = re.search(r"All players gain\s+(\d+|a|an|one|two|three)\s+(\[\w+\])\s+from the supply", t, re.I)
    if m and m.group(2) in FOOD_TAGS:
        n = _to_int(m.group(1)) or 1
        effects.append(Effect(kind=EffectKind.ALL_PLAYERS_GAIN_FOOD, amount=n, food=FOOD_TAGS[m.group(2)], raw_text=m.group(0)))

    # Pattern 9: "All players draw 1 [card]"
    m = re.search(r"All players draw\s+(\d+|a|an|one|two|three)\s+\[card\]", t, re.I)
    if m:
        n = _to_int(m.group(1)) or 1
        effects.append(Effect(kind=EffectKind.ALL_PLAYERS_DRAW, amount=n, raw_text=m.group(0)))

    # Pattern 10: "Draw N bonus cards" -- e.g. Abbott's Booby
    m = re.search(r"Draw\s+(\d+|a|an|one|two|three)\s+bonus cards", t, re.I)
    if m:
        n = _to_int(m.group(1)) or 1
        effects.append(Effect(kind=EffectKind.DRAW_BONUS, amount=n, raw_text=m.group(0)))

    # Pattern 11: "Discard 1 [egg] from any of your other birds to gain N [wild] from the supply"
    m = re.search(
        r"Discard 1 \[egg\] from any of your other birds to gain\s+(\d+|a|an|one|two|three)\s+\[wild\] from the supply",
        t, re.I,
    )
    if m:
        n = _to_int(m.group(1)) or 1
        effects.append(Effect(kind=EffectKind.DISCARD_EGG_FOR_WILD, amount=n, raw_text=m.group(0)))

    if not effects:
        effects.append(Effect(kind=EffectKind.UNIMPLEMENTED, raw_text=text))

    return Power(color=color, effects=effects, raw_text=text)


# ---------------------------------------------------------------------------
# Loaders

def _parse_food_cost(record: dict) -> tuple[dict[Food, int], int]:
    cost: dict[Food, int] = {}
    for tag_label, food in [("Invertebrate", Food.INVERTEBRATE), ("Seed", Food.SEED),
                            ("Fish", Food.FISH), ("Fruit", Food.FRUIT), ("Rodent", Food.RODENT)]:
        v = record.get(tag_label)
        if isinstance(v, (int, float)) and v > 0:
            cost[food] = int(v)
    wild = record.get("Wild (food)")
    wild_n = int(wild) if isinstance(wild, (int, float)) and wild > 0 else 0
    return cost, wild_n


def _parse_nest(s: Optional[str]) -> NestType:
    if not s:
        return NestType.NONE
    s = s.lower().strip()
    for n in NestType:
        if n.value == s:
            return n
    if s == "wild":
        return NestType.STAR
    return NestType.NONE


def _bonus_categories_for_bird(record: dict, bonus_records: list[dict]) -> tuple[str, ...]:
    out: list[str] = []
    for b in bonus_records:
        if b.get("Set") != "core":
            continue
        name = b.get("Bonus card")
        if record.get(name) == "X":
            out.append(name)
    return tuple(out)


def load_all() -> tuple[list[Bird], list[BonusCard], list[EndRoundGoal]]:
    base = resources.files("wingspan.data")
    master = json.loads((base / "master.json").read_text(encoding="utf-8"))
    bonus_raw = json.loads((base / "bonus.json").read_text(encoding="utf-8"))
    goals_raw = json.loads((base / "goals.json").read_text(encoding="utf-8"))

    birds: list[Bird] = []
    for r in master:
        if r.get("Set") != "core":
            continue
        color_str = (r.get("Color") or "none").lower()
        try:
            color = PowerColor(color_str)
        except ValueError:
            color = PowerColor.NONE
        habitats: list[Habitat] = []
        for h, tag in [(Habitat.FOREST, "Forest"), (Habitat.GRASSLAND, "Grassland"), (Habitat.WETLAND, "Wetland")]:
            if r.get(tag) == "X":
                habitats.append(h)
        if not habitats:
            continue  # malformed record - skip
        cost, wild = _parse_food_cost(r)
        total = int(r.get("Total food cost") or sum(cost.values()) + wild)
        flocking = r.get("Flocking") == "X"
        predator = isinstance(r.get("Predator"), (int, float))
        pts = int(r.get("Victory points") or 0)
        eggs = int(r.get("Egg limit") or 0)
        wing = int(r.get("Wingspan") or 0)
        nest = _parse_nest(r.get("Nest type"))
        bird = Bird(
            id=int(r["id"]),
            name=r["Common name"],
            scientific_name=r.get("Scientific name") or "",
            color=color,
            points=pts,
            nest=nest,
            egg_limit=eggs,
            wingspan_cm=wing,
            habitats=tuple(habitats),
            food_cost=cost,
            wild_food_cost=wild,
            total_food_cost=total,
            flocking=flocking,
            predator=predator,
            is_swift_start=(r.get("Swift Start") == "X"),
            raw_power_text=r.get("Power text") or "",
            power=parse_power(color, r.get("Power text") or ""),
            bonus_categories=_bonus_categories_for_bird(r, bonus_raw),
        )
        birds.append(bird)

    bonuses: list[BonusCard] = []
    for b in bonus_raw:
        if b.get("Set") != "core":
            continue
        bonuses.append(BonusCard(
            id=int(b["id"]),
            name=b["Bonus card"],
            condition=b.get("Condition") or "",
            explanatory=b.get("Explanatory text") or "",
            vp_text=b.get("VP") or "",
            thresholds=_parse_bonus_thresholds(b.get("VP") or ""),
        ))

    goals: list[EndRoundGoal] = []
    for g in goals_raw:
        if g.get("Set") != "core":
            continue
        desc = g.get("Goal") or ""
        category = _goal_category(desc)
        # 2P payout values: from PDF rules, 1st=5, 2nd=2 by default
        goals.append(EndRoundGoal(
            id=int(g["id"]),
            description=desc,
            category=category,
            payouts_2p=(5, 2),
        ))
    return birds, bonuses, goals


def _parse_bonus_thresholds(vp_text: str) -> list[tuple[int, int]]:
    """Parse strings like '2 to 3 birds: 3[point]; 4+ birds: 7[point]'."""
    out: list[tuple[int, int]] = []
    for chunk in vp_text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        m = re.match(r"(\d+)\s*\+\s*birds?\s*:\s*(\d+)", chunk, re.I)
        if m:
            out.append((int(m.group(1)), int(m.group(2))))
            continue
        m = re.match(r"(\d+)\s*to\s*(\d+)\s*birds?\s*:\s*(\d+)", chunk, re.I)
        if m:
            out.append((int(m.group(1)), int(m.group(3))))
            continue
        m = re.match(r"(\d+)\s*birds?\s*:\s*(\d+)", chunk, re.I)
        if m:
            out.append((int(m.group(1)), int(m.group(2))))
    # sort by threshold ascending
    out.sort(key=lambda x: x[0])
    return out


def _goal_category(desc: str) -> str:
    """Compress goal descriptions to a short tag the engine can dispatch on."""
    d = desc.lower()
    if "[bird] in [forest]" in d: return "birds_forest"
    if "[bird] in [grassland]" in d: return "birds_grassland"
    if "[bird] in [wetland]" in d: return "birds_wetland"
    if "[egg] in [forest]" in d: return "eggs_forest"
    if "[egg] in [grassland]" in d: return "eggs_grassland"
    if "[egg] in [wetland]" in d: return "eggs_wetland"
    if "[egg] in [bowl]" in d: return "eggs_bowl"
    if "[egg] in [cavity]" in d: return "eggs_cavity"
    if "[egg] in [ground]" in d: return "eggs_ground"
    if "[egg] in [platform]" in d: return "eggs_platform"
    if "[bowl] [bird] with [egg]" in d: return "bowl_birds_with_eggs"
    if "[cavity] [bird] with [egg]" in d: return "cavity_birds_with_eggs"
    if "[ground] [bird] with [egg]" in d: return "ground_birds_with_eggs"
    if "[platform] [bird] with [egg]" in d: return "platform_birds_with_eggs"
    if "[card]" in d and "tucked" in d: return "tucked_cards"
    if "wingspan" in d and "less" in d: return "wingspan_under_30"
    if "wingspan" in d and "more" in d: return "wingspan_over_65"
    return "unknown:" + d[:30]


def power_coverage(birds: list[Bird]) -> tuple[int, int]:
    """Return ``(implemented, total)``."""
    impl = sum(
        1
        for b in birds
        if b.power.effects and not any(e.kind == EffectKind.UNIMPLEMENTED for e in b.power.effects)
    )
    return impl, len(birds)
