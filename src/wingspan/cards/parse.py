"""Card data loaders and parsers.

This module is responsible for turning the raw wingsearch JSON files in
``wingspan.data`` into the parsed card models defined in
:mod:`wingspan.cards.schema`. It owns two related responsibilities:

- :func:`load_all` reads the bundled JSON files, validates each core-set
  row into an input record (:class:`schema.BirdRecord`,
  :class:`schema.BonusRecord`, :class:`schema.GoalRecord`), and converts
  each record into its parsed counterpart by calling ``record.load()``.
- The various ``parse_*`` helpers turn individual raw fields (power text,
  food costs, habitat columns, VP thresholds, goal descriptions, ...)
  into typed values. These are called from the ``.load()`` methods on the
  input records.

Wingspan's printed power text uses a small inline-icon language
(``[seed]``, ``[bird]``, ``[forest]`` etc.). :func:`parse_power`
recognises the patterns we model and emits a structured ``Power``
consisting of one or more ``Effect`` records. Anything we don't
recognise becomes a single ``EffectKind.UNIMPLEMENTED`` effect so the
simulator can still run.
"""

from __future__ import annotations

import json
import re
from importlib import resources
from importlib.resources import abc

import pydantic

from wingspan.cards import schema

# ---------------------------------------------------------------------------
# Tag tables. Tag conventions used by wingsearch:
#   [seed]       seed icon
#   [invertebrate] [fish] [fruit] [rodent] [nectar] [wild]
#   [card]       bird card
#   [egg]        egg
#   [die]        birdfeeder die
#   [forest] [grassland] [wetland]
#   [bird] [bowl] [cavity] [ground] [platform]

FOOD_TAGS = {
    "[invertebrate]": schema.Food.INVERTEBRATE,
    "[seed]": schema.Food.SEED,
    "[fish]": schema.Food.FISH,
    "[fruit]": schema.Food.FRUIT,
    "[rodent]": schema.Food.RODENT,
}
HABITAT_TAGS = {
    "[forest]": schema.Habitat.FOREST,
    "[grassland]": schema.Habitat.GRASSLAND,
    "[wetland]": schema.Habitat.WETLAND,
}
NEST_TAGS = {
    "bowl": schema.NestType.BOWL,
    "cavity": schema.NestType.CAVITY,
    "ground": schema.NestType.GROUND,
    "platform": schema.NestType.PLATFORM,
}

_NUM_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5}


def parse_power(color: schema.PowerColor, text: str) -> schema.Power:
    """Best-effort parser. Recognises a small set of common patterns and
    returns ``UNIMPLEMENTED`` for everything else. Idempotent and safe to
    call once per bird at load time."""
    text = (text or "").strip()
    if not text:
        return schema.Power(color=color, effects=(), raw_text="")
    normalized = _normalize(text)
    effects = _extract_effects(normalized)
    if not effects:
        effects.append(
            schema.Effect(kind=schema.EffectKind.UNIMPLEMENTED, raw_text=text)
        )
    return schema.Power(color=color, effects=tuple(effects), raw_text=text)


###### PRIVATE #######

#### Text normalization ####


def _normalize(text: str) -> str:
    return text.replace("—", "-").replace("“", '"').replace("”", '"')


def _to_int(tok: str) -> int | None:
    if tok.isdigit():
        return int(tok)
    return _NUM_WORDS.get(tok.lower())


#### Pattern dispatch ####


def _extract_effects(t: str) -> list[schema.Effect]:
    """Apply each recognized pattern in turn, accumulating matched effects.

    Order matters: more specific patterns must run before less specific
    overlapping patterns (e.g. the "or"-disjunction birdfeeder pattern runs
    before the generic ``Gain N [food] from the birdfeeder``)."""
    effects: list[schema.Effect] = []
    for matcher in _PATTERN_MATCHERS:
        eff = matcher(t)
        if eff is not None:
            effects.append(eff)
    return effects


#### Pattern matchers ####

# Each matcher takes the normalized power text and returns an Effect (or None
# if the pattern does not match). They are independent — multiple may fire on
# the same text (no early return). Ordering is significant when patterns
# overlap (see _extract_effects).


def _m_gain_food_supply(t: str) -> schema.Effect | None:
    m = re.search(
        r"Gain\s+(\d+|a|an|one|two|three)\s+(\[\w+\])\s+from the supply",
        t,
        re.I,
    )
    if m and m.group(2) in FOOD_TAGS:
        n = _to_int(m.group(1)) or 1
        return schema.Effect(
            kind=schema.EffectKind.GAIN_FOOD_SUPPLY,
            amount=n,
            food=FOOD_TAGS[m.group(2)],
            raw_text=m.group(0),
        )
    return None


def _m_gain_food_from_feeder_choice(t: str) -> schema.Effect | None:
    # "Gain 1 [foodA] or [foodB] from the birdfeeder" -- Indigo Bunting etc.
    # Matched before the more permissive birdfeeder pattern.
    m = re.search(
        r"Gain 1\s+(\[\w+\])\s+or\s+(\[\w+\])\s+from the birdfeeder",
        t,
        re.I,
    )
    if m and m.group(1) in FOOD_TAGS and m.group(2) in FOOD_TAGS:
        return schema.Effect(
            kind=schema.EffectKind.GAIN_FOOD_FROM_FEEDER_CHOICE,
            amount=1,
            food_a=FOOD_TAGS[m.group(1)],
            food_b=FOOD_TAGS[m.group(2)],
            raw_text=m.group(0),
        )
    return None


def _m_gain_die_any(t: str) -> schema.Effect | None:
    # American Redstart. Anchored so it only matches the unqualified wording.
    m = re.match(r"^Gain 1 \[die\] from the birdfeeder\.?$", t, re.I)
    if m:
        return schema.Effect(
            kind=schema.EffectKind.GAIN_DIE_ANY,
            amount=1,
            raw_text=m.group(0),
        )
    return None


def _m_gain_food_birdfeeder(t: str) -> schema.Effect | None:
    m = re.search(
        r"Gain\s+(\d+|a|an|one|two|three)\s+(\[\w+\])\s+from the birdfeeder",
        t,
        re.I,
    )
    if m and m.group(2) in FOOD_TAGS:
        n = _to_int(m.group(1)) or 1
        return schema.Effect(
            kind=schema.EffectKind.GAIN_FOOD_BIRDFEEDER,
            amount=n,
            food=FOOD_TAGS[m.group(2)],
            raw_text=m.group(0),
        )
    return None


def _m_lay_egg_on_this(t: str) -> schema.Effect | None:
    m = re.search(
        r"Lay\s+(\d+|a|an|one|two|three)\s+\[egg\] on this bird",
        t,
        re.I,
    )
    if m:
        n = _to_int(m.group(1)) or 1
        return schema.Effect(
            kind=schema.EffectKind.LAY_EGG_ON_THIS,
            amount=n,
            raw_text=m.group(0),
        )
    return None


def _m_lay_egg_any(t: str) -> schema.Effect | None:
    m = re.search(
        r"Lay\s+(\d+|a|an|one|two|three)\s+\[egg\] on any bird",
        t,
        re.I,
    )
    if m:
        n = _to_int(m.group(1)) or 1
        return schema.Effect(
            kind=schema.EffectKind.LAY_EGG_ANY,
            amount=n,
            raw_text=m.group(0),
        )
    return None


def _m_draw_cards(t: str) -> schema.Effect | None:
    # Excludes "All players draw" (handled separately).
    if re.search(r"All players draw", t, re.I):
        return None
    m = re.search(r"Draw\s+(\d+|a|an|one|two|three)\s+\[card\]", t, re.I)
    if m:
        n = _to_int(m.group(1)) or 1
        return schema.Effect(
            kind=schema.EffectKind.DRAW_CARDS,
            amount=n,
            raw_text=m.group(0),
        )
    return None


def _m_cache_food(t: str) -> schema.Effect | None:
    m = re.search(
        r"Cache\s+(\d+|a|an|one|two|three)\s+(\[\w+\])\s+from the supply on this bird",
        t,
        re.I,
    )
    if m and m.group(2) in FOOD_TAGS:
        n = _to_int(m.group(1)) or 1
        return schema.Effect(
            kind=schema.EffectKind.CACHE_FOOD,
            amount=n,
            food=FOOD_TAGS[m.group(2)],
            raw_text=m.group(0),
        )
    return None


def _m_tuck_from_hand(t: str) -> schema.Effect | None:
    m = re.search(
        r"Tuck\s+(\d+|a|an|one|two|three)\s+\[card\] from your hand behind this",
        t,
        re.I,
    )
    if m:
        n = _to_int(m.group(1)) or 1
        return schema.Effect(
            kind=schema.EffectKind.TUCK_FROM_HAND,
            amount=n,
            raw_text=m.group(0),
        )
    return None


def _m_play_additional_bird(t: str) -> schema.Effect | None:
    m = re.search(r"Play an additional bird in your (\[\w+\])", t, re.I)
    if m and m.group(1) in HABITAT_TAGS:
        return schema.Effect(
            kind=schema.EffectKind.PLAY_ADDITIONAL_BIRD,
            habitat=HABITAT_TAGS[m.group(1)],
            raw_text=m.group(0),
        )
    return None


def _m_all_players_gain_food(t: str) -> schema.Effect | None:
    m = re.search(
        r"All players gain\s+(\d+|a|an|one|two|three)\s+(\[\w+\])\s+from the supply",
        t,
        re.I,
    )
    if m and m.group(2) in FOOD_TAGS:
        n = _to_int(m.group(1)) or 1
        return schema.Effect(
            kind=schema.EffectKind.ALL_PLAYERS_GAIN_FOOD,
            amount=n,
            food=FOOD_TAGS[m.group(2)],
            raw_text=m.group(0),
        )
    return None


def _m_all_players_draw(t: str) -> schema.Effect | None:
    m = re.search(
        r"All players draw\s+(\d+|a|an|one|two|three)\s+\[card\]",
        t,
        re.I,
    )
    if m:
        n = _to_int(m.group(1)) or 1
        return schema.Effect(
            kind=schema.EffectKind.ALL_PLAYERS_DRAW,
            amount=n,
            raw_text=m.group(0),
        )
    return None


def _m_draw_bonus(t: str) -> schema.Effect | None:
    m = re.search(
        r"Draw\s+(\d+|a|an|one|two|three)\s+bonus cards",
        t,
        re.I,
    )
    if m:
        n = _to_int(m.group(1)) or 1
        return schema.Effect(
            kind=schema.EffectKind.DRAW_BONUS,
            amount=n,
            raw_text=m.group(0),
        )
    return None


def _m_discard_egg_for_wild(t: str) -> schema.Effect | None:
    m = re.search(
        r"Discard 1 \[egg\] from any of your other birds to gain"
        r"\s+(\d+|a|an|one|two|three)\s+\[wild\] from the supply",
        t,
        re.I,
    )
    if m:
        n = _to_int(m.group(1)) or 1
        return schema.Effect(
            kind=schema.EffectKind.DISCARD_EGG_FOR_WILD,
            amount=n,
            raw_text=m.group(0),
        )
    return None


def _m_each_player_gains_die(t: str) -> schema.Effect | None:
    # Anna's / Ruby-Throated Hummingbird.
    m = re.search(
        r"Each player gains\s+(\d+|a|an|one|two|three)\s+\[die\]"
        r"\s+from the birdfeeder, starting with the player of your choice",
        t,
        re.I,
    )
    if m:
        n = _to_int(m.group(1)) or 1
        return schema.Effect(
            kind=schema.EffectKind.EACH_PLAYER_GAINS_DIE_CHOOSE_ORDER,
            amount=n,
            raw_text=m.group(0),
        )
    return None


def _m_all_players_lay_egg_on_nest(t: str) -> schema.Effect | None:
    # "All players lay 1 [egg] on any 1 [<nest>] bird.
    #  (You may lay 1 [egg] on 1 additional [<nest>] bird.)"
    # ``amount`` encodes the optional second-sentence extra for the active
    # player (0 when absent, 1 when present).
    m = re.search(
        r"All players lay\s+\d+\s+\[egg\]\s+on any\s+\d+\s+\[(bowl|cavity|ground|platform)\] bird\."
        r"(?:\s+You may lay\s+\d+\s+\[egg\]\s+on\s+\d+\s+additional)?",
        t,
        re.I,
    )
    if m:
        nest = NEST_TAGS[m.group(1).lower()]
        extra_for_self = 1 if "additional" in m.group(0).lower() else 0
        return schema.Effect(
            kind=schema.EffectKind.ALL_PLAYERS_LAY_EGG_ON_NEST,
            nest=nest,
            amount=extra_for_self,
            raw_text=m.group(0),
        )
    return None


def _m_draw_from_tray_all(t: str) -> schema.Effect | None:
    # Brant.
    m = re.search(r"Draw the (\d+) face-up \[card\] in the bird tray", t, re.I)
    if m:
        n = int(m.group(1))
        return schema.Effect(
            kind=schema.EffectKind.DRAW_FROM_TRAY_ALL,
            amount=n,
            raw_text=m.group(0),
        )
    return None


def _m_trade_wild(t: str) -> schema.Effect | None:
    # Green Heron.
    m = re.search(r"Trade 1 \[wild\] for any other type from the supply", t, re.I)
    if m:
        return schema.Effect(
            kind=schema.EffectKind.TRADE_WILD_FOOD,
            raw_text=m.group(0),
        )
    return None


def _m_fewest_forest_gains_die(t: str) -> schema.Effect | None:
    # Hermit Thrush.
    m = re.search(
        r"Player\(s\) with the fewest birds in their \[forest\] gain 1 \[die\] from birdfeeder",
        t,
        re.I,
    )
    if m:
        return schema.Effect(
            kind=schema.EffectKind.FEWEST_FOREST_GAINS_DIE,
            raw_text=m.group(0),
        )
    return None


def _m_play_additional_bird_here(t: str) -> schema.Effect | None:
    # House Wren.
    m = re.search(r"Play an additional bird in this bird.{1,4}s habitat", t, re.I)
    if m:
        return schema.Effect(
            kind=schema.EffectKind.PLAY_ADDITIONAL_BIRD_HERE,
            raw_text=m.group(0),
        )
    return None


def _m_draw_n_plus_one_draft(t: str) -> schema.Effect | None:
    # American Oystercatcher.
    m = re.search(r"Draw \[card\] equal to the number of players \+1", t, re.I)
    if m:
        return schema.Effect(
            kind=schema.EffectKind.DRAW_N_PLUS_ONE_DRAFT,
            raw_text=m.group(0),
        )
    return None


def _m_draw_bonus_keep(t: str) -> schema.Effect | None:
    m = re.search(
        r"Draw\s+(\d+|a|an|one|two|three)\s+new bonus cards and keep\s+(\d+|a|an|one|two|three)",
        t,
        re.I,
    )
    if m:
        n = _to_int(m.group(1)) or 1
        k = _to_int(m.group(2)) or 1
        return schema.Effect(
            kind=schema.EffectKind.DRAW_BONUS_KEEP,
            amount=n,
            keep_count=k,
            raw_text=m.group(0),
        )
    return None


def _m_lay_egg_all_nest(t: str) -> schema.Effect | None:
    # Bobolink etc.
    m = re.search(
        r"Lay\s+(\d+|a|an|one|two|three)\s+\[egg\]"
        r" on each of your birds with a \[(bowl|cavity|ground|platform)\] nest",
        t,
        re.I,
    )
    if m:
        n = _to_int(m.group(1)) or 1
        return schema.Effect(
            kind=schema.EffectKind.LAY_EGG_ALL_NEST,
            amount=n,
            nest=NEST_TAGS[m.group(2).lower()],
            raw_text=m.group(0),
        )
    return None


def _m_gain_all_food_feeder(t: str) -> schema.Effect | None:
    # Bald Eagle, Northern Flicker.
    m = re.search(r"Gain all (\[\w+\]) that are in the birdfeeder", t, re.I)
    if m and m.group(1) in FOOD_TAGS:
        return schema.Effect(
            kind=schema.EffectKind.GAIN_ALL_FOOD_FEEDER,
            food=FOOD_TAGS[m.group(1)],
            raw_text=m.group(0),
        )
    return None


def _m_tuck_from_deck_paid(t: str) -> schema.Effect | None:
    m = re.search(
        r"Discard 1 (\[\w+\]) to tuck\s+(\d+|a|an|one|two|three)\s+\[card\] from the deck behind this bird",
        t,
        re.I,
    )
    if m and m.group(1) in FOOD_TAGS:
        n = _to_int(m.group(2)) or 1
        return schema.Effect(
            kind=schema.EffectKind.TUCK_FROM_DECK_PAID,
            amount=n,
            food=FOOD_TAGS[m.group(1)],
            raw_text=m.group(0),
        )
    return None


def _m_predator_hunt(t: str) -> schema.Effect | None:
    m = re.search(
        r"Look at a \[card\] from the deck\. If less than\s+(\d+)\s*cm,"
        r" tuck it behind this bird\. If not, discard it",
        t,
        re.I,
    )
    if m:
        return schema.Effect(
            kind=schema.EffectKind.PREDATOR_HUNT,
            max_wingspan_cm=int(m.group(1)),
            raw_text=m.group(0),
        )
    return None


def _m_move_rightmost(t: str) -> schema.Effect | None:
    m = re.search(
        r"If this bird is to the right of all other birds in its habitat,"
        r" move it to another habitat",
        t,
        re.I,
    )
    if m:
        return schema.Effect(
            kind=schema.EffectKind.MOVE_BIRD_IF_RIGHTMOST,
            raw_text=m.group(0),
        )
    return None


def _m_repeat_brown(t: str) -> schema.Effect | None:
    # Gray Catbird, Northern Mockingbird.
    m = re.search(r"Repeat a brown power on another bird in this habitat", t, re.I)
    if m:
        return schema.Effect(
            kind=schema.EffectKind.REPEAT_BROWN_POWER,
            raw_text=m.group(0),
        )
    return None


def _m_repeat_predator(t: str) -> schema.Effect | None:
    # Hooded Merganser.
    m = re.search(r"Repeat 1 \[predator\] power in this habitat", t, re.I)
    if m:
        return schema.Effect(
            kind=schema.EffectKind.REPEAT_PREDATOR_POWER,
            raw_text=m.group(0),
        )
    return None


def _m_pink_lay_egg_on_nest(t: str) -> schema.Effect | None:
    m = re.search(
        r"When another player takes the .lay eggs. action,"
        r"\s*lay 1 \[egg\] on \w+ bird with a \[(bowl|cavity|ground|platform)\] nest",
        t,
        re.I,
    )
    if m:
        return schema.Effect(
            kind=schema.EffectKind.PINK_LAY_EGG_ON_NEST,
            nest=NEST_TAGS[m.group(1).lower()],
            raw_text=m.group(0),
        )
    return None


def _m_pink_predator_feeder(t: str) -> schema.Effect | None:
    m = re.search(
        r"When another player's \[predator\] succeeds, gain 1 \[die\] from the birdfeeder",
        t,
        re.I,
    )
    if m:
        return schema.Effect(
            kind=schema.EffectKind.PINK_PREDATOR_FEEDER,
            raw_text=m.group(0),
        )
    return None


_PATTERN_MATCHERS = (
    _m_gain_food_supply,
    _m_gain_food_from_feeder_choice,
    _m_gain_die_any,
    _m_gain_food_birdfeeder,
    _m_lay_egg_on_this,
    _m_lay_egg_any,
    _m_draw_cards,
    _m_cache_food,
    _m_tuck_from_hand,
    _m_play_additional_bird,
    _m_all_players_gain_food,
    _m_all_players_draw,
    _m_draw_bonus,
    _m_discard_egg_for_wild,
    _m_each_player_gains_die,
    _m_all_players_lay_egg_on_nest,
    _m_draw_from_tray_all,
    _m_trade_wild,
    _m_fewest_forest_gains_die,
    _m_play_additional_bird_here,
    _m_draw_n_plus_one_draft,
    _m_draw_bonus_keep,
    _m_lay_egg_all_nest,
    _m_gain_all_food_feeder,
    _m_tuck_from_deck_paid,
    _m_predator_hunt,
    _m_move_rightmost,
    _m_repeat_brown,
    _m_repeat_predator,
    _m_pink_lay_egg_on_nest,
    _m_pink_predator_feeder,
)


# ---------------------------------------------------------------------------
# Top-level loader


def load_all() -> (
    tuple[list[schema.Bird], list[schema.BonusCard], list[schema.EndRoundGoal]]
):
    """Read every core-set bird, bonus card, and end-of-round goal from the
    bundled JSON data. Returns three parallel lists in source order."""
    base = resources.files("wingspan.data")
    bird_records = _load_core_records(base / "master.json", schema.BirdRecord)
    bonus_records = _load_core_records(base / "bonus.json", schema.BonusRecord)
    goal_records = _load_core_records(base / "goals.json", schema.GoalRecord)
    birds = [b for b in (r.load(bonus_records) for r in bird_records) if b is not None]
    bonuses = [b.load() for b in bonus_records]
    goals = [g.load() for g in goal_records]
    return birds, bonuses, goals


def power_coverage(birds: list[schema.Bird]) -> tuple[int, int]:
    """Return ``(implemented, total)``. Birds with no power text are counted
    as implemented (there is nothing to model)."""
    impl = sum(
        1
        for b in birds
        if not any(e.kind == schema.EffectKind.UNIMPLEMENTED for e in b.power.effects)
    )
    return impl, len(birds)


# ---------------------------------------------------------------------------
# Record-field parsers (called from the ``.load()`` methods on
# ``schema.BirdRecord`` / ``schema.BonusRecord`` / ``schema.GoalRecord``)


def parse_power_color(raw: str | None) -> schema.PowerColor:
    """Map a raw color string (e.g. ``"brown"``) to a ``PowerColor`` enum,
    defaulting to ``NONE`` for unknown or missing values."""
    s = (raw or "none").lower()
    try:
        return schema.PowerColor(s)
    except ValueError:
        return schema.PowerColor.NONE


def parse_habitats(record: schema.BirdRecord) -> list[schema.Habitat]:
    """Return the habitats the bird may live in, in canonical order."""
    out: list[schema.Habitat] = []
    for h, v in [
        (schema.Habitat.FOREST, record.forest),
        (schema.Habitat.GRASSLAND, record.grassland),
        (schema.Habitat.WETLAND, record.wetland),
    ]:
        if v == "X":
            out.append(h)
    return out


def parse_food_cost(record: schema.BirdRecord) -> schema.BirdCost:
    """Return the :class:`schema.BirdCost` printed on a bird record."""
    vec: list[int] = [0] * schema.N_FOODS
    for v, food in [
        (record.invertebrate, schema.Food.INVERTEBRATE),
        (record.seed, schema.Food.SEED),
        (record.fish, schema.Food.FISH),
        (record.fruit, schema.Food.FRUIT),
        (record.rodent, schema.Food.RODENT),
    ]:
        if v is not None and v > 0:
            vec[schema.food_index(food)] = int(v)
    wild = record.wild_food
    wild_n = int(wild) if wild is not None and wild > 0 else 0
    return schema.BirdCost(counts=(vec[0], vec[1], vec[2], vec[3], vec[4], wild_n))


def parse_nest(s: str | None) -> schema.NestType:
    """Map a raw nest-type string to a :class:`NestType` enum."""
    if not s:
        return schema.NestType.NONE
    s = s.lower().strip()
    for n in schema.NestType:
        if n.value == s:
            return n
    if s == "wild":
        return schema.NestType.STAR
    return schema.NestType.NONE


def bonus_categories_for_bird(
    record: schema.BirdRecord, bonuses: list[schema.BonusRecord]
) -> tuple[str, ...]:
    """Return the names of all core-set bonus cards whose category column
    is marked ``"X"`` on this bird record. Bonus-card column names are
    dynamic (one per bonus card) so they live in :attr:`model_extra`."""
    out: list[str] = []
    extras = record.model_extra or {}
    for b in bonuses:
        if b.card_set != "core":
            continue
        if extras.get(b.bonus_card) == "X":
            out.append(b.bonus_card)
    return tuple(out)


def parse_bonus_thresholds(vp_text: str) -> tuple[tuple[int, int], ...]:
    """Parse strings like ``'2 to 3 birds: 3[point]; 4+ birds: 7[point]'``
    into ascending ``(min_count, vp)`` pairs."""
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
    out.sort(key=lambda x: x[0])
    return tuple(out)


# Exact-match table from a goal's raw description to the scoring-engine
# tag it dispatches on. Keys are the verbatim ``"Goal"`` strings from
# ``goals.json`` for the core set.
_GOAL_CATEGORIES: dict[str, str] = {
    "[bird] in [forest]": "birds_forest",
    "[bird] in [grassland]": "birds_grassland",
    "[bird] in [wetland]": "birds_wetland",
    "[egg] in [forest]": "eggs_forest",
    "[egg] in [grassland]": "eggs_grassland",
    "[egg] in [wetland]": "eggs_wetland",
    "[egg] in [bowl]": "eggs_bowl",
    "[egg] in [cavity]": "eggs_cavity",
    "[egg] in [ground]": "eggs_ground",
    "[egg] in [platform]": "eggs_platform",
    "[bowl] [bird] with [egg]": "bowl_birds_with_eggs",
    "[cavity] [bird] with [egg]": "cavity_birds_with_eggs",
    "[ground] [bird] with [egg]": "ground_birds_with_eggs",
    "[platform] [bird] with [egg]": "platform_birds_with_eggs",
}


def goal_category(desc: str) -> str:
    """Look up the scoring-engine tag for a goal description.
    Unknown descriptions return a synthetic ``"unknown:..."`` tag that
    scoring treats as zero points."""
    return _GOAL_CATEGORIES.get(desc, "unknown:" + desc[:30].lower())


###### PRIVATE #######


def _load_core_records[R: pydantic.BaseModel](
    path: abc.Traversable, model: type[R]
) -> list[R]:
    """Load and validate only the core-set rows from a wingsearch JSON file.

    Non-core rows are skipped before validation: other expansions use values
    (e.g. ``"Wingspan": "*"`` for variable-wingspan birds) that fall outside
    the core-set schema and would otherwise fail validation here."""
    return [
        model.model_validate(r)
        for r in json.loads(path.read_text(encoding="utf-8"))
        if r.get("Set") == "core"
    ]
