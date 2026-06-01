"""Record-field parsers.

These turn individual raw wingsearch fields (power color, habitats, food cost,
nest, bonus categories, VP thresholds, goal descriptions) into typed values.
They are called from the ``.load()`` methods on the ``schema`` input records.
"""

from __future__ import annotations

import re

from wingspan.cards import schema

# A few bonus cards are named differently in ``bonus.json`` than in the
# per-bird qualification columns of ``master.json`` — wingsearch sourced the
# two files from different printings, and Wingspan renamed several bonus cards
# between them. Map the bonus-card name to the ``master.json`` column that
# marks its qualifying birds so the lookup in ``bonus_categories_for_bird``
# resolves. Without this, "Omnivore Specialist" (column "Omnivore Expert")
# would tag zero birds and silently score 0 VP for the rest of the game.
_BONUS_COLUMN_ALIASES = {
    "Omnivore Specialist": "Omnivore Expert",
}


def parse_power_color(raw: str | None) -> schema.PowerColor:
    """Map a raw color string (e.g. ``"brown"``) to a ``PowerColor`` enum,
    defaulting to ``NONE`` for unknown or missing values."""
    lowered = (raw or "none").lower()
    try:
        return schema.PowerColor(lowered)
    except ValueError:
        return schema.PowerColor.NONE


def parse_habitats(record: schema.BirdRecord) -> list[schema.Habitat]:
    """Return the habitats the bird may live in, in canonical order."""
    out: list[schema.Habitat] = []
    for habitat, marker in [
        (schema.Habitat.FOREST, record.forest),
        (schema.Habitat.GRASSLAND, record.grassland),
        (schema.Habitat.WETLAND, record.wetland),
    ]:
        if marker == "X":
            out.append(habitat)
    return out


def parse_food_cost(record: schema.BirdRecord) -> schema.BirdCost:
    """Return the :class:`schema.BirdCost` printed on a bird record."""
    vec: list[int] = [0] * schema.N_FOODS
    for amount, food in [
        (record.invertebrate, schema.Food.INVERTEBRATE),
        (record.seed, schema.Food.SEED),
        (record.fish, schema.Food.FISH),
        (record.fruit, schema.Food.FRUIT),
        (record.rodent, schema.Food.RODENT),
    ]:
        if amount is not None and amount > 0:
            vec[schema.food_index(food)] = int(amount)
    wild = record.wild_food
    wild_n = int(wild) if wild is not None and wild > 0 else 0
    return schema.BirdCost(counts=(vec[0], vec[1], vec[2], vec[3], vec[4], wild_n))


def parse_nest(raw: str | None) -> schema.NestType:
    """Map a raw nest-type string to a :class:`NestType` enum."""
    if not raw:
        return schema.NestType.NONE
    normalized = raw.lower().strip()
    for nest_type in schema.NestType:
        if nest_type.value == normalized:
            return nest_type
    if normalized == "wild":
        return schema.NestType.STAR
    return schema.NestType.NONE


def bonus_categories_for_bird(
    record: schema.BirdRecord, bonuses: list[schema.BonusRecord]
) -> tuple[str, ...]:
    """Return the names of all core-set bonus cards whose category column
    is marked ``"X"`` on this bird record. Bonus-card column names are
    dynamic (one per bonus card) so they live in :attr:`model_extra`; a
    card renamed between the two source files is resolved through
    :data:`_BONUS_COLUMN_ALIASES`. The returned name is always the
    ``bonus.json`` card name, so it matches ``BonusCard.name`` downstream."""
    out: list[str] = []
    extras = record.model_extra or {}
    for bonus in bonuses:
        if bonus.card_set != "core":
            continue
        column = _BONUS_COLUMN_ALIASES.get(bonus.bonus_card, bonus.bonus_card)
        if extras.get(column) == "X":
            out.append(bonus.bonus_card)
    return tuple(out)


def parse_bonus_per_bird(vp_text: str) -> int | None:
    """Parse a per-bird payout like ``'2[point] per bird'`` into the VP each
    qualifying bird earns, or ``None`` for a tiered card.

    Per-bird and tiered payouts are mutually exclusive in the core set; a
    tiered string (``'... birds: N[point]'``) never matches this pattern."""
    match = re.search(r"(\d+)\s*\[point\]\s*per bird", vp_text, re.I)
    return int(match.group(1)) if match else None


def parse_bonus_thresholds(vp_text: str) -> tuple[tuple[int, int], ...]:
    """Parse strings like ``'2 to 3 birds: 3[point]; 4+ birds: 7[point]'``
    into ascending ``(min_count, vp)`` pairs. Per-bird cards (handled by
    :func:`parse_bonus_per_bird`) yield no thresholds."""
    out: list[tuple[int, int]] = []
    for chunk in vp_text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        match = re.match(r"(\d+)\s*\+\s*birds?\s*:\s*(\d+)", chunk, re.I)
        if match:
            out.append((int(match.group(1)), int(match.group(2))))
            continue
        match = re.match(r"(\d+)\s*to\s*(\d+)\s*birds?\s*:\s*(\d+)", chunk, re.I)
        if match:
            out.append((int(match.group(1)), int(match.group(3))))
            continue
        match = re.match(r"(\d+)\s*birds?\s*:\s*(\d+)", chunk, re.I)
        if match:
            out.append((int(match.group(1)), int(match.group(2))))
    out.sort(key=lambda pair: pair[0])
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
    "total [bird]": "total_birds",
    "sets of [egg][egg][egg] in [wetland][grassland][forest]": "egg_sets_3habitats",
}


def goal_category(desc: str) -> str:
    """Look up the scoring-engine tag for a goal description.
    Unknown descriptions return a synthetic ``"unknown:..."`` tag that
    scoring treats as zero points."""
    return _GOAL_CATEGORIES.get(desc, "unknown:" + desc[:30].lower())
