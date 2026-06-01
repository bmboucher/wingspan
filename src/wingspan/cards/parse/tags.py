"""Inline-icon tag tables and number-word parsing shared by the power matchers.

Wingsearch power text uses ``[seed]``/``[forest]``/``bowl`` style tokens; these
tables map them to the typed ``schema`` enums. ``to_int`` resolves a count token
(a digit or a number-word like "two").
"""

from __future__ import annotations

from wingspan.cards import schema

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


def to_int(tok: str) -> int | None:
    """Parse a number token (digit or number-word) to an int, or None."""
    if tok.isdigit():
        return int(tok)
    return _NUM_WORDS.get(tok.lower())
