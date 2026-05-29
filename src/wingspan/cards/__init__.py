"""Bird, bonus-card, and end-of-round-goal definitions loaded from wingsearch data.

Public surface is re-exported so callers can keep writing
``from wingspan.cards import Bird, Food, parse_power, load_all, ...``.
Internally the package is split into two modules:

- ``schema`` — enums, ``Effect``/``Power`` IR, all Pydantic models
  (``Bird``/``BonusCard``/``EndRoundGoal`` parsed cards plus
  ``BirdRecord``/``BonusRecord``/``GoalRecord`` raw JSON inputs)
- ``parse``  — JSON file loader (``load_all``), record-field parsers, and
  the bird power-text parser
"""

from wingspan.cards.parse import (
    FOOD_TAGS,
    HABITAT_TAGS,
    NEST_TAGS,
    load_all,
    parse_power,
    power_coverage,
)
from wingspan.cards.schema import (
    ALL_FOODS,
    ALL_HABITATS,
    N_FOODS,
    Bird,
    BirdCost,
    BirdRecord,
    BonusCard,
    BonusRecord,
    Effect,
    EffectKind,
    EndRoundGoal,
    Food,
    GoalRecord,
    Habitat,
    NestType,
    Power,
    PowerColor,
    food_index,
)

__all__ = [
    "ALL_FOODS",
    "ALL_HABITATS",
    "Bird",
    "BirdCost",
    "BirdRecord",
    "BonusCard",
    "BonusRecord",
    "Effect",
    "EffectKind",
    "EndRoundGoal",
    "FOOD_TAGS",
    "Food",
    "GoalRecord",
    "HABITAT_TAGS",
    "Habitat",
    "NEST_TAGS",
    "N_FOODS",
    "NestType",
    "Power",
    "PowerColor",
    "food_index",
    "load_all",
    "parse_power",
    "power_coverage",
]
