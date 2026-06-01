"""Card data loaders and parsers.

Turns the bundled wingsearch JSON (``wingspan.data``) into the parsed card models
in :mod:`wingspan.cards.schema`, and parses printed bird power text into the
structured ``Power`` IR. The public surface is re-exported here so callers keep
writing ``from wingspan.cards import parse`` then ``parse.load_all`` /
``parse.parse_power`` / ``parse.parse_food_cost`` etc.

Submodules:

- ``tags``         — inline-icon tag tables + number-word parsing
- ``registry``     — the ordered matcher registries + ``@pattern`` decorators
- ``power``        — ``parse_power`` + normalization + dispatch
- ``matchers`` / ``pink_matchers`` — the power-text pattern matchers
- ``loader``       — ``load_all`` / ``power_coverage`` (the JSON loader)
- ``catalog``      — stable card -> dense-index maps for the RL encoder
- ``fields``       — record-field parsers (``parse_*``, ``goal_category``)
"""

# ``matchers`` and ``pink_matchers`` are imported for their ``@registry.pattern``
# side effects; importing ``matchers`` first keeps the general patterns ahead of
# the pink ones in the shared order-sensitive list.
from wingspan.cards.parse import matchers, pink_matchers
from wingspan.cards.parse.catalog import (
    bird_index,
    bonus_index,
    n_birds,
    n_bonus_cards,
)
from wingspan.cards.parse.fields import (
    bonus_categories_for_bird,
    goal_category,
    parse_bonus_per_bird,
    parse_bonus_thresholds,
    parse_food_cost,
    parse_habitats,
    parse_nest,
    parse_power_color,
)
from wingspan.cards.parse.loader import load_all, power_coverage
from wingspan.cards.parse.power import parse_power
from wingspan.cards.parse.tags import FOOD_TAGS, HABITAT_TAGS, NEST_TAGS

_ = (matchers, pink_matchers)

__all__ = [
    "FOOD_TAGS",
    "HABITAT_TAGS",
    "NEST_TAGS",
    "bird_index",
    "bonus_categories_for_bird",
    "bonus_index",
    "goal_category",
    "load_all",
    "n_birds",
    "n_bonus_cards",
    "parse_bonus_per_bird",
    "parse_bonus_thresholds",
    "parse_food_cost",
    "parse_habitats",
    "parse_nest",
    "parse_power",
    "parse_power_color",
    "power_coverage",
]
