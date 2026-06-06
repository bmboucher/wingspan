# cards — Immutable card definitions

Schema models, enums, and the public API for all card data. Raw JSON is loaded
from `data/` and parsed via the `parse/` subpackage. All card objects are frozen
Pydantic models — they are never mutated after load.

## Modules

**`__init__.py`** — re-exports the public surface: `Bird`, `BonusCard`, `EndRoundGoal`,
`Food`, `Habitat`, `NestType`, `PowerColor`, `EffectKind`, `Power`, `ALL_FOODS`,
`ALL_HABITATS`, `food_index`, `nest_matches`, `parse_power`, `load_all`,
`power_coverage`. Import from here, not from `schema` or `parse` directly.

**`schema.py`** — All enums and Pydantic models:
- `Habitat` (`FOREST`, `GRASSLAND`, `WETLAND`), `Food` (`INVERTEBRATE`, `SEED`,
  `FISH`, `FRUIT`, `RODENT`), `NestType`, `PowerColor`.
- `EffectKind` — ~60+ generic power-pattern variants (e.g. `GAIN_FOOD_SUPPLY`,
  `LAY_EGG_ON_THIS`, `DRAW_CARDS`, `TUCK_CARD`, `UNIMPLEMENTED`). The canonical
  list of what the engine can dispatch.
- `Power(color, effect: EffectKind, metadata)` — parsed IR for a single bird power.
- `Bird(name, habitat, wingspan, food_cost: FoodPool, egg_limit, nest_type,
  power_color, powers)` — frozen, the main card object.
- `BonusCard(name, explanatory, scoring_rule)`, `EndRoundGoal(name, category,
  explanatory)` — other card types.
- `BirdRecord`, `BonusRecord`, `GoalRecord` — raw-JSON input models with
  `Field(alias=...)` and `extra="allow"`; each exposes a `.load()` method that
  returns the corresponding typed card object.
- `ALL_FOODS`, `ALL_HABITATS`, `N_FOODS` — canonical orderings (append-only;
  part of the encoder's checkpoint format).
- `food_index(food) -> int`, `nest_matches(bird_nest, target) -> bool`.

## Subpackage

**`parse/`** — JSON loader and power-text parser.
See [`parse/INDEX.md`](parse/INDEX.md).
