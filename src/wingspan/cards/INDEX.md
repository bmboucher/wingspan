# cards — Immutable card definitions

Schema models, enums, and the public API for all card data. Raw JSON is loaded
from `data/` and parsed via the `parse/` subpackage. All card objects are frozen
Pydantic models — they are never mutated after load.

## Modules

**`__init__.py`** — re-exports the public surface (`__all__` has 31 names,
including `Bird`, `BonusCard`, `EndRoundGoal`, `Food`, `Habitat`, `NestType`,
`PowerColor`, `EffectKind`, `Power`, `ALL_FOODS`, `ALL_HABITATS`, `food_index`,
`nest_matches`, `parse_power`, `load_all`, `power_coverage`, plus the raw-record
and ordering/index helpers). Import from here, not from `schema` or `parse`
directly.

**`schema.py`** — All enums and Pydantic models:
- `Habitat` (`FOREST`, `GRASSLAND`, `WETLAND`), `Food` (`INVERTEBRATE`, `SEED`,
  `FISH`, `FRUIT`, `RODENT`), `NestType`, `PowerColor`.
- `EffectKind` — 46 generic power-pattern variants (e.g. `GAIN_FOOD_SUPPLY`,
  `LAY_EGG_ON_THIS`, `DRAW_CARDS`, `TUCK_FROM_HAND`, `UNIMPLEMENTED`). The canonical
  list of what the engine can dispatch.
- `Power(color, effect: EffectKind, metadata)` — parsed IR for a single bird power.
- `Bird(id, name, scientific_name, color: PowerColor, points, nest: NestType,
  egg_limit, wingspan_cm, habitats: tuple[Habitat, ...], food_cost: BirdCost,
  flocking, predator, raw_power_text, power: Power, bonus_categories)` —
  frozen, the main card object. Note `food_cost` is `BirdCost`, an
  immutable 6-slot payment vector — not the `state.py` `FoodPool`.
- `BonusCard(id, name, condition, explanatory, vp_text, thresholds, per_bird_vp)`,
  `EndRoundGoal(id, description, category, tile_id)` — other card types.
- `BirdRecord`, `BonusRecord`, `GoalRecord` — raw-JSON input models with
  `Field(alias=...)`; each exposes a `.load()` method that returns the
  corresponding typed card object. Only `BirdRecord` declares
  `extra="allow"` (the JSON source has extra columns per printed card);
  `BonusRecord` and `GoalRecord` have no `model_config`.
- `ALL_FOODS`, `ALL_HABITATS`, `N_FOODS` — canonical orderings (append-only;
  part of the encoder's checkpoint format).
- `food_index(food) -> int`, `nest_matches(bird_nest, target) -> bool`.

**`lookup.py`** — Fuzzy name → card resolution for interactive entry points
(the `wingspan aid` session). Three-tier match over the catalog: exact
normalized name → normalized-prefix (unique or ambiguous) → `difflib` fuzzy
candidates. `find_birds(query)`, `find_bonus_cards(query)`, and
`find_goals(query)` (goals have no name, so matching is against each goal's
category tag + description) each return a list of candidates — empty (no
match), one (resolved), or many (ambiguous). `find_food(query)` is a plain
alias-table lookup (`Food | None`). `parse_bird_list(text)` splits a
comma-separated hand entry into per-token `find_birds` results.

## Subpackage

**`parse/`** — JSON loader and power-text parser.
See [`parse/INDEX.md`](parse/INDEX.md).
