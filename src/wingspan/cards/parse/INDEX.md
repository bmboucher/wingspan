# cards.parse — JSON loader + power-text parser

Three-step pipeline: normalize raw power text → match against ordered pattern
registry → emit `EffectKind` IR stored on the `Power` model. The public entry
point is `power.parse_power`; `loader.load_all` drives the full card load from
`data/*.json`. See `docs/BIRDS.md` for the full matcher/handler reference.

## Modules

**`tags.py`** — Inline-icon tag tables and number-word parsing. `FOOD_TAGS`,
`HABITAT_TAGS`, `NEST_TAGS` map wingsearch icon strings to enum values.
`word_to_int(word) -> int` converts "one"/"two"/… to integers. Used by matchers
and field parsers throughout.

**`registry.py`** — Ordered pattern registries. `@pattern(regex)` and
`@pink_pattern(regex)` decorators append matchers to `MATCHERS` /
`PINK_MATCHERS` in source order. `first_match(text, matchers) -> Power | None`
runs them in order and returns the first hit. Order matters: more-specific
patterns must be registered before catch-all ones.

**`power.py`** — `parse_power(text: str, color: PowerColor) -> Power` is the
public API; it normalises whitespace/case, strips icon tags via `tags.py`, then
dispatches to the appropriate registry based on color. Returns an
`UNIMPLEMENTED` power for unrecognised text (never raises).

**`matchers.py`** — `@pattern`-decorated functions for brown/white/yellow
power text. Each matcher is a function `(match: re.Match) -> Power` that
constructs the `Power` IR. Grouped by EffectKind family; consult `docs/BIRDS.md`
for which birds each pattern covers.

**`pink_matchers.py`** — `@pink_pattern`-decorated functions for pink
(between-turns) reactive power text. Same shape as `matchers.py` but registered
in `PINK_MATCHERS`. Pink powers are dispatched from `engine.reactors`.

**`loader.py`** — `load_all() -> tuple[list[Bird], list[BonusCard], list[EndRoundGoal]]`
reads the three `data/*.json` files, validates each record via `*Record.load()`,
and returns typed card lists. `power_coverage() -> dict[str, int]` tallies
`UNIMPLEMENTED` powers per bird for the coverage report.

**`catalog.py`** — Stable card-to-dense-index maps for the encoder:
`bird_index(bird) -> int`, `bonus_index(bc) -> int`. Maps are built once on
first call and frozen; their order is part of the encoder checkpoint format
(append-only). Also exports `N_BIRDS`, `N_BONUS_CARDS`.

**`fields.py`** — Record-field parsers: `parse_food_cost(raw) -> FoodPool`,
`parse_habitat(raw) -> list[Habitat]`, `parse_nest_type(raw) -> NestType`,
`goal_category(name) -> str`. Called by `*Record.load()` methods in
`schema.py`.
