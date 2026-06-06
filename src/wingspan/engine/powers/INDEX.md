# engine.powers — Bird-power dispatch

Registry, dispatcher, and handler modules — one module per `EffectKind` family.
Three-step pattern: `EffectKind` variant in `cards.schema` → `@registry.handles`
decorator → handler function that receives the `Engine` and mutates `engine.state`.
Pink powers are dispatched from `engine.reactors`, not here. See `docs/BIRDS.md`
for the full implementation reference and per-bird coverage map.

## Modules

**`__init__.py`**

**`registry.py`** — The handler lookup table:
- `_HANDLERS: dict[EffectKind, Callable]` — populated by `@registry.handles`.
- `@handles(kind: EffectKind)` — decorator that registers a function as the
  handler for a given `EffectKind`. One handler per kind; re-registering raises.
- `handler_for(kind: EffectKind) -> Callable | None` — lookup; returns `None`
  for `UNIMPLEMENTED` (the caller logs the no-op).

**`dispatch.py`** — Top-level dispatch entry points:
- `dispatch_power(engine, agent, power: Power)` — looks up and calls the handler
  via `registry.handler_for`, logging unimplemented powers.
- `apply_effect(engine, agent, kind: EffectKind, **kwargs)` — lower-level call
  when the effect kind and params are known directly (used by multi-step powers).
- `lay_one_egg_on_nest(engine, agent, player_id)` — shared helper called by
  several handlers that involve laying a single egg.

**`grants.py`** — Handlers for direct food/card/egg grants: `GAIN_FOOD_SUPPLY`,
`GAIN_FOOD_BIRDFEEDER`, `GAIN_FOOD_TRAY`, `DRAW_CARDS`, `LAY_EGG_ON_THIS`,
`GAIN_CARD_FROM_HAND` and related variants. These are the most common power
family.

**`egg_trade.py`** — Handlers for egg-exchange powers: `TUCK_FOR_EGG`,
`CACHE_FOR_EGG`, `EGG_FOR_FOOD` and related patterns. Each handler offers an
`AcceptExchangeDecision` (optional-then-commit pattern) then executes the
exchange on acceptance.

**`multi_actor.py`** — Handlers for all-players / each-player effects:
`ALL_PLAYERS_GAIN_FOOD`, `EACH_PLAYER_DRAW_CARD`, etc. Iterates seats and
delegates to `dispatch.apply_effect` per player.

**`tray_trade.py`** — Handlers for tray-trade powers: `TRADE_FOOD_FOR_CARD`,
`DISCARD_FOOD_GAIN_CARD` and similar. Manages tray interactions and the
optional-then-commit accept/decline flow.

**`drafting.py`** — Handlers for card-draw and drafting variants: `DRAW_AND_KEEP`,
`DRAW_FROM_DECK`, `DRAW_TUCKED_CARD` and related. Presents `DrawSourceChoice`
or `BirdChoice` decisions as appropriate.

**`nest_aggregate.py`** — Handlers for nest-aggregate scoring powers:
`COUNT_BOWL_NESTS`, `COUNT_EGGS_IN_HABITAT`, etc. Read-only aggregations over
the board that award food or points.

**`predator_repeat.py`** — Handlers for predator-attack and repeat-action powers:
`PREDATOR`, `REPEAT_BROWN`, `ACTIVATE_ANOTHER`. `PREDATOR` presents a
`PlayerIdChoice` to select the target; `REPEAT_BROWN` re-dispatches another
bird's power.
