# CLAUDE.md

Project-specific guidance for working in the Wingspan simulator. The global
style rules in `~/.claude/CLAUDE.md` already apply (Pydantic-first, absolute
module-qualified imports, Python 3.12+ syntax, no `Any`/payload bags); this
file documents the patterns the current codebase has settled on so future
changes stay consistent.

## What this project is

- Wingspan core-set simulator (180 birds, 26 bonus cards, 16 round goals,
  2-player automa-free) plus an RL training pipeline.
- The long-term goal is to run enough self-play training to answer
  analytical questions about the game — card power rankings, bonus-card
  value, food / habitat economy, opening-hand selection, etc. The current
  surface (manual CLI, random self-play with logs, REINFORCE training
  entry point) is the starting point, not the destination; design for
  scaling up training (more episodes, smarter algorithms, richer
  introspection) rather than for the minimum that runs today.
- Bird powers are modelled by a small library of generic `EffectKind`
  patterns. All 180 core-set birds are currently modelled; the parser also
  defines an `UNIMPLEMENTED` fallback effect (no-op at runtime, surfaced in
  the coverage report) so any future addition that doesn't yet have a
  pattern stays non-fatal.

## Run / test

```
pip install -e ".[dev]"                               # runtime + pyright/black/isort/pytest
python -m wingspan.cli manual                         # human vs random
python -m wingspan.cli random --log game.log          # watch a random game
python -m wingspan.train --device cuda --episodes 32  # one training cycle
python -m pytest tests/
```

PyTorch with CUDA is needed for training runs; everything else runs on CPU.

## Quality gate (run after every change, in this order)

Every change must clear this gate before it is considered finished. The
config lives in `pyproject.toml` (`[tool.pyright]`, `[tool.black]`,
`[tool.isort]`), so the editor (Pylance) and the CLI report identically and
no flags are needed.

1. **Type-check first — strict pyright must be clean.** Run `pyright` from
   the repo root; it reads `typeCheckingMode = "strict"` from `pyproject.toml`
   and checks `src/` and `tests/`. Do **not** finalize while it reports any
   error. (Strict surfaces rules the old default mode hid:
   `reportUnknownParameterType` / "Return type is unknown",
   `reportMissingParameterType`, `reportUnknownVariableType`,
   `reportUnnecessaryIsInstance`, `reportMissingTypeArgument`. `torch`'s
   under-exporting stubs are silenced via `reportPrivateImportUsage = false`
   — don't re-enable it.)
2. **Then format — isort, then black.** Only once types are clean:
   ```
   python -m isort src tests
   python -m black src tests
   ```
   `isort` runs first (its `profile = "black"` keeps the two compatible).
3. **Re-run pyright and the tests** to confirm formatting changed nothing:
   `pyright` then `python -m pytest tests/`.

Invocation note (Windows): call the formatters as `python -m black` /
`python -m isort`, not the bare `black` / `isort` shims — the console-script
`.exe` sometimes fails to install on this machine, but the module entry
points always work. `pyright` is invoked directly.

For type-checking patterns specific to this repo's scripted test agents (the
generic `Agent` protocol, when a `typing.cast` is required), see the
"Agent protocol" and "Test conventions" sections below.

## Package layout

```
src/wingspan/
  __init__.py            # version only
  cli.py                 # argparse entry points (main_manual, main_random)
  state.py               # GameState, Player, Board, FoodPool, PlayedBird, Birdfeeder, new_game
  decisions.py           # Decision[C] hierarchy + Choice hierarchy + MainAction
  encode.py              # state/choice tensor encoders for RL
  model.py               # PyTorch PolicyValueNet
  train.py               # self-play + REINFORCE
  data/*.json            # wingsearch card data (bundled)

  cards/                 # immutable card definitions
    __init__.py          # re-exports the public surface (Bird, Food, parse_power, load_all, ...)
    schema.py            # enums, Effect/Power IR, Bird/BonusCard/EndRoundGoal models,
                         #   BirdRecord/BonusRecord/GoalRecord raw-JSON record models
    parse.py             # JSON loader (load_all), per-field parsers, power-text parser

  engine/                # mutation logic
    __init__.py          # re-exports Engine, Agent, print_coverage_report
    core.py              # Engine class, Agent protocol, turn loop, setup, ask plumbing
    actions.py           # do_play_bird / do_gain_food / do_lay_eggs / do_draw_cards
    powers.py            # dispatch_power + apply_effect (one big EffectKind switch)
    reactors.py          # pink between-turn reactor hooks
    scoring.py           # score_round_goal, final_scoring
    helpers.py           # cost_meets, enumerate_payments — pure functions

  agents/
    __init__.py          # re-exports random_agent, cli_agent, mixed_agents
    base.py              # random_agent
    cli.py               # cli_agent + mixed_agents (hotseat helper)

tests/                   # pytest; tests prepend src/ to sys.path themselves
```

## Architectural patterns to preserve

### Pydantic v2 BaseModel for *all* structured data

Every record-shaped object is a `pydantic.BaseModel`. This includes the
immutable card data (`Bird`, `BonusCard`, `EndRoundGoal`, `Effect`, `Power`,
`BirdCost`), the mutable game state (`GameState`, `Player`, `Board`,
`FoodPool`, `PlayedBird`, `Birdfeeder`), every `Choice` / `Decision`
subclass, and the raw-JSON input records (`BirdRecord`, `BonusRecord`,
`GoalRecord`). Do not introduce dataclasses, `TypedDict`, or bare
`dict[str, ...]` for new records — extend the existing models or add a
sibling.

Patterns already in use, worth following:

- Frozen models (`model_config = ConfigDict(frozen=True)`) for immutable
  card data and IR; mutation-friendly models (defaults) for game state.
- `arbitrary_types_allowed=True` is used sparingly (e.g. `random.Random`
  on `GameState`, `cards.Bird` as a non-Pydantic identity in `Player.hand`
  is fine since `Bird` *is* a Pydantic model). Don't add it elsewhere.
- Raw wingsearch JSON rows are modelled as `*Record` BaseModels with
  `Field(alias="...")` for the printed column names and `extra="allow"`
  where the JSON carries dynamic columns (e.g. one column per bonus card on
  `BirdRecord`). Each record exposes a `.load()` that returns the parsed
  card model; the conversion helpers live in `cards.parse` and are imported
  lazily inside `.load()` to avoid a top-level cycle.
- Vector-shaped pools (`FoodPool`, `BirdCost`) expose a dict-like surface
  (`__getitem__`, `items()`, `total()`, `from_dict`, `from_specific`) so
  call sites read naturally; the underlying storage is a fixed-length list
  / tuple aligned to `cards.ALL_FOODS`. Keep that two-layer shape when
  adding new pool-like types — internal vector, dict-like external API.

### Imports: module-qualified, never symbol-level

Per the global rule. Concretely in this repo:

- `from wingspan import cards, decisions, state` then write `cards.Bird`,
  `state.Player`, `decisions.MainAction`. **Not** `from wingspan.cards
  import Bird`.
- Sibling engine submodules group together:
  `from wingspan.engine import actions, helpers, powers, reactors, scoring`.
- `from wingspan.engine import core as engine_core` is the convention when
  the bare name `core` would be ambiguous; otherwise `from wingspan.engine
  import core` is fine.
- The `__init__.py` files re-export the package's public surface so the
  `cards.Bird` / `engine.Engine` / `agents.random_agent` qualifications
  resolve through the package. Keep new public names listed in the
  package's `__all__` and imported in the `__init__.py`.
- Standard library too: `import typing` then `typing.Any`, not
  `from typing import Any`. `from __future__ import annotations` at the
  top of every module.

### Python 3.12+ syntax

- PEP 695 generics. `class Decision[C: Choice](pydantic.BaseModel): ...`
  and `def agent[C: decisions.Choice](...) -> C: ...`. No `TypeVar` /
  `Generic[T]` pair.
- `enum.StrEnum` for every enum (`Habitat`, `Food`, `NestType`, `PowerColor`,
  `EffectKind`, `MainAction`).
- `X | None` and `X | Y`, never `Optional[X]` / `Union[X, Y]`.
- `Annotated[list[C], Field(min_length=1)]` for declarative validation on
  collection fields (see `Decision.choices`). Reserve `@model_validator`
  for genuine cross-field invariants.

### The decision/choice system

Agents resolve decisions, not raw action ints. The shape is fixed:

- `Choice` is the abstract base. One subclass per *data shape*
  (`BirdChoice`, `HabitatChoice`, `FoodChoice`, `BoardTargetChoice`,
  `FoodPaymentChoice`, `SkipChoice`, `PayCostChoice`, `SetupChoice`, ...).
  Every option's data is reachable through named typed attributes — no
  opaque payload tuple, no `Any` carrier.
- `Decision[C: Choice]` is generic in the Choice subtype it accepts.
  Decisions that may be declined parameterize with a union including
  `SkipChoice` (e.g. `Decision[BoardTargetChoice | SkipChoice]`); consumers
  branch via `isinstance`.
- Every decision point is a concrete `Decision` subclass
  (`PlayBirdPickCardDecision`, `GainFoodPickDieDecision`, ...). Decisions
  that need extra context add typed fields directly
  (`SetupDecision.dealt_cards`, `SetupDecision.dealt_bonus`).
- `ALL_DECISION_CLASSES` is the stable iteration order for the encoder's
  decision-class one-hot stripe. Append new subclasses to the end so
  existing trained checkpoints stay aligned.

When adding a new decision point: define the Choice subclass first (or
reuse an existing one), then the `Decision[C]` subclass, then add it to
`ALL_DECISION_CLASSES`, then teach `encode.py` how to featurize it.

### Engine = orchestrator; sibling modules = free functions

`engine.core.Engine` owns the top-level turn loop, setup phase, and the
`ask` plumbing that routes a Decision through an Agent. Everything else
(main actions, power dispatch, pink reactors, scoring) lives in sibling
modules and is called as **free functions whose first argument is the
Engine**. The Engine does not have `_do_play_bird` / `_dispatch_power`
methods that wrap them — call `actions.do_play_bird(engine, agent)`
directly. New action / power / scoring logic should follow the same shape.

Import cycle handling: sibling engine modules need the Engine type for
annotations but Engine imports them at runtime. Use `if typing.TYPE_CHECKING:
from wingspan.engine import core` and annotate parameters as
`engine: "core.Engine"`. Don't move logic into `core.py` just to avoid this.

### The Agent protocol

`Agent` lives in `engine.core` and is a `typing.Protocol` with a generic
`__call__`:

```python
class Agent(typing.Protocol):
    def __call__[C: decisions.Choice](
        self, engine: "Engine", decision: decisions.Decision[C], /,
    ) -> C: ...
```

This keeps `Agent` non-generic at the use site (so `list[Agent]` and
`agent: Agent` parameters typecheck) while letting each call's return type
track the Decision's parameterization. New agent implementations live in
`wingspan.agents` and follow the same `def agent[C: decisions.Choice](
engine, decision) -> C:` shape — `random_agent` in `agents.base` is the
reference. The Engine routes opponent-prompting effects through
`engine.agent_for(player)`, so opponent powers don't need to thread agents
through every method signature.

### Bird powers: parser + dispatcher pair

Adding support for a new bird power is a three-step pattern:

1. Add a new `EffectKind` variant in `cards.schema`. If the effect needs
   data the existing carriers don't cover (`amount`, `food`, `habitat`,
   `keep_count`, `max_wingspan_cm`, `nest`, `food_a`, `food_b`), add a new
   typed carrier field to `Effect` — don't reach for a generic payload.
2. Add a pattern matcher in `cards.parse` and register it in
   `_PATTERN_MATCHERS`. Matchers are independent; ordering matters when
   patterns overlap (more specific first).
3. Add a handler in `engine.powers._HANDLERS` keyed by the new
   `EffectKind`. Pink (between-turn) effects are dispatched from
   `engine.reactors`, not from `apply_effect`; if the new effect is a pink
   reactor, register it there and have `apply_effect` treat it as a
   silent no-op.

Keep the `UNIMPLEMENTED` fallback in place — every core-set bird is
modelled today, but the fallback is what lets future expansion cards (or
any newly-discovered parser gap) stay non-fatal. Tests of specific bird
powers live in `tests/test_powers_*.py` and follow a per-power file
pattern.

### Public constants

Action / track / cost constants live at the top of `state.py`
(`ROUND_CUBES`, `ROW_SLOTS`, `BIRDFEEDER_DICE`, `STARTING_HAND_SIZE`,
`GAIN_FOOD_TRACK`, `LAY_EGGS_TRACK`, `DRAW_CARDS_TRACK`, `EGG_COSTS`,
`FULL_ROW_EGG_COST`, ...). Encoder normalisation scales live at the top of
`encode.py` as module-private constants. Don't sprinkle magic numbers in
function bodies — promote them.

### Per-turn scratch state

Cross-action turn state (e.g. `+1 extra play in this habitat` from House
Wren) lives on `GameState` as explicit fields (`turn_extra_plays`,
`turn_extra_play_habitat`) and is reset by `GameState.reset_turn_state()`
at the start of every turn. Don't introduce parallel scratch dicts —
extend `GameState` with a named typed field.

## Test conventions

- Tests prepend `src/` to `sys.path` themselves (see `test_smoke.py`); new
  tests should match so `pytest tests/` works from the repo root without
  install.
- One file per power (`tests/test_powers_*.py`); the cross-power smoke
  test is `test_smoke.py`. Encoder and food-payment helpers each have
  their own dedicated test file.
- The training-cycle smoke test (`test_train_one_epoch_cpu`) writes to
  `checkpoints/_test.pt`; don't add other tests that race for that path.

## Things to avoid

- Don't replace `cards.Food` / `cards.Habitat` / etc. enums with strings.
  The enums are `StrEnum`, so JSON serialisation already gives the string
  for free, and the type checker catches typos.
- Don't add `Decision`/`Choice` payload tuples or `Any`-typed context
  fields. Add a typed field to the subclass instead, or define a new
  Choice subclass.
- Don't bypass `Engine.ask`. It validates the agent's answer against the
  offered choices; constructing a Choice directly and acting on it skips
  that check.
- Don't add `model_config = ConfigDict(...)` to a Pydantic model unless
  you actually need a non-default behavior. A bare model is the preferred
  shape.
- Don't add `_do_*` wrapper methods on Engine that just delegate to
  `actions.do_*`. Call the free function directly.
