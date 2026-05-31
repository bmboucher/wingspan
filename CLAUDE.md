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

## Making changes: the worktree workflow

Substantive code changes are implemented in an isolated git **worktree**, not
directly in the main working directory, so the main checkout stays usable and
nothing lands on `main` until the user says so. The shape is fixed:

**plan → user approves → implement in a worktree → quality gate passes there →
return and ask to merge → merge into `main` → gate passes again → commit + push.**

**When this applies.** Any change that goes through plan-and-approve. Edit the
main working directory directly (no worktree) only for trivial edits — one-line
fixes, comment/doc tweaks — or when the user explicitly says to just edit in
place. When in doubt, use the worktree.

**1. Plan and get approval.** Don't touch code until the user approves the plan
(standard plan-mode flow).

**2. Commit any pre-existing work in `main` first.** Before creating the
worktree, run `git status` in the main working directory. If anything is
uncommitted (staged, unstaged, or untracked), commit it **all in one commit** on
the current branch with a descriptive message summarizing what it is —
autonomously, no need to ask. Never stash or discard. This guarantees the
worktree branches from a clean, current `main` and keeps the later merge clean.

**3. Create the worktree from the current local `HEAD`.** The worktree must
branch from the commit you just made — local `HEAD`, not `origin/main` (which is
often behind). The harness default base ref is `fresh` (origin), so branch
explicitly, then switch the session into the worktree with `EnterWorktree` using
its `path` (this CLAUDE.md instruction is the sanctioned trigger for that tool):

```
git worktree add .claude/worktrees/<name> -b wt/<feature-slug> HEAD
```

then `EnterWorktree(path=".claude/worktrees/<name>")`. All subsequent edits and
commands now run inside the worktree.

**4. Implement the change.** Make all edits in the worktree. The main working
directory is untouched, so the user can keep working there.

**5. Run the quality gate — in the worktree.** Run the full gate from
**"Quality gate"** below (pyright → isort → black → pyright + `pytest`) from the
worktree root. Because the tests add their own `src/` to `sys.path` via
`__file__` and pyright/black/isort resolve `src`/`tests` relative to the
invocation directory, the gate checks and formats the worktree's copy. (`.venv/`
is gitignored and is **not** copied into the worktree, so the gate relies on the
project venv at `C:\Repos\wingspan\.venv` — the same one the main dir uses; if a
tool isn't found from the worktree, invoke it via that venv's `Scripts\` or
activate it first.) Do not return until the gate is clean — fix every failure
inside the worktree.

**6. Commit, then return and ask before merging.** Once the gate is green,
commit all worktree changes on the feature branch (uncommitted work is not
merged). Then stop: report that the change is implemented and passing in the
worktree, and ask the user when they're ready to merge into `main`. Do **not**
merge on your own initiative.

**7. On the user's go-ahead, merge into `main`.**

- a. `ExitWorktree` with `action: "keep"` to return the session to the main
  working directory with the feature branch left intact.
- b. **Re-check `main` for uncommitted changes** (the user may have kept working
  while you did). If any exist, commit them in one descriptive commit first —
  same rule as step 2 — before merging.
- c. Bring the feature branch into `main` with a squash merge:
  `git merge --squash wt/<feature-slug>`. Resolve any merge conflicts in the
  working tree, preserving both intents; if a conflict is genuinely ambiguous,
  surface it to the user instead of guessing.
- d. Re-run the full quality gate in the main working directory (black may
  reformat the merged result). It must be clean before you commit.
- e. `git add -A` and commit the merged result with a descriptive message
  covering the change, then `git push` to `origin/main`.
- f. Clean up the now-merged worktree and branch:
  `git worktree remove .claude/worktrees/<name>` and
  `git branch -D wt/<feature-slug>` (squash merges aren't recorded as merges, so
  `-D` is needed). If anything in 7c–7e fails, leave the worktree in place for
  inspection and tell the user.

## Run / test

```
pip install -e ".[dev]"                               # runtime + pyright/black/isort/pytest
python -m wingspan.cli manual                         # human vs random
python -m wingspan.cli random --log game.log          # watch a random game
python -m wingspan.train --episodes 32                # one (legacy) training cycle
python -m wingspan.training --device cpu              # live training dashboard ("FLYWAY CONTROL")
python -m wingspan.training --config                  # interactive configurator ("FLIGHT PLAN")
python -m pytest tests/
```

Training is CPU-only — collection fans out across worker processes and the
gradient update is small, so no GPU is required (CUDA still works for one-off
experiments but is not the supported path).
`python -m wingspan.train` is the original minimal REINFORCE cycle;
`python -m wingspan.training` (the `wingspan.training` package, console script
`wingspan-dashboard`) is the `top`-style live training + monitoring app that
implements the TRAINING.md Phase-0/1 program (length-bucketed update, advantage
normalization, paired eval vs random, resumable checkpoints) behind a `rich`
dashboard. Collection is fastest on `--device cpu` (TRAINING.md §1.4).

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

  training/              # live training + monitoring dashboard ("FLYWAY CONTROL")
    __main__.py / app.py # entry point: argparse (+ --config) -> worker thread + rich.Live loop
    config.py            # TrainConfig (self-describing hyperparameters, §5.1)
    artifacts.py         # shared on-disk filenames (LAST/BEST/OPPONENT ckpt, metrics+games logs, model_config/process json)
    runmeta.py           # model_config.json + dated process_<stamp>.json sidecar writers (torch-free)
    metrics.py           # ScoreBreakdown / FamilyCounts / EvalResult / IterationMetrics / GameOutcome (games.jsonl row)
    runstate.py          # RunState: the shared live snapshot the dashboard reads
    policy.py            # single-decision sample (collect) + greedy (eval)
    collect.py           # self-play game -> recorded steps + score breakdown
    learner.py           # length-bucketed REINFORCE + advantage norm (§3.3, §4.2a)
    evaluate.py          # paired-game strength vs random + 95% CI (§7)
    loop.py              # TrainingLoop orchestrator (collect/update/eval/checkpoint)
    theme.py             # palette + glyph constants ("wetland dawn")
    charts.py            # braille convergence chart, family histogram, sparklines
    dashboard.py         # the five-band Layout + per-region renderers
    configure/           # interactive "FLIGHT PLAN" configurator (python -m wingspan.training --config)
      fields.py          # FieldSpec hierarchy + FIELD_SPECS + read/format/commit/nudge
      runs.py            # RunSummary + inspect_run / archive_run / clear_run / list_archives
      state.py           # ConfiguratorState + Mode/Outcome/ConfirmPrompt value-objects
      keys.py            # cross-platform raw single-key reader (msvcrt / termios), non-blocking
      screen.py          # the rich Layout + per-region renderers + the modal
      controller.py      # run_configurator Live loop + console-free build_initial_state / dispatch

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
