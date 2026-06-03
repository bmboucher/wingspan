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
directly in the main working directory. The shape is fixed:

**plan → user approves → create worktree → implement → gate passes → commit →
report ready → human authorizes (deletes lock) → merge → done.**

**When this applies.** Any change that goes through plan-and-approve. Edit the
main working directory directly (no worktree) only for trivial edits — one-line
fixes, comment/doc tweaks — or when the user explicitly says to just edit in
place. When in doubt, use the worktree.

### Step 1 — Plan and get approval

Don't touch code until the user approves the plan (standard plan-mode flow).

### Step 2 — Create the worktree

```
bash scripts/create_worktree.sh <feature-slug>
```

This commits any pre-existing dirty state in `main`, creates the worktree at
`.claude/worktrees/<slug>` on branch `wt/<slug>` from the current local `HEAD`,
installs a **fresh `.venv`** inside the worktree (wall-clock ~30–60s, zero token
cost), and creates a **merge-auth lock** at `<slug>.lock` in the repo root.

Then switch the session into the worktree (this CLAUDE.md is the sanctioned
trigger for `EnterWorktree`):

```
EnterWorktree(path=".claude/worktrees/<slug>")
```

All subsequent edits run inside the worktree.

### Step 3 — Implement the change

Make all edits inside the worktree. The main working directory is untouched.

### Step 4 — Run the quality gate

```
bash scripts/quality_gate.sh
```

When run from inside the worktree (after `EnterWorktree`) this gates the
worktree's own code using the worktree's own `.venv` (installed by
`create_worktree.sh`). Do not proceed until the gate is clean.

### Step 5 — Commit and report ready

Commit all worktree changes on the feature branch (uncommitted work is not
merged). Then **stop**: report that the change is implemented and passing, and
tell the user to delete the lock file when they're ready to merge.

Do **not** merge on your own initiative. Do **not** delete the lock file.

### Step 6 — Merge (after human authorization)

The human deletes the lock file to authorize, then merges in one of two ways:

**Manual merge (human runs):**
```
bash scripts/merge_worktree.sh <slug>
```

**Automated merge (Claude subprocess via `claude -p`):**
```
bash scripts/auto_merge_worktree.sh <slug>
```

Both scripts handle: committing any new dirty state in `main`, squash-merging
the feature branch, running the quality gate on the merged result, committing,
pushing, and removing the worktree + branch. On any failure they reset `main` to
a clean state and report what needs fixing.

If the human asks you to merge during your session (after they've deleted the
lock), `ExitWorktree(action="keep")` first, then run `merge_worktree.sh` from
the main working directory.

## Merge-auth lock files

`create_worktree.sh` creates `<slug>.lock` in the repo root to block premature
merging. The human deletes it to authorize. `merge_worktree.sh` refuses to run
while it exists. Lock files are gitignored via `*.lock`.

**Claude must NEVER delete, move, or modify any `*.lock` file in the repo
root.** A lock being absent means a human reviewed and approved the merge;
Claude deleting one bypasses that review entirely. Claude may fail to create the
lock (the script handles it) — that is acceptable — but removing an existing
lock is not. This rule applies even if asked, even if the lock seems stale.

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

## Quality gate

Run from the current directory (worktree or repo root):

```
bash scripts/quality_gate.sh
```

Or pass an explicit path:

```
bash scripts/quality_gate.sh .claude/worktrees/<slug>
```

The gate runs five steps in order: `pyright` (strict) → `isort` → `black` →
`pyright` (post-format) → `pytest`. It stops at the first `pyright` failure so
you don't format broken code. Config lives in `pyproject.toml`
(`[tool.pyright]`, `[tool.black]`, `[tool.isort]`); no flags needed. `pyright`
is the globally-installed npm binary; formatters run as `python -m isort` /
`python -m black` via the target directory's own `.venv`.

For faster iteration while fixing a specific problem, use `--only`:

```
bash scripts/quality_gate.sh --only pyright          # type-check only (one pass)
bash scripts/quality_gate.sh --only pytest           # tests only
bash scripts/quality_gate.sh --only format           # isort + black only
bash scripts/quality_gate.sh --only pyright,pytest   # type-check + tests, skip format
# (same pattern with an explicit path)
bash scripts/quality_gate.sh .claude/worktrees/<slug> --only pyright
```

Always run the full gate (no `--only`) before committing. Do NOT call pyright,
pytest, isort, or black directly — use this script so the step ordering and
Python path are always correct.

Every change must pass the gate before it is considered finished. Do not
finalize while pyright reports any error — strict mode surfaces
`reportUnknownParameterType`, `reportMissingParameterType`,
`reportUnknownVariableType`, `reportUnnecessaryIsInstance`,
`reportMissingTypeArgument`. (`reportPrivateImportUsage = false` silences
torch's under-exporting stubs — don't re-enable it.)

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
  encode/                # state/choice tensor encoders for RL (package)
    layout.py            # feature dims, stripe offsets, normalization scales (the chain)
    state_encode.py      # encode_state / state_size + per-aspect state summaries
    choice_encode.py     # encode_choices + per-Choice featurizers + stripe fillers
  architecture.py        # ModelArchitecture + ActivationName (torch-free network topology descriptor)
  model.py               # PyTorch PolicyValueNet (built from a ModelArchitecture)
  train.py               # self-play + REINFORCE
  data/*.json            # wingsearch card data (bundled)

  cards/                 # immutable card definitions
    __init__.py          # re-exports the public surface (Bird, Food, parse_power, load_all, ...)
    schema.py            # enums, Effect/Power IR, Bird/BonusCard/EndRoundGoal models,
                         #   BirdRecord/BonusRecord/GoalRecord raw-JSON record models
    parse/               # JSON loader + power-text parser (package)
      tags.py            # inline-icon tag tables + number-word parsing
      registry.py        # ordered @pattern / @pink_pattern matcher registries
      power.py           # parse_power + normalization + dispatch
      matchers.py        # general power-text matchers (pink_matchers.py: reactive ones)
      loader.py          # load_all / power_coverage (the JSON loader)
      catalog.py         # stable card -> dense-index maps for the encoder
      fields.py          # record-field parsers (parse_*, goal_category)

  engine/                # mutation logic
    __init__.py          # re-exports Engine, Agent, print_coverage_report
    core.py              # Engine class, Agent protocol, turn loop, setup, ask plumbing
    actions.py           # do_play_bird / do_gain_food / do_lay_eggs / do_draw_cards
    powers/              # bird-power dispatch (package)
      registry.py        # _HANDLERS table + @registry.handles decorator + handler_for
      dispatch.py        # dispatch_power / apply_effect (registry lookup) / lay_one_egg_on_nest
      grants.py egg_trade.py multi_actor.py tray_trade.py drafting.py
      nest_aggregate.py predator_repeat.py   # @handles handlers grouped by family
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
    runmeta.py           # model_config.json (full topology, reconstitutable) + dated process_<stamp>.json sidecars (torch-free); read_model_config reader
    metrics.py           # ScoreBreakdown / FamilyCounts / EvalResult / IterationMetrics / GameOutcome (games.jsonl row)
    runstate.py          # RunState: the shared live snapshot the dashboard reads
    policy.py            # single-decision sample (collect) + greedy (eval)
    collect.py           # self-play game -> recorded steps + score breakdown
    learner.py           # length-bucketed REINFORCE + advantage norm (§3.3, §4.2a)
    evaluate.py          # paired-game strength vs random + 95% CI (§7)
    loop.py              # TrainingLoop orchestrator (collect/update/eval/checkpoint)
    theme.py             # palette + glyph constants ("wetland dawn")
    charts/              # custom rich renderables (package)
      geometry.py        # layout constants (gutter, inset width, ...)
      braille.py         # the 2x4-dot braille bitmap canvas
      text_helpers.py    # sparkline / eighth-block bar / human-count
      convergence_chart.py  # GettingBetterChart + its drawing helpers
      histogram.py       # FamilyHistogram
      insets.py          # the docked eval inset + narrow-panel strip
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
`ALL_DECISION_CLASSES`, then teach `encode/choice_encode.py` how to featurize it.

### Configurable network topology

The network shape is data-driven, not hard-coded. `architecture.ModelArchitecture`
(top-level, torch-free) is the single descriptor of the topology: per-block
hidden-width lists for the four blocks (`trunk_layers`, `choice_layers`,
`head_layers`, `value_layers`) plus the `activation`, `dropout`, `layernorm`, and
`card_embed_dim` handles. `PolicyValueNet` builds every block from it
(`_build_body` / `_build_readout`), `TrainConfig` mirrors the same fields flat
(so the configurator edits each independently) and assembles them via its `arch`
property, and `runmeta.write_model_config` / `read_model_config` serialize the
full descriptor to `model_config.json` so a run's network reads at a glance and
reconstitutes via `PolicyValueNet.from_model_config`.

Invariants to preserve when extending it: the trunk and choice encoder must end
at the same width `H` (the cross-field rule on `ModelArchitecture`, since their
outputs are concatenated to `2H` for the scorers); `ShapeKey` /
`architecture_key` must include any new field that changes a tensor shape (a
FRESH change — old checkpoints then restart cleanly via
`loop._architecture_matches`), while shape-preserving knobs like `activation` /
`dropout` stay out of it (REGIME, resumable). In the configurator, a per-layer
width list is a `LayersField` (type widths to set sizes, ←/→ to add/remove a
layer); scalar handles reuse the existing `IntField` / `FloatField` /
`ChoiceField`. The `arch` property is deliberately *not* named `architecture` —
that would shadow the imported `architecture` module in `TrainConfig`'s field
annotations and break pydantic's hint resolution.

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
2. Add a pattern matcher in `cards.parse.matchers` and decorate it with
   `@registry.pattern` (or `@registry.pink_pattern` in `pink_matchers` for a
   reactive one). Matchers are independent; **registration order = source
   order**, and ordering matters when patterns overlap (more specific first) —
   see `cards.parse.registry`.
3. Add a handler in the matching `engine.powers` submodule (grouped by family)
   and decorate it with `@registry.handles(EffectKind.X)`; the package
   `__init__` imports every handler submodule so the table self-populates. Pink
   (between-turn) effects are dispatched from `engine.reactors`, not from
   `apply_effect`; if the new effect is a pink reactor, register it there and
   have `apply_effect` treat it as a silent no-op.

Keep the `UNIMPLEMENTED` fallback in place — every core-set bird is
modelled today, but the fallback is what lets future expansion cards (or
any newly-discovered parser gap) stay non-fatal. Tests of specific bird
powers live in `tests/test_powers_*.py` and follow a per-power file
pattern.

### Public constants

Action / track / cost constants live at the top of `state.py`
(`ROUND_CUBES`, `ROW_SLOTS`, `BIRDFEEDER_DICE`, `STARTING_HAND_SIZE`,
`GAIN_FOOD_TRACK`, `LAY_EGGS_TRACK`, `DRAW_CARDS_TRACK`, `EGG_COSTS`,
`FULL_ROW_EGG_COST`, ...). Encoder feature dims, stripe offsets, and
normalisation scales live in `encode/layout.py` (the whole chain in one file).
Chart layout sizes live in `training/charts/geometry.py`. Don't sprinkle magic
numbers in function bodies — promote them.

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

- **Never delete, move, or modify `*.lock` files in the repo root.** These are
  human-authorization tokens for the merge workflow. Their absence signals
  approval; Claude removing one silently bypasses human review. See "Merge-auth
  lock files" above.
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
