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
  surface (manual CLI, configurable self-play matchups, the FLIGHT PLAN
  training dashboard, round-robin tournaments between trained runs, cloud
  runs) is the starting point, not the destination; design for scaling up
  training (more episodes, smarter algorithms, richer introspection) rather
  than for the minimum that runs today.
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

While iterating, run individual sections with arguments passed through to the
underlying tool — e.g. `bash scripts/quality_gate.sh --pytest
tests/test_smoke.py` for a single test file. See "Quality gate" below for the
full argument reference. Always finish with the full gate (no section flags).

If the gate exits `2`, or any workflow script itself malfunctions, that is an
infrastructure problem, not a code problem — **stop and ask the user to fix
it** (see "Script failures: stop, don't circumvent" below).

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
the feature branch, refreshing main's `.venv` (`pip install -e ".[dev]"`) when
the merge changed `pyproject.toml`, running the quality gate on the merged
result, committing, pushing, and removing the worktree + branch. On any failure
they reset `main` to a clean state and report what needs fixing.

If the human asks you to merge during your session (after they've deleted the
lock), `ExitWorktree(action="keep")` first, then run `merge_worktree.sh` from
the main working directory.

### Workflow script reference

**`bash scripts/create_worktree.sh <feature-slug>`** — run from the repo root.
Takes exactly one argument, the feature slug. Commits any dirty state in
`main`, creates `.claude/worktrees/<slug>` on branch `wt/<slug>` from local
`HEAD`, installs a fresh `.venv` inside the worktree, and writes the merge-auth
lock `<slug>.lock` in the repo root. Fails without touching anything if the
worktree directory or branch already exists.

**`bash scripts/quality_gate.sh [target-dir] [--pyright [args…]]
[--format [paths…]] [--pytest [args…]]`** — the only sanctioned way to run
pyright / isort / black / pytest. Full reference in "Quality gate" below.

**`bash scripts/merge_worktree.sh <feature-slug>`** — run from the main working
directory (`ExitWorktree(action="keep")` first if the session is inside the
worktree), and only after the human has deleted `<slug>.lock`. Squash-merges
`wt/<slug>` into `main`, refreshes main's `.venv` if the merge changed
`pyproject.toml`, runs the full gate on the merged result, commits, pushes, and
removes the worktree + branch; on any failure it resets `main` clean. Exit
codes:

| Exit | Meaning | What Claude does |
|------|---------|------------------|
| 0 | merged, pushed, cleaned up | report done |
| 1 | merge-auth lock still present | stop — human authorization missing |
| 2 | merge conflicts | fix in the worktree, commit there, retry |
| 3 | gate failed on the merged result | fix in the worktree, commit there, retry |
| 4 | preflight failure (worktree/branch missing, or worktree has uncommitted changes) | commit the worktree work if that is the cause; otherwise stop and report |
| 5 | gate or venv refresh could not run (infrastructure failure) | **stop — ask the user to fix the environment** |

**`bash scripts/auto_merge_worktree.sh <feature-slug>`** — fully automated
variant the *human* runs: loops `merge_worktree.sh`, spawning `claude -p`
subprocesses to fix conflicts / gate failures, up to 5 attempts. Requires the
lock to be already deleted and `claude` on PATH. Stops immediately (no
subprocess) on exit 1 (lock present) or exit 5 (infrastructure failure).

## Merge-auth lock files

`create_worktree.sh` creates `<slug>.lock` in the repo root to block premature
merging. The human deletes it to authorize. `merge_worktree.sh` refuses to run
while it exists. Lock files are gitignored via `*.lock`.

**Claude must NEVER delete, move, or modify any `*.lock` file in the repo
root.** A lock being absent means a human reviewed and approved the merge;
Claude deleting one bypasses that review entirely. Claude may fail to create the
lock (the script handles it) — that is acceptable — but removing an existing
lock is not. This rule applies even if asked, even if the lock seems stale.

## Script failures: stop, don't circumvent

The workflow scripts above are the **only** sanctioned interface to the build
tools and the merge process. Distinguish two kinds of failure:

- **Genuine check failures** (gate exit `1`: pyright type errors, failing
  tests) are normal feature work. Fix the code, rerun the gate.
- **Infrastructure failures** (gate exit `2`; merge exit `5`; a script crashing
  with a bash error; the venv install failing; `pyright` or `claude` missing
  from PATH; CRLF-mangled scripts) mean the tooling itself is broken. **Stop
  working immediately, show the user the failing output verbatim, and ask them
  to fix the script or environment.** Do not continue toward a merge until
  they have.

Never do any of the following to get past a failing script:

- Run `pyright`, `pytest`, `isort`, or `black` directly instead of through
  `quality_gate.sh`. Its section flags and pass-through arguments cover every
  invocation needed — a single test file, a `-k` filter, a one-module
  type-check (see "Quality gate" below).
- Hand-roll the scripts' jobs with raw `git worktree add` /
  `git merge --squash` / `pip install` commands.
- Edit the workflow scripts mid-feature to make an error go away. Changing the
  scripts is a legitimate change, but it is its own plan-and-approve feature —
  never a workaround embedded in another one.
- Declare a change finished "except for the gate" because the gate wouldn't
  run.

## Run / test

```
pip install -e ".[dev]"                               # runtime + pyright/black/isort/pytest
wingspan play                                         # human vs random
wingspan random --log game.log                        # watch a random game
wingspan selfplay --p0 best --p1 random               # trained-checkpoint matchups
wingspan dashboard --device cpu                       # FLIGHT PLAN (config → training)
wingspan tournament                                   # round-robin between trained AIs
wingspan inspect --checkpoint-dir checkpoints         # model introspection report
wingspan cloud --config run.yaml                      # headless S3-persisted training
wingspan monitor --bucket <bucket> --prefix runs      # FLOCK WATCH roster
python -m pytest tests/
```

Training is CPU-only — collection fans out across worker processes and the
gradient update is small, so no GPU is required (CUDA still works for one-off
experiments but is not the supported path).
`wingspan dashboard` (or `python -m wingspan.training`) opens the FLIGHT PLAN
screen first — tune any hyperparameters, then start or resume a run, which
transitions into the live training display. Collection is fastest on
`--device cpu` (TRAINING.md §1.4). Containerized S3-persisted runs go through
`wingspan cloud` (see `deploy/`); `wingspan monitor` is the read-only FLOCK
WATCH roster of cloud runs.

## Quality gate

Run from the current directory (worktree or repo root):

```
bash scripts/quality_gate.sh
```

Or pass an explicit path:

```
bash scripts/quality_gate.sh .claude/worktrees/<slug>
```

The full gate runs five steps in order: `pyright` (strict) → `isort` → `black`
→ `pyright` (post-format) → `pytest`. It stops at the first `pyright` failure
so you don't format broken code. Config lives in `pyproject.toml`
(`[tool.pyright]`, `[tool.black]`, `[tool.isort]`); no flags needed. `pyright`
is the globally-installed npm binary; formatters run as `python -m isort` /
`python -m black` via the target directory's own `.venv`.

For faster iteration while fixing a specific problem, run individual sections.
Everything after a section flag (up to the next section flag) is passed
verbatim to the underlying tool — so the gate covers single-file, single-test,
and keyword-filtered runs, and there is never a reason to invoke the tools
directly:

```
bash scripts/quality_gate.sh --pyright                          # type-check only (one pass)
bash scripts/quality_gate.sh --pytest                           # full test suite only
bash scripts/quality_gate.sh --pytest tests/test_smoke.py       # a single test file
bash scripts/quality_gate.sh --pytest tests/test_encode.py -k state   # filter tests by name
bash scripts/quality_gate.sh --pytest -x -q                     # any pytest flags pass through
bash scripts/quality_gate.sh --pyright src/wingspan/state.py    # type-check one file
bash scripts/quality_gate.sh --format                           # isort + black only
bash scripts/quality_gate.sh --pyright --pytest                 # types + tests, skip format
# (same pattern with an explicit path; target-dir goes before the first flag)
bash scripts/quality_gate.sh .claude/worktrees/<slug> --pytest tests/test_smoke.py
```

Defaults when a section gets no arguments: `--pytest` → `tests/ -n 8 --dist
load` (parallel across 8 pytest-xdist worker processes; override the count with
the `WINGSPAN_PYTEST_WORKERS` env var, `0` = serial), `--format` → `src tests`,
`--pyright` → the `pyproject.toml` config. Explicit pytest arguments replace
the default entirely, so a targeted run like `--pytest tests/test_smoke.py`
stays serial and never pays the multi-worker torch-import startup. Steps always
execute in the canonical gate order regardless of flag order; requesting both
`--pyright` and `--format` also runs the post-format pyright pass.

Exit codes:

- `0` — gate passed.
- `1` — genuine check failure (pyright errors or failing tests). Normal
  feature work: fix the code and rerun.
- `2` — infrastructure/usage failure (missing venv, `pyright` not on PATH, bad
  target dir, invalid arguments). **Not a code problem — stop and ask the user
  to fix it** (see "Script failures: stop, don't circumvent").

Always run the full gate (no section flags) before committing. Do NOT call
pyright, pytest, isort, or black directly — use this script so the step
ordering and Python path are always correct.

### Coverage (on-demand, not in the gate)

Coverage is fully configured (`[tool.coverage.*]` in `pyproject.toml`) but
deliberately **not** part of the gate — plain gate runs pay zero coverage
overhead. When you want a report, run it through the pytest pass-through:

```
bash scripts/quality_gate.sh --pytest tests/ --cov --cov-report=term-missing --cov-report=html
```

`source_pkgs = ["wingspan"]` keys measurement on the import package (works in
worktrees and without an install), and `concurrency = ["multiprocessing",
"thread"]` + `parallel = true` capture the mp_collect pool workers' lines too;
pytest-cov combines the per-process data files at session end. The HTML report
lands in `htmlcov/`; all coverage artifacts (`htmlcov/`, `.coverage*`,
`coverage.xml`) are gitignored. Branch coverage is on, so a coverage run is
noticeably slower than a plain run.

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
  cli.py                 # argparse entry points (manual / random / selfplay / tournament dispatch)
  state.py               # GameState, Player, Board, FoodPool, PlayedBird, Birdfeeder, new_game
  decisions.py           # Decision[C] hierarchy + Choice hierarchy + MainAction + judgment families
  architecture.py        # ModelArchitecture + ActivationName (torch-free network topology descriptor)
  model.py               # PyTorch PolicyValueNet (built from a ModelArchitecture)
  mlp.py                 # shared MLP body/readout builders (policy net + setup net build identical stacks)
  hand_model.py          # stateless multi-card set-embedder helpers (hand / tray / setup kept-set)
  selfplay.py            # selfplay CLI: per-seat agent matchups over trained checkpoints
  introspect.py          # model introspection CLI (vector layout, architecture, parameters)
  report.py              # standalone HTML model-summary report generator
  data/*.json            # wingsearch card data (bundled)

  encode/                # state/choice tensor encoders for RL (package)
    layout.py            # feature dims, stripe offsets, normalization scales (the chain)
    stripes.py           # programmatic stripe registry for the state/choice vectors
    state_encode.py      # encode_state / state_size + per-aspect state summaries
    choice_encode.py     # encode_choices + per-Choice featurizers + stripe fillers

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
    display.py           # human-readable formatters for cards and game state
    interactive.py       # terminal selection-form widget for the interactive CLI

  setup_model/           # the separately-trained setup model (value-regression bandit)
    architecture.py      # SetupArchitecture topology descriptor (+ its shape_key)
    candidates.py        # the keep options the setup model scores + selection
    encode.py            # per-candidate feature encoder
    stripes.py           # programmatic stripe registry for the setup input vector
    generate.py          # random-setup generation (the pre-model training phase)
    record.py            # the setup training sample + its on-disk store

  instrumentation/       # general-purpose event-callback instrumentation for games
    config.py            # serializable instrumentation config + per-run context
    dispatcher.py        # the live event router an Engine holds
    events.py            # event taxonomy + per-shape handler base classes
    registry.py          # config-class-name <-> handler bijection
    handlers/            # card_visits (per-bird play tallies), decision_logger (JSONL rows)

  training/              # live training + monitoring dashboard ("FLYWAY CONTROL")
    __main__.py / app.py # entry point: argparse (+ --config) -> worker thread + rich.Live loop
    config.py            # TrainConfig (self-describing hyperparameters, §5.1)
    artifacts.py         # shared on-disk filenames (LAST/BEST/OPPONENT ckpt, metrics+games logs, model_config/process json)
    runmeta.py           # model_config.json (full topology, reconstitutable) + dated process_<stamp>.json sidecars (torch-free); read_model_config reader
    metrics.py           # ScoreBreakdown / FamilyCounts / EvalResult / IterationMetrics / GameOutcome (games.jsonl row)
    metrics_log.py       # cached reader for the append-only metrics.jsonl history
    runstate.py          # RunState: the shared live snapshot the dashboard reads (+ RunProgress)
    steps.py             # Step: the recorded self-play transition the learner consumes
    policy.py            # single-decision sample (collect) + greedy (eval)
    collect.py           # baseline single-game collector -> recorded steps + score breakdown
    mp_collect.py        # process-parallel collection (the CPU path; see COLLECTORS.md)
    batched_collect.py   # batched-forward collection (the CUDA path; see COLLECTORS.md)
    learner.py           # length-bucketed REINFORCE + advantage norm (§3.3, §4.2a)
    setup_net.py         # SetupNet: the setup model's MLP value-regressor
    setup_learner.py     # setup-model updates: offline fit + on-policy MSE
    setup_runmeta.py     # setup_config.json descriptor sidecar
    evaluate.py          # paired-game strength vs the reference opponent + 95% CI (§7)
    convergence.py       # series + axis-window math for the convergence charts
    sysmon.py            # host telemetry sampling for the SYSTEM band
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
      arch_diagram.py    # the live ARCHITECTURE diagram

  tournament/            # round-robin tournament between trained AIs (wingspan-tournament)
    app.py               # entry point: pick competitors, play live, write the report
    participants.py      # competitor specs, on-disk run discovery, agent loading
    schedule.py          # the round-robin game schedule
    runner.py            # plays the scheduled games (process-parallel, sequential fallback)
    elo.py results.py state.py dashboard.py picker.py config.py

  cloud/                 # containerized, S3-persisted training runs + monitor
    runner.py            # headless supervisor (wingspan-cloud)
    runfile.py           # the single YAML run-file configuring one cloud run
    s3sync.py            # the S3 persistence sidecar around the loop
    status.py            # the compact monitoring snapshot of a run
    monitor.py           # "FLOCK WATCH" read-only roster of cloud runs (wingspan-monitor)

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
  (`PlayBirdDecision`, `GainFoodDecision`, `PayBirdFoodDecision`, ...). Decisions
  that need extra context add typed fields directly
  (`SetupDecision.dealt_cards`, `SetupDecision.dealt_bonus`).
- `ALL_DECISION_CLASSES` is the stable iteration order for the encoder's
  decision-class one-hot stripe. Append new subclasses at the end — but
  keep `SetupDecision` last (the `include_setup` truncation contract) —
  reordering or removing entries shifts the stripe indices, a FRESH
  (checkpoint-invalidating) change. See "Checkpoint compatibility policy".

When adding a new decision point: define the Choice subclass first (or
reuse an existing one), then the `Decision[C]` subclass, then add it to
`ALL_DECISION_CLASSES`, then teach `encode/choice_encode.py` how to featurize it.

**Optional-then-commit pattern.** Any effect a player may decline must flow
through `AcceptExchangeDecision` (SKIP_OPTIONAL family) before any follow-up
decisions. The accept row is a `PayCostChoice` carrying the full exchange
ledger (`paid_*` / `gained_*` and `opp_gained_*` for shared-benefit powers
that also help the opponent); the skip row is a `SkipChoice`. Follow-up
decisions — *which* egg to give up, *which* food to spend, *where* to lay —
are then presented without a skip: the commitment is settled. Conditionally-
optional effects (those that only matter when a specific round goal is active,
e.g. `birds_no_eggs`) should offer `AcceptExchangeDecision` only under that
condition; outside it, execute the effect as mandatory so the SKIP_OPTIONAL
head is not trained on trivially-obvious non-decisions.

**Keep `DECISIONS.md` in sync.** `DECISIONS.md` is the per-family modelling
report (engine call sites, choice-vector contents, and intra-family variation
for every scoring head). Any change to the decision/choice taxonomy, the
`_DECISION_FAMILY` mapping, the choice-vector stripes or featurizers, the
engine's decision call sites (new power handlers / reactors / conversions), or
the setup / `split_setup_bonus` config axes must update `DECISIONS.md` in the
same change — its closing "Maintaining this document" section maps each kind of
code change to the sections to refresh.

### Configurable network topology

The network shape is data-driven, not hard-coded. `architecture.ModelArchitecture`
(top-level, torch-free) is the single descriptor of the topology: per-block
hidden-width lists (`trunk_layers`, `choice_layers`, `head_layers`,
`value_layers`, `card_encoder_layers`, `hand_encoder_layers`, and the optional
per-family override `per_family_head_layers`, resolved via `head_layers_for`)
plus the `activation`, `dropout`, `layernorm`, `card_embed_dim`,
`use_distinct_hand_model`, `hand_embed_dim` (resolved via `hand_embed_width`;
`None` means "match `card_embed_dim`"), and `tray_set_embedding` handles.
`PolicyValueNet` builds every block from it via the shared `mlp.build_body` /
`mlp.build_readout` recipes (factored into `mlp.py` so the setup net builds
byte-identical stacks), `TrainConfig` mirrors the same fields flat
(so the configurator edits each independently) and assembles them via its `arch`
property, and `runmeta.write_model_config` / `read_model_config` serialize the
full descriptor to `model_config.json` so a run's network reads at a glance and
reconstitutes via `PolicyValueNet.from_model_config`.

Invariants to preserve when extending it: the trunk ends at width `M`
(`trunk_embed_width`) and the choice encoder at width `N` (`choice_embed_width`)
— they are independent, and their outputs are concatenated to `M+N` for the
scorers; `ShapeKey` / `architecture_key` must include any new field that changes
a tensor shape (a FRESH change — mismatched checkpoints then restart cleanly via
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

## Checkpoint compatibility policy

The June 2026 compatibility cutoff: loaders tolerate **no** artifact written
before it. From that point on, compatibility with everything the *current* code
writes is strict and deliberate. The rules:

- **Every artifact is self-describing, and loaders refuse what isn't.** Every
  checkpoint embeds its `config` (`setup.pt` embeds `setup_config`); every run
  directory carries `model_config.json` (+ `setup_config.json` when the setup
  model is on). A payload with no embedded config is UNREADABLE in the
  configurator and starts fresh (with an alarm) at the resume gate; a run dir
  with no descriptor cannot be seated in tournaments or introspected. Never add
  an "assume compatible" branch, a missing-key fallback, a second on-disk
  location for the same datum, or a ghost entry kept only for index stability.
- **FRESH vs REGIME is the gate.** `architecture_key` / `ShapeKey` (and the
  setup twins) cover everything that changes a tensor shape; a mismatch refuses
  the weights and restarts cleanly (FRESH). Shape-preserving knobs
  (`activation`, `dropout`, learning rates, cadences) stay out of the key and
  resume freely (REGIME). Any new field that changes a tensor shape must be
  added to the key.
- **The stable orders are part of the checkpoint format.** `ALL_DECISION_CLASSES`,
  `ALL_DECISION_FAMILIES`, the `encode/layout.py` offset chain, and the
  `cards.parse.catalog` card-index maps are what trained weights are aligned
  to. Append-only; reordering, renumbering, or removing an entry is a FRESH
  break for every checkpoint and must be a deliberate, called-out decision.
- **New fields on persisted models default — that is the one sanctioned
  back-compat mechanism.** When adding a field to a model that is persisted in
  checkpoints or logs (`RunProgress`, `IterationMetrics`, `ModelConfig`,
  `GameOutcome`, ...), give it a default so artifacts already written by
  current-era runs keep loading, and comment the field with why the default
  exists. Required fields stay required.
- Crash-survivability tolerance is fine and stays (e.g. `metrics_log` skipping
  a truncated final line, archiving a partial run dir): that guards the
  *current* format against interruption, not an old format against age.

## Test conventions

- Tests prepend `src/` to `sys.path` themselves (see `test_smoke.py`); new
  tests should match so `pytest tests/` works from the repo root without
  install.
- One file per power (`tests/test_powers_*.py`); the cross-power smoke
  test is `test_smoke.py`. Encoder and food-payment helpers each have
  their own dedicated test file. The training-cycle smoke coverage is the
  collect → update pair in `test_model_and_self_play.py`.

## Things to avoid

- **Never delete, move, or modify `*.lock` files in the repo root.** These are
  human-authorization tokens for the merge workflow. Their absence signals
  approval; Claude removing one silently bypasses human review. See "Merge-auth
  lock files" above.
- **Never circumvent the workflow scripts.** If `quality_gate.sh` exits `2`,
  `merge_worktree.sh` exits `5`, or any workflow script itself malfunctions,
  stop and ask the user — do not fall back to running pyright / pytest / isort
  / black directly, hand-rolling git worktree or merge commands, or patching
  the scripts mid-feature. See "Script failures: stop, don't circumvent".
- Don't add tolerant-parse fallbacks, "assume compatible" branches, or ghost
  entries for old artifact formats. Loaders refuse what the current code
  doesn't write — see "Checkpoint compatibility policy".
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
