# CLAUDE.md

Project-specific guidance for working in the Wingspan simulator. The global
style rules in `~/.claude/CLAUDE.md` already apply (Pydantic-first, absolute
module-qualified imports, Python 3.12+ syntax, no `Any`/payload bags); this
file documents the patterns the current codebase has settled on so future
changes stay consistent.

## What this project is

A Wingspan core-set simulator (180 birds, 26 bonus cards, 16 round goals,
2-player automa-free) plus an RL training pipeline. **See `README.md`** for
the general description, the full CLI reference, and the annotated package
layout (§ "How it's organized") — skim the layout before adding a module so
new code lands in the right package, and update it there when the layout
changes. The long-term goal is self-play training at scale to answer
analytical questions about the game (card power rankings, bonus-card value,
food/habitat economy) — design for scaling up training, not for the minimum
that runs today. All 180 bird powers are modelled via generic `EffectKind`
patterns; the `UNIMPLEMENTED` fallback (runtime no-op, surfaced in the
coverage report) keeps unparsed future additions non-fatal.

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

See `README.md` for the full CLI (`wingspan play / random / selfplay /
dashboard / tournament / inspect / cloud / monitor`) and training notes;
`TRAINING.md` is the training program. Setup: `pip install -e ".[dev]"`.
Training is CPU-only — collection fans out across worker processes and
`--device cpu` is fastest (TRAINING.md §1.4). Run tests through the quality
gate below, not bare pytest.

## Quality gate

```
bash scripts/quality_gate.sh [target-dir]
```

Run from the current directory (worktree or repo root), or pass an explicit
target dir. Five steps in order: `pyright` (strict) → `isort` → `black` →
`pyright` (post-format) → `pytest`. Config lives in `pyproject.toml`;
`pyright` is the globally-installed npm binary; formatters, pytest, and
coverage run via the target directory's own `.venv`.

For faster iteration, run individual sections; everything after a section flag
passes verbatim to the underlying tool, so there is never a reason to invoke
the tools directly:

```
bash scripts/quality_gate.sh --pyright src/wingspan/state.py   # types only / one file
bash scripts/quality_gate.sh --format                          # isort + black only
bash scripts/quality_gate.sh --pytest tests/test_encode.py -k state -x -q
bash scripts/quality_gate.sh --coverage                        # full gate + coverage regression
```

No-argument defaults: `--pytest` → `tests/ -n 8 --dist load` (worker count via
`WINGSPAN_PYTEST_WORKERS`, `0` = serial; explicit args replace the default, so
targeted runs stay serial), `--format` → `src tests`. Steps always execute in
canonical gate order regardless of flag order.

Exit codes: `0` — passed. `1` — genuine check failure (pyright errors, failing
tests, or coverage regression): fix the code and rerun. `2` — infrastructure/usage
failure (missing venv, `pyright` not on PATH, bad arguments): **not a code
problem — stop and ask the user to fix it**.

Always run the full gate (no section flags) before committing; every change
must pass it to be considered finished. Strict-mode pyright must be completely
clean (`reportPrivateImportUsage = false` silences torch's under-exporting
stubs — don't re-enable it).

**Coverage regression check.** Pass `--coverage` (no args) to run pytest
serially with `--cov --cov-report=term-missing` in a single pass, then compare
the TOTAL percentage against `coverage_baseline.txt` in the repo root.
`merge_worktree.sh` always passes `--coverage`; it is not needed during
worktree iteration. The baseline ratchets upward:

- Coverage improves → baseline file updated automatically; commit it with your
  change.
- Coverage unchanged → gate passes silently.
- Coverage drops → gate fails (exit 1). Either add tests to recover coverage,
  or report the drop to the user so they can decide whether to lower the
  baseline manually (see "Things to avoid").

**First run (baseline absent):** the gate creates `coverage_baseline.txt`
automatically and passes. Commit that file to lock the regression floor.

Modules excluded from coverage measurement (CLI entry points, SVG/chart
rendering, AWS/S3 integration) are listed in `[tool.coverage.run] omit` in
`pyproject.toml`. To include a module, remove it from that list.

## Architectural patterns to preserve

### Pydantic v2 BaseModel for *all* structured data

Every record-shaped object is a `pydantic.BaseModel` — immutable card data,
mutable game state, every `Choice` / `Decision` subclass, and the raw-JSON
`*Record` input models. No dataclasses, `TypedDict`, or bare `dict[str, ...]`
for new records. Conventions: frozen models for immutable card data and IR,
default mutability for game state; `arbitrary_types_allowed=True` only where
already used (e.g. `random.Random` on `GameState`); `*Record` models use
`Field(alias=...)` + `extra="allow"`, and their `.load()` imports the
`cards.parse` helpers lazily to avoid a cycle; vector-shaped pools
(`FoodPool`, `BirdCost`) keep the two-layer shape — fixed-length internal
vector aligned to `cards.ALL_FOODS`, dict-like external API.

### Imports: module-qualified, never symbol-level

Per the global rule: `from wingspan import cards, decisions, state` then
`cards.Bird` — never `from wingspan.cards import Bird`. Group sibling engine
submodules (`from wingspan.engine import actions, helpers, powers`); alias
(`import core as engine_core`) when the bare name would be ambiguous.
`__init__.py` files re-export each package's public surface — keep `__all__`
updated. Standard library too (`import typing` → `typing.Any`), and
`from __future__ import annotations` at the top of every module.

### Python 3.12+ syntax

PEP 695 generics (`class Decision[C: Choice](pydantic.BaseModel)`, `def
agent[C: decisions.Choice](...) -> C`) — no `TypeVar` / `Generic[T]`.
`enum.StrEnum` for every enum. `X | None`, never `Optional[X]`.
`Annotated[list[C], Field(min_length=1)]` for declarative single-field
validation; reserve `@model_validator` for genuine cross-field invariants.

### The decision/choice system

Agents resolve decisions, not raw action ints:

- `Choice` is the abstract base; one subclass per *data shape* with named
  typed attributes — no opaque payload tuples, no `Any` carriers.
- `Decision[C: Choice]` is generic in the Choice it accepts; declinable
  decisions include `SkipChoice` in the union and consumers branch via
  `isinstance`. Extra context = typed fields on the Decision subclass.
- `ALL_DECISION_CLASSES` is the encoder's stable one-hot order: append new
  subclasses at the end, keep `SetupDecision` last (the `include_setup`
  truncation contract); reordering or removing entries is a FRESH change
  (see "Checkpoint compatibility policy").
- New decision point: Choice subclass (or reuse one) → `Decision[C]` subclass
  → `ALL_DECISION_CLASSES` → featurize in `encode/choice_encode.py`.

**Optional-then-commit.** Any declinable effect flows through
`AcceptExchangeDecision` (SKIP_OPTIONAL family) first: the accept row is a
`PayCostChoice` carrying the full exchange ledger (`paid_*` / `gained_*` /
`opp_gained_*`); the skip row is a `SkipChoice`; follow-ups are then presented
without a skip — the commitment is settled. Conditionally-optional effects
(e.g. `birds_no_eggs` under its round goal) offer the accept decision only
under that condition and run as mandatory otherwise, so the SKIP_OPTIONAL head
isn't trained on trivially-obvious non-decisions.

**Keep `DECISIONS.md` in sync.** Any change to the decision/choice taxonomy,
`_DECISION_FAMILY`, the choice-vector stripes or featurizers, the engine's
decision call sites, or the setup / `split_setup_bonus` config axes must
update `DECISIONS.md` in the same change — its "Maintaining this document"
section maps each kind of code change to the sections to refresh.

### Configurable network topology

`architecture.ModelArchitecture` (top-level, torch-free) is the single
topology descriptor: per-block hidden-width lists (`trunk_layers`,
`choice_layers`, `head_layers`, `value_layers`, `card_encoder_layers`,
`hand_encoder_layers`, `per_family_head_layers`) plus scalar handles
(`activation`, `dropout`, `layernorm`, `card_embed_dim`, ...).
`PolicyValueNet` builds every block from it via the shared `mlp.build_body` /
`mlp.build_readout` recipes (the setup net builds byte-identical stacks);
`TrainConfig` mirrors the fields flat, assembling them via its `arch` property
— deliberately *not* named `architecture`, which would shadow the imported
module; `runmeta` serializes it to `model_config.json` for
`PolicyValueNet.from_model_config`. Invariants: the trunk (width `M`) and
choice encoder (width `N`) are independent, concatenated to `M+N` for the
scorers; any new field that changes a tensor shape must join `ShapeKey` /
`architecture_key` (FRESH); shape-preserving knobs stay out of it (REGIME).

### Engine = orchestrator; sibling modules = free functions

`engine.core.Engine` owns the turn loop, setup phase, and the `ask` plumbing
that routes a Decision through an Agent. Everything else (actions, power
dispatch, pink reactors, scoring) lives in sibling modules as **free functions
whose first argument is the Engine** — call `actions.do_play_bird(engine,
agent)` directly; no `_do_*` wrapper methods on Engine. Sibling modules break
the import cycle with `if typing.TYPE_CHECKING: from wingspan.engine import
core` and `engine: "core.Engine"` annotations — don't move logic into
`core.py` just to avoid this.

### The Agent protocol

`Agent` (in `engine.core`) is a `typing.Protocol` with a generic `__call__`:
`def __call__[C: decisions.Choice](self, engine, decision:
decisions.Decision[C], /) -> C`. This keeps `Agent` non-generic at use sites
(`list[Agent]` typechecks) while each call's return type tracks the Decision's
parameterization. New agents live in `wingspan.agents` and follow the same
generic-function shape — `agents.base.random_agent` is the reference.
Opponent-prompting effects route through `engine.agent_for(player)`.

### Bird powers: parser + dispatcher pair

Supporting a new bird power is a three-step pattern: (1) add an `EffectKind`
variant in `cards.schema`, adding a new typed carrier field to `Effect` if the
existing ones don't cover its data — no generic payload; (2) add a pattern
matcher in `cards.parse.matchers` with `@registry.pattern`
(`@registry.pink_pattern` in `pink_matchers` for reactive ones) — registration
order = source order, more specific patterns first when they overlap; (3) add
a handler in the matching `engine.powers` submodule with
`@registry.handles(EffectKind.X)`. Pink (between-turn) effects dispatch from
`engine.reactors`, not `apply_effect`, which treats them as silent no-ops.
Keep the `UNIMPLEMENTED` fallback in place — it's what lets future expansion
cards or parser gaps stay non-fatal.

### Public constants and per-turn scratch state

Action / track / cost constants live at the top of `state.py`; encoder feature
dims, stripe offsets, and normalisation scales in `encode/layout.py` (the
whole chain in one file); chart layout sizes in `training/charts/geometry.py`.
No magic numbers in function bodies — promote them. Cross-action turn state
(e.g. House Wren's extra play) lives on `GameState` as explicit typed fields
(`turn_extra_plays`, ...) reset by `GameState.reset_turn_state()` each turn —
no parallel scratch dicts.

## Checkpoint compatibility policy

The June 2026 compatibility cutoff: loaders tolerate **no** artifact written
before it. From the cutoff on, compatibility is governed by the **artifact
version** below — a deliberate, versioned guarantee, never ad-hoc tolerance:

### Artifact version (`wingspan.version.MODEL_VERSION`)

Every persisted artifact (`model_config.json`, `setup_config.json`, and every
`.pt` payload) is stamped with a `MAJOR.MINOR` artifact version. This is
distinct from the package release version (`wingspan.__version__`) — one
tracks the codebase, the other the on-disk artifact format.

- **The guarantee: load + play.** At code version X.Y, artifacts with the same
  MAJOR and MINOR ≤ Y must load and play games (inference / eval /
  tournament). A different MAJOR, or a MINOR newer than the code, is refused
  cleanly with `version.IncompatibleArtifactError`.
- **Enforcement is deliberately asymmetric.** The hard version check guards
  the *inference* loaders (`runmeta.read_model_config`,
  `setup_runmeta.read_setup_config`, `selfplay._load_policy_net` /
  `_load_setup_net`, `tournament.participants.load_player`). The *resume*
  loaders (`loop_resume`, `loop_setup`, `loop_eval.load_opponent`) keep the
  graceful `architecture_key` gate — mismatch starts fresh, never crashes.
  Training resume across versions is **not** promised.
- **Compat is version-number-specific checks, never config flags.** A shim
  for an older same-major encoding lives in the `wingspan.compat` package, one
  module per superseded era (shape: `if artifact older than the change:
  regenerate the encoding without the new field` — see `compat.v0_0`, which
  regenerates the pre-0.1 choice rows and provides the frozen-era
  `PolicyValueNetV00`; the loaders route by
  `compat.v0_0.uses_v0_0_choice_encoding`). Do not add `TrainConfig` axes to
  toggle old behaviors. Inference call sites must encode through the net
  (`net.encode_state` / `net.encode_choices`), never by pairing the live
  encoder with a spec by hand — that is what lets a compat-era net carry its
  own geometry.
- **A MINOR bump is required for every FRESH-type change** (see below), and
  must: (a) bump `MODEL_VERSION`; (b) add the version-specific shim; (c)
  capture a new fixture set under `tests/data/compat/v<X.Y>/` from a run at
  the new version (see that directory's READMEs); (d) extend the compat tests
  so **every retained fixture set still loads and plays**. All same-MAJOR
  fixture sets are retained.
- **A MAJOR bump is the deliberate escape hatch**: it drops the accumulated
  shims and deletes the old fixture sets. It must be its own called-out,
  user-approved decision — never a side effect.
- The fixture sets under `tests/data/compat/` are the only checkpoints
  committed to git: gzip-compressed (`*.pt.gz`) and **Git LFS**-tracked via
  `.gitattributes`, with the config JSONs committed plain. New fixture sets
  must follow the same shape (see the v0.0 README in that directory).

### Format rules

- **Every artifact is self-describing; loaders refuse what isn't.** Every
  checkpoint embeds its `config` (`setup.pt` embeds `setup_config`) and its
  `version`; every run directory carries `model_config.json`
  (+ `setup_config.json` when the setup model is on). Never add an "assume
  compatible" branch, a second on-disk location for the same datum, or a
  ghost entry kept only for index stability.
- **FRESH vs REGIME still gates resume.** `architecture_key` / `ShapeKey` (and
  the setup twins) cover everything that changes a tensor shape; a mismatch
  refuses the weights and restarts cleanly (FRESH) — and, per the versioning
  rules above, shipping such a change requires the MINOR bump + shim +
  fixture set. Shape-preserving knobs (`activation`, `dropout`, learning
  rates, cadences) stay out of the key, resume freely, and need no version
  bump (REGIME).
- **The stable orders are part of the checkpoint format.**
  `ALL_DECISION_CLASSES`, `ALL_DECISION_FAMILIES`, the `encode/layout.py`
  offset chain, and the `cards.parse.catalog` card-index maps are append-only;
  reordering, renumbering, or removing an entry is a FRESH break for every
  checkpoint and must be a deliberate, called-out decision.
- **New fields on persisted models default — the one sanctioned back-compat
  mechanism** so current-era artifacts keep loading; comment why the default
  exists. Required fields stay required. (The `version` field itself works
  this way: absence reads as `version.PRE_VERSIONING_VERSION`, pinned `"0.0"`
  forever while `MODEL_VERSION` advances.)
- Crash-survivability tolerance is fine and stays (e.g. `metrics_log` skipping
  a truncated final line): it guards the *current* format against
  interruption, not an old format against age.

## Test conventions

- Tests prepend `src/` to `sys.path` themselves (see `test_smoke.py`) so
  `pytest tests/` works from the repo root without install; new tests match.
- One file per power (`tests/test_powers_*.py`); the cross-power smoke test is
  `test_smoke.py`; encoder and food-payment helpers have dedicated files; the
  training-cycle smoke coverage is the collect → update pair in
  `test_model_and_self_play.py`.

## Things to avoid

- **Never delete, move, or modify `*.lock` files in the repo root** — they are
  human-authorization tokens for the merge workflow (see "Merge-auth lock
  files").
- **Never circumvent the workflow scripts** — no direct pyright / pytest /
  isort / black, no hand-rolled worktree or merge commands, no mid-feature
  script edits (see "Script failures: stop, don't circumvent").
- No tolerant-parse fallbacks, "assume compatible" branches, or ghost entries
  for old artifact formats (see "Checkpoint compatibility policy").
- Don't replace the `cards.Food` / `cards.Habitat` / etc. `StrEnum`s with
  strings — JSON serialisation is already free and the type checker catches
  typos.
- No `Decision`/`Choice` payload tuples or `Any`-typed context fields — add a
  typed field to the subclass or define a new Choice subclass.
- Don't bypass `Engine.ask` — it validates the agent's answer against the
  offered choices; constructing a Choice directly skips that check.
- No `model_config = ConfigDict(...)` unless a non-default behavior is
  actually needed; a bare model is the preferred shape.
- No `_do_*` wrapper methods on Engine that just delegate to `actions.do_*` —
  call the free function directly.
- **Never edit `coverage_baseline.txt` to lower the percentage.** The ratchet
  is a one-way floor. If coverage drops (e.g. deleted dead code, or a module
  removed from the `omit` list), report the impact and the gate's message to
  the user and let them decide whether to update the baseline manually.
