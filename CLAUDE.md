# CLAUDE.md

Project-specific guidance for the Wingspan simulator. Global style rules in `~/.claude/CLAUDE.md` apply (Pydantic-first, module-qualified imports, Python 3.12+, no `Any` bags) — this file covers only wingspan-specific patterns.

## What this project is

A 2-player Wingspan core-set simulator (180 birds, 26 bonus cards, 16 round goals) plus an RL self-play training pipeline. The long-term goal is answering analytical questions about the game (card power rankings, food economy) via scale — **design for scaling up training, not the minimum that runs today**. See `README.md` for the CLI reference and `docs/PROJECT.md` for the package map.

## Documentation files

Load these when working in the relevant area. Update them in the same commit that updates the code — they are not kept automatically in sync.

| File | What it covers | Update when… |
| ---- | -------------- | ------------ |
| `docs/PROJECT.md` | Top-level package map (links to each subpackage's `INDEX.md`) | Adding or renaming a top-level package; structural refactors |
| `src/wingspan/*/INDEX.md` | Per-subpackage module detail: classes, key methods, state | Adding or renaming a module within a subpackage |
| `docs/DECISIONS.md` | Decision/choice taxonomy, `ALL_DECISION_CLASSES` ordering, choice-vector stripes, engine call sites | Adding a `Decision`/`Choice` subclass; changing a featurizer; reordering `ALL_DECISION_CLASSES` |
| `docs/GAMELOG.md` | Six event types, sub-event taxonomy, open-event-stack rule, call-site map | Adding a `GameEvent`/`SubEvent` subclass; wiring a new call site; changing a renderer |
| `docs/BIRDS.md` | All 180 birds + 26 bonus cards: `EffectKind` patterns, handler mappings, implementation gaps | Adding an `EffectKind` variant, a matcher, or a power handler |
| `docs/BONUSES.md` | All 26 bonus cards and 16 round goals: scoring rules, state/choice encoding, what advances each | Adding a bonus category, a goal category, or changing the delta/encoding logic |
| `docs/TRAINING.md` | Training program, hyperparameter guidance, Phase 0–3 roadmap | Changing the training approach, convergence criteria, or phased plan |
| `docs/RESEARCH.md` | Research agenda, per-project feasibility verdicts | Adding/completing a research project; updating a feasibility gap assessment |
| `docs/VERSIONING.md` | Artifact version changelog, FRESH/REGIME distinction, compat shim rules | Bumping `MODEL_VERSION`; adding a compat shim; capturing a new fixture set |
| `docs/QUALITY.md` | Quality gate section flags, coverage regression details, merge exit codes | Reference only — load when iterating on the gate |

## Making changes: the worktree workflow

Substantive code changes follow a fixed shape:

**plan → user approves → create worktree → implement → gate passes → commit → report ready → human authorizes (deletes lock) → merge → done.**

**When this applies.** Any change that goes through plan-and-approve. For trivial edits (one-liners, doc tweaks), edit the main working directory directly — no worktree — then **commit and push with a descriptive commit message**. When in doubt, use the worktree.

### Steps

1. **Plan and get approval.** Don't touch code until the user approves the plan.

2. **Create the worktree and enter it.**
   ```
   bash scripts/create_worktree.sh <feature-slug>
   EnterWorktree(path=".claude/worktrees/<slug>")
   ```
   The script commits any dirty state in `main`, creates the worktree on branch `wt/<slug>`, installs a fresh `.venv`, and writes `<slug>.lock` in the repo root (the merge-auth lock). All subsequent edits happen inside the worktree.

3. **Implement the change** inside the worktree. The main working directory is untouched.

4. **Run the quality gate.**
   ```
   bash scripts/quality_gate.sh
   ```
   Exit `0` = passed. Exit `1` = code problem — fix and rerun. Exit `2` = infrastructure failure — **stop, show the user the output verbatim, and wait for them to fix it**. The gate is quiet by default (one summary line per step); add `--debug` to see full tool output when a failure summary isn't enough to diagnose the problem. See `docs/QUALITY.md` for section flags and targeted-run examples.

5. **Commit and stop.** Commit all changes on the feature branch. Report that the change is ready and tell the user to delete the lock file to authorize the merge. Do **not** merge or delete the lock yourself.

6. **Merge (after human authorization).** Once the human deletes `<slug>.lock`:
   ```
   bash scripts/merge_worktree.sh <slug>       # human runs directly, or:
   bash scripts/auto_merge_worktree.sh <slug>  # automated: loops merge with claude -p
   ```
   If asked to merge during your session, `ExitWorktree(action="keep")` first, then run `merge_worktree.sh` from the main working directory.

### Hard rules

- **Never delete, move, or modify any `*.lock` file in the repo root.** A lock being absent means a human reviewed and approved; Claude removing one bypasses that entirely. This applies even if asked, even if the lock seems stale.
- **`quality_gate.sh` is the only sanctioned way to run pyright / isort / black / pytest.** Do not call them directly, hand-roll the gate with raw git/pip commands, or edit the workflow scripts mid-feature. Changing the scripts is its own plan-and-approve feature.
- **Infrastructure failures (exit `2`, crashing bash, missing PATH tools) mean stop.** Do not continue toward a merge until the user fixes the environment.

## Architectural patterns to preserve

### Decision/choice system

Agents resolve decisions, not raw action ints. `Choice` is the abstract base — one subclass per data shape with named typed attributes; no opaque payload tuples or `Any` carriers. `Decision[C: Choice]` is generic in the Choice it accepts; declinable decisions include `SkipChoice` in the union (consumers branch via `isinstance`).

`ALL_DECISION_CLASSES` is the encoder's stable one-hot order: append new subclasses at the end, keep `SetupDecision` last; reordering or removing entries is a FRESH change (`docs/VERSIONING.md`). Adding a decision point: Choice subclass → `Decision[C]` subclass → `ALL_DECISION_CLASSES` → featurize in `encode/choice_encode.py`. See `docs/DECISIONS.md`.

**Optional-then-commit.** Declinable effects flow through `AcceptExchangeDecision` first (the accept row carries the full exchange ledger; the skip row is `SkipChoice`); follow-ups are presented without a skip once the commitment is settled.

### Engine = orchestrator; sibling modules = free functions

`engine.core.Engine` owns the turn loop, setup, and the `ask` plumbing. Everything else (actions, powers, pink reactors, scoring) lives in sibling modules as **free functions whose first argument is the Engine** — `actions.do_play_bird(engine, agent)` directly; no `_do_*` wrapper methods on Engine. Break the import cycle with `if typing.TYPE_CHECKING` and forward-reference annotations.

### The Agent protocol

`Agent` is a `typing.Protocol` with `def __call__[C: decisions.Choice](self, engine, decision: decisions.Decision[C], /) -> C` — non-generic at use sites (`list[Agent]`) while call-site return types are tracked. New agents in `wingspan.agents`; `agents.base.random_agent` is the reference.

### Bird powers: parser + dispatcher pair

`EffectKind` variant in `cards.schema` → `@registry.pattern` matcher in `cards.parse.matchers` (pink: `@registry.pink_pattern` in `pink_matchers`) → `@registry.handles(EffectKind.X)` handler in `engine.powers`. Pink effects dispatch from `engine.reactors`. Keep the `UNIMPLEMENTED` fallback in place. See `docs/BIRDS.md`.

### Network topology and versioning

`architecture.ModelArchitecture` is the single topology descriptor; `PolicyValueNet` builds all blocks from it; `TrainConfig` mirrors the fields via its `arch` property. Any new field that changes a tensor shape must join `ShapeKey` / `architecture_key` (FRESH change requiring a `MODEL_VERSION` bump + compat shim); shape-preserving knobs stay out of it (REGIME). See `docs/VERSIONING.md`.

**The rehydration guarantee.** Loading a run's frozen config under any later same-MAJOR code must reconstitute a model that *computes identically* — old artifacts never adopt new behavior. The shim trigger is **any code change that would make a rehydrated artifact behave differently**, not just a shape change: the real line is config-carried (travels in the artifact, safely REGIME) vs code-carried (lives in the live codebase — featurizer math, slice offsets, inference branches — must be era-gated even when shape is unchanged). Freeze *all* geometry a net derives, not just `encode_state`. The lone exception is the engine: shared by both seats, it can't fork by version, so its choice-calc/apply/present changes are accepted drift. See `docs/VERSIONING.md`.

**Era-pinned training resume.** The guarantee extends to training: a run carries `TrainConfig.encoding_version`, trains as the era's net class at the era's dims, and stamps every artifact it writes with its own era — so a FRESH bump never orphans an in-flight run. The era is adopted from the run directory (configurator seeding + `loop_resume.adopt_checkpoint_era`), never hand-set; superseded eras are therefore *producing* paths — new training features must work at every same-MAJOR era or refuse one explicitly. See `docs/VERSIONING.md`.

### Constants and per-turn scratch state

Action/track/cost constants at the top of `state.py`; encoder dims and stripe offsets in `encode/layout.py`; chart sizes in `training/charts/geometry.py`. No magic numbers — promote them. Per-turn scratch lives on `GameState` as explicit typed fields reset by `GameState.reset_turn_state()`.

## Test conventions

- Tests prepend `src/` to `sys.path` (see `test_smoke.py`) — new tests match this pattern.
- One file per power (`tests/test_powers_*.py`); cross-power smoke test is `test_smoke.py`; training-cycle coverage in `test_model_and_self_play.py`.

## Things to avoid

- **Never delete, move, or modify `*.lock` files** (see workflow above).
- **Never circumvent the workflow scripts** — no direct tool calls, no hand-rolled worktree/merge, no mid-feature script edits.
- No tolerant-parse fallbacks or ghost entries for old artifact formats (see `docs/VERSIONING.md`).
- No `Decision`/`Choice` payload tuples or `Any`-typed context fields — add a typed field to the subclass or define a new Choice subclass.
- Don't bypass `Engine.ask` — it validates the agent's answer against the offered choices.
- No `_do_*` wrapper methods on Engine that just delegate to `actions.do_*`.
- **Never lower `coverage_baseline.txt`** — it's a one-way ratchet; report drops to the user and let them decide.
