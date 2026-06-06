# Quality gate reference

Full reference for `scripts/quality_gate.sh`. Load this file when you need to
iterate on the gate (targeted runs, section flags, coverage), or when
`merge_worktree.sh` exits with a non-zero code and you need to understand what
happened.

## What the gate does

```
bash scripts/quality_gate.sh [target-dir]
```

Run from the current directory (worktree or repo root), or pass an explicit
target dir. Five steps in order:

1. `pyright` (strict) — type-check `src/`
2. `isort` — import ordering
3. `black` — formatting
4. `pyright` (post-format) — re-check after any formatter changes
5. `pytest` — full test suite

Config lives in `pyproject.toml`. `pyright` is the globally-installed npm
binary; formatters, pytest, and coverage run via the target directory's own
`.venv`.

## Exit codes

| Code | Meaning | What to do |
|------|---------|------------|
| 0 | All checks passed | proceed |
| 1 | Genuine check failure (pyright errors, failing tests, coverage regression) | fix the code and rerun |
| 2 | Infrastructure failure (missing venv, `pyright` not on PATH, bad arguments) | **stop — ask the user to fix the environment** |

Always run the full gate (no section flags) before committing. Strict-mode
pyright must be completely clean (`reportPrivateImportUsage = false` silences
torch's under-exporting stubs — don't re-enable it).

## Section flags (targeted runs)

Everything after a section flag passes verbatim to the underlying tool. Steps
always execute in canonical gate order regardless of flag order.

```
bash scripts/quality_gate.sh --pyright src/wingspan/state.py   # types only / one file
bash scripts/quality_gate.sh --format                          # isort + black only
bash scripts/quality_gate.sh --pytest tests/test_encode.py -k state -x -q
bash scripts/quality_gate.sh --coverage                        # full gate + coverage regression
```

No-argument defaults:
- `--pytest` → `tests/ -n 8 --dist load`
  - Worker count via `WINGSPAN_PYTEST_WORKERS` env var; `0` = serial.
  - Explicit args replace the default, so targeted runs stay serial.
- `--format` → `src tests`

## Coverage regression check

Pass `--coverage` (no args) to run pytest serially with `--cov
--cov-report=term-missing`, then compare the TOTAL percentage against
`coverage_baseline.txt` in the repo root.

`merge_worktree.sh` always passes `--coverage`; it is not needed during worktree
iteration.

The baseline ratchets upward:
- Coverage improves → baseline file updated automatically; commit it with your change.
- Coverage unchanged → gate passes silently.
- Coverage drops → gate fails (exit 1). Either add tests to recover coverage,
  or report the drop to the user so they can decide whether to lower the
  baseline manually.

**First run (baseline absent):** the gate creates `coverage_baseline.txt`
automatically and passes. Commit that file to lock the regression floor.

Modules excluded from coverage measurement (CLI entry points, SVG/chart
rendering, AWS/S3 integration) are listed in `[tool.coverage.run] omit` in
`pyproject.toml`. To include a module, remove it from that list.

**Never edit `coverage_baseline.txt` to lower the percentage.** The ratchet is
a one-way floor. Report a coverage drop to the user and let them decide.

## merge_worktree.sh exit codes

`bash scripts/merge_worktree.sh <feature-slug>` — run from the main working
directory only, after the human has deleted `<slug>.lock`.

| Exit | Meaning | What Claude does |
|------|---------|------------------|
| 0 | merged, pushed, cleaned up | report done |
| 1 | merge-auth lock still present | stop — human authorization missing |
| 2 | merge conflicts | fix in the worktree, commit there, retry |
| 3 | gate failed on the merged result | fix in the worktree, commit there, retry |
| 4 | preflight failure (worktree/branch missing, or worktree has uncommitted changes) | commit the worktree work if that is the cause; otherwise stop and report |
| 5 | gate or venv refresh could not run (infrastructure failure) | **stop — ask the user to fix the environment** |

`bash scripts/auto_merge_worktree.sh <feature-slug>` — fully automated variant
the *human* runs: loops `merge_worktree.sh`, spawning `claude -p` subprocesses
to fix conflicts / gate failures, up to 5 attempts. Requires the lock to be
already deleted and `claude` on PATH. Stops immediately on exit 1 (lock
present) or exit 5 (infrastructure failure).
