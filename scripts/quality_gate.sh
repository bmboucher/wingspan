#!/usr/bin/env bash
# Run the quality gate: pyright -> isort -> black -> pyright -> pytest.
#
# Usage:
#   bash scripts/quality_gate.sh [target-dir] [--pyright [args...]] [--format [paths...]] [--pytest [args...]] [--coverage]
#
# target-dir  Directory to check (default: repo root, derived from this script's
#             location). Must come BEFORE the first section flag. Pass a worktree
#             path to gate a worktree before merging.
#
# Section flags select which steps run. With no section flags the full gate runs:
# pyright -> isort -> black -> pyright -> pytest. Steps always execute in that
# canonical order regardless of the order flags appear on the command line.
#
# Every argument after a section flag (up to the next section flag) is passed
# verbatim to the underlying tool:
#
#   --pyright [args...]   pyright       (default: no args — pyproject.toml config)
#   --format  [paths...]  isort + black (default paths: src tests)
#   --pytest  [args...]   pytest        (default args: tests/ -n $WINGSPAN_PYTEST_WORKERS
#                                        --dist load — parallel via pytest-xdist.
#                                        Explicit args replace the default entirely,
#                                        so targeted runs stay serial unless the
#                                        caller passes -n themselves. Override the
#                                        worker count with WINGSPAN_PYTEST_WORKERS;
#                                        0 = serial.)
#   --coverage            Run pytest serially with --cov in a single pass, then
#                         check TOTAL coverage against coverage_baseline.txt.
#                         Takes no arguments. Implies --pytest (uses the full
#                         serial+cov run instead of the default parallel run).
#                         Used by merge_worktree.sh; not needed during worktree
#                         iteration.
#
# Examples:
#   bash scripts/quality_gate.sh                                # full gate (fast, no coverage)
#   bash scripts/quality_gate.sh --coverage                     # full gate + coverage regression
#   bash scripts/quality_gate.sh --pyright                      # type-check only
#   bash scripts/quality_gate.sh --pytest tests/test_smoke.py   # one test file
#   bash scripts/quality_gate.sh --pytest -k house_wren -x      # pytest flags
#   bash scripts/quality_gate.sh --pyright --pytest             # types + tests
#   bash scripts/quality_gate.sh .claude/worktrees/<slug> --pytest tests/test_smoke.py
#
# Exit codes:
#   0  gate passed
#   1  genuine check failure (type errors or failing tests) — fix the code, rerun
#   2  infrastructure/usage error (missing venv, pyright not on PATH, bad target
#      dir, invalid arguments) — NOT a code problem; stop and ask the user to fix
#      the environment, do not run the underlying tools directly
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

EXIT_CHECK_FAILED=1
EXIT_INFRA=2

# pytest-xdist worker count for the *default* full-suite pytest run (full gate,
# or a bare --pytest). Sized for the suite's shape: wall-clock is bounded by a
# few unsplittable multi-second tests, and each worker pays a one-time torch
# import, so more workers than this buys little. 0 = run serial.
WINGSPAN_PYTEST_WORKERS="${WINGSPAN_PYTEST_WORKERS:-8}"

# Report an environment/script problem (as opposed to a genuine check failure)
# and exit with the dedicated infrastructure code.
infra_error() {
    echo "ERROR: $*" >&2
    echo >&2
    echo "This is an environment/script problem, NOT a code problem." >&2
    echo "STOP and ask the user to fix it before continuing. Do not run pyright," >&2
    echo "pytest, isort, or black directly, and do not work around the gate." >&2
    exit "$EXIT_INFRA"
}

# Print the header comment block (everything between the shebang and the first
# non-comment line) as the usage text.
usage() {
    awk 'NR > 1 && !/^#/ { exit } NR > 1 { sub(/^# ?/, ""); print }' "${BASH_SOURCE[0]}"
}

# ---- Argument parsing ----

TARGET_DIR="$REPO_ROOT"
RUN_PYRIGHT=false
RUN_FORMAT=false
RUN_PYTEST=false
RUN_COVERAGE=false
PYRIGHT_ARGS=()
FORMAT_ARGS=()
PYTEST_ARGS=()
SECTION=""   # section that bare args currently belong to ("" = before any flag)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pyright)  SECTION="pyright"; RUN_PYRIGHT=true ;;
        --format)   SECTION="format";  RUN_FORMAT=true ;;
        --pytest)   SECTION="pytest";  RUN_PYTEST=true ;;
        --coverage) SECTION="";        RUN_COVERAGE=true ;;
        --only|--only=*)
            infra_error "--only has been removed. Use section flags instead: --pyright / --format / --pytest (arguments after each flag are passed to that tool, e.g. --pytest tests/test_smoke.py)."
            ;;
        *)
            if [ -n "$SECTION" ]; then
                case "$SECTION" in
                    pyright) PYRIGHT_ARGS+=("$1") ;;
                    format)  FORMAT_ARGS+=("$1") ;;
                    pytest)  PYTEST_ARGS+=("$1") ;;
                esac
            elif [[ "$1" == "-h" || "$1" == "--help" ]]; then
                usage
                exit 0
            elif [[ "$1" == -* ]]; then
                infra_error "Unknown flag: $1 (section flags: --pyright / --format / --pytest / --coverage; run with --help for usage)"
            else
                TARGET_DIR="$1"
            fi
            ;;
    esac
    shift
done

# --coverage implies --pytest (the coverage run IS the pytest run).
if [ "$RUN_COVERAGE" = true ]; then
    RUN_PYTEST=true
fi

# Default args for sections that were requested bare.
if [ ${#FORMAT_ARGS[@]} -eq 0 ]; then FORMAT_ARGS=(src tests); fi
if [ ${#PYTEST_ARGS[@]} -eq 0 ]; then
    if ! [[ "$WINGSPAN_PYTEST_WORKERS" =~ ^[0-9]+$ ]]; then
        infra_error "WINGSPAN_PYTEST_WORKERS must be a non-negative integer (got: $WINGSPAN_PYTEST_WORKERS)"
    fi
    if [ "$RUN_COVERAGE" = true ]; then
        # Coverage run: serial for clean term-missing output, with --cov.
        PYTEST_ARGS=(tests/ -p no:xdist --cov --cov-report=term-missing)
    elif [ "$WINGSPAN_PYTEST_WORKERS" -gt 0 ]; then
        PYTEST_ARGS=(tests/ -n "$WINGSPAN_PYTEST_WORKERS" --dist load)
    else
        PYTEST_ARGS=(tests/)
    fi
fi

# No section flags at all -> full gate.
FULL_GATE=false
if [ "$RUN_PYRIGHT" = false ] && [ "$RUN_FORMAT" = false ] && [ "$RUN_PYTEST" = false ]; then
    FULL_GATE=true
    RUN_PYRIGHT=true
    RUN_FORMAT=true
    RUN_PYTEST=true
fi

# ---- Helpers ----

header() {
    echo
    echo "========================================"
    echo "  $*"
    echo "========================================"
    echo
}

# ---- Preflight ----

# Resolve to absolute path so PYTHON stays valid after cd "$TARGET_DIR".
RESOLVED_TARGET_DIR="$(cd "$TARGET_DIR" 2>/dev/null && pwd)" || {
    infra_error "Target directory not found: $TARGET_DIR"
}
TARGET_DIR="$RESOLVED_TARGET_DIR"

PYTHON="$TARGET_DIR/.venv/Scripts/python.exe"

if [ ! -f "$PYTHON" ]; then
    infra_error "No venv found at $TARGET_DIR/.venv
       Main repo:  pip install -e '.[dev]'
       Worktree:   bash scripts/create_worktree.sh <slug>  (venv is set up automatically)"
fi

if ! command -v pyright &> /dev/null; then
    infra_error "pyright not found in PATH (install: npm install -g pyright)"
fi

cd "$TARGET_DIR"
if [ "$FULL_GATE" = true ]; then
    if [ "$RUN_COVERAGE" = true ]; then
        echo "==== QUALITY GATE (with coverage): $TARGET_DIR ===="
    else
        echo "==== QUALITY GATE: $TARGET_DIR ===="
    fi
else
    STEPS=""
    [ "$RUN_PYRIGHT" = true ] && STEPS="$STEPS pyright"
    [ "$RUN_FORMAT" = true ] && STEPS="$STEPS format"
    [ "$RUN_PYTEST" = true ] && STEPS="$STEPS pytest"
    [ "$RUN_COVERAGE" = true ] && STEPS="$STEPS coverage"
    echo "==== QUALITY GATE: $TARGET_DIR  [steps:$STEPS] ===="
fi

FAILED=0

# ---- Step 1: pyright (initial) ----

if [ "$RUN_PYRIGHT" = true ]; then
    header "pyright (strict type check)"
    if ! pyright "${PYRIGHT_ARGS[@]}"; then
        echo
        echo "GATE FAILED at pyright — fix type errors before continuing."
        # Stop early: if format is also requested we don't want to format broken code.
        exit "$EXIT_CHECK_FAILED"
    fi
fi

# ---- Step 2+3: isort + black ----

if [ "$RUN_FORMAT" = true ]; then
    header "isort (import sort)"
    if ! "$PYTHON" -m isort "${FORMAT_ARGS[@]}"; then
        echo
        echo "GATE FAILED at isort."
        exit "$EXIT_CHECK_FAILED"
    fi

    header "black (format)"
    if ! "$PYTHON" -m black "${FORMAT_ARGS[@]}"; then
        echo
        echo "GATE FAILED at black."
        exit "$EXIT_CHECK_FAILED"
    fi
fi

# ---- Step 4: pyright post-format (full gate, or when both pyright+format requested) ----

if [ "$RUN_PYRIGHT" = true ] && [ "$RUN_FORMAT" = true ]; then
    header "pyright (post-format verification)"
    if ! pyright "${PYRIGHT_ARGS[@]}"; then
        echo
        echo "GATE FAILED at post-format pyright — formatting introduced a type error."
        FAILED=1
    fi
fi

# ---- Step 5: pytest ----
#
# When --coverage is set, PYTEST_ARGS already include --cov/--cov-report so the
# output from this single run is also what we parse for the regression check.
# We tee it to a temp file so the regression check can extract the TOTAL line
# while still printing live output to the terminal.

if [ "$RUN_PYTEST" = true ]; then
    header "pytest"
    if [ "$RUN_COVERAGE" = true ]; then
        COVERAGE_OUTPUT_FILE="$(mktemp)"
        "$PYTHON" -m pytest "${PYTEST_ARGS[@]}" 2>&1 | tee "$COVERAGE_OUTPUT_FILE"
        PYTEST_EXIT="${PIPESTATUS[0]}"
        if [ "$PYTEST_EXIT" -ne 0 ]; then
            echo
            echo "GATE FAILED at pytest — tests failed."
            rm -f "$COVERAGE_OUTPUT_FILE"
            COVERAGE_OUTPUT_FILE=""
            FAILED=1
        fi
    else
        if ! "$PYTHON" -m pytest "${PYTEST_ARGS[@]}"; then
            echo
            echo "GATE FAILED at pytest — tests failed."
            FAILED=1
        fi
    fi
fi

# ---- Step 6: coverage regression check (--coverage only) ----
#
# Runs after Step 5 when --coverage was passed and all prior steps passed.
# PYTEST_ARGS already included --cov/--cov-report; COVERAGE_OUTPUT_FILE holds
# the tee'd output from that single run.
#
# Ratchet behaviour:
#   Baseline absent       -> create it, pass (first-run establishment).
#   Coverage >= baseline  -> update file if improved, pass.
#   Coverage < baseline   -> print regression message, set FAILED=1.
#
# BASELINE_FILE always resolves to the main repo root even from a worktree
# because REPO_ROOT is derived from this script's own location, not CWD.

BASELINE_FILE="$REPO_ROOT/coverage_baseline.txt"

if [ "$RUN_COVERAGE" = true ] && [ "$FAILED" -eq 0 ] && [ -n "${COVERAGE_OUTPUT_FILE:-}" ]; then
    header "coverage (regression check)"

    # Extract TOTAL line: "TOTAL   NNN  NNN  NNN  NNN  78%"  (last field = %)
    COVERAGE_PCT="$(grep -E '^TOTAL\s' "$COVERAGE_OUTPUT_FILE" \
        | awk '{print $NF}' \
        | tr -d '%')"
    rm -f "$COVERAGE_OUTPUT_FILE"

    if [ -z "$COVERAGE_PCT" ]; then
        echo
        echo "WARNING: could not extract TOTAL coverage line — skipping regression check."
    elif [ ! -f "$BASELINE_FILE" ]; then
        # First run: establish the baseline.
        echo "$COVERAGE_PCT" > "$BASELINE_FILE"
        echo
        echo "Coverage baseline established: ${COVERAGE_PCT}%"
        echo "  Written to: $BASELINE_FILE"
        echo "  Commit this file to lock the regression floor."
    else
        BASELINE_PCT="$(cat "$BASELINE_FILE" | tr -d '[:space:]')"

        # Floating-point comparisons via awk (bash only does integer math).
        REGRESSED="$(awk -v cur="$COVERAGE_PCT" -v base="$BASELINE_PCT" \
            'BEGIN { print (cur + 0 < base + 0) ? "1" : "0" }')"
        IMPROVED="$(awk -v cur="$COVERAGE_PCT" -v base="$BASELINE_PCT" \
            'BEGIN { print (cur + 0 > base + 0) ? "1" : "0" }')"

        if [ "$REGRESSED" = "1" ]; then
            echo
            echo "COVERAGE REGRESSION: ${COVERAGE_PCT}% is below baseline ${BASELINE_PCT}%"
            echo
            echo "  Options:"
            echo "    1. Add tests to recover coverage, then rerun the gate."
            echo "    2. If the drop is intentional, report the impact to the user"
            echo "       so they can update coverage_baseline.txt manually."
            echo "  Do NOT edit coverage_baseline.txt downward yourself."
            FAILED=1
        elif [ "$IMPROVED" = "1" ]; then
            echo "$COVERAGE_PCT" > "$BASELINE_FILE"
            echo
            echo "Coverage improved: ${BASELINE_PCT}% -> ${COVERAGE_PCT}%"
            echo "  Baseline updated. Commit coverage_baseline.txt with your change."
        else
            echo
            echo "Coverage: ${COVERAGE_PCT}% (matches baseline — OK)"
        fi
    fi
fi

# ---- Summary ----

echo
echo "========================================"
if [ $FAILED -eq 0 ]; then
    echo "  QUALITY GATE PASSED"
else
    echo "  QUALITY GATE FAILED"
fi
echo "========================================"

if [ $FAILED -ne 0 ]; then
    exit "$EXIT_CHECK_FAILED"
fi
exit 0
