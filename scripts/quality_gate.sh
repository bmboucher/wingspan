#!/usr/bin/env bash
# Run the quality gate: pyright -> isort -> black -> pyright -> pytest.
#
# Usage:
#   bash scripts/quality_gate.sh [target-dir] [--only <steps>]
#
# target-dir  Directory to check (default: repo root, derived from this script's location).
#             Pass a worktree path to gate a worktree before merging.
#
# --only      Comma-separated subset of steps to run: pyright, format, pytest
#             Examples:
#               --only pyright          type-check only (one pass)
#               --only pytest           tests only
#               --only format           isort + black only
#               --only pyright,pytest   type-check + tests, skip format
#             Default (no --only): full gate — pyright → isort → black → pyright → pytest
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# ---- Argument parsing ----

TARGET_DIR="$REPO_ROOT"
ONLY_STEPS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --only)
            shift
            ONLY_STEPS="${1:-}"
            shift
            ;;
        --only=*)
            ONLY_STEPS="${1#--only=}"
            shift
            ;;
        -*)
            echo "ERROR: Unknown flag: $1" >&2
            echo "Usage: bash scripts/quality_gate.sh [target-dir] [--only pyright|format|pytest]" >&2
            exit 1
            ;;
        *)
            TARGET_DIR="$1"
            shift
            ;;
    esac
done

# ---- Helpers ----

# Returns 0 (true) if the given step should run given the --only filter.
should_run() {
    local step="$1"
    [ -z "$ONLY_STEPS" ] || echo "$ONLY_STEPS" | tr ',' '\n' | grep -qx "$step"
}

header() {
    echo
    echo "========================================"
    echo "  $*"
    echo "========================================"
    echo
}

# ---- Preflight ----

PYTHON="$TARGET_DIR/.venv/Scripts/python.exe"

if [ ! -f "$PYTHON" ]; then
    echo "ERROR: No venv found at $TARGET_DIR/.venv" >&2
    echo "       Main repo:  pip install -e '.[dev]'" >&2
    echo "       Worktree:   bash scripts/create_worktree.sh <slug>  (venv is set up automatically)" >&2
    exit 1
fi

if ! command -v pyright &> /dev/null; then
    echo "ERROR: pyright not found in PATH (install: npm install -g pyright)" >&2
    exit 1
fi

if [ ! -d "$TARGET_DIR" ]; then
    echo "ERROR: Target directory not found: $TARGET_DIR" >&2
    exit 1
fi

cd "$TARGET_DIR"
if [ -n "$ONLY_STEPS" ]; then
    echo "==== QUALITY GATE: $TARGET_DIR  [steps: $ONLY_STEPS] ===="
else
    echo "==== QUALITY GATE: $TARGET_DIR ===="
fi

FAILED=0

# ---- Step 1: pyright (initial) ----

if should_run "pyright"; then
    header "pyright (strict type check)"
    if ! pyright; then
        echo
        echo "GATE FAILED at pyright — fix type errors before continuing."
        # Stop early only when doing the full gate or when only checking types.
        # If format is also requested, we still want to skip to avoid formatting broken code.
        exit 1
    fi
fi

# ---- Step 2+3: isort + black ----

if should_run "format"; then
    header "isort (import sort)"
    "$PYTHON" -m isort src tests

    header "black (format)"
    "$PYTHON" -m black src tests
fi

# ---- Step 4: pyright post-format (only in full gate or when both pyright+format requested) ----

RUN_POST_FORMAT_PYRIGHT=false
if [ -z "$ONLY_STEPS" ]; then
    RUN_POST_FORMAT_PYRIGHT=true
elif should_run "pyright" && should_run "format"; then
    RUN_POST_FORMAT_PYRIGHT=true
fi

if [ "$RUN_POST_FORMAT_PYRIGHT" = "true" ]; then
    header "pyright (post-format verification)"
    if ! pyright; then
        echo
        echo "GATE FAILED at post-format pyright — formatting introduced a type error."
        FAILED=1
    fi
fi

# ---- Step 5: pytest ----

if should_run "pytest"; then
    header "pytest"
    if ! "$PYTHON" -m pytest tests/; then
        echo
        echo "GATE FAILED at pytest — tests failed."
        FAILED=1
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

exit $FAILED
