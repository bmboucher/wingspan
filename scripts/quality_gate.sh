#!/usr/bin/env bash
# Run the full quality gate: pyright -> isort -> black -> pyright -> pytest.
# Usage: bash scripts/quality_gate.sh [target-dir]
#   target-dir  Directory to check (default: repo root, derived from this script's location).
#               Pass a worktree path to gate a worktree before merging.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
TARGET_DIR="${1:-$REPO_ROOT}"
PYTHON="$REPO_ROOT/.venv/Scripts/python.exe"

# ---- Preflight ----

if [ ! -f "$PYTHON" ]; then
    echo "ERROR: Python venv not found at $REPO_ROOT/.venv" >&2
    echo "       Run: pip install -e \".[dev]\"" >&2
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
echo "==== QUALITY GATE: $TARGET_DIR ===="

# ---- Helpers ----

header() {
    echo
    echo "========================================"
    echo "  $*"
    echo "========================================"
    echo
}

FAILED=0

# ---- Step 1: pyright (initial) ----

header "STEP 1/5: pyright (strict type check)"
if ! pyright; then
    echo
    echo "GATE FAILED at step 1 — fix type errors before formatting."
    exit 1
fi

# ---- Step 2: isort ----

header "STEP 2/5: isort (import sort)"
"$PYTHON" -m isort src tests

# ---- Step 3: black ----

header "STEP 3/5: black (format)"
"$PYTHON" -m black src tests

# ---- Step 4: pyright (post-format) ----

header "STEP 4/5: pyright (post-format verification)"
if ! pyright; then
    echo
    echo "GATE FAILED at step 4 — formatting introduced a type error."
    FAILED=1
fi

# ---- Step 5: pytest ----

header "STEP 5/5: pytest"
if ! "$PYTHON" -m pytest tests/; then
    echo
    echo "GATE FAILED at step 5 — tests failed."
    FAILED=1
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
