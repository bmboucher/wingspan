#!/usr/bin/env bash
# Run the quality gate: pyright -> isort -> black -> pyright -> pytest.
#
# Usage:
#   bash scripts/quality_gate.sh [target-dir] [--pyright [args...]] [--format [paths...]] [--pytest [args...]]
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
#   --pytest  [args...]   pytest        (default args: tests/)
#
# Examples:
#   bash scripts/quality_gate.sh                                # full gate
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
PYRIGHT_ARGS=()
FORMAT_ARGS=()
PYTEST_ARGS=()
SECTION=""   # section that bare args currently belong to ("" = before any flag)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pyright) SECTION="pyright"; RUN_PYRIGHT=true ;;
        --format)  SECTION="format";  RUN_FORMAT=true ;;
        --pytest)  SECTION="pytest";  RUN_PYTEST=true ;;
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
                infra_error "Unknown flag: $1 (section flags: --pyright / --format / --pytest; run with --help for usage)"
            else
                TARGET_DIR="$1"
            fi
            ;;
    esac
    shift
done

# Default args for sections that were requested bare.
if [ ${#FORMAT_ARGS[@]} -eq 0 ]; then FORMAT_ARGS=(src tests); fi
if [ ${#PYTEST_ARGS[@]} -eq 0 ]; then PYTEST_ARGS=(tests/); fi

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
    echo "==== QUALITY GATE: $TARGET_DIR ===="
else
    STEPS=""
    [ "$RUN_PYRIGHT" = true ] && STEPS="$STEPS pyright"
    [ "$RUN_FORMAT" = true ] && STEPS="$STEPS format"
    [ "$RUN_PYTEST" = true ] && STEPS="$STEPS pytest"
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

if [ "$RUN_PYTEST" = true ]; then
    header "pytest"
    if ! "$PYTHON" -m pytest "${PYTEST_ARGS[@]}"; then
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

if [ $FAILED -ne 0 ]; then
    exit "$EXIT_CHECK_FAILED"
fi
exit 0
