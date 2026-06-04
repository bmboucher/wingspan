#!/usr/bin/env bash
# Squash-merge a worktree into main, refresh main's venv if the merge changed
# pyproject.toml, run the quality gate, commit + push, clean up.
# Usage: bash scripts/merge_worktree.sh <feature-slug>
#
# Exit codes:
#   0  — merge complete, pushed, worktree removed
#   1  — merge-auth lock is present (human must delete it first)
#   2  — git conflicts during squash merge (fix in worktree, retry)
#   3  — quality gate failed after merge (fix in worktree, retry)
#   4  — worktree or branch not found / other preflight failure
#   5  — quality gate or venv update could not run (infrastructure failure,
#        e.g. missing venv, pyright not on PATH, or pip install failing) — a
#        human must fix the environment; do NOT attempt to work around the gate
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

SLUG="${1:?Usage: bash scripts/merge_worktree.sh <feature-slug>}"
WORKTREE_DIR=".claude/worktrees/$SLUG"
BRANCH="wt/$SLUG"
LOCK_FILE="$SLUG.lock"

# ---- Step 1: Check merge-auth lock ----

if [ -f "$LOCK_FILE" ]; then
    echo "MERGE BLOCKED: Authorization lock exists."
    echo "  Lock:     $LOCK_FILE"
    echo "  Worktree: $WORKTREE_DIR"
    echo
    echo "Review the changes, then delete the lock to authorize:"
    echo "  rm \"$LOCK_FILE\""
    exit 1
fi

# ---- Step 2: Preflight ----

if [ ! -d "$WORKTREE_DIR" ]; then
    echo "ERROR: Worktree not found at $WORKTREE_DIR" >&2
    exit 4
fi

if ! git show-ref --verify --quiet "refs/heads/$BRANCH"; then
    echo "ERROR: Branch $BRANCH not found." >&2
    exit 4
fi

# Ensure worktree has a clean, committed HEAD (nothing should be dangling)
WORKTREE_DIRTY=$(cd "$WORKTREE_DIR" && git status --porcelain)
if [ -n "$WORKTREE_DIRTY" ]; then
    echo "ERROR: Worktree $WORKTREE_DIR has uncommitted changes." >&2
    echo "       Commit them in the worktree first:" >&2
    echo "         cd $WORKTREE_DIR && git add -A && git commit -m \"...\"" >&2
    exit 4
fi

# ---- Step 3: Commit any dirty state in main ----

if ! git diff --quiet || ! git diff --cached --quiet \
        || [ -n "$(git ls-files --others --exclude-standard)" ]; then
    echo "==== Committing uncommitted changes in main before merge ===="
    git add -A
    git commit -m "WIP: pre-merge snapshot before merging $SLUG"
    echo
fi

# ---- Step 4: Squash merge ----

echo "==== Squash-merging $BRANCH into main ===="
if ! git merge --squash "$BRANCH" 2>&1; then
    # Capture conflicting files before resetting
    CONFLICTING=$(git diff --name-only --diff-filter=U 2>/dev/null || echo "(unknown)")
    echo
    echo "MERGE CONFLICTS detected. Resetting main to pre-merge state."
    git reset --hard HEAD
    echo
    echo "Conflicting files:"
    echo "$CONFLICTING" | sed 's/^/  /'
    echo
    echo "Fix the conflicts by updating the worktree ($WORKTREE_DIR) so these"
    echo "files no longer conflict with main, commit the fix, then retry."
    exit 2
fi

# ---- Step 5: Sync main's venv with the merged dependency set ----

# The squash merge stages its changes, so a staged pyproject.toml means the
# dependency set (or project metadata) may have changed. The gate below runs
# with main's .venv — refresh it first, or a feature that adds a dev
# dependency could never pass the merge gate.
if ! git diff --cached --quiet -- pyproject.toml; then
    echo
    echo "==== pyproject.toml changed; updating main's venv ===="
    MAIN_PYTHON="$REPO_ROOT/.venv/Scripts/python.exe"
    if [ ! -f "$MAIN_PYTHON" ] || ! "$MAIN_PYTHON" -m pip install --quiet -e ".[dev]"; then
        echo
        echo "VENV UPDATE FAILED (infrastructure failure). Rolling back squash merge."
        git reset --hard HEAD
        echo
        echo "Main's .venv may be partially updated. A human must fix the environment"
        echo "(e.g. run: pip install -e '.[dev]' in the repo root), then retry the merge."
        echo "Do NOT attempt to work around the gate."
        exit 5
    fi
fi

# ---- Step 6: Quality gate ----

echo
echo "==== Running quality gate on merged result ===="
bash "$SCRIPT_DIR/quality_gate.sh"
GATE_STATUS=$?
if [ "$GATE_STATUS" -eq 2 ]; then
    echo
    echo "QUALITY GATE COULD NOT RUN (infrastructure failure). Rolling back squash merge."
    git reset --hard HEAD
    echo
    echo "This is an environment/script problem, not a code problem. A human must"
    echo "fix the environment (see the gate output above), then retry the merge."
    echo "Do NOT attempt to work around the gate."
    exit 5
elif [ "$GATE_STATUS" -ne 0 ]; then
    echo
    echo "Quality gate failed. Rolling back squash merge."
    git reset --hard HEAD
    echo
    echo "Fix the issues in the worktree ($WORKTREE_DIR), commit the fix, then retry."
    exit 3
fi

# ---- Step 7: Commit, push ----

echo
echo "==== Committing merge ===="
git add -A
git commit -m "$(cat << EOF
Merge $SLUG

Squash-merged from branch $BRANCH.
EOF
)"

echo "==== Pushing to origin/main ===="
git push origin main

# ---- Step 8: Clean up worktree and branch ----

echo "==== Cleaning up worktree and branch ===="
git worktree remove "$WORKTREE_DIR"
git branch -D "$BRANCH"

echo
echo "==== MERGE COMPLETE: $SLUG ===="
