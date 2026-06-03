#!/usr/bin/env bash
# Automated worktree merge: loop merge_worktree.sh and invoke `claude -p` to fix
# any conflicts or quality-gate failures, retrying until success or giving up.
#
# Usage: bash scripts/auto_merge_worktree.sh <feature-slug>
#
# Prerequisites:
#   - The human must have ALREADY deleted <slug>.lock from the repo root
#     (this script refuses to run if the lock is still present)
#   - `claude` CLI must be in PATH
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

SLUG="${1:?Usage: bash scripts/auto_merge_worktree.sh <feature-slug>}"
WORKTREE_DIR=".claude/worktrees/$SLUG"
BRANCH="wt/$SLUG"
LOCK_FILE="$SLUG.lock"
MAX_RETRIES=5

# ---- Guard: lock must be gone ----

if [ -f "$LOCK_FILE" ]; then
    echo "BLOCKED: Merge lock still exists — human authorization required."
    echo "  Lock:     $LOCK_FILE"
    echo "  Delete it to authorize: rm \"$LOCK_FILE\""
    exit 1
fi

if ! command -v claude &> /dev/null; then
    echo "ERROR: 'claude' CLI not found in PATH." >&2
    echo "       Install Claude Code: https://claude.ai/code" >&2
    exit 1
fi

# ---- Main loop ----

for attempt in $(seq 1 "$MAX_RETRIES"); do
    echo
    echo "========================================"
    echo "  AUTO-MERGE ATTEMPT $attempt / $MAX_RETRIES"
    echo "========================================"
    echo

    MERGE_OUTPUT=$(bash "$SCRIPT_DIR/merge_worktree.sh" "$SLUG" 2>&1) || MERGE_STATUS=$?
    MERGE_STATUS="${MERGE_STATUS:-0}"

    echo "$MERGE_OUTPUT"

    if [ "$MERGE_STATUS" -eq 0 ]; then
        echo
        echo "==== AUTO-MERGE SUCCEEDED on attempt $attempt ===="
        exit 0
    fi

    # Lock appeared mid-run — stop immediately, do not call Claude
    if [ "$MERGE_STATUS" -eq 1 ]; then
        echo
        echo "BLOCKED: A merge lock appeared. Stopping auto-merge."
        exit 1
    fi

    if [ "$attempt" -ge "$MAX_RETRIES" ]; then
        echo
        echo "==== MAX RETRIES REACHED ($MAX_RETRIES). Manual intervention required. ===="
        exit 1
    fi

    # ---- Determine failure kind and build Claude prompt ----

    if [ "$MERGE_STATUS" -eq 2 ]; then
        FAILURE_KIND="git merge conflicts"
    elif [ "$MERGE_STATUS" -eq 3 ]; then
        FAILURE_KIND="quality gate failure (pyright errors or test failures)"
    else
        FAILURE_KIND="unexpected error (exit $MERGE_STATUS)"
    fi

    echo
    echo "==== Invoking Claude to fix $FAILURE_KIND (attempt $attempt) ===="
    echo

    # Write prompt to a temp file to avoid quoting hell
    PROMPT_FILE=$(mktemp)
    cat > "$PROMPT_FILE" << PROMPT
Automated merge of worktree '$SLUG' into main FAILED on attempt $attempt.
Failure type: $FAILURE_KIND

Output from scripts/merge_worktree.sh:
--- BEGIN OUTPUT ---
$MERGE_OUTPUT
--- END OUTPUT ---

The worktree is at: $WORKTREE_DIR  (branch: $BRANCH)
The main working directory is: $REPO_ROOT

YOUR TASK:
1. Diagnose the root cause from the output above.
2. Fix it by editing files INSIDE THE WORKTREE at $WORKTREE_DIR/src/ or $WORKTREE_DIR/tests/.
   - For git conflicts: update the worktree files so they no longer conflict with main's current HEAD.
   - For pyright errors: fix the type annotations in the worktree source.
   - For test failures: fix the code or tests in the worktree.
   - Do NOT edit files in $REPO_ROOT/src/ or $REPO_ROOT/tests/ (those are main — changes there would be overwritten by the merge).
3. Verify your fix by running the quality gate on the worktree:
     bash scripts/quality_gate.sh "$WORKTREE_DIR"
4. Once the gate passes, commit your changes inside the worktree:
     cd "$WORKTREE_DIR" && git add -A && git commit -m "Fix: <describe what you fixed>"
5. STOP. Do not run scripts/merge_worktree.sh yourself — the calling script will retry it.

HARD CONSTRAINTS:
- NEVER delete, move, or modify any *.lock file in the repo root
- NEVER push to any remote
- NEVER edit files outside of $WORKTREE_DIR (except to read context from $REPO_ROOT)
PROMPT

    claude -p "$(cat "$PROMPT_FILE")"
    rm -f "$PROMPT_FILE"
done

echo
echo "==== AUTO-MERGE FAILED after $MAX_RETRIES attempts ===="
exit 1
