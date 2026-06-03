#!/usr/bin/env bash
# Create a feature worktree from the current HEAD of main and set up the merge-auth lock.
# Usage: bash scripts/create_worktree.sh <feature-slug>
#
# After this script runs, implement changes in .claude/worktrees/<slug>, then pass the
# quality gate there.  When done, report ready and wait — do NOT merge until the human
# deletes the lock file.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

SLUG="${1:?Usage: bash scripts/create_worktree.sh <feature-slug>}"
WORKTREE_DIR=".claude/worktrees/$SLUG"
BRANCH="wt/$SLUG"
LOCK_FILE="$SLUG.lock"

# ---- Preflight ----

if [ -d "$WORKTREE_DIR" ]; then
    echo "ERROR: Worktree already exists at $WORKTREE_DIR" >&2
    echo "       Remove it first: git worktree remove $WORKTREE_DIR" >&2
    exit 1
fi

if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
    echo "ERROR: Branch $BRANCH already exists." >&2
    echo "       Delete it first: git branch -D $BRANCH" >&2
    exit 1
fi

# ---- Commit any dirty state in main ----

if ! git diff --quiet || ! git diff --cached --quiet \
        || [ -n "$(git ls-files --others --exclude-standard)" ]; then
    echo "==== Committing uncommitted changes in main before branching ===="
    git add -A
    git commit -m "WIP: pre-worktree snapshot for $SLUG"
    echo
fi

# ---- Create worktree from current HEAD ----

echo "==== Creating worktree ===="
echo "  Dir:    $WORKTREE_DIR"
echo "  Branch: $BRANCH"
echo "  Base:   $(git rev-parse --short HEAD) ($(git log -1 --format='%s'))"
echo
git worktree add "$WORKTREE_DIR" -b "$BRANCH" HEAD

# ---- Set up fresh venv in worktree ----

echo "==== Installing fresh venv in worktree (this takes ~30-60s) ===="
MAIN_PYTHON="$REPO_ROOT/.venv/Scripts/python.exe"
if [ ! -f "$MAIN_PYTHON" ]; then
    echo "ERROR: Main repo venv not found at $REPO_ROOT/.venv" >&2
    echo "       Run: pip install -e '.[dev]'  in the repo root first." >&2
    exit 1
fi
"$MAIN_PYTHON" -m venv "$WORKTREE_DIR/.venv"
(cd "$WORKTREE_DIR" && "$WORKTREE_DIR/.venv/Scripts/pip" install --quiet -e ".[dev]")
echo "  Venv ready: $WORKTREE_DIR/.venv"
echo

# ---- Create merge-auth lock ----

cat > "$LOCK_FILE" << EOF
Merge of '$BRANCH' into 'main' has NOT been authorized.

Review the changes at:  $WORKTREE_DIR
When satisfied, delete this file to authorize the merge:

    rm "$LOCK_FILE"

Then run:
    bash scripts/merge_worktree.sh $SLUG
  or (fully automated):
    bash scripts/auto_merge_worktree.sh $SLUG

Created: $(date -u '+%Y-%m-%dT%H:%M:%SZ')
EOF

# ---- Summary ----

echo
echo "==== Worktree ready ===="
echo "  Worktree: $WORKTREE_DIR"
echo "  Branch:   $BRANCH"
echo "  Lock:     $LOCK_FILE  (delete to authorize merge)"
echo
echo "Implement your changes in $WORKTREE_DIR, run the quality gate:"
echo "  bash scripts/quality_gate.sh $WORKTREE_DIR"
echo
echo "When the gate passes, commit inside the worktree and report ready."
echo "Do NOT merge until the human deletes the lock file."
