#!/usr/bin/env bash
# ── QQ_Cameras → GitHub sync ──────────────────────────────────────────────────
# Double-click this file in Finder to commit & push any changes to GitHub.
# ─────────────────────────────────────────────────────────────────────────────

REPO="$HOME/Documents/QuitQuito/Cameras"

cd "$REPO" || { echo "ERROR: Could not cd to $REPO"; read -r -p "Press Enter..."; exit 1; }

echo "── QQ_Cameras git sync ──────────────────────────"
git status --short

# Nothing to commit?
if git diff --quiet && git diff --cached --quiet && \
   [ -z "$(git ls-files --others --exclude-standard)" ]; then
    echo ""
    echo "Nothing to commit — already up to date."
    read -r -p "Press Enter to close..."
    exit 0
fi

# Auto-generate commit message with timestamp
MSG="chore: auto-sync $(date '+%Y-%m-%d %H:%M')"
echo ""
echo "Committing: $MSG"
git add -A
git commit -m "$MSG"

echo ""
echo "Pushing to origin..."
git push

echo ""
echo "Done! ✓"
read -r -p "Press Enter to close..."
