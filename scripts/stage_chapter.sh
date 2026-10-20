#!/usr/bin/env bash
# Stage ONE chapter (its folder + its test files) so it can be committed and pushed on its own.
# Usage (from the project root):  bash scripts/stage_chapter.sh 3
# It only runs `git add` -- you commit and push yourself.
set -e
cd "$(dirname "$0")/.."
[ -z "$1" ] && { echo "usage: bash scripts/stage_chapter.sh <chapter-number>"; exit 1; }
N=$(printf "%02d" "$1")
DIR=$(ls -d Chapter-$N-* 2>/dev/null | head -1)
[ -z "$DIR" ] && { echo "No folder Chapter-$N-* found"; exit 1; }
git add "$DIR"
for f in tests/R/test-ch$N.R tests/python/test_ch$N.py tests/reference/make_reference_ch$N.R \
         tests/reference/ch${N}_reference.json python/fpp_ch$N.py data/extra_ch$N; do
  [ -e "$f" ] && git add "$f"
done
TITLE=$(echo "$DIR" | sed -E "s/Chapter-$N-//; s/-/ /g")
echo "Staged $DIR and its tests. Next:"
echo "  git commit -m \"Chapter $((10#$N)): $TITLE\""
echo "  git push"
git status --short
