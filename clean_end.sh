#!/bin/bash
cd /mnt/d/DVCH13-1
mv commit_final_fixes.sh _archive/scripts/ 2>/dev/null
git add -A
git -c core.pager=cat commit --quiet -m "chore: retira script temporal" 2>/dev/null
git push origin main 2>&1 | tail -1
echo "pendientes: $(git status --short | wc -l)"
git rev-parse HEAD
git rev-parse origin/main
