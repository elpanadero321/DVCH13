#!/bin/bash
cd /mnt/d/DVCH13-1
mv commit_chainfix.sh final_state.sh _archive/scripts/ 2>/dev/null
git add -A
git -c core.pager=cat commit -m "chore: retira scripts temporales del estado final" 2>/dev/null
git push origin main 2>&1 | tail -2
echo "=== pendientes: ==="
git status --short | wc -l
