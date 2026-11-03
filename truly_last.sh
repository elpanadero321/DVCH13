#!/bin/bash
cd /mnt/d/DVCH13-1
git rm --cached -q last_clean.sh 2>/dev/null
mv last_clean.sh _archive/scripts/ 2>/dev/null
git add -A
git -c core.pager=cat commit -m "chore: retira last_clean.sh del arbol" 2>/dev/null
git push origin main 2>&1 | tail -2
echo "=== pendientes finales ==="
git status --short | wc -l
echo "=== veredicto ==="
git rev-parse HEAD
git rev-parse origin/main
