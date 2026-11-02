#!/bin/bash
cd /mnt/d/DVCH13-1
for f in deep_verify.py deep_verify2.py deep_verify3.py deep_verify4.py \
         find_missing.py recover3.py check_git_files.py; do
  [ -f "$f" ] && mv "$f" _archive/scripts/
done
git add -A
git -c core.pager=cat commit -m "chore: retira scripts temporales de verificacion" 2>/dev/null
git push origin main 2>&1 | tail -2
echo "=== arbol de trabajo limpio? ==="
git status --short | wc -l
