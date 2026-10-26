#!/bin/bash
cd /mnt/d/DVCH13-1 || exit 1
for i in 1 2 3 4; do
  printf 'chain %s: %s lines\n' "$i" "$(wc -l < dvch_prod.$i.txt)"
done
echo '=== process tree ==='
pgrep -af prterun | head -3
pgrep -af run_dvch_cobaya | head -5
