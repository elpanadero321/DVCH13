#!/usr/bin/env bash
set -u
cd /mnt/d/DVCH13-1
echo "=== ls dvch_prod* ==="
ls -la --time-style=long-iso dvch_prod* 2>/dev/null
echo "=== wc run.log ==="
wc -l dvch_prod_run.log 2>/dev/null
echo "=== run.log head+tail ==="
head -30 dvch_prod_run.log 2>/dev/null
echo "-----"
tail -30 dvch_prod_run.log 2>/dev/null
echo "=== /tmp/launch_prod.out ==="
cat /tmp/launch_prod.out 2>/dev/null
echo "=== any mpirun/python ==="
pgrep -af 'mpirun|run_dvch' || echo NONE