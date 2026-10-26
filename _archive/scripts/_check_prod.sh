#!/usr/bin/env bash
set -u
cd /mnt/d/DVCH13-1
echo "=== run.log tail ==="
tail -20 dvch_prod_run.log 2>/dev/null
echo "=== covmat mentions ==="
grep -c "covmat" dvch_prod_run.log 2>/dev/null
echo "=== covmat ERROR? ==="
grep -n "Can.t open covmat" dvch_prod_run.log 2>/dev/null || echo "NO_COVMAT_ERROR"
echo "=== chains ==="
wc -l dvch_prod.[1-4].txt 2>/dev/null
echo "=== progress ==="
cat dvch_prod.progress 2>/dev/null
echo "=== procs ==="
pgrep -af 'run_dvch_cobaya_full_highl' | head