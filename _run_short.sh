#!/usr/bin/env bash
set -u
cd /mnt/d/DVCH13-1
tr -d '\r' < env.sh > /tmp/dvch_env.sh
# shellcheck disable=SC1091
source /tmp/dvch_env.sh
echo "== short chain (12 samples) =="
timeout 1500 python3 run_dvch_cobaya_short.py 2>&1 | grep -viE 'SyntaxWarning|invalid escape|Did you mean|raw string' | tail -50
echo "== SHORT_EXIT=$? =="
echo "== output files =="
ls -la --time-style=long-iso dvch_planck_short_chain* 2>/dev/null