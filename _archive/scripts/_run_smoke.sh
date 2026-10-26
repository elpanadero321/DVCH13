#!/usr/bin/env bash
set -u
cd /mnt/d/DVCH13-1
# normalize env.sh (created on Windows -> CRLF) into a POSIX temp and source it
tr -d '\r' < env.sh > /tmp/dvch_env.sh
# shellcheck disable=SC1091
source /tmp/dvch_env.sh
echo "== PYTHONPATH=$PYTHONPATH =="
echo "== run smoke =="
python3 dvch_planck_clik_smoke.py \
  --camb-root "$DVCH_CAMB_ROOT" \
  --clik-egg "$DVCH_CLIK_EGG" \
  --likelihood "$DVCH_PLANCK_LIKELIHOOD" 2>&1 | grep -viE 'SyntaxWarning|invalid escape|Did you mean|raw string' | tail -40
echo "== EXIT_SMOKE=$? =="