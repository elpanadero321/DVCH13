#!/bin/bash
cd /mnt/d/DVCH13-1 || exit 9
source env.sh > /tmp/envV.out 2>&1
echo "=== 1. verificacion CAMB vs referencia SIN switch ==="
python3 dvch_camb_patch_verify.py 2>&1 | tail -7
echo
echo "=== 2. verificacion CAMB vs referencia CON switch (zc=5, p=10) ==="
DVCH_ZC=5 DVCH_P=10 python3 dvch_camb_patch_verify.py 2>&1 | tail -7
echo
echo "=== 3. chequeo de punto fijo CMB CON switch ==="
DVCH_ZC=5 DVCH_P=10 timeout 1200 python3 cmb_npoint_check.py 2>&1 | grep -E 'chi2|delta|Best|convergencia' | head -6
echo "SWITCH_TEST_DONE"
