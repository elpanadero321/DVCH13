#!/bin/bash
# Coste CMB del switch de desacoplamiento: para cada z_c, chi2 en el punto fijo
# (cosmologia del mejor ajuste LCDM del CMB, n/beta = mejor ajuste tardio de ese z_c).
cd /mnt/d/DVCH13-1 || exit 9
source env.sh > /tmp/envT.out 2>&1
CASES="1.0:0.3239:0.2894 1.5:0.4371:0.6267 2.0:0.4280:0.5617 5.0:0.4226:0.5257 1000000.0:0.4226:0.5257"
for CASE in $CASES; do
    zc=$(echo $CASE | cut -d: -f1)
    nn=$(echo $CASE | cut -d: -f2)
    bb=$(echo $CASE | cut -d: -f3)
    echo "=== z_c = $zc (n=$nn, beta=$bb) ==="
    DVCH_ZC=$zc DVCH_P=10 DVCH_N=$nn DVCH_BETA=$bb timeout 1800 python3 cmb_npoint_check.py 2>&1 \
        | grep -E "chi2 =|delta" | head -3
done
echo "CMB_TRADEOFF_DONE"
