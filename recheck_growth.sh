#!/bin/bash
# Recalcula sigma8/S8 y el robustness scan para confirmar los valores correctos
cd /mnt/d/DVCH13-1
echo "=== recalculando growth diagnostic (sigma8/S8) ==="
python3 -c "
import dvch_growth_diagnostic as g
try:
    res = g.main() if hasattr(g, 'main') else None
except Exception as e:
    print('main() fallo:', e)
" 2>&1 | head -8
echo ""
echo "=== recalculando robustness scan ==="
python3 -c "
import dvch_robustness_scan as r
try:
    res = r.main() if hasattr(r, 'main') else None
except Exception as e:
    print('main() fallo:', e)
" 2>&1 | head -6
echo ""
echo "=== valores actuales en CSV ==="
cat dvch_sigma8_s8_summary.csv
grep viable dvch_robustness_summary.csv
