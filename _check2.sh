#!/usr/bin/env bash
set -u
B=/mnt/d/DVCH-external/dvch_portable_bundle.tar.gz
echo "=== 1. covmats dentro del bundle ==="
tar -tzf "$B" | grep -E 'repo/[^/]*\.covmat$' | sed 's#.*/##'
echo "=== 2. dvch_prod_wide.covmat en bundle? ==="
tar -tzf "$B" | grep -E 'dvch_prod_wide\.covmat$' || echo FALTA_dvch_prod_wide
echo "=== 3. dvch_prod_wide.covmat local (bytes + 3 primeras lineas) ==="
ls -la /mnt/d/DVCH13-1/dvch_prod_wide.covmat
head -n 3 /mnt/d/DVCH13-1/dvch_prod_wide.covmat
echo "=== 4. devcontainer.json: bytes y lineas ==="
wc -c -l /mnt/d/DVCH13-1/.devcontainer/devcontainer.json
echo "--- contenido completo ---"
cat /mnt/d/DVCH13-1/.devcontainer/devcontainer.json
echo "=== 5. devcontainer.json es JSON valido? ==="
python3 -c "import json;json.load(open('/mnt/d/DVCH13-1/.devcontainer/devcontainer.json'));print('JSON OK')" 2>&1 | tail -n 2
echo "=== 6. config local de la corrida cancelada (yaml efectivo) ==="
grep -nE 'max_samples|burn_in|learn_proposal|Rminus1_stop|covmat|output' /mnt/d/DVCH13-1/dvch_prod.input.yaml 2>/dev/null | head -n 20
echo "=== 7. aceptacion estimada (accepted vs steps) ==="
grep -m1 'Progress' /mnt/d/DVCH13-1/dvch_prod_run.log
grep 'Progress' /mnt/d/DVCH13-1/dvch_prod_run.log | tail -n 1
echo "=== FIN ==="