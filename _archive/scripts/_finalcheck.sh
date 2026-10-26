#!/usr/bin/env bash
set -u
echo "=== A. devcontainer.json es JSON valido? ==="
python3 -c "import json;d=json.load(open('/mnt/d/DVCH13-1/.devcontainer/devcontainer.json'));print('JSON OK');print('cpus=',d['hostRequirements']['cpus'],'mem=',d['hostRequirements']['memory'])" 2>&1 | tail -n 3
echo "=== B. run_codespace.sh: sintaxis bash ==="
bash -n /mnt/d/DVCH13-1/deploy/run_codespace.sh && echo "run_codespace.sh OK"
echo "=== C. covmat presente en la raiz del repo (viaja por GitHub)? ==="
ls -la /mnt/d/DVCH13-1/dvch_prod_wide.covmat
echo "=== FIN ==="