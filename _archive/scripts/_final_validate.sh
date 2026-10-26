#!/usr/bin/env bash
set -u
cd /mnt/d/DVCH13-1 || exit 1

echo "=== 1. devcontainer.json valido? ==="
python3 - <<'PY'
import json, re, pathlib
p = pathlib.Path(".devcontainer/devcontainer.json")
s = p.read_text()
# quita comentarios // (no strings con // en este archivo)
s = re.sub(r'^\s*//.*$', '', s, flags=re.M)
try:
    json.loads(s)
    print("devcontainer.json: JSON OK")
except Exception as e:
    print("devcontainer.json: FALLO ->", e)
PY

echo
echo "=== 2. syntax de scripts deploy/ ==="
for f in deploy/setup_cloud.sh deploy/upload_bundle.sh deploy/docker_run.sh deploy/entrypoint.sh deploy/bootstrap_remote.sh; do
    if tr -d '\r' < "$f" | bash -n 2>/dev/null; then
        echo "OK   : $f"
    else
        echo "FAIL : $f"
    fi
done

echo
echo "=== 3. contenido deploy/ + .devcontainer/ ==="
ls -1 deploy/
echo "--- .devcontainer ---"
ls -1 .devcontainer/

echo
echo "=== 4. bundle presente? ==="
ls -lh /mnt/d/DVCH-external/dvch_portable_bundle.tar.gz 2>/dev/null || echo "NO_BUNDLE"

echo
echo "=== FIN VALIDACION ==="
EOF