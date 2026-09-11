#!/usr/bin/env bash
set -u
cd /mnt/d/DVCH13-1 || exit 1

echo "=== 1. devcontainer.json valido? ==="
python3 - <<'PY'
import json, re, pathlib
s = pathlib.Path(".devcontainer/devcontainer.json").read_text()
s = re.sub(r'^\s*//.*$', '', s, flags=re.M)
try:
    json.loads(s)
    print("devcontainer.json: JSON OK")
except Exception as e:
    print("devcontainer.json: FALLO ->", e)
PY

echo
echo "=== 2. syntax scripts deploy/ ==="
for f in deploy/setup_cloud.sh deploy/upload_bundle.sh deploy/docker_run.sh deploy/entrypoint.sh deploy/bootstrap_remote.sh; do
    tr -d '\r' < "$f" | bash -n 2>/dev/null && echo "OK   : $f" || echo "FAIL : $f"
done
echo "=== FIN ==="