#!/usr/bin/env bash
set -u
ROOT=/mnt/d/DVCH13-1
echo "=== 1. Normalizando CRLF -> LF en deploy/*.sh y devcontainer.json ==="
for f in "$ROOT"/deploy/*.sh "$ROOT"/.devcontainer/devcontainer.json; do
    sed -i 's/\r$//' "$f"
    echo "  normalizado: $(basename "$f")"
done
echo "=== 2. Quedan CRLF? ==="
if grep -lU $'\r' "$ROOT"/deploy/*.sh "$ROOT"/.devcontainer/devcontainer.json 2>/dev/null; then
    echo "  TODAVIA HAY CRLF (revisar)"
else
    echo "  sin CRLF (OK)"
fi
echo "=== 3. devcontainer.json valido (JSONC: se quitan comentarios //) ==="
python3 - "$ROOT/.devcontainer/devcontainer.json" <<'PY'
import re, json, sys
raw = open(sys.argv[1], encoding='utf-8').read()
clean = re.sub(r'^\s*//.*$', '', raw, flags=re.M)
try:
    d = json.loads(clean)
    print("  JSONC OK")
    print("  cpus =", d['hostRequirements']['cpus'], "| memory =", d['hostRequirements']['memory'])
    print("  features =", list(d.get('features', {}).keys()))
    print("  remoteEnv.OMP_NUM_THREADS =", d['remoteEnv']['OMP_NUM_THREADS'])
except Exception as e:
    print("  JSONC ERROR:", e)
PY
echo "=== 4. Sintaxis de TODOS los scripts de deploy ==="
for f in "$ROOT"/deploy/*.sh; do
    if bash -n "$f" 2>/tmp/e; then
        echo "  OK   $(basename "$f")"
    else
        echo "  FAIL $(basename "$f"): $(cat /tmp/e)"
    fi
done
echo "=== 5. covmat en la raiz del repo (viaja por GitHub) ==="
ls -la "$ROOT"/dvch_prod_wide.covmat
echo "=== FIN ==="