#!/usr/bin/env bash
set -u
ROOT=/mnt/d/DVCH13-1
for f in "$ROOT"/deploy/gcp_launch.sh "$ROOT"/deploy/gcp_remote_run.sh; do
    sed -i 's/\r$//' "$f"
    echo "normalizado: $(basename "$f")"
done
echo "=== CRLF restante? ==="
grep -lU $'\r' "$ROOT"/deploy/gcp_launch.sh "$ROOT"/deploy/gcp_remote_run.sh 2>/dev/null && echo "HAY CRLF" || echo "sin CRLF (OK)"
echo "=== sintaxis ==="
for f in "$ROOT"/deploy/gcp_launch.sh "$ROOT"/deploy/gcp_remote_run.sh; do
    bash -n "$f" && echo "OK   $(basename "$f")" || echo "FAIL $(basename "$f")"
done
echo "=== FIN ==="