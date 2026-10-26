#!/usr/bin/env bash
set -u
echo "=== bundle listo? ==="
ls -lh /mnt/d/DVCH-external/dvch_portable_bundle.tar.gz 2>/dev/null || echo NO_BUNDLE
echo "=== deploy/ ==="
ls /mnt/d/DVCH13-1/deploy/ 2>/dev/null
echo "=== herramientas de subida en WSL ==="
for t in gh gcloud rclone curl wget scp git; do
  if command -v "$t" >/dev/null 2>&1; then
    echo "$t: OK ($(command -v $t))"
  else
    echo "$t: NO instalado"
  fi
done
echo "=== python local ==="
python3 --version 2>&1
EOF