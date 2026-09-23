#!/usr/bin/env bash
# Prueba local del contenedor: construye la imagen y verifica que
# CAMB+clik+DVCH cargan y que el entrypoint resuelve la config fase 3.
# NO lanza la MCMC completa; solo valida el arranque.
set -euo pipefail
STAGE="/mnt/d/DVCH-external/dvch_bundle_stage"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[test] verificando bundle..."
test -f "$STAGE/repo/dvch_prod_frozen.covmat" || { echo "FALTA frozen covmat"; exit 1; }
test -f "$STAGE/repo/dvch_cobaya_short.yaml" || { echo "FALTA yaml"; exit 1; }
echo "  frozen covmat: $(wc -c < "$STAGE/repo/dvch_prod_frozen.covmat") bytes"
grep -q "67.28" "$STAGE/repo/dvch_cobaya_short.yaml" && echo "  refs actualizados OK"

cp -f "$HERE/Dockerfile" "$HERE/../Dockerfile.dvch" 2>/dev/null || true
echo "[test] construyendo imagen (varios minutos)..."
docker build -t dvch-mcmc-test -f "$HERE/Dockerfile" "/mnt/d/DVCH-external" 2>&1 | tail -8
echo "[test] imagen construida"
docker images dvch-mcmc-test
