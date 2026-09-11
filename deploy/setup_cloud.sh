#!/usr/bin/env bash
# ============================================================
# setup_cloud.sh — se ejecuta EN LA NUBE (GitHub Codespaces,
# Colab, GCP, etc). Descarga el bundle y arranca la MCMC.
#
# Uso (dentro de la nube):
#   export DVCH_BUNDLE_URL='https://0x0.st/xxxx.tar.gz'
#   bash deploy/setup_cloud.sh
#
# Elige la ruta mas rapida disponible:
#   1) Docker (python:3.14-slim)  -> reproduccion exacta, sin recompilar
#   2) Python 3.14 local          -> usa los .so del bundle
#   3) Recompila clik+CAMB        -> fallback si no hay 3.14
# ============================================================
set -euo pipefail

: "${DVCH_BUNDLE_URL:?Define DVCH_BUNDLE_URL con la URL del bundle}"
WORK="${DVCH_WORK:-$HOME/dvch}"

echo "[setup] workdir: $WORK"
mkdir -p "$WORK"
cd "$WORK"

# --- 1. Descargar y descomprimir el bundle -------------------
if [[ ! -d "$WORK/dvch_bundle_stage" ]]; then
    echo "[setup] descargando bundle ..."
    curl -L --fail -o bundle.tar.gz "$DVCH_BUNDLE_URL"
    echo "[setup] descomprimiendo ..."
    tar -xzf bundle.tar.gz
fi
[[ -d "$WORK/dvch_bundle_stage" ]] || { echo "ERROR: bundle incompleto"; exit 1; }
echo "[setup] bundle OK: $(du -sh dvch_bundle_stage | cut -f1)"

# --- 2. Elegir motor -----------------------------------------
if command -v docker >/dev/null 2>&1; then
    echo "[setup] RUTA 1: Docker (python:3.14-slim)"
    cp deploy/Dockerfile "$WORK/Dockerfile.dvch" 2>/dev/null || true
    cp deploy/entrypoint.sh "$WORK/dvch_bundle_stage/entrypoint.sh"
    docker build -t dvch-mcmc -f "$WORK/Dockerfile.dvch" "$WORK"
    exec docker run --rm -it --name dvch-mcmc-run \
        -e DVCH_NCHAINS="${DVCH_NCHAINS:-$(nproc)}" \
        -e DVCH_MAX_SAMPLES="${DVCH_MAX_SAMPLES:-2000}" \
        -e DVCH_BURN_IN="${DVCH_BURN_IN:-200}" \
        -v "$WORK/dvch_bundle_stage/repo:/opt/dvch/repo" \
        dvch-mcmc
fi

PYV="$(python3 --version 2>&1 | awk '{print $2}')"
echo "[setup] python local: $PYV"

if [[ "$PYV" == 3.14.* ]]; then
    echo "[setup] RUTA 2: Python 3.14 local, uso los .so del bundle"
    bash deploy/bootstrap_remote.sh
    exit $?
fi

echo "[setup] RUTA 3: sin Docker ni Py3.14 -> recompilando (mas lento)"
echo "[setup] instala primero: sudo apt-get install -y gfortran libopenmpi-dev openmpi-bin"
bash deploy/bootstrap_remote.sh || true
echo "[setup] Si fallo, revisa README_REMOTE.md secciones 3-4 (recompilar clik+CAMB)."
EOF