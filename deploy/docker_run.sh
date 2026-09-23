#!/usr/bin/env bash
# ============================================================
# docker_run.sh — construye y lanza el contenedor DVCH en
# cualquier nube con Docker. Un solo comando.
#
# Configuracion "definitiva" (fase 3): covmat optimo congelado,
# learn_proposal=false, Rminus1_stop=0.03, refs = medias posteriores.
# El reparto de hilos lo calcula entrypoint.sh (cadenas = nproc/4,
# ~4 hilos OMP por cadena: CAMB satura a 4 hilos).
#
# Uso (en la maquina de nube, tras copiar el bundle + deploy/):
#   tar -xzf dvch_portable_bundle.tar.gz
#   bash deploy/docker_run.sh
#
# Variables opcionales:
#   DVCH_NCHAINS=8   (default: nproc/4)
#   DVCH_RMINUS1_STOP=0.03
# ============================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# El Dockerfile necesita un contexto de build con 'dvch_bundle_stage/' y
# 'deploy/' como hermanos. Se soportan dos layouts:
#   (a) deploy/ ya es hermano de dvch_bundle_stage/ -> contexto = dirname(HERE)
#   (b) deploy/ anidado en el bundle (<stage>/repo/deploy/, el caso del
#       tarball): se resuelve el stage y se publica deploy/ a su lado.
if [[ -d "$(dirname "$HERE")/dvch_bundle_stage" ]]; then
    BUNDLE_DIR="$(dirname "$HERE")/dvch_bundle_stage"
    CTX="$(dirname "$HERE")"
else
    BUNDLE_DIR="$(cd "$HERE/../.." && pwd)"
    CTX="$(dirname "$BUNDLE_DIR")"
    if [[ ! -d "$CTX/deploy" ]]; then
        echo "[docker] publicando deploy/ junto al bundle (contexto de build)"
        cp -r "$HERE" "$CTX/deploy"
    fi
fi

if [[ ! -d "$BUNDLE_DIR" ]]; then
    echo "ERROR: no encuentro $BUNDLE_DIR" >&2
    echo "       Descomprime primero: tar -xzf dvch_portable_bundle.tar.gz" >&2
    exit 1
fi
cp "$HERE/Dockerfile" "$CTX/Dockerfile.dvch"
cp "$HERE/entrypoint.sh" "$CTX/dvch_bundle_stage/entrypoint.sh"

echo "[docker] construyendo imagen dvch-mcmc"
docker build -t dvch-mcmc -f "$CTX/Dockerfile.dvch" "$CTX"

echo "[docker] lanzando MCMC definitiva (frozen covmat, R-1 stop 0.03)"
docker run --rm -it \
    --name dvch-mcmc-run \
    -e DVCH_NCHAINS="${DVCH_NCHAINS:-}" \
    -e DVCH_MAX_SAMPLES="${DVCH_MAX_SAMPLES:-}" \
    -e DVCH_BURN_IN="${DVCH_BURN_IN:-}" \
    -e DVCH_LEARN_PROPOSAL="${DVCH_LEARN_PROPOSAL:-}" \
    -e DVCH_RMINUS1_STOP="${DVCH_RMINUS1_STOP:-}" \
    -e DVCH_COVMAT="${DVCH_COVMAT:-}" \
    -e DVCH_CHAIN_SEED="${DVCH_CHAIN_SEED:-}" \
    -e OMP_NUM_THREADS="${OMP_NUM_THREADS:-}" \
    -v "$CTX/dvch_bundle_stage/repo:/opt/dvch/repo" \
    dvch-mcmc

# Al terminar, las cadenas quedan en:
#   $CTX/dvch_bundle_stage/repo/dvch_prod.[1-N].txt
echo "[docker] listo. Cadenas en $CTX/dvch_bundle_stage/repo/"