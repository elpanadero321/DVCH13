#!/usr/bin/env bash
# ============================================================
# docker_run.sh — construye y lanza el contenedor DVCH en
# cualquier nube con Docker. Un solo comando.
#
# Uso (en la maquina de nube, tras copiar el bundle + deploy/):
#   tar -xzf dvch_portable_bundle.tar.gz
#   bash deploy/docker_run.sh
#
# Variables opcionales:
#   DVCH_NCHAINS=8   (default: nproc del host)
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

echo "[docker] lanzando MCMC FULL (sin recortes)"
docker run --rm -it \
    --name dvch-mcmc-run \
    -e DVCH_NCHAINS="${DVCH_NCHAINS:-}" \
    -e DVCH_MAX_SAMPLES=2000 \
    -e DVCH_BURN_IN=200 \
    -v "$CTX/dvch_bundle_stage/repo:/opt/dvch/repo" \
    dvch-mcmc

# Al terminar, las cadenas quedan en:
#   $CTX/dvch_bundle_stage/repo/dvch_prod.[1-N].txt
echo "[docker] listo. Cadenas en $CTX/dvch_bundle_stage/repo/"