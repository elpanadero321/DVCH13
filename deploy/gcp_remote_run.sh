#!/usr/bin/env bash
# ============================================================
# gcp_remote_run.sh — se ejecuta EN la VM de GCP (no en tu PC).
# Lo lanza deploy/gcp_launch.sh por SSH. No lo corras a mano.
# Descarga el bundle, lo verifica, construye la imagen Docker
# (python:3.14-slim, sin recompilar) y arranca la MCMC FULL.
# ============================================================
set -euo pipefail

: "${DVCH_BUNDLE_URL:?Falta DVCH_BUNDLE_URL}"
NCHAINS="${DVCH_NCHAINS:-4}"
MAXS="${DVCH_MAX_SAMPLES:-2000}"
BURN="${DVCH_BURN_IN:-200}"
COVMAT="${DVCH_COVMAT:-dvch_prod_wide.covmat}"
KIT="$HOME/deploy_kit"
WORK="$HOME/dvch"

echo "[remote] workdir: $WORK"
mkdir -p "$WORK"
cd "$WORK"

# --- 1. Bundle ---
if [[ ! -d dvch_bundle_stage ]]; then
    echo "[remote] descargando bundle (~181 MB)..."
    curl -L --fail -o bundle.tar.gz "$DVCH_BUNDLE_URL"
    echo "[remote] descomprimiendo..."
    tar -xzf bundle.tar.gz
fi
[[ -d dvch_bundle_stage ]] || { echo "ERROR: bundle incompleto"; exit 1; }
echo "[remote] bundle OK: $(du -sh dvch_bundle_stage | cut -f1)"

# --- 2. Contexto de build para Docker ---
# El Dockerfile espera 'dvch_bundle_stage/' y 'deploy/entrypoint.sh' en el contexto ($WORK).
mkdir -p deploy
cp -f "$KIT/Dockerfile" "$WORK/Dockerfile"
cp -f "$KIT/entrypoint.sh" "$WORK/deploy/entrypoint.sh"
chmod +x "$WORK/deploy/entrypoint.sh"

# --- 3. Covmat de arranque (viaja en el repo / lo sube gcp_launch.sh) ---
if [[ ! -f "dvch_bundle_stage/repo/$COVMAT" ]]; then
    if [[ -f "$HOME/$COVMAT" ]]; then
        cp "$HOME/$COVMAT" "dvch_bundle_stage/repo/$COVMAT"
        echo "[remote] covmat colocado: $COVMAT ($(wc -c < "dvch_bundle_stage/repo/$COVMAT") bytes)"
    else
        echo "[remote] [aviso] falta $COVMAT -> la aceptacion sera PEOR" >&2
    fi
fi

# --- 4. Build (verifica CAMB + clik dentro de la imagen) ---
echo "[remote] construyendo imagen docker (varios minutos)..."
sudo docker build -t dvch-mcmc -f "$WORK/Dockerfile" "$WORK"

# --- 5. Lanzar la MCMC (desacoplada: sobrevive al cierre de SSH) ---
echo "[remote] lanzando MCMC: $NCHAINS cadenas x $MAXS (burn_in $BURN)"
sudo docker rm -f dvch-mcmc-run >/dev/null 2>&1 || true
setsid nohup sudo docker run --rm --name dvch-mcmc-run \
    -e DVCH_NCHAINS="$NCHAINS" \
    -e DVCH_MAX_SAMPLES="$MAXS" \
    -e DVCH_BURN_IN="$BURN" \
    -v "$WORK/dvch_bundle_stage/repo:/opt/dvch/repo" \
    dvch-mcmc > "$WORK/dvch_prod_run.log" 2>&1 < /dev/null &
echo "[remote] MCMC lanzada. Log: $WORK/dvch_prod_run.log"