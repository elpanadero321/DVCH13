#!/usr/bin/env bash
# ============================================================
# entrypoint.sh — arranca la MCMC DVCH FULL dentro del contenedor.
# 4 cadenas x 2000 muestras, full plik, SIN recortes.
# ============================================================
set -euo pipefail

cd /opt/dvch/repo
# shellcheck disable=SC1091
source /opt/dvch/env_remote.sh

export DVCH_MAX_SAMPLES="${DVCH_MAX_SAMPLES:-2000}"
export DVCH_BURN_IN="${DVCH_BURN_IN:-200}"
export DVCH_LEARN_PROPOSAL="${DVCH_LEARN_PROPOSAL:-true}"
export DVCH_COVMAT="dvch_prod_wide.covmat"
export DVCH_CHAIN_OUTPUT="dvch_prod"
export DVCH_CHAIN_SEED="${DVCH_CHAIN_SEED:-1}"
# 1 hilo por proceso: en nube lo que importa es el paralelismo entre cadenas.
export OMP_NUM_THREADS=1

NCHAINS="${DVCH_NCHAINS:-$(nproc)}"
echo "[entrypoint] nproc=$(nproc)  cadenas=$NCHAINS  max_samples=$DVCH_MAX_SAMPLES"

exec mpirun --allow-run-as-root -np "$NCHAINS" \
    python3 -u run_dvch_cobaya_full_highl.py