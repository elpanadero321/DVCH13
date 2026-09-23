#!/usr/bin/env bash
# ============================================================
# entrypoint.sh — arranca la MCMC DVCH definitiva dentro del contenedor.
#
# Configuracion "definitiva" (fase 3), heredada del analisis local:
#   - covmat = propuesta OPTIMA ya aprendida y verificada
#     (coincide con la covarianza posterior real a pocos %), congelada
#     => learn_proposal=false: cadena de Markov limpia y homogenea.
#   - Rminus1_stop = 0.03  (ESS ~1400/param; R-1<0.01 costaria ~33 dias).
#   - refs = medias posteriores reales => arranque en el modo, burn-in minimo.
#   - 1 hilo OMP por proceso: en nube lo que importa es el paralelismo
#     entre cadenas (mas cadenas = menor piso de ruido del R-1).
#
# Todo es sobreescribible por variables de entorno.
# ============================================================
set -euo pipefail

cd /opt/dvch/repo
# shellcheck disable=SC1091
source /opt/dvch/env_remote.sh

# YAML con refs = medias posteriores reales (arranque en el modo) SOLO para la
# corrida fresca de cloud. El runner lo selecciona via DVCH_SHORT_YAML.
export DVCH_SHORT_YAML="${DVCH_SHORT_YAML:-dvch_cobaya_cloud.yaml}"

export DVCH_MAX_SAMPLES="${DVCH_MAX_SAMPLES:-40000}"
export DVCH_BURN_IN="${DVCH_BURN_IN:-500}"
export DVCH_LEARN_PROPOSAL="${DVCH_LEARN_PROPOSAL:-false}"
export DVCH_COVMAT="${DVCH_COVMAT:-dvch_prod_frozen.covmat}"
export DVCH_RMINUS1_STOP="${DVCH_RMINUS1_STOP:-0.03}"
export DVCH_LEARN_EVERY="${DVCH_LEARN_EVERY:-10d}"
export DVCH_CHAIN_OUTPUT="${DVCH_CHAIN_OUTPUT:-dvch_prod}"
export DVCH_CHAIN_SEED="${DVCH_CHAIN_SEED:-1}"
export DVCH_RESUME="${DVCH_RESUME:-false}"

# Reparto de hilos. CAMB satura a ~4 hilos (benchmark: 20s a 1 hilo, 10.9s a 2,
# 8.2s a 3, 6.6s a 4). Como el R-1 de Cobaya necesita cierta longitud por cadena
# (R-1 ~ C(m)*tau/n_tail, y C(m) baja con m), el optimo es maximizar el numero
# de cadenas dandole a cada una ~4 hilos (saturacion de CAMB), no 1 hilo/cadena.
NPROC="$(nproc)"
NCHAINS="${DVCH_NCHAINS:-$(( NPROC / 4 ))}"
[[ "$NCHAINS" -ge 4 ]] || NCHAINS=4
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(( NPROC / NCHAINS ))}"
[[ "$OMP_NUM_THREADS" -ge 1 ]] || OMP_NUM_THREADS=1

echo "[entrypoint] nproc=$NPROC  cadenas=$NCHAINS  OMP_threads=$OMP_NUM_THREADS  max_samples=$DVCH_MAX_SAMPLES  R-1_stop=$DVCH_RMINUS1_STOP  covmat=$DVCH_COVMAT"

exec mpirun --allow-run-as-root -np "$NCHAINS" \
    python3 -u run_dvch_cobaya_full_highl.py
