#!/bin/bash
# Definitive phase-3 resume of dvch_prod.
#
# Context (see DVCH_CONTEXTO_SESION.md / _r1_diagnostic.py):
#   - The learned proposal is verified optimal (matches the empirical posterior
#     covariance to within a few %), tau~45 is the random-walk floor for 26
#     params, and Cobaya's R-1 = C(m,p)*tau/n_tail ~ 47.6/ESS_per_param.
#   - R-1 < 0.01 would need ESS~4760/param (~33 days). Chosen budget:
#     Rminus1_stop = 0.03  ->  ESS ~1400/param, ~9 days.
#   - learn_proposal=false freezes the optimal proposal, so the resumed phase
#     is a clean time-homogeneous Markov chain (no further adaptation).
#   - Rminus1_stop / max_samples were ALSO patched into dvch_prod.updated.{yaml,
#     dill_pickle} because Cobaya restores those from the old run info on resume
#     (_at_resume_prefer_old); env vars alone would be silently ignored.
#
# OMP_NUM_THREADS=3: 4 mpi procs x 3 threads = 12 cores. Benchmarked best of the
# 4-proc layouts (4x3: 11.8s/eval vs 4x12: 12.7s, 4x2: 19.4s, 4x1: 27.2s).
set -eo pipefail
cd "$(dirname "$0")"

# Machine-specific paths (CAMB root, clik egg, Planck data, LD_LIBRARY_PATH).
source ./env.sh >/dev/null 2>&1 || {
    echo "ERROR: could not source ./env.sh" >&2; exit 1; }

if [[ -n "${DVCH_CLIK_LIB_DIR:-}" ]]; then
    export LD_LIBRARY_PATH="${DVCH_CLIK_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# NOTE: OMP_NUM_THREADS is deliberately UNSET. An end-to-end measurement showed
# the previous phase (OMP unset -> CAMB default of 12 threads/proc, nominally
# oversubscribed 4x12 on 12 cores) ran at ~507 accepted samples/hr, whereas
# OMP_NUM_THREADS=3 (benchmarked as the "best" 4-proc layout) only reached
# 333/hr. The microbenchmark's thread ordering does not hold for the real
# workload, so we keep the configuration that was actually measured fastest.
export DVCH_MAX_SAMPLES="${DVCH_MAX_SAMPLES:-40000}"
export DVCH_BURN_IN="${DVCH_BURN_IN:-0}"           # burn-in already consumed in phase 1/2
export DVCH_LEARN_PROPOSAL="${DVCH_LEARN_PROPOSAL:-false}"
export DVCH_RMINUS1_STOP="${DVCH_RMINUS1_STOP:-0.03}"
export DVCH_LEARN_EVERY="${DVCH_LEARN_EVERY:-10d}"
export DVCH_COVMAT="${DVCH_COVMAT:-dvch_prod_learned.covmat}"
export DVCH_CHAIN_OUTPUT=dvch_prod
export DVCH_CHAIN_SEED=1
export DVCH_RESUME=true

echo "[resume_prod.sh] $(date -Is)  OMP_NUM_THREADS=$OMP_NUM_THREADS  np=4"
mpirun -np 4 python3 run_dvch_cobaya_full_highl.py 2>&1 | tee -a dvch_prod_run.log
