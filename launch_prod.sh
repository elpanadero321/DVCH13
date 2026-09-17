#!/bin/bash
# Launch the 4-chain MPI production run for DVCH Planck MCMC.
set -eo pipefail
cd "$(dirname "$0")"

# Portable: read the clik lib directory from DVCH_CLIK_LIB_DIR (see env.sh.example).
if [[ -n "${DVCH_CLIK_LIB_DIR:-}" ]]; then
    export LD_LIBRARY_PATH="${DVCH_CLIK_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
elif [[ -z "${LD_LIBRARY_PATH:-}" ]]; then
    echo "ERROR: set DVCH_CLIK_LIB_DIR (or LD_LIBRARY_PATH) to the directory" >&2
    echo "       containing libclik.so (e.g. .../plc-3.1/lib)." >&2
    exit 1
fi

export DVCH_MAX_SAMPLES=2000
export DVCH_BURN_IN=200
export DVCH_LEARN_PROPOSAL=true
# Re-enable Cobaya's Gelman-Rubin convergence stop.  The previous default of
# Rminus1_stop=100 was vacuous ("R-1 < 100" holds for any chain), so the run
# could only ever stop on max_samples; 0.01 matches Cobaya's own default and
# the repo's documented criterion "R-hat < 1.01" (RUN_FULL_BATTERY.md S6).
export DVCH_RMINUS1_STOP=0.01
# Input covmat MUST NOT share its name with DVCH_CHAIN_OUTPUT: Cobaya's cleanup
# regexp 'dvch_prod[._]covmat$' deletes any input covmat named dvch_prod.covmat,
# which then aborts the run with "Can't open covmat file 'dvch_prod.covmat'"
# (see dvch_prod_run.log and §9.3 point 4 of DVCH_CONTEXTO_SESION.md).
export DVCH_COVMAT=dvch_prod_wide.covmat
export DVCH_CHAIN_OUTPUT=dvch_prod
export DVCH_CHAIN_SEED=1
# Foreground mode: DVCH_PROD_FG=1 blocks until the MCMC finishes (no nohup, no
# backgrounding), so an orchestrator can run convergence diagnostics right after.
# Default (unset/0) keeps the original background behaviour.
if [[ "${DVCH_PROD_FG:-0}" == "1" ]]; then
    echo "MODO FOREGROUND (DVCH_PROD_FG=1): el MCMC corre en primer plano y este"
    echo "script bloquea hasta que termine. Log: dvch_prod_run.log"
    mpirun -np 4 python3 run_dvch_cobaya_full_highl.py 2>&1 | tee dvch_prod_run.log
else
    echo "MODO BACKGROUND (DVCH_PROD_FG no es 1): el MCMC se lanza en segundo plano."
    nohup mpirun -np 4 python3 run_dvch_cobaya_full_highl.py > dvch_prod_run.log 2>&1 &
    echo "Launched PID: $!"
fi