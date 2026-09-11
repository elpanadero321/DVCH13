#!/bin/bash
# Launch the 4-chain MPI production run for DVCH Planck MCMC.
set -e
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
# Input covmat MUST NOT share its name with DVCH_CHAIN_OUTPUT: Cobaya's cleanup
# regexp 'dvch_prod[._]covmat$' deletes any input covmat named dvch_prod.covmat,
# which then aborts the run with "Can't open covmat file 'dvch_prod.covmat'"
# (see dvch_prod_run.log and §9.3 point 4 of DVCH_CONTEXTO_SESION.md).
export DVCH_COVMAT=dvch_prod_wide.covmat
export DVCH_CHAIN_OUTPUT=dvch_prod
export DVCH_CHAIN_SEED=1
nohup mpirun -np 4 python3 run_dvch_cobaya_full_highl.py > dvch_prod_run.log 2>&1 &
echo "Launched PID: $!"