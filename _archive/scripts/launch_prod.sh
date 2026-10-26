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

# All sampler knobs are overridable from the environment (e.g. phase-2 resume
# with a learned covmat and a larger max_samples); the values below are the
# phase-1 defaults used only when the caller did not export them.
export DVCH_MAX_SAMPLES="${DVCH_MAX_SAMPLES:-2000}"
export DVCH_BURN_IN="${DVCH_BURN_IN:-200}"
export DVCH_LEARN_PROPOSAL="${DVCH_LEARN_PROPOSAL:-true}"
export DVCH_COVMAT="${DVCH_COVMAT:-dvch_prod_wide.covmat}"
export DVCH_LEARN_RMINUS1_MAX="${DVCH_LEARN_RMINUS1_MAX:-2.0}"
# Re-enable Cobaya's Gelman-Rubin convergence stop.  The previous default of
# Rminus1_stop=100 was vacuous ("R-1 < 100" holds for any chain), so the run
# could only ever stop on max_samples; 0.01 matches Cobaya's own default and
# the repo's documented criterion "R-hat < 1.01" (RUN_FULL_BATTERY.md S6).
export DVCH_RMINUS1_STOP=0.01
# Cadence of the automatic checkpoint + convergence/proposal-learning check.
# Cobaya resolves 'd' units as *number_of_params, so Nd = N*27 accepted samples
# *per chain*; a checkpoint is written only once ALL chains sit on a multiple of
# that number. Cobaya's default 40d (=1080/chain) left a ~16 h gap with no
# checkpoint after startup, and the dummy checkpoint written at startup keeps
# burn_in=200 until the first check, so a stop in that window re-discards 200
# samples per chain on resume. 10d (=270/chain) is the first multiple of 27
# safely above the counts already reached, so the first checkpoint lands ~1.5 h
# after a resume and refreshes it every few hours thereafter.
export DVCH_LEARN_EVERY=10d
# Input covmat MUST NOT share its name with DVCH_CHAIN_OUTPUT: Cobaya's cleanup
# regexp 'dvch_prod[._]covmat$' deletes any input covmat named dvch_prod.covmat,
# which then aborts the run with "Can't open covmat file 'dvch_prod.covmat'"
# (see dvch_prod_run.log and §9.3 point 4 of DVCH_CONTEXTO_SESION.md).
# dvch_prod_learned.covmat (phase 2) is the empirical covmat of the 8000 phase-1
# samples; it does not match the cleanup regexp, so it survives.
export DVCH_CHAIN_OUTPUT=dvch_prod
export DVCH_CHAIN_SEED=1
# Resume: Cobaya checkpoints continuously. If you stop this run (or the PC
# shuts down), just re-run launch_prod.sh: it continues the same chains from
# the last checkpoint instead of restarting. Set DVCH_RESUME=false to force a
# fresh start from scratch.
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