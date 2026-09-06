#!/bin/bash
# Launch the 4-chain MPI production run for DVCH Planck MCMC.
set -e
cd /mnt/d/DVCH13-1
export LD_LIBRARY_PATH=/home/danieproyect/plc-3.1/lib
export DVCH_MAX_SAMPLES=2000
export DVCH_BURN_IN=200
export DVCH_LEARN_PROPOSAL=true
export DVCH_COVMAT=dvch_prod.covmat
export DVCH_CHAIN_OUTPUT=dvch_prod
export DVCH_CHAIN_SEED=1
nohup mpirun -np 4 python3 run_dvch_cobaya_full_highl.py > dvch_prod_run.log 2>&1 &
echo "Launched PID: $!"