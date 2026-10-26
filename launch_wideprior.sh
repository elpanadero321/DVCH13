#!/bin/bash
# MCMC definitivo del CMB con el prior de DVCH_n ENSANCHADO a [0.01, 0.99]
# (igual que el fit de expansion) para resolver la tension:
# ¿el CMB excluye n=0.47 o simplemente no lo exploro (prior [0.01,0.25])?
#
# Parte de la covmat aprendida en la corrida anterior para ahorrar burn-in.
set -eo pipefail
cd "$(dirname "$0")"
source ./env.sh >/dev/null 2>&1

export DVCH_SHORT_YAML=dvch_cobaya_wideprior.yaml
export DVCH_CHAIN_OUTPUT=dvch_prod_wideprior
export DVCH_MAX_SAMPLES=400000
export DVCH_BURN_IN=0
export DVCH_LEARN_PROPOSAL=true
export DVCH_LEARN_RMINUS1_MAX=30.0
export DVCH_LEARN_EVERY=2d
export DVCH_RMINUS1_STOP=0.03
export DVCH_CHAIN_SEED=2
export DVCH_RESUME=${DVCH_RESUME:-true}

# covmat aprendida (absoluta en /tmp)
COV=/tmp/dvch_wideprior.covmat
if [ ! -f "$COV" ]; then
  cp -f dvch_prod_widebeta5.covmat "$COV" 2>/dev/null || cp -f dvch_prod_widebeta.covmat "$COV"
fi
export DVCH_COVMAT="$COV"

exec mpirun -np 5 python3 run_dvch_cobaya_full_highl.py
