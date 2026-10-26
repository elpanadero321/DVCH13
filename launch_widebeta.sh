#!/bin/bash
# Lanzador de la cadena de produccion con el prior de DVCH_beta corregido.
set -eo pipefail
cd "$(dirname "$0")"
source ./env.sh >/dev/null 2>&1

export DVCH_SHORT_YAML=dvch_cobaya_short.yaml
export DVCH_CHAIN_OUTPUT=dvch_prod_widebeta
export DVCH_COVMAT=dvch_widebeta_input.covmat
export DVCH_MAX_SAMPLES=40000
export DVCH_BURN_IN=0
export DVCH_LEARN_PROPOSAL=true
export DVCH_LEARN_RMINUS1_MAX=30.0
export DVCH_LEARN_EVERY=2d
export DVCH_RMINUS1_STOP=0.03
export DVCH_CHAIN_SEED=1
export DVCH_RESUME=false

# NO borrar covmats de entrada: cobaya con force=True elimina todo lo que
# coincide con '<output>[._]covmat$', por eso el de entrada se llama
# dvch_widebeta_input.covmat (prefijo distinto a dvch_prod_widebeta).
rm -f dvch_prod_widebeta.[0-9]*.txt \
      dvch_prod_widebeta.checkpoint dvch_prod_widebeta.progress

exec mpirun -np 4 python3 run_dvch_cobaya_full_highl.py
