#!/bin/bash
# Reanuda la cadena widebeta tras un apagado/reinicio.
# cobaya restaura el estado desde dvch_prod_widebeta.checkpoint y descarta
# automaticamente las muestras inconsistentes.
#
# IMPORTANTE: NO borra las cadenas existentes (a diferencia de launch_widebeta.sh)
set -eo pipefail
cd "$(dirname "$0")"
source ./env.sh >/dev/null 2>&1

export DVCH_SHORT_YAML=dvch_cobaya_short.yaml
export DVCH_CHAIN_OUTPUT=dvch_prod_widebeta
export DVCH_COVMAT=dvch_widebeta_input.covmat
export DVCH_MAX_SAMPLES=200000
export DVCH_BURN_IN=0
export DVCH_LEARN_PROPOSAL=true
export DVCH_LEARN_RMINUS1_MAX=30.0
export DVCH_LEARN_EVERY=2d
export DVCH_NO_BLOCKING=1
export DVCH_RMINUS1_STOP=0.03
export DVCH_CHAIN_SEED=1
export DVCH_RESUME=true

exec mpirun -np 4 python3 run_dvch_cobaya_full_highl.py
