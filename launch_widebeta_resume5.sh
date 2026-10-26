#!/bin/bash
# Reanuda la cadena widebeta de 5 cadenas (dvch_prod_widebeta5) tras un apagon.
# cobaya restaura el estado desde dvch_prod_widebeta5.checkpoint.
#
# IMPORTANTE: NO borrar las cadenas existentes. NO usar launch_widebeta.sh.
set -eo pipefail
cd "$(dirname "$0")"
source ./env.sh >/dev/null 2>&1

export DVCH_SHORT_YAML=dvch_cobaya_short.yaml
export DVCH_CHAIN_OUTPUT=dvch_prod_widebeta5
export DVCH_MAX_SAMPLES=400000
export DVCH_BURN_IN=0
export DVCH_LEARN_PROPOSAL=true
export DVCH_LEARN_RMINUS1_MAX=30.0
export DVCH_LEARN_EVERY=2d
export DVCH_NO_BLOCKING=1
export DVCH_RMINUS1_STOP=0.03
export DVCH_CHAIN_SEED=1
export DVCH_RESUME=true

# covmat aprendida (ruta absoluta en /tmp por robustez frente a WSL2)
COV=/tmp/dvch_widebeta5.covmat
if [ -f "$COV" ]; then
  export DVCH_COVMAT="$COV"
fi

exec mpirun -np 5 python3 run_dvch_cobaya_full_highl.py
