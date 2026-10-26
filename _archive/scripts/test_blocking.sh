#!/bin/bash
# Test corto: valida que DVCH_NO_BLOCKING=1 activa measure_speeds + oversampling
# y mejora la mezcla en la direccion problematica. NO toca la cadena principal.
set -eo pipefail
cd "$(dirname "$0")"
source ./env.sh >/dev/null 2>&1

export DVCH_SHORT_YAML=dvch_cobaya_short.yaml
export DVCH_CHAIN_OUTPUT=dvch_test_blocking
export DVCH_MAX_SAMPLES=300
export DVCH_BURN_IN=0
export DVCH_LEARN_PROPOSAL=true
export DVCH_LEARN_RMINUS1_MAX=30.0
export DVCH_LEARN_EVERY=1d
export DVCH_NO_BLOCKING=1
export DVCH_RMINUS1_STOP=0.03
export DVCH_CHAIN_SEED=1
export DVCH_RESUME=false

# covmat aprendida de la cadena principal como punto de partida.
# Path ABSOLUTO en /tmp: /mnt/d a veces tiene retardos de flushing en WSL2.
COV=/tmp/dvch_test_blocking.covmat
cp -f dvch_prod_widebeta.covmat "$COV"
export DVCH_COVMAT="$COV"

rm -f dvch_test_blocking.[0-9]*.txt dvch_test_blocking.checkpoint \
      dvch_test_blocking.progress
# NO borrar dvch_test_blocking.covmat

exec mpirun -np 4 python3 run_dvch_cobaya_full_highl.py
