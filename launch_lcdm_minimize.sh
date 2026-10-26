#!/bin/bash
# Comparacion LCDM vs DVCH: encuentra el minimo chi2 de LCDM (modo minimize).
# LCDM = DVCH con DVCH_n=0 y DVCH_beta=0 (caso anidado, misma likelihood).
set -eo pipefail
cd "$(dirname "$0")"
source ./env.sh >/dev/null 2>&1

export DVCH_SHORT_YAML=dvch_cobaya_short.yaml
export DVCH_MODE=minimize
export DVCH_CHAIN_OUTPUT=dvch_lcdm_min
export DVCH_RESUME=false

exec mpirun -np 1 python3 run_dvch_cobaya_lcdm.py
