#!/bin/bash
# Ajuste del CMB con n=0.47, beta=1.04 FIJOS (los valores que prefiere la
# expansion). Compara con el best-fit de LCDM ya calculado (chi2=2768.32).
# Resuelve la tension: ¿el CMB rechaza n=0.47 o no lo exploraba?
set -eo pipefail
cd "$(dirname "$0")"
source ./env.sh >/dev/null 2>&1

export DVCH_SHORT_YAML=dvch_cobaya_short.yaml
export DVCH_MODE=minimize
export DVCH_CHAIN_OUTPUT=dvch_n47_min
export DVCH_RESUME=false

# n=0.47, beta=1.04 fijos se inyectan via variables de entorno
export DVCH_FIX_N=0.47
export DVCH_FIX_BETA=1.04

exec mpirun -np 1 python3 run_dvch_cobaya_lcdm.py
