#!/bin/bash
# PASO 0 (segun INSTRUCCIONES_PARA_IA.md): minimiza el chi2 del CMB con
# alpha LIBRE (CAMB estandar, DE desacoplada ppf). Caso anidado del de
# arriba (alpha=0).
set -eo pipefail
cd "$(dirname "$0")"
source ./env.sh >/dev/null 2>&1

export ALPHA_MODE=free
export ALPHA_IGNORE_PRIOR=true
export ALPHA_CHAIN_OUTPUT=alpha_free_min
# Arrancar desde el minimo de LCDM si existe (mejor convergencia).
if [ -f alpha_lcdm_min_free.bestfit.txt ]; then
  export ALPHA_REF_MIN=alpha_lcdm_min_free.bestfit.txt
fi

exec python3 run_alpha_cobaya.py
