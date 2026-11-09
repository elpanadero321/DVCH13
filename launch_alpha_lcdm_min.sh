#!/bin/bash
# PASO 0 (segun INSTRUCCIONES_PARA_IA.md): veredicto de fondo del modelo
# alpha usando CAMB ESTANDAR (ppf con rho_DE(a) = OL + alpha*(1-a)^2).
# Modo "lcdm" (env ALPHA_MODE=lcdm): alpha=0 fijado; mismo camino ppf que
# el modelo -> control de anidamiento (debe dar ~2766.8 con nuisance libre).
set -eo pipefail
cd "$(dirname "$0")"
source ./env.sh >/dev/null 2>&1

export ALPHA_MODE=lcdm
export ALPHA_IGNORE_PRIOR=true
export ALPHA_CHAIN_OUTPUT=alpha_lcdm_min_free

exec python3 run_alpha_cobaya.py
