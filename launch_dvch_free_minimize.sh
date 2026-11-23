#!/bin/bash
# Minimiza chi2 del CMB con DVCH_n y DVCH_beta LIBRES (caso anidado).
# Resuelve el problema de contabilidad: el chi2 "DVCH" existente
# (2769.28) viene del minimo de la CADENA, no de un minimizador, y por
# tanto no se puede comparar con el chi2 de LCDM del minimizador
# (2768.32): en un modelo anidado el minimo verdadero de DVCH tiene que
# ser <= el de LCDM. Este script lo calcula.
#
# AVISO: el CAMB parcheado deja el fondo identico a LCDM
# (dvch_camb_patch_check.py), de modo que el resultado mide el mejor
# ajuste del TERMINO DE PERTURBACION del CDM, no de la cerradura de fondo
# del paper.
set -eo pipefail
cd "$(dirname "$0")"
source ./env.sh >/dev/null 2>&1

export DVCH_SHORT_YAML=dvch_cobaya_short.yaml
export DVCH_MODE=minimize
export DVCH_CHAIN_OUTPUT=dvch_dvch_free_min
export DVCH_RESUME=false
export DVCH_IGNORE_PRIOR=true

exec mpirun -np 1 python3 run_dvch_cobaya_full_highl.py
