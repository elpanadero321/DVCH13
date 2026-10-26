#!/bin/bash
# Lanzamiento de la cadena widebeta DEFINITIVA con 5 cadenas.
#
# Por que una cadena nueva y no reanudar la anterior (28k muestras)?
#  - cobaya NO permite reanudar con distinto numero de cadenas.
#  - El R-1 multivariado estaba estancado en ~0.19 (real, no ruido del
#    estimador: suelo de ruido con 4 cadenas = 0.003).
#  - Con 4 cadenas la matriz de covarianza de medias tiene rango <= 3 sobre
#    26 parametros; anadir cadenas es la unica forma de romper ese techo.
#  - Las 28k muestras NO se pierden: su valor esta en la covmat aprendida
#    (dvch_prod_widebeta.covmat), que se usa como punto de partida.
#    Eso ahorra toda la fase de burn-in/learning inicial.
#
# IMPORTANTE: NO usar launch_widebeta.sh (fuerza max_samples=32 y borra).
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
export DVCH_RESUME=false

# Semillas DISTINTAS por cadena: con seed unico, cobaya deriva una semilla
# independiente por cadena a partir de ella, pero dejamos constancia.
# (cobaya ya garantiza semillas diferentes entre cadenas a partir de un seed
#  raiz; no es necesario fijar una por proceso.)

# covmat aprendida de la cadena de 4 (en /tmp por seguridad frente a WSL2)
COV=/tmp/dvch_widebeta5.covmat
cp -f dvch_prod_widebeta.covmat "$COV"
export DVCH_COVMAT="$COV"

exec mpirun -np 5 python3 run_dvch_cobaya_full_highl.py
