#!/usr/bin/env bash
# ============================================================
# DVCH Pipeline — variables de entorno REALES (WSL2 Ubuntu)
# Generado a partir de env.sh.example con las rutas verificadas en disco.
# Uso:  source env.sh   (ANTES de arrancar python)
# ============================================================

# --- CAMB patcheado (raíz del repo CAMB, ya compilado con DVCH) ---
export DVCH_CAMB_ROOT="/mnt/d/DVCH-external/CAMB-master"

# --- clik (egg de Python DESCOMPRIMIDO, no el .zip) ---
# El egg como zip hace que zipimport cargue clik/lkl.py (fuente) en vez del .so
# y `clik.clik` no existe. Por eso se usa el directorio extraído.
export DVCH_CLIK_EGG="/home/danieproyect/plc-3.1/clik-egg"

# --- Datos Planck (archivos .clik; son directorios con _mdb + clik) ---
export DVCH_PLANCK_LIKELIHOOD="/mnt/d/DVCH-external/planck-data/baseline/plc_3.0/hi_l/plik/plik_rd12_HM_v22b_TTTEEE.clik"
export DVCH_PLANCK_LOWL="/mnt/d/DVCH-external/planck-data/baseline/plc_3.0/low_l/commander/commander_dx12_v3_2_29.clik"
export DVCH_PLANCK_LOWE="/mnt/d/DVCH-external/planck-data/baseline/plc_3.0/low_l/simall/simall_100x143_offlike5_EE_Aplanck_B.clik"
export DVCH_PLANCK_LENSING="/mnt/d/DVCH-external/planck-data/baseline/plc_3.0/lensing/smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_consext8.clik_lensing"

# --- Directorio con libclik.so (CRÍTICO para el linker dinámico) ---
export DVCH_CLIK_LIB_DIR="/home/danieproyect/plc-3.1/lib"
export LD_LIBRARY_PATH="${DVCH_CLIK_LIB_DIR}:${LD_LIBRARY_PATH:-}"

# --- Que Cobaya importe el CAMB patcheado (y no el de PyPI) ---
export PYTHONPATH="${DVCH_CAMB_ROOT}:${PYTHONPATH:-}"

# --- Parámetros de la cadena MCMC (opcional, con defaults) ---
# export DVCH_MAX_SAMPLES=400
# export DVCH_BURN_IN=200
# export DVCH_LEARN_PROPOSAL=false
# export DVCH_RMINUS1_STOP=0.01
# export DVCH_COVMAT=dvch_prod_wide.covmat
# export DVCH_CHAIN_OUTPUT=dvch_prod
# export DVCH_CHAIN_SEED=42

echo "[env.sh] DVCH_CAMB_ROOT=$DVCH_CAMB_ROOT"
echo "[env.sh] DVCH_CLIK_EGG=$DVCH_CLIK_EGG"
echo "[env.sh] LD_LIBRARY_PATH=$LD_LIBRARY_PATH"