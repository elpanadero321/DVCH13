#!/usr/bin/env bash
# ============================================================
# bootstrap_remote.sh — Instala y arranca la MCMC DVCH FULL
# (4 cadenas x 2000 muestras, SIN recortes) en una maquina Linux
# nueva (servidor / HPC / VM nube).
#
# Uso en la maquina destino (Ubuntu 22.04+):
#   tar -xzf dvch_portable_bundle.tar.gz
#   cd dvch_bundle_stage
#   bash repo/deploy/bootstrap_remote.sh
#
# Requisitos: sudo, >= 8 cores, >= 8 GB RAM, ~5 GB disco.
# ============================================================
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "[boot] bundle root: $BUNDLE_ROOT"

echo "[boot] 1/5 instalando dependencias del sistema"
sudo apt update
sudo apt install -y gfortran build-essential python3 python3-pip python3-venv \
    libopenmpi-dev openmpi-bin liblapack-dev libfftw3-dev rsync

echo "[boot] 2/5 creando venv e instalando deps Python"
# CRITICO: clik y CAMB del bundle estan compilados contra CPython 3.14
# (archivos clik/lkl.cpython-314-x86_64-linux-gnu.so). Con otra version de
# Python el import falla con "incompatible ABI". Fijamos 3.14.
PY_MAJOR_MINOR="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if [[ "$PY_MAJOR_MINOR" != "3.14" ]]; then
    echo "[boot] ERROR: se requiere Python 3.14 (encontrado $PY_MAJOR_MINOR)." >&2
    echo "       El bundle trae .so compiladas para cpython-314." >&2
    echo "       Instala Python 3.14 (p.ej. via deadsnakes o pyenv) y reintenta." >&2
    exit 1
fi
python3 -m venv "$BUNDLE_ROOT/.venv"
# shellcheck disable=SC1091
source "$BUNDLE_ROOT/.venv/bin/activate"
pip install --upgrade pip
pip install numpy scipy pandas matplotlib getdist mpi4py pyyaml \
            cobaya==3.6.2

echo "[boot] 3/5 instalando CAMB parcheado incluido en el bundle"
pip install -e "$BUNDLE_ROOT/CAMB-master"

echo "[boot] 4/5 smoke test (debe imprimir DVCH OK)"
export DVCH_CAMB_ROOT="$BUNDLE_ROOT/CAMB-master"
export PYTHONPATH="$DVCH_CAMB_ROOT:${PYTHONPATH:-}"
python3 - <<'PY'
import camb
p = camb.CAMBparams()
p.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.12)
p.DVCH_flag = True; p.DVCH_n = 0.2; p.DVCH_beta = 1e-4
print("DVCH OK")
PY

echo "[boot] 5/5 lanzando MCMC FULL (4 cadenas x 2000, full plik)"
cd "$BUNDLE_ROOT/repo"
# shellcheck disable=SC1091
source "$BUNDLE_ROOT/env_remote.sh"
export DVCH_MAX_SAMPLES=2000
export DVCH_BURN_IN=200
export DVCH_LEARN_PROPOSAL=true
export DVCH_COVMAT=dvch_prod_wide.covmat
export DVCH_CHAIN_OUTPUT=dvch_prod
export DVCH_CHAIN_SEED=1

# -np N = numero de cadenas; sube si el servidor tiene mas cores.
NCHAINS="${DVCH_NCHAINS:-4}"
setsid bash -c "cd '$BUNDLE_ROOT/repo' && source '$BUNDLE_ROOT/env_remote.sh' && \
  export DVCH_MAX_SAMPLES=2000 DVCH_BURN_IN=200 DVCH_LEARN_PROPOSAL=true \
  DVCH_COVMAT=dvch_prod_wide.covmat DVCH_CHAIN_OUTPUT=dvch_prod DVCH_CHAIN_SEED=1 && \
  mpirun -np $NCHAINS python3 -u run_dvch_cobaya_full_highl.py" \
  > dvch_prod_run.log 2>&1 < /dev/null &
disown
echo "[boot] MCMC lanzada. Log: $BUNDLE_ROOT/repo/dvch_prod_run.log"
echo "[boot] Monitorea con: tail -f $BUNDLE_ROOT/repo/dvch_prod_run.log"
EOF