#!/usr/bin/env bash
# ============================================================
# run_codespace.sh — arranca la MCMC DVCH dentro de un GitHub
# Codespace de forma robusta (4 cores / 16 GB).
#
# Uso (dentro del Codespace, en la raiz del repo clonado):
#   export DVCH_BUNDLE_URL='https://litter.catbox.moe/9jijot.gz'
#   bash deploy/run_codespace.sh
#
# Variables:
#   DVCH_NCHAINS=4          cadenas MPI (default 4)
#   DVCH_MAX_SAMPLES=2000   muestras por cadena
#   DVCH_BURN_IN=200
#   DVCH_WORK=$HOME/dvch
#
# Lanza la MCMC DESACOPLADA (setsid+nohup) para que sobreviva si
# cierras la pestana, y aborta con mensaje claro si falta algo.
# ============================================================
set -euo pipefail

BUNDLE_URL="${DVCH_BUNDLE_URL:?Define DVCH_BUNDLE_URL con la URL del bundle}"
WORK="${DVCH_WORK:-$HOME/dvch}"
NCHAINS="${DVCH_NCHAINS:-4}"
MAXS="${DVCH_MAX_SAMPLES:-2000}"
BURN="${DVCH_BURN_IN:-200}"

echo "=== DVCH en Codespaces ==="
echo "cores       : $(nproc)"
echo "RAM total   : $(free -h | awk '/^Mem:/ {print $2}')"
echo "nchains     : $NCHAINS"
echo "max_samples : $MAXS   burn_in: $BURN"

if [[ "$NCHAINS" -gt "$(nproc)" ]]; then
    echo "[aviso] pides $NCHAINS cadenas y solo hay $(nproc) cores: se saturara."
    echo "[aviso] se baja a $(nproc) cadenas."
    NCHAINS="$(nproc)"
fi

mkdir -p "$WORK"
cd "$WORK"

if [[ ! -d dvch_bundle_stage ]]; then
    echo "[1/5] descargando bundle (~181 MB)..."
    curl -L --fail -o bundle.tar.gz "$BUNDLE_URL"
    echo "[2/5] descomprimiendo..."
    tar -xzf bundle.tar.gz
fi
[[ -d dvch_bundle_stage ]] || { echo "ERROR: bundle incompleto" >&2; exit 1; }
cd dvch_bundle_stage
echo "[3/5] bundle OK: $(du -sh . | cut -f1)"

# --- covmat de arranque ---
# El bundle NO incluye dvch_prod_wide.covmat (se excluyo al empaquetar). Ese
# covmat vive en la raiz del repo de GitHub, asi que lo copiamos al bundle.
COVMAT="dvch_prod_wide.covmat"
if [[ ! -f "repo/$COVMAT" ]]; then
    echo "[aviso] el bundle no trae repo/$COVMAT; buscandolo en el repo..."
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
    for cand in "$REPO_ROOT/$COVMAT" "$SCRIPT_DIR/$COVMAT" /workspaces/*/"$COVMAT"; do
        if [[ -f "$cand" ]]; then
            cp "$cand" "repo/$COVMAT"
            echo "[covmat] copiado desde $cand ($(wc -c < "repo/$COVMAT") bytes)"
            break
        fi
    done
fi
if [[ ! -f "repo/$COVMAT" ]]; then
    echo "[aviso] SIGUE faltando repo/$COVMAT -- la aceptacion sera PEOR." >&2
    echo "[aviso] para arreglarlo: sube dvch_prod_wide.covmat al repo de GitHub." >&2
    ALT="$(ls -S repo/*.covmat 2>/dev/null | head -1 || true)"
    if [[ -z "$ALT" ]]; then
        echo "ERROR: no hay ningun .covmat; la MCMC no puede arrancar." >&2
        echo "       Regenera el bundle con deploy/package_dvch.sh (sin excluir covmats)." >&2
        exit 1
    fi
    COVMAT="$(basename "$ALT")"
    echo "[aviso] uso alternativa: $COVMAT"
fi

# shellcheck disable=SC1091
source ./env_remote.sh
# 1 hilo por cadena: lo que importa es el paralelismo entre cadenas.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# --- requisitos del sistema ---
if ! command -v mpirun >/dev/null 2>&1; then
    echo "[4/5] instalando OpenMPI..."
    sudo apt-get update -qq && sudo apt-get install -y -qq openmpi-bin libopenmpi-dev
fi

# --- Python debe ser 3.14 (los .so del bundle son cpython-314) ---
PYV="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo none)"
if [[ "$PYV" != "3.14" ]]; then
    echo "ERROR: este Codespace tiene Python $PYV, pero el bundle necesita 3.14." >&2
    echo "       Reconstruye el Codespace (el devcontainer ya pide 3.14)." >&2
    echo "       Si insiste, borra el Codespace y crea uno nuevo desde este repo." >&2
    exit 1
fi

echo "[4/5] instalando deps Python (2-4 min)..."
python3 -m pip install --quiet --disable-pip-version-check \
    numpy scipy pandas matplotlib getdist mpi4py pyyaml 'cobaya==3.6.2'
python3 -m pip install --quiet -e ./CAMB-master

echo "[5/5] smoke test (debe imprimir DVCH OK y CLIK OK)..."
python3 -c "import clik; print('CLIK OK')"
python3 - <<'PY'
import camb
p = camb.CAMBparams()
p.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.12)
p.DVCH_flag = True; p.DVCH_n = 0.2; p.DVCH_beta = 1e-4
print("DVCH OK")
PY

cd repo
export DVCH_MAX_SAMPLES="$MAXS" DVCH_BURN_IN="$BURN" DVCH_LEARN_PROPOSAL=true
export DVCH_COVMAT="$COVMAT" DVCH_CHAIN_OUTPUT=dvch_prod DVCH_CHAIN_SEED=1

echo "[lanza] mpirun -np $NCHAINS (desacoplado: sobrevive al cierre de pestana)"
setsid nohup mpirun --allow-run-as-root -np "$NCHAINS" \
    python3 -u run_dvch_cobaya_full_highl.py \
    > dvch_prod_run.log 2>&1 < /dev/null &
disown || true

sleep 8
echo "--- procesos ---"
pgrep -af run_dvch_cobaya_full_highl || echo "(arrancando...)"
echo "--- ultimas lineas del log ---"
tail -n 5 dvch_prod_run.log 2>/dev/null || true
echo
echo "Monitorear : tail -f $PWD/dvch_prod_run.log"
echo "Diagnostico: python3 dvch_planck_chain_diagnostics.py dvch_prod.[1-4].txt"