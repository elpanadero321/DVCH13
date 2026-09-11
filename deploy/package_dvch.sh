#!/usr/bin/env bash
# ============================================================
# package_dvch.sh — Empaqueta el pipeline DVCH COMPLETO y portable
# para correrlo en OTRA maquina Linux (servidor / HPC / VM nube).
#
# Incluye: repo DVCH + CAMB parcheado YA COMPILADO + egg clik +
#          libclik.so + datos Planck .clik.
# NO recorta nada: el arranque en destino es 4 cadenas x 2000 muestras.
#
# Uso (dentro de WSL):
#   bash deploy/package_dvch.sh
# Salida:
#   /mnt/d/DVCH-external/dvch_portable_bundle.tar.gz
# ============================================================
set -euo pipefail

REPO="/mnt/d/DVCH13-1"
CAMB="/mnt/d/DVCH-external/CAMB-master"
CLIK_EGG="/home/danieproyect/plc-3.1/clik-egg"
CLIK_LIB="/home/danieproyect/plc-3.1/lib"
PLANCK_DATA="/mnt/d/DVCH-external/planck-data"
STAGE="/mnt/d/DVCH-external/dvch_bundle_stage"
OUT="/mnt/d/DVCH-external/dvch_portable_bundle.tar.gz"

echo "[pkg] limpiando stage previo"
rm -rf "$STAGE"
mkdir -p "$STAGE"

echo "[pkg] copiando repo (sin salidas de cadena)"
mkdir -p "$STAGE/repo"
rsync -a --delete \
  --exclude 'dvch_prod*' \
  --exclude 'dvch_planck_short_chain*' \
  --exclude '*.pyc' \
  --exclude '__pycache__' \
  --exclude '.git' \
  "$REPO"/ "$STAGE/repo"/

echo "[pkg] copiando CAMB parcheado (ya compilado: .so + build)"
rsync -a --delete \
  --exclude '.git' \
  --exclude '*.pyc' \
  "$CAMB"/ "$STAGE/CAMB-master"/

echo "[pkg] copiando clik egg descomprimido + libclik.so"
mkdir -p "$STAGE/clik"
rsync -a "$CLIK_EGG"/ "$STAGE/clik/clik-egg"/
rsync -a "$CLIK_LIB"/ "$STAGE/clik/lib"/

echo "[pkg] copiando datos Planck .clik"
rsync -a "$PLANCK_DATA"/ "$STAGE/planck-data"/

echo "[pkg] escribiendo env.sh del destino (rutas relativas al bundle)"
cat > "$STAGE/env_remote.sh" <<'EOF'
#!/usr/bin/env bash
# Rutas relativas a la raiz del bundle desplegado.
# Ajusta DVCH_BUNDLE_ROOT si lo descomprimes en otro sitio.
export DVCH_BUNDLE_ROOT="${DVCH_BUNDLE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export DVCH_CAMB_ROOT="${DVCH_BUNDLE_ROOT}/CAMB-master"
export DVCH_CLIK_EGG="${DVCH_BUNDLE_ROOT}/clik/clik-egg"
export DVCH_CLIK_LIB_DIR="${DVCH_BUNDLE_ROOT}/clik/lib"
export LD_LIBRARY_PATH="${DVCH_CLIK_LIB_DIR}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${DVCH_CAMB_ROOT}:${PYTHONPATH:-}"
export DVCH_PLANCK_LIKELIHOOD="${DVCH_BUNDLE_ROOT}/planck-data/baseline/plc_3.0/hi_l/plik/plik_rd12_HM_v22b_TTTEEE.clik"
export DVCH_PLANCK_LOWL="${DVCH_BUNDLE_ROOT}/planck-data/baseline/plc_3.0/low_l/commander/commander_dx12_v3_2_29.clik"
export DVCH_PLANCK_LOWE="${DVCH_BUNDLE_ROOT}/planck-data/baseline/plc_3.0/low_l/simall/simall_100x143_offlike5_EE_Aplanck_B.clik"
export DVCH_PLANCK_LENSING="${DVCH_BUNDLE_ROOT}/planck-data/baseline/plc_3.0/lensing/smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_consext8.clik_lensing"
echo "[env_remote] CAMB=$DVCH_CAMB_ROOT"
echo "[env_remote] CLIK=$DVCH_CLIK_EGG"
echo "[env_remote] LIB=$LD_LIBRARY_PATH"
EOF
chmod +x "$STAGE/env_remote.sh"

echo "[pkg] empaquetando en $OUT (esto puede tardar)"
tar -czf "$OUT" -C "$(dirname "$STAGE")" "$(basename "$STAGE")"

echo "[pkg] LISTO:"
ls -lh "$OUT"
echo "[pkg] tamano del stage:"
du -sh "$STAGE"
EOF