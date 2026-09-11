#!/usr/bin/env bash
# Read-only readiness probe for the DVCH Planck MCMC (run inside WSL).
set -u
CAMB=/mnt/d/DVCH-external/CAMB-master
PLANCK=/mnt/d/DVCH-external/planck-data/baseline/plc_3.0
CLIKLIB=/home/danieproyect/plc-3.1/lib
EGG=/home/danieproyect/plc-3.1/dist/clik-3.1-py3.14-linux-x86_64.egg

echo "== python / tools =="
python3 --version
which mpirun gfortran

echo "== cobaya/mpi4py =="
python3 -m pip list 2>/dev/null | grep -iE 'cobaya|getdist|mpi4py|camb|clik'

echo "== import camb (patched, cwd=CAMB) =="
cd "$CAMB"
python3 - <<'PY'
import camb, os
print("camb", camb.__version__)
print("file", camb.__file__)
print("so", [f for f in os.listdir(os.path.dirname(camb.__file__)) if f.endswith(".so")])
p = camb.CAMBparams()
try:
    p.DVCH_flag = True
    p.DVCH_n = 0.2
    p.DVCH_beta = 1e-4
    print("DVCH_OK")
except Exception as e:
    print("DVCH_FAIL", repr(e))
PY

echo "== clik load =="
export LD_LIBRARY_PATH="$CLIKLIB:${LD_LIBRARY_PATH:-}"
python3 - <<PY
import sys
sys.path.insert(0, "$EGG")
try:
    import clik
    print("clik ok", clik.__file__)
    h = clik.clik("$PLANCK/hi_l/plik/plik_rd12_HM_v22b_TTTEEE.clik")
    print("plik initialized")
except Exception as e:
    print("clik_fail", repr(e))
PY

echo "== planck files =="
for f in \
  "$PLANCK/hi_l/plik/plik_rd12_HM_v22b_TTTEEE.clik" \
  "$PLANCK/low_l/commander/commander_dx12_v3_2_29.clik" \
  "$PLANCK/low_l/simall/simall_100x143_offlike5_EE_Aplanck_B.clik" \
  "$PLANCK/lensing/smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_consext8.clik_lensing"; do
  [ -f "$f" ] && echo "OK  $f" || echo "MISS $f"
done