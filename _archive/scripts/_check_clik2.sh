#!/usr/bin/env bash
set -u
PLANCK=/mnt/d/DVCH-external/planck-data/baseline/plc_3.0
CLIKLIB=/home/danieproyect/plc-3.1/lib
EGG=/home/danieproyect/plc-3.1/dist/clik-3.1-py3.14-linux-x86_64.egg

export LD_LIBRARY_PATH="$CLIKLIB:${LD_LIBRARY_PATH:-}"

echo "== is egg a zip? =="
file "$EGG"
python3 - <<PY
import zipfile, sys
z = zipfile.ZipFile("$EGG")
names = [n for n in z.namelist() if n.startswith("clik/") and (n.endswith(".py") or n.endswith(".so"))]
print("clik files:", [n for n in names][:30])
init = z.read("clik/__init__.py").decode("utf-8", "replace")
print("---- __init__.py (head) ----")
print("\n".join(init.splitlines()[:60]))
PY

echo "== exact runner-style clik use =="
python3 - <<PY
import sys
sys.path.insert(0, "$EGG")
try:
    import clik
    import clik.hpy as hpy
    print("clik import ok")
    h = clik.clik("$PLANCK/hi_l/plik/plik_rd12_HM_v22b_TTTEEE.clik")
    print("plik ok, extra names:", len(h.extra_parameter_names))
except Exception as e:
    import traceback; traceback.print_exc()
PY