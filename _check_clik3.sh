#!/usr/bin/env bash
set -u
PLANCK=/mnt/d/DVCH-external/planck-data/baseline/plc_3.0
CLIKLIB=/home/danieproyect/plc-3.1/lib
EGG=/home/danieproyect/plc-3.1/dist/clik-3.1-py3.14-linux-x86_64.egg
EX=/tmp/clikegg

export LD_LIBRARY_PATH="$CLIKLIB:${LD_LIBRARY_PATH:-}"
rm -rf "$EX"; mkdir -p "$EX"
python3 -m zipfile -e "$EGG" "$EX"
echo "== extracted =="
ls "$EX/clik" | head

echo "== clik from unpacked egg =="
python3 - <<PY
import sys
sys.path.insert(0, "$EX")
try:
    import clik
    print("clik file:", clik.__file__)
    print("has clik:", hasattr(clik, "clik"))
    h = clik.clik("$PLANCK/hi_l/plik/plik_rd12_HM_v22b_TTTEEE.clik")
    print("PLIK OK, extra names:", len(h.extra_parameter_names))
    import numpy as np
    hpy = None
    import clik.hpy as hpy
    print("hpy ok")
except Exception as e:
    import traceback; traceback.print_exc()
PY