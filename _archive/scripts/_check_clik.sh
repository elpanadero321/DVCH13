#!/usr/bin/env bash
set -u
PLANCK=/mnt/d/DVCH-external/planck-data/baseline/plc_3.0
CLIKLIB=/home/danieproyect/plc-3.1/lib
EGG=/home/danieproyect/plc-3.1/dist/clik-3.1-py3.14-linux-x86_64.egg

echo "== which libclik gets dlopened =="
export LD_LIBRARY_PATH="$CLIKLIB:${LD_LIBRARY_PATH:-}"
ldd "$CLIKLIB/libclik.so" 2>&1 | grep -iE 'not found|clik' | head
ls "$CLIKLIB" | head

echo "== egg contents =="
ls "$EGG/clik" | head -40

echo "== try clik.clik =="
python3 - <<PY
import sys, os, glob
sys.path.insert(0, "$EGG")
import clik
print("clik pkg:", clik.__file__)
print("has clik attr:", hasattr(clik, "clik"))
import clik.lkl as lkl
print("lkl file:", lkl.__file__)
print("lkl attrs:", [a for a in dir(lkl) if not a.startswith("__")][:20])
try:
    import clik._clik as _c
    print("_clik ok", _c.__file__)
except Exception as e:
    print("_clik fail", repr(e))
PY