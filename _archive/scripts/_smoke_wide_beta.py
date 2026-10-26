"""Smoke test: CAMB-DVCH con beta en la zona late-time (beta ~ 0.4-0.5).

Verifica que el CAMB patcheado calcula Cls sin explotar cuando DVCH_beta
se acerca al valor que encontro el fit late-time (beta ~ 0.495).
"""
import os
import sys

os.environ.setdefault("DVCH_CAMB_ROOT", "/mnt/d/DVCH-external/CAMB-master")
sys.path.insert(0, os.environ["DVCH_CAMB_ROOT"])

import numpy as np
import camb

print("camb importado de:", os.path.dirname(camb.__file__))

cases = [
    (0.07, 1e-4, "casi-LCDM (beta pequeno)"),
    (0.07, 0.1, "beta intermedio"),
    (0.07, 0.4, "beta ~ late-time"),
    (0.07, 0.495, "beta = mediana late-time"),
    (0.07, 0.9, "beta cerca del limite fisico"),
]

for n, beta, lbl in cases:
    try:
        p = camb.CAMBparams()
        p.set_cosmology(H0=67.3, ombh2=0.0224, omch2=0.120, tau=0.054)
        p.DVCH_flag = True
        p.DVCH_n = n
        p.DVCH_beta = beta
        p.InitPower.set_params(As=2.1e-9, ns=0.965)
        p.set_for_lmax(2508)
        r = camb.get_results(p)
        cl = r.get_total_cls(2000)
        tt2500 = cl[2000, 0]
        print(f"  OK  n={n:<5} beta={beta:<6} -> TT(l=2500)={tt2500:.4g}   [{lbl}]")
    except Exception as e:
        print(f"  FAIL n={n:<5} beta={beta:<6} -> {type(e).__name__}: {e}   [{lbl}]")

print("\nSmoke test completado.")
