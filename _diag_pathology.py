"""Mapea la frontera de la patologia de la clausura DVCH:
materia/(Om*(1+z)^3) debe ser ~1 (crecimiento estandar a^-3).
Si colapsa (<<1), el pasado no tiene epoca de materia y el CMB se destruye.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dvch_vs_lcdm_real as dvr  # noqa: E402

CASES = [
    (0.3021, 0.518, 1.104),   # mediana tardia en cosmologia CMB
    (0.3021, 0.4226, 0.5257),  # best-fit tardio (sin switch) en cosmologia CMB
    (0.3021, 0.097, 0.101),   # best-fit del CMB
    (0.38, 0.4226, 0.5257),   # best-fit tardio en su cosmologia
    (0.38, 0.097, 0.101),     # corner CMB en cosmologia tardia
    (0.38, 0.3239, 0.2894),   # switch z_c=1
]
ZS = np.array([1.0, 2.0, 10.0, 100.0, 1000.0])

print("Om      n      beta   " + "".join(f"z={z:<8}" for z in ZS))
for om, n, b in CASES:
    mg = dvr._integrate_dvch_matter(om, n, b, dvr._ZGRID)
    if mg is None:
        print(f"{om:.4f} {n:.4f} {b:.4f}  INTEGRACION FALLIDA")
        continue
    mt = np.interp(ZS, dvr._ZGRID, mg)
    row = "".join(f"{m/(om*(1+z)**3):<10.4f}" for z, m in zip(ZS, mt))
    print(f"{om:.4f} {n:.4f} {b:.4f} {row}")
