"""Evalua la clausura DVCH (referencia Python) en la cosmologia del CMB
(Om=0.3021, H0=67.36) y compara con la H(z) que da CAMB. Distingue:
(a) bug del parche CAMB vs (b) la clausura predice de verdad H(z) grande.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dvch_vs_lcdm_real as dvr  # noqa: E402

N = float(os.environ.get("DVCH_N", 0.518))
BETA = float(os.environ.get("DVCH_BETA", 1.104))
OM_M0 = 0.3021
OM_R0 = dvr.Omega_r0
OL0 = 1.0 - OM_M0 - OM_R0
H0 = 67.36

zs = np.array([0.0, 0.3, 1.0, 2.0, 10.0, 100.0, 1000.0, 1100.0])
e2 = dvr.E2_dvch(zs, OM_M0, N, BETA)
e2_lcdm = OM_R0 * (1 + zs) ** 4 + OM_M0 * (1 + zs) ** 3 + OL0
print(f"n={N} beta={BETA} Om_m0={OM_M0:.4f} Om_l0={OL0:.4f} Om_r0={OM_R0:.3e}")
print("  z     H_DVCH   H_LCDM    dH/H")
for z, a, b in zip(zs, np.sqrt(e2), np.sqrt(e2_lcdm)):
    print(f"{z:7.1f} {a*H0:8.2f} {b*H0:8.2f} {(a-b)/b:+8.2e}")

# x(z) = matter(z)/Om: cuanto se desvia la materia de (1+z)^3
grid = dvr._dvch_grid(OM_M0, N, BETA)
matter = None
if grid is not None:
    # reintegro para obtener matter(z) en mis z
    matter = np.interp(zs, dvr._ZGRID, dvr._integrate_dvch_matter(
        OM_M0, N, BETA, dvr._ZGRID))
if matter is not None:
    print("\n  z     matter/(Om*(1+z)^3)")
    for z, m in zip(zs, matter):
        print(f"{z:7.1f} {m/(OM_M0*(1+z)**3):8.4f}")
print("\nCAMB deca: H(1)=-9.3%, H(1000)=-11% (patch). Ref Python arriba.")
