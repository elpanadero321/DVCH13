"""Integra la ODE de conservacion DVCH hasta z=1100 (paso fino) y compara
con la H(z) de CAMB. _ZGRID en dvch_vs_lcdm_real.py solo llega a z=2.61
(datos tardios), asi que aqui se usa una malla propia.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dvch_vs_lcdm_real as dvr  # noqa: E402


def rk4_matter(Om, n, beta, zmax=1100.0, h=0.01):
    """matter(z) por RK4 con la misma ODE que el Fortran (m(0)=Om)."""
    OL = 1.0 - Om - dvr.Omega_r0
    z = 0.0
    m = Om
    out_z, out_m = [0.0], [Om]
    while z < zmax - 1e-9:
        hh = min(h, zmax - z)
        k1 = dvr._dmatter_dz(z, m, Om, n, beta, OL)
        k2 = dvr._dmatter_dz(z + 0.5 * hh, m + 0.5 * hh * k1, Om, n, beta, OL)
        k3 = dvr._dmatter_dz(z + 0.5 * hh, m + 0.5 * hh * k2, Om, n, beta, OL)
        k4 = dvr._dmatter_dz(z + hh, m + hh * k3, Om, n, beta, OL)
        m = m + (hh / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        z = z + hh
        if not math.isfinite(m) or m <= 0.0:
            return np.array(out_z), np.array(out_m), False
        out_z.append(z)
        out_m.append(m)
    return np.array(out_z), np.array(out_m), True


def main():
    om, n, b = 0.3021, 0.518, 1.104
    OL = 1.0 - om - dvr.Omega_r0
    zg, mg, ok = rk4_matter(om, n, b)
    print(f"integracion OK={ok}  zmax={zg[-1]:.1f}  x(zmax)={mg[-1]/om:.3e}")
    print(" matter/(Om(1+z)^3) y E2 vs LCDM:")
    for z in [1.0, 2.0, 3.0, 10.0, 100.0, 500.0, 1000.0]:
        m = np.interp(z, zg, mg)
        x = m / om
        e2 = dvr.Omega_r0 * (1 + z) ** 4 + m + OL * (x ** n) * (1 + b) / (
            1 + b * (dvr.Omega_r0 * (1 + z) ** 4 + m + OL))
        e2l = dvr.Omega_r0 * (1 + z) ** 4 + om * (1 + z) ** 3 + OL
        print(f"  z={z:6}: x/(1+z)^3={x/(1+z)**3:8.4f}  "
              f"H/H0={np.sqrt(e2):9.3f} vs {np.sqrt(e2l):9.3f}  "
              f"dH/H={(np.sqrt(e2)-np.sqrt(e2l))/np.sqrt(e2l):+8.2e}")


if __name__ == "__main__":
    main()
