"""Paso 0 (segun INSTRUCCIONES): el fondo del modelo alpha se puede meter en
CAMB estandar como DE desacoplada con
    rho_DE(a) = OL + alpha*(1-a)^2,
    w_eff(a) = -1 + 2*alpha*a*(1-a) / (3*(OL + alpha*(1-a)^2))
Usando PPF solo para el fondo (theta*, r_drag, D_A no dependen de las
perturbaciones). Verifica el claim: Delta theta*/theta* ~= +0.78% (alpha=0.597)
a H0 fijo, y el H0 que conserva theta*.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get("DVCH_CAMB_ROOT",
                                   "/mnt/d/DVCH-external/CAMB-master"))
import camb
from camb import get_results
from camb.model import CAMBparams

OMCH2 = 0.1430 - 0.02237  # wm h2 = 0.1430 fijado (Planck)
OMBH2 = 0.02237
TAU, AS, NS = 0.0544, 2.1e-9, 0.965


def params_for(h0, alpha, omch2=OMCH2):
    p = CAMBparams()
    p.WantTensors = False
    p.DoLensing = True
    p.set_for_lmax(2500, lens_potential_accuracy=1)
    p.InitPower.set_params(As=AS, ns=NS)
    p.set_cosmology(H0=h0, ombh2=OMBH2, omch2=omch2, omk=0.0, tau=TAU,
                    mnu=0.06, neutrino_hierarchy="degenerate")
    if alpha != 0.0:
        a = np.geomspace(1e-5, 1.0, 2000)
        h = h0 / 100.0
        OL = 1.0 - OMBH2 / h**2 - omch2 / h**2
        w = -1.0 + 2.0 * alpha * a * (1.0 - a) / (
            3.0 * (OL + alpha * (1.0 - a) ** 2))
        # anadir radiacion al total que ve CAMB (la ponen ellas)
        p.set_dark_energy_w_a(a, w, dark_energy_model="ppf")
    return p


def theta_of(h0, alpha):
    res = get_results(params_for(h0, alpha))
    d = res.get_derived_params()
    return d.get("thetastar", np.nan), d.get("rdrag", np.nan), d.get("DAstar",
                                                                     np.nan)


def main():
    print(f"wm h2 = {OMCH2 + OMBH2:.4f} (fijado, Planck)")
    th0, rd0, da0 = theta_of(67.36, 0.0)
    print(f"LCDM (H0=67.36):  theta*={100*th0:.4f}  r_drag={rd0:.2f}  "
          f"D_A*={da0:.1f}")

    for alpha in (0.3, 0.597, 1.0):
        # a) H0 fijo
        tha, rda, daa = theta_of(67.36, alpha)
        print(f"\nalpha={alpha}: H0=67.36 fijo -> theta*={100*tha:.4f} "
              f"({100*(tha/th0-1):+.2f}%), r_drag={rda:.2f}, D_A*={daa:.1f}")
        # b) H0 que conserva theta* (cota de H0 y wm)
        lo, hi = 60.0, 70.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if theta_of(mid, alpha)[0] > th0:
                lo = mid
            else:
                hi = mid
        h_ok = 0.5 * (lo + hi)
        th_ok, rd_ok, da_ok = theta_of(h_ok, alpha)
        om = 0.1430 / (h_ok / 100.0) ** 2
        print(f"   H0 que conserva theta*: {h_ok:.2f} -> Om={om:.3f}, "
              f"theta*={100*th_ok:.4f} (diff {100*(th_ok/th0-1):+.4f}%), "
              f"r_drag={rd_ok:.2f}")

    # w_eff en el mejor ajuste tardio
    alpha = 0.597
    h = 67.36 / 100.0
    OL = 1.0 - OMBH2 / h**2 - OMCH2 / h**2
    print(f"\nw_eff (alpha={alpha}, OL={OL:.3f}):")
    for z in (0.0, 0.25, 0.5, 1.0, 2.0):
        a = 1.0 / (1.0 + z)
        w = -1.0 + 2.0 * alpha * a * (1.0 - a) / (
            3.0 * (OL + alpha * (1.0 - a) ** 2))
        print(f"  w_eff(z={z:.2f}) = {w:.4f}")


if __name__ == "__main__":
    main()
