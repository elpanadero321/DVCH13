"""Diagnostico: compara espectros CMB y parametros derivados entre LCDM y DVCH
(fondo modificado, fuente de perturbaciones desactivada) para localizar el
origen del chi2 gigantesco (+5.9e4) en el CMB."""
import os
import sys

import numpy as np
sys.path.insert(0, os.environ.get("DVCH_CAMB_ROOT", "/mnt/d/DVCH-external/CAMB-master"))
import camb
from camb import get_results, model
from camb.model import CAMBparams

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_diag_cmb.csv")


def base_params():
    p = CAMBparams()
    p.WantTensors = False
    p.DoLensing = True
    p.min_l = 2
    p.max_l = 2508
    p.set_for_lmax(2508, lens_potential_accuracy=1)
    p.InitPower.set_params(As=2.1e-9, ns=0.965)
    # LCDM: mejor ajuste del paper (sin DVCH)
    p.set_cosmology(H0=67.36, ombh2=0.02237, omch2=0.1200, omk=0.0,
                    tau=0.0544, mnu=0.06, neutrino_hierarchy="degenerate")
    return p


def get_cl(pars, tag):
    res = get_results(pars)
    cl = res.get_total_cls(2508)  # lmax=2508, unlensed+lensed totales
    derived = res.get_derived_params()
    theta = 100 * derived.get("thetastar", np.nan)
    rd = derived.get("rdrag", np.nan)
    zdrag = derived.get("zdrag", np.nan)
    da = derived.get("DAstar", np.nan)
    ze_i = derived.get("zei" if False else "", np.nan)
    print(f"[{tag}] theta*={theta:.4f}  r_drag={rd:.2f}  z_drag={zdrag:.2f}  "
          f"D_A*={da:.1f}")
    # H(z) en algunos puntos
    zs = np.array([0.0, 0.5, 1.0, 2.0, 10.0, 100.0, 1000.0])
    arr = res.h_of_z(zs)
    for z, h in zip(zs, arr):
        print(f"    H({z:>6}) = {h[0] if h.ndim else h:.3f}")
    return cl, res


def main():
    pars = base_params()
    cl_ref, res_ref = get_cl(pars, "LCDM")

    pars_d = base_params()
    pars_d.DVCH_flag = True
    pars_d.DVCH_n = float(os.environ.get("DVCH_N", 0.518))
    pars_d.DVCH_beta = float(os.environ.get("DVCH_BETA", 1.104))
    cl_d, res_d = get_cl(pars_d, f"DVCH n={pars_d.DVCH_n} b={pars_d.DVCH_beta}")

    ls = np.arange(2, 2509)
    tt_ref = cl_ref[2:, 0]
    tt_d = cl_d[2:, 0]
    if not np.all(np.isfinite(tt_d)):
        print("!!! DVCH: valores no finitos en Cl_TT !!!")
    rel = np.where(tt_ref > 0, (tt_d - tt_ref) / tt_ref, 0.0)
    for l in [220, 500, 1000, 1500, 2000]:
        print(f"  l={l}: rel diff TT = {rel[l-2]:+.4e}")

    #_posicion del primer pico
    i = int(np.argmax(tt_ref[10:400])) + 10
    print(f"  pico LCDM en l={ls[i]}, DVCH en l={ls[int(np.argmax(tt_d[10:400]))+10]}")

    # comparar H(z) directamente
    zs = np.array([0.0, 0.5, 1.0, 2.0, 10.0, 100.0, 1000.0])
    h_ref = np.array([h[0] if np.ndim(h) else h for h in res_ref.h_of_z(zs)])
    h_d = np.array([h[0] if np.ndim(h) else h for h in res_d.h_of_z(zs)])
    print("  H(z) rel diff:")
    for z, a, b in zip(zs, h_ref, h_d):
        print(f"    z={z:>6}: {a:8.3f} vs {b:8.3f}  ({(b-a)/a:+.2e})")


if __name__ == "__main__":
    main()
