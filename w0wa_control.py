#!/usr/bin/env python3
"""
Prueba de CONTROL: w0waCDM (energia oscura evolutiva) sobre los mismos
datos reales, con el mismo metodo.

Motivo cientifico: DESI DR1/DR2 + Pantheon+ ya prefieren w0waCDM frente a
LCDM por una cantidad similar. Si DVCH solo iguala a w0waCDM, esta
reproduciendo un fenomeno conocido. Si lo supera, es un resultado nuevo.

Ademas: tests de robustez por subset (SNe solo, BAO solo, CC solo)
y sensibilidad al prior de n.
"""
import os
import numpy as np
import dynesty
import dvch_vs_lcdm_real as m
from dvch_vs_lcdm_real import (load_pantheon, chi2_cc, chi2_bao, chi2_sne)

sne = load_pantheon()
Z_SN, MB, DMB, COV, CINV = sne
print(f"Pantheon+: {len(Z_SN)} SNe\n")

# ------------------------------------------------------------------
# w0waCDM: E2 = Or(1+z)^4 + Om(1+z)^3 + OL*(1+z)^(3(1+w0+wa))*exp(-3wa z/(1+z))
# ------------------------------------------------------------------
def E2_w0wa(z, Om, w0, wa):
    OL = 1.0 - Om - m.Omega_r0
    if OL <= 0:
        return np.nan
    opz = 1.0 + np.asarray(z, dtype=float)
    exponent = 3.0 * (1.0 + w0 + wa)
    de = OL * opz**exponent * np.exp(-3.0 * wa * (opz - 1.0) / opz)
    return m.Omega_r0 * opz**4 + Om * opz**3 + de


def chi2_all_w0wa(Om, H0, M, w0, wa, subsets=("cc", "bao", "sne")):
    total = 0.0
    if "cc" in subsets:
        for z, H_obs, sigma in m.cc_data:
            e2 = E2_w0wa(z, Om, w0, wa)
            if np.isnan(e2) or e2 <= 0:
                return 1e10
            total += ((H0 * np.sqrt(e2) - H_obs) / sigma) ** 2
    if "bao" in subsets:
        zs = np.array([d[0] for d in m.desi_data])
        e2g = E2_w0wa(m._ZGRID, Om, w0, wa)
        if np.any(np.isnan(e2g)) or np.any(e2g <= 0):
            return 1e10
        inv_e = 1.0 / np.sqrt(e2g)
        cum = np.concatenate([[0.0], np.cumsum((inv_e[1:] + inv_e[:-1]) * 0.5
                                                * np.diff(m._ZGRID))])
        integ = np.interp(zs, m._ZGRID, cum)
        dm = m.c_speed / H0 * integ
        for i, (z, _, val, sigma) in enumerate(m.desi_data):
            E = np.sqrt(float(E2_w0wa(z, Om, w0, wa)))
            th = dm[i] / m.RD_FID  # todos DM_rd
            total += ((th - val) / sigma) ** 2
    if "sne" in subsets:
        e2g = E2_w0wa(m._ZGRID, Om, w0, wa)
        if np.any(np.isnan(e2g)) or np.any(e2g <= 0):
            return 1e10
        inv_e = 1.0 / np.sqrt(e2g)
        cum = np.concatenate([[0.0], np.cumsum((inv_e[1:] + inv_e[:-1]) * 0.5
                                                * np.diff(m._ZGRID))])
        integ = np.interp(Z_SN, m._ZGRID, cum)
        dl = (1.0 + Z_SN) * m.c_speed / H0 * integ
        if np.any(dl <= 0):
            return 1e10
        mu = 5.0 * np.log10(dl) + 25.0 + M
        r = MB - mu
        total += float(r @ CINV @ r)
    return total


BOUNDS = [(0.10, 0.50), (60.0, 80.0), (-25.0, -18.0), (-3.0, 1.0), (-3.0, 2.0)]
NAMES = ["Om", "H0", "M", "w0", "wa"]


def run_nested(subsets, label, nlive=400):
    def logpost(p):
        Om, H0, M, w0, wa = p
        for (lo, hi), v in zip(BOUNDS, p):
            if not (lo < v < hi):
                return -np.inf
        c2 = chi2_all_w0wa(Om, H0, M, w0, wa, subsets)
        if c2 > 1e9:
            return -np.inf
        return -0.5 * c2

    def ptform(u):
        return np.array([lo + (hi - lo) * ui
                         for ui, (lo, hi) in zip(u, BOUNDS)])

    print(f"  (corriendo nested sampling: {label})")
    sampler = dynesty.NestedSampler(logpost, ptform, 5, nlive=nlive,
                                    sample="unif", bound="multi")
    sampler.run_nested(print_progress=False)
    res = sampler.results
    from dynesty import utils as dyfunc
    weights = np.exp(res.logwt - res.logwt[-1])
    means = dyfunc.mean_and_cov(res.samples, weights)[0]
    print(f"\n=== {label} ===")
    print(f"  log Z = {res.logz[-1]:.3f} +/- {res.logzerr[-1]:.3f}")
    for nm, val in zip(NAMES, means):
        print(f"  {nm:6s} = {val:.4f}")
    return res.logz[-1], res.logzerr[-1], means


if __name__ == "__main__":
    print("########## MODELO DE CONTROL: w0waCDM ##########\n")
    r = run_nested(("cc", "bao", "sne"), "w0waCDM (todos los datos)")
    with open("logz_w0wa_cache.txt", "w") as fh:
        fh.write(f"{r[0]} {r[1]}")
