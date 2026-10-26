#!/usr/bin/env python3
# Evidencia Bayesiana (log Z) via muestreo anidado con dynesty.
# Compara DVCH vs LCDM sobre datos reales (Pantheon+ + DESI DR1 + CC).
import os
import numpy as np
import dynesty
import dvch_vs_lcdm_real as m
from dvch_vs_lcdm_real import (load_pantheon, chi2_cc, chi2_bao, chi2_sne,
                               make_logpost, log_prior)

sne = load_pantheon()
z, mb, dmb, cov, cinv = sne
print(f"Pantheon+: {len(z)} SNe")

NAMES = {"lcdm": ["Om", "H0", "M"], "dvch": ["Om", "H0", "M", "n", "beta"]}
BOUNDS = {"lcdm": [(0.10, 0.50), (60.0, 80.0), (-25.0, -18.0)],
          "dvch": [(0.10, 0.50), (60.0, 80.0), (-25.0, -18.0),
                   (0.01, 0.99), (0.0, 2.0)]}

for model in ["lcdm", "dvch"]:
    if model == "lcdm" and os.path.exists("logz_lcdm_cache.txt"):
        globals()["logz_lcdm"] = [float(x) for x in
                                  open("logz_lcdm_cache.txt").read().split()]
        print(f"\nLCDM (cache): log Z = {logz_lcdm[0]:.3f} +/- {logz_lcdm[1]:.3f}")
        continue
    ndim = len(BOUNDS[model])
    logpost = make_logpost(model, sne)

    def ptform(u):
        return np.array([lo + (hi - lo) * u[i]
                         for i, (lo, hi) in enumerate(BOUNDS[model])])

    print(f"\n=== Nested sampling {model.upper()} ({ndim} dims) ===")
    sampler = dynesty.NestedSampler(logpost, ptform, ndim, nlive=400,
                                    sample="unif", bound="multi")
    sampler.run_nested(print_progress=False)
    res = sampler.results
    logz = res.logz[-1]
    logz_err = res.logzerr[-1]
    print(f"  log Z = {logz:.3f} +/- {logz_err:.3f}")
    print(f"  N iter: {len(res.logz)} puntos muertos")
    # mejores parametros
    from dynesty import utils as dyfunc
    weights = np.exp(res.logwt - res.logwt[-1])
    samples = res.samples
    means = dyfunc.mean_and_cov(samples, weights)[0]
    for nm, val in zip(NAMES[model], means):
        print(f"  {nm:6s} = {val:.4f}")

    globals()[f"logz_{model}"] = [logz, logz_err]
    if model == "lcdm":
        with open("logz_lcdm_cache.txt", "w") as fh:
            fh.write(f"{logz} {logz_err}")

print("\n=== VEREDICTO ===")
dlz = logz_dvch[0] - logz_lcdm[0]
err = (logz_dvch[1]**2 + logz_lcdm[1]**2) ** 0.5
print(f"  delta log Z (DVCH - LCDM) = {dlz:.3f} +/- {err:.3f}")
print("  (Jeffreys: >5 evidencia muy fuerte a favor de DVCH)")
