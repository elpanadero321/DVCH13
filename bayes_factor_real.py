#!/usr/bin/env python3
# Evidencia Bayesiana (log Z) via muestreo anidado con dynesty.
# Compara LCDM, w0waCDM (control) y DVCH sobre los mismos datos reales
# (Pantheon+ cortada + DESI DR1 con correlaciones + CC de 31 puntos),
# con el mismo prior plano marginalizado sobre r_d.
# Las posteriores estan definidas en _bayes_worker.py (picklables) y el
# muestreo se paraleliza con un pool de procesos.
import os

import numpy as np
import pandas as pd

import w0wa_control as wc
import _bayes_worker as worker
from dvch_vs_lcdm_real import PARAM_NAMES

NDIM = {"lcdm": 4, "w0wa": 6, "dvch": 6}
NAMES = {"lcdm": PARAM_NAMES["lcdm"], "w0wa": wc.NAMES,
         "dvch": PARAM_NAMES["dvch"]}


def run_nested(model, nlive=200, nprocs=None):
    import dynesty
    from dynesty.pool import Pool

    nproc = nprocs or min(10, os.cpu_count() or 1)
    logl = getattr(worker, f"logpost_{model}")
    ptform = getattr(worker, f"ptform_{model}")
    print(f"  (pool de {nproc} procesos)")
    with Pool(nproc, logl, ptform) as pool:
        sampler = dynesty.NestedSampler(pool.loglike, pool.prior_transform,
                                        NDIM[model], pool=pool, nlive=nlive,
                                        sample="rslice", bound="multi",
                                        bootstrap=0)
        sampler.run_nested(print_progress=False)
    return sampler.results


def main(nlive=200):
    out = {}
    for model in ["lcdm", "w0wa", "dvch"]:
        print(f"\n=== Nested sampling {model.upper()} "
              f"({NDIM[model]} dims, {nlive} live points, pool) ===")
        res = run_nested(model, nlive)
        logz, logzerr = float(res.logz[-1]), float(res.logzerr[-1])
        from dynesty import utils as dyfunc
        weights = np.exp(res.logwt - res.logwt[-1])
        means = dyfunc.mean_and_cov(res.samples, weights)[0]
        print(f"  log Z = {logz:.3f} +/- {logzerr:.3f}")
        for nm, val in zip(NAMES[model], means):
            print(f"  {nm:6s} = {val:.4f}")
        out[model] = (logz, logzerr)

    print("\n=== VEREDICTO (evidencia bayesiana) ===")
    rows = []
    for model in ["lcdm", "w0wa", "dvch"]:
        rows.append({"model": model, "logZ": round(out[model][0], 3),
                     "logZ_err": round(out[model][1], 3)})
        print(f"  {model:6s}: log Z = {out[model][0]:.3f} "
              f"+/- {out[model][1]:.3f}")
    dlz = out["dvch"][0] - out["lcdm"][0]
    err = (out["dvch"][1]**2 + out["lcdm"][1]**2) ** 0.5
    print(f"  delta log Z (DVCH - LCDM) = {dlz:+.3f} +/- {err:.3f}")
    print(f"  delta log Z (DVCH - w0waCDM) = "
          f"{out['dvch'][0]-out['w0wa'][0]:+.3f}")
    print("  (Jeffreys: |dlnZ|>5 evidencia muy fuerte; >2.5 fuerte; "
          ">1.3 sustancial)")

    df = pd.DataFrame(rows)
    df.to_csv("dvch_evidence_summary.csv", index=False)
    print("\nWrote dvch_evidence_summary.csv")


if __name__ == "__main__":
    main()
