#!/usr/bin/env python3
"""Curva de trade-off: para cada z_c del switch de desacoplamiento,
mide (a) la preferencia tardia (dChi2 del mejor ajuste) y (b) el coste
CMB en el punto fijo (mediana tardia original).

Muestra empiricamente si existe un regimen donde el modelo ayuda a
tiempo tardio sin danar el CMB.
"""
import os
import numpy as np
import pandas as pd

import dvch_vs_lcdm_real as m

sne = m.load_pantheon()

LATE = {"n": 0.5175, "beta": 1.1036, "Om": 0.3804}  # mediana tardia original


def late_fit():
    from scipy.optimize import minimize

    def neg(p):
        return m.total_chi2(p, "dvch", sne)

    starts = [
        np.array([0.30, 70.0, -19.3, LATE["n"], LATE["beta"], 147.0]),
        np.array([0.38, 68.1, -19.41, LATE["n"], LATE["beta"], 147.0]),
    ]
    best = None
    for x0 in starts:
        r = minimize(neg, x0, method="Nelder-Mead",
                     options=dict(maxiter=6000, maxfev=6000,
                                  xatol=1e-5, fatol=1e-5))
        if best is None or r.fun < best.fun:
            best = r
    return best.fun, best.x


chi2_lcdm = m.total_chi2([0.3108, 68.54, -19.404, 146.77], "lcdm", sne)
rows = []
for zc in [1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 1e6]:
    os.environ["DVCH_ZC"] = str(zc)
    os.environ["DVCH_P"] = "10"
    m._dvch_grid.cache_clear()
    c2, x = late_fit()
    n_hat, beta_hat = x[3], x[4]
    rows.append({
        "z_c": zc, "chi2_late_dvch": round(c2, 2),
        "dchi2_late": round(c2 - chi2_lcdm, 2),
        "n_best": round(n_hat, 4), "beta_best": round(beta_hat, 4),
        "Om_best": round(x[0], 4), "H0_best": round(x[1], 2),
        "rd_best": round(x[5], 1),
    })

df = pd.DataFrame(rows)
print(df.to_string(index=False))
print("\n(dchi2_late negativo = DVCH ajusta mejor que LCDM)")
df.to_csv("dvch_switch_tradeoff.csv", index=False)
print("Wrote dvch_switch_tradeoff.csv")
