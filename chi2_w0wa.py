#!/usr/bin/env python3
# Calcula chi2_min de w0waCDM sobre los mismos datos reales.
# Estrategia: optimizacion local (Nelder-Mead en 6 dims, espacio de los
# parametros fisicos) partiendo de varios puntos. Es la misma medida
# "independiente del prior" que tenemos para DVCH/LCDM.
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import w0wa_control as w
from w0wa_control import chi2_all_w0wa, BOUNDS


def neg(p):
    return chi2_all_w0wa(p[0], p[1], p[2], p[3], p[4], p[5],
                         subsets=("cc", "bao", "sne"))


def main():
    print("optimizando chi2_min de w0waCDM (CC+BAO+SNe reales)...")
    starts = [
        np.array([0.30, 70.0, -19.3, -1.0, 0.0, 147.0]),
        np.array([0.25, 68.5, -19.4, -0.7, -0.6, 147.0]),
        np.array([0.32, 67.0, -19.2, -0.9, 0.3, 140.0]),
        np.array([0.28, 72.0, -19.5, -1.2, -0.4, 155.0]),
    ]
    best = None
    for x0 in starts:
        r = minimize(neg, x0, method="Nelder-Mead",
                     options=dict(maxiter=8000, maxfev=8000,
                                  xatol=1e-5, fatol=1e-5))
        if best is None or r.fun < best.fun:
            best = r
    chi2 = best.fun
    print(f"\nchi2_min(w0waCDM) = {chi2:.2f}")
    print(f"  Om={best.x[0]:.4f} H0={best.x[1]:.3f} M={best.x[2]:.4f} "
          f"w0={best.x[3]:.4f} wa={best.x[4]:.4f} rd={best.x[5]:.2f}")

    # Comparacion prior-independiente (lee los chi2_min oficiales del CSV)
    try:
        df = pd.read_csv("dvch_late_time_fit_summary.csv")
        chi2_lcdm = float(df.loc[df.model == "lcdm", "chi2_min"].iloc[0])
        chi2_dvch = float(df.loc[df.model == "dvch", "chi2_min"].iloc[0])
    except Exception as exc:
        print(f"  (no se pudo leer dvch_late_time_fit_summary.csv: {exc})")
        return
    print("\n=== Comparacion de calidad de ajuste (chi2_min, sin prior) ===")
    print(f"  LCDM    : {chi2_lcdm:.2f}")
    print(f"  w0waCDM : {chi2:.2f}  (dChi2 vs LCDM = {chi2-chi2_lcdm:+.2f})")
    print(f"  DVCH    : {chi2_dvch:.2f}  (dChi2 vs LCDM = {chi2_dvch-chi2_lcdm:+.2f})")
    print(f"\n  DVCH - w0waCDM = {chi2_dvch - chi2:+.2f}  (<0 => DVCH ajusta mejor)")


if __name__ == "__main__":
    main()
