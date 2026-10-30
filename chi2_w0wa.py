#!/usr/bin/env python3
# Calcula chi2_min de w0waCDM sobre los mismos datos reales.
# Estrategia: optimizacion local (Nelder-Mead en 5 dims, espacio de los
# parametros fisicos) partiendo del mejor punto que ya conoce el modelo.
# Es la misma medida "independiente del prior" que tenemos para DVCH/LCDM.
import numpy as np
from scipy.optimize import minimize
import w0wa_control as w
from w0wa_control import chi2_all_w0wa, BOUNDS

sne = w.sne


def neg(p):
    return chi2_all_w0wa(p[0], p[1], p[2], p[3], p[4],
                         subsets=("cc", "bao", "sne"))


def main():
    print("optimizando chi2_min de w0waCDM (CC+BAO+SNe reales)...")
    # punto de partida: el mejor de la corrida de nested (medias posteriores)
    x0 = np.array([0.2501, 68.5139, -19.3732, -0.6815, -0.6412])
    r = minimize(neg, x0, method="Nelder-Mead",
                 options=dict(maxiter=4000, xatol=1e-4, fatol=1e-4))
    chi2 = r.fun
    print(f"\nchi2_min(w0waCDM) = {chi2:.2f}")
    print(f"  Om={r.x[0]:.4f} H0={r.x[1]:.3f} M={r.x[2]:.4f} "
          f"w0={r.x[3]:.4f} wa={r.x[4]:.4f}")
    # Comparacion prior-independiente
    chi2_lcdm = 1832.45
    chi2_dvch = 1807.67
    print("\n=== Comparacion de calidad de ajuste (chi2_min, sin prior) ===")
    print(f"  LCDM    : {chi2_lcdm:.2f}")
    print(f"  w0waCDM : {chi2:.2f}  (dChi2 vs LCDM = {chi2-chi2_lcdm:+.2f})")
    print(f"  DVCH    : {chi2_dvch:.2f}  (dChi2 vs LCDM = {chi2_dvch-chi2_lcdm:+.2f})")
    print(f"\n  DVCH - w0waCDM = {chi2_dvch - chi2:+.2f}  (<0 => DVCH ajusta mejor)")


if __name__ == "__main__":
    main()
