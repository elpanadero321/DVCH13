#!/usr/bin/env python3
# Comparacion LCDM vs DVCH: chi2_min, delta chi2, AIC, BIC, LRT.
import numpy as np

chi2_lcdm = 2768.3182
chi2_dvch = 2769.28   # min chi2 de la cadena DVCH (117k muestras)

# Parmetros ajustados (libres) en cada caso
k_lcdm = 24   # 6 cosmologicos + 18 nuisance
k_dvch = 26   # 8 cosmologicos + 18 nuisance
d_k = k_dvch - k_lcdm  # = 2

print("=== Resultados del best-fit ===")
print(f"  LCDM (n=0, beta=0): chi2_min = {chi2_lcdm:.2f}  ({k_lcdm} params)")
print(f"  DVCH (n, beta libres): chi2_min = {chi2_dvch:.2f}  ({k_dvch} params)")
print()
print("=== Comparacion ===")
dchi2 = chi2_dvch - chi2_lcdm
print(f"  delta chi2 = chi2_DVCH - chi2_LCDM = {dchi2:.2f}")
print()
# LRT: para modelos anidados, 2*Delta chi2 ~ chi2_{d_k} bajo H0 (LCDM)
from math import erf, sqrt
def chi2_sf(x, k):
    # survival function P(chi2_k > x) via series for even k
    if k % 2 == 0:
        m = k // 2
        s = 0.0
        term = np.exp(-x/2)
        for j in range(1, m):
            term = term * (x/2) / j
            s += term
        return s + np.exp(-x/2)
    else:
        # odd k: use regularized incomplete gamma via scipy-free approximation
        from math import gamma, exp
        # P(chi2_k > x) = Q(k/2, x/2) -- use scipy if available
        try:
            from scipy.stats import chi2 as chi2dist
            return float(chi2dist.sf(x, k))
        except Exception:
            return float('nan')
p = chi2_sf(dchi2, d_k)
print(f"  Prueba de razon de verosimilitud (LRT): 2*dchi2 = {2*dchi2:.2f} con {d_k} dof")
print(f"  p-valor ~ {p:.3f}")
print()
# AIC = chi2 + 2k ; BIC = chi2 + k*ln(n_datos)
n_data = 7527  # tamano del vector de datos plik TTTEEE
aic_lcdm = chi2_lcdm + 2*k_lcdm
aic_dvch = chi2_dvch + 2*k_dvch
bic_lcdm = chi2_lcdm + k_lcdm*np.log(n_data)
bic_dvch = chi2_dvch + k_dvch*np.log(n_data)
print(f"  AIC: LCDM = {aic_lcdm:.1f}, DVCH = {aic_dvch:.1f}, delta = {aic_dvch - aic_lcdm:+.1f}")
print(f"  BIC: LCDM = {bic_lcdm:.1f}, DVCH = {bic_dvch:.1f}, delta = {bic_dvch - bic_lcdm:+.1f}")
print()
print("=== Interpretacion ===")
if dchi2 > 0:
    print(f"  DVCH NO mejora el ajuste: su chi2 es {dchi2:.2f} PEOR que el de LCDM")
    print(f"  con {d_k} parametros MAS. Los datos no favorecen DVCH.")
else:
    print(f"  DVCH mejora el ajuste en {-dchi2:.2f} con {d_k} parametros extra.")
