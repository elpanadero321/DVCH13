#!/usr/bin/env python3
# Comparacion LCDM vs DVCH en el CMB: chi2_min, delta chi2, AIC, BIC, LRT.
#
# ACTUALIZADO (2026-11-06, CAMB con el parche de FONDO aplicado):
# los dos minimizadores con el sector nuisance oficial libre (cobaya
# BOBYQA) sobre el CAMB parcheado de fondo:
#   chi2_min LCDM = 2766.81  (n=0, beta=0: LCDM puro a traves del CAMB
#   parcheado, identico al estandar)
#   chi2_min DVCH = 2780.85  (n=0.097, beta=0.101)
#   dChi2 = +14.04 (2 dof), p ~ 9e-4: el CMB EXCLUYE la interaccion del
#   cierre DVCH; n=0 (LCDM exacto) esta fuera del prior muestreado
#   [0.01, 0.99], de modo que el mejor punto PERMITIDO es peor que LCDM.
# El +0.96 que se citaba antes era un limite de la cadena (no un
# minimizador) tomado con un CAMB ademas sin parchear el fondo.
import numpy as np
from scipy.stats import chi2 as chi2dist

# Resultados de los minimizadores (nuisance oficial libre, ignore_prior):
# dvch_lcdm_min_free.bestfit.txt y dvch_dvch_free_min.bestfit.txt
chi2_lcdm = 2766.81
chi2_dvch = 2780.85
n_best, beta_best = 0.097, 0.101

k_lcdm = 24  # 6 cosmologicos + 18 nuisance oficiales libres
k_dvch = 26  # + DVCH_n, DVCH_beta
d_k = k_dvch - k_lcdm  # = 2
n_data = 7527  # tamano del vector de datos plik TTTEEE (+lowl/lowE)

print("=== Resultados del best-fit (minimizadores, nuisance oficial libre) ===")
print(f"  LCDM (n=0, beta=0):  chi2_min = {chi2_lcdm:.2f}  ({k_lcdm} params)")
print(f"  DVCH (n, beta lib.): chi2_min = {chi2_dvch:.2f}  ({k_dvch} params)")
print(f"  mejor ajuste DVCH: n = {n_best}, beta = {beta_best}")

dchi2 = chi2_dvch - chi2_lcdm
print("\n=== Comparacion anidada (chi2 de likelihood, libre de prior) ===")
print(f"  delta chi2 = chi2_DVCH - chi2_LCDM = {dchi2:+.2f}  ({d_k} params extra)")
p = float(chi2dist.sf(dchi2, d_k))
print(f"  LRT: estadistico = {dchi2:.2f} con {d_k} dof -> p ~ {p:.3g}")
print("  El CMB EXCLUYE la interaccion del cierre DVCH: el mejor punto")
print(f"  permitido esta {dchi2:.1f} peor que LCDM (~{float(chi2dist.isf(p, 1))**0.5:.1f}sigma equivalente).")
print("  Nota: n=0 (LCDM exacto) esta fuera del prior muestreado")
print("  n in [0.01, 0.99], de modo que DVCH no puede alcanzar la calidad")
print("  de LCDM aunque su limite n->0 lo recupere exactamente.")
print("  Perfil de punto fijo (cmb_npoint_profile.py): a la cosmologia del")
print("  mejor ajuste LCDM, el coste crece muy bruscamente con n y beta:")
print("    n=0.01 -> +185, n=0.02 -> +761, n=0.05 -> +5.2e3, n=0.2 -> +1.3e5")
print("    (beta=0); con beta=0.53 el coste base ya es > +4e4. En la")
print("    mediana tardia (n=0.518, beta=1.104): dChi2 ~ +5.9e4.")

aic_lcdm = chi2_lcdm + 2 * k_lcdm
aic_dvch = chi2_dvch + 2 * k_dvch
bic_lcdm = chi2_lcdm + k_lcdm * np.log(n_data)
bic_dvch = chi2_dvch + k_dvch * np.log(n_data)
print(f"\n  AIC: LCDM = {aic_lcdm:.1f}, DVCH = {aic_dvch:.1f}, "
      f"delta = {aic_dvch - aic_lcdm:+.1f}")
print(f"  BIC: LCDM = {bic_lcdm:.1f}, DVCH = {bic_dvch:.1f}, "
      f"delta = {bic_dvch - bic_lcdm:+.1f}")
print("  (delta AIC/BIC > 0 => LCDM preferido)")
