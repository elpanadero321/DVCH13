#!/usr/bin/env python3
# Comparacion LCDM vs DVCH en el CMB: chi2_min, delta chi2, AIC, BIC, LRT.
#
# ACTUALIZADO (2026-11-06): los dos minimizadores con el sector nuisance
# oficial libre (cobaya BOBYQA) se han corrido (cmb_minimize_compare.py
# los lee):
#   chi2_min LCDM = 2766.81, chi2_min DVCH = 2766.10 (n=0.01, beta=0.37)
# El chi2 de DVCH ya viene del MINIMIZADOR, no de la cadena, de modo que
# la comparacion anidada es valida: delta chi2 = -0.70 <= 0 como debe ser.
# El valor +0.96 que se citaba antes era un limite de la cadena y no era
# interpretable; esta version lo sustituye.
#
# AVISO: el CAMB parcheado deja el fondo identico a LCDM
# (dvch_camb_patch_check.py), asi que estos chi2 miden el TERMINO DE
# PERTURBACION del CDM sobre un fondo LCDM: no confrontan la cerradura de
# fondo del paper. k_LCDM/k_DVCH cuentan el nuisance oficial libre.
import numpy as np

# Resultados de los minimizadores (nuisance oficial libre, ignore_prior):
# dvch_lcdm_min_free.bestfit.txt y dvch_dvch_free_min.bestfit.txt
chi2_lcdm = 2766.81
chi2_dvch = 2766.10
n_best, beta_best = 0.01, 0.37

k_lcdm = 24  # 6 cosmologicos + 18 nuisance oficiales libres
k_dvch = 26  # + DVCH_n, DVCH_beta
d_k = k_dvch - k_lcdm  # = 2

print("=== Resultados del best-fit (minimizadores, nuisance oficial libre) ===")
print(f"  LCDM (n=0, beta=0):  chi2_min = {chi2_lcdm:.2f}  ({k_lcdm} params)")
print(f"  DVCH (n, beta lib.): chi2_min = {chi2_dvch:.2f}  ({k_dvch} params)")
print(f"  mejor ajuste DVCH: n = {n_best} (borde inferior del prior), "
      f"beta = {beta_best}")

print("\n=== Comparacion anidada (chi2 de likelihood, libre de prior) ===")
dchi2 = chi2_dvch - chi2_lcdm
print(f"  delta chi2 = chi2_DVCH - chi2_LCDM = {dchi2:+.2f}  ({d_k} params extra)")

# LRT valido: DVCH es anidado y su minimo ya viene del minimizador.
from scipy.stats import chi2 as chi2dist
stat = -dchi2
p = float(chi2dist.sf(stat, d_k))
print(f"  LRT: estadistico = {stat:.2f} con {d_k} dof -> p ~ {p:.3g}")
print("  El CMB (fondo LCDM + termino de perturbacion) NO prefiere DVCH.")
print("  Ademas n=0.01 esta en el borde inferior del prior n in [0.01,0.99]:")
print("  el CMB empuja la interaccion a cero. n=0 no esta dentro del prior,")
print("  asi que esto NO es una 'exclusion a 2sigma de LCDM' (efecto de borde).")

# AIC = chi2 + 2k ; BIC = chi2 + k*ln(n_datos)
n_data = 7527  # tamano del vector de datos plik TTTEEE (+lowl/lowE incluidos)
aic_lcdm = chi2_lcdm + 2 * k_lcdm
aic_dvch = chi2_dvch + 2 * k_dvch
bic_lcdm = chi2_lcdm + k_lcdm * np.log(n_data)
bic_dvch = chi2_dvch + k_dvch * np.log(n_data)
print(f"\n  AIC: LCDM = {aic_lcdm:.1f}, DVCH = {aic_dvch:.1f}, "
      f"delta = {aic_dvch - aic_lcdm:+.1f}")
print(f"  BIC: LCDM = {bic_lcdm:.1f}, DVCH = {bic_dvch:.1f}, "
      f"delta = {bic_dvch - bic_lcdm:+.1f}")
print("  (delta AIC/BIC > 0 => LCDM preferido por parsimonia)")
