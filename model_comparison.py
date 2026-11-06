#!/usr/bin/env python3
# Comparacion LCDM vs DVCH en el CMB: chi2_min, delta chi2, AIC, BIC, LRT.
#
# AVISO DE INTEGRIDAD (2026-11-05): el chi2 de DVCH (2769.28) sale del
# minimo de la CADENA MCMC, mientras que el de LCDM (2768.32) sale del
# MINIMIZADOR. En un modelo anidado, el chi2 minimo verdadero de DVCH
# tiene que ser <= el de LCDM; un delta chi2 positivo solo significa que
# la cadena no encontro el minimo (o que el minimizador de DVCH no se ha
# corrido). Por eso el LRT y el p-valor NO son interpretables hasta que
# se corra el minimizador tambien para DVCH. Este script lo deja claro
# en vez de imprimir un p-valor sin sentido.
import numpy as np

chi2_lcdm = 2768.3182       # minimizador (launch_lcdm_minimize.sh)
chi2_dvch = 2769.28         # cadena MCMC (NO minimizador) -> pendiente

k_lcdm = 24   # 6 cosmologicos + 18 nuisance
k_dvch = 26   # 8 cosmologicos + 18 nuisance
d_k = k_dvch - k_lcdm  # = 2

print("=== Resultados del best-fit ===")
print(f"  LCDM (n=0, beta=0): chi2_min = {chi2_lcdm:.2f}  ({k_lcdm} params)")
print(f"  DVCH (n, beta libres): chi2 = {chi2_dvch:.2f}  ({k_dvch} params)")
print("  [chi2 de DVCH viene de la cadena, no de un minimizador:]")
print("  [no es comparable con el de LCDM hasta repetir el minimizador]")

print("\n=== Comparacion ===")
dchi2 = chi2_dvch - chi2_lcdm
print(f"  delta chi2 = chi2_DVCH - chi2_LCDM = {dchi2:.2f}")

if dchi2 > 0:
    print("\n  AVISO: DVCH es un modelo anidado que contiene a LCDM, de modo")
    print("  que en su minimo verdadero su chi2 no puede ser mayor. El valor")
    print("  positivo indica que la cadena no encontro el minimo. Hay que")
    print("  correr el minimizador tambien para DVCH antes de comparar.")
else:
    # LRT: para modelos anidados, -2*(lnL_DVCH - lnL_LCDM) = -dchi2 ~ chi2_{d_k}
    from scipy.stats import chi2 as chi2dist
    stat = -dchi2
    p = float(chi2dist.sf(stat, d_k))
    print(f"\n  Prueba de razon de verosimilitud (LRT): "
          f"estadistico = {stat:.2f} con {d_k} dof")
    print(f"  p-valor ~ {p:.3g}")

# AIC = chi2 + 2k ; BIC = chi2 + k*ln(n_datos)
n_data = 7527  # tamano del vector de datos plik TTTEEE
aic_lcdm = chi2_lcdm + 2 * k_lcdm
aic_dvch = chi2_dvch + 2 * k_dvch
bic_lcdm = chi2_lcdm + k_lcdm * np.log(n_data)
bic_dvch = chi2_dvch + k_dvch * np.log(n_data)
print(f"\n  AIC: LCDM = {aic_lcdm:.1f}, DVCH = {aic_dvch:.1f}, "
      f"delta = {aic_dvch - aic_lcdm:+.1f}")
print(f"  BIC: LCDM = {bic_lcdm:.1f}, DVCH = {bic_dvch:.1f}, "
      f"delta = {bic_dvch - bic_lcdm:+.1f}")

print("\n=== Interpretacion ===")
if dchi2 > 0:
    print(f"  DVCH NO mejora el ajuste segun los numeros disponibles")
    print(f"  (chi2 {dchi2:.2f} peor) con {d_k} parametros MAS, pero el")
    print(f"  numero no es fiable: falta el minimizador de DVCH.")
else:
    print(f"  DVCH mejora el ajuste en {-dchi2:.2f} con {d_k} parametros extra.")
    print(f"  delta AIC = {aic_dvch - aic_lcdm:+.1f} "
          f"(<0 => DVCH preferido; >+2 => LCDM preferido).")
