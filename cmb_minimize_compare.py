#!/usr/bin/env python3
"""Compara los chi2_min de los minimizadores de Planck (nuisance oficial libre).

Lee los productos de cobaya minimize (ignore_prior=True -> .bestfit):
  - dvch_lcdm_min_free.bestfit   (LCDM, n=0, beta=0)
  - dvch_dvch_free_min.bestfit   (DVCH, n y beta libres)

El LCDM anidado (n=0, beta=0) a traves del CAMB parcheado es EXACTAMENTE
LCDM (la cerradura con n=beta=0 da U=1 y x ~ (1+z)^3 con precision de
bit), de modo que su chi2_min es valido para el CAMB parcheado o no.

Resultado (2026-11-06, CAMB con el parche de FONDO aplicado):
  chi2_min LCDM = 2766.81, chi2_min DVCH = 2780.85 (n=0.097, beta=0.101)
  -> dChi2 = +14.04, p ~ 5.4e-4: el CMB EXCLUYE la interaccion
     (n=0 estaria fuera del prior [0.01, 0.99]).
"""
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))


def read_bests(path):
    """Cobaya .bestfit.txt (getdist) -> (chi2, {param: value}).

    El chi2 es el de la likelihood (chi2__dvch_plik), no el del
    posterior: la cabecera '-log(Like)' del .bestfit es el posterior
    (incluye los priors Gaussianos del nuisance).
    """
    if not os.path.exists(path):
        return None
    names = None
    values = None
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            names = line.lstrip("#").split()
        elif names is not None:
            values = line.split()
            break
    if names is None or values is None or "chi2__dvch_plik" not in names:
        return None
    idx = names.index("chi2__dvch_plik")
    chi2 = float(values[idx])
    params = {}
    for i, nm in enumerate(names):
        if nm in ("weight", "minuslogpost", "minuslogprior",
                  "minuslogprior__0", "minuslogprior__SZ",
                  "chi2", "chi2__dvch_plik"):
            continue
        params[nm] = float(values[i])
    return chi2, params


def main():
    lcdm = read_bests(os.path.join(_HERE, "dvch_lcdm_min_free.bestfit.txt"))
    dvch = read_bests(os.path.join(_HERE, "dvch_dvch_free_min.bestfit.txt"))

    print("=== Comparacion CMB: minimizadores con nuisance oficial libre ===")
    for name, res in (("LCDM (n=0, beta=0)", lcdm), ("DVCH (n, beta libres)", dvch)):
        if res is None:
            print(f"  {name}: archivo .bestfit no encontrado (pendiente)")
        else:
            print(f"  {name}: chi2 = {res[0]:.2f}")

    if lcdm and dvch:
        dchi2 = dvch[0] - lcdm[0]
        n_best = dvch[1].get("DVCH_n")
        beta_best = dvch[1].get("DVCH_beta")
        print(f"\n  Delta chi2 (DVCH - LCDM) = {dchi2:+.2f} (2 params extra)")
        print(f"  Mejor ajuste DVCH: n = {n_best}, beta = {beta_best}")
        if dchi2 > 0:
            print("  -> el mejor punto PERMITIDO de DVCH es peor que LCDM:")
            print("     la interaccion del cierre es excluida por el CMB.")
            print("     (n=0, que recuperaria LCDM exactamente, esta fuera")
            print("      del prior muestreado n in [0.01, 0.99].)")
            from scipy.stats import chi2 as chi2dist
            p = float(chi2dist.sf(dchi2, 2))
            print(f"  LRT (2 dof): p = {p:.3g}")
        else:
            print("  (minimo del modelo anidado mejor o igual que LCDM)")
    print("\n  El LCDM anidado (n=0, beta=0) a traves del CAMB parcheado es")
    print("  exactamente LCDM, de modo que su chi2 es directamente comparable.")


if __name__ == "__main__":
    main()
