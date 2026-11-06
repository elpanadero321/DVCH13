#!/usr/bin/env python3
"""Compara los chi2_min de los minimizadores de Planck (nuisance oficial libre).

Lee los productos de cobaya minimize (ignore_prior=True -> .bestfit):
  - dvch_lcdm_min_free.bestfit   (LCDM, n=0, beta=0)
  - dvch_dvch_free_min.bestfit   (DVCH, n y beta libres)

En un modelo anidado, chi2_min(DVCH) <= chi2_min(LCDM) tiene que
cumplirse; el valor +0.96 que se citaba antes venia del minimo de la
CADENA, no de un minimizador. Esta comparacion es la correcta.

AVISO: el CAMB parcheado deja el fondo identico a LCDM
(dvch_camb_patch_check.py), de modo que estos chi2 miden el ajuste del
TERMINO DE PERTURBACION del CDM sobre un fondo LCDM, no la cerradura de
fondo del paper.
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
        if n_best is not None and n_best <= 0.0101:
            print("  -> n esta en el borde inferior del prior [0.01, 0.99]:")
            print("     el CMB no quiere interaccion (quieres n -> 0).")
        if dchi2 > 0:
            print("  AVISO: en un modelo anidado el minimo de DVCH no puede")
            print("  ser mayor; esto indica un fallo del minimizador.")
        else:
            from scipy.stats import chi2 as chi2dist
            p = float(chi2dist.sf(-dchi2, 2))
            print(f"  LRT (2 dof): estadistico = {-dchi2:.2f}, p = {p:.3g}")
    print("\n  Recuerda: estos chi2 miden el termino de perturbacion del CDM")
    print("  (el fondo del CAMB parcheado es LCDM), no la cerradura DVCH.")


if __name__ == "__main__":
    main()
