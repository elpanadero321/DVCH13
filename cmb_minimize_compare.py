#!/usr/bin/env python3
"""Compara los chi2_min de los minimizadores de Planck (nuisance oficial libre).

Lee los archivos .minimum producidos por cobaya:
  - dvch_lcdm_min_free.minimum   (LCDM, n=0, beta=0)
  - dvch_dvch_free_min.minimum   (DVCH, n y beta libres)

En un modelo anidado, chi2_min(DVCH) <= chi2_min(LCDM) tiene que
cumplirse; el valor positivo que se citaba antes (+0.96) venia del
minimo de la cadena, no de un minimizador. Este script reporta la
comparacion correcta y, si falta algun archivo, lo dice claramente.

AVISO: el CAMB parcheado deja el fondo identico a LCDM
(dvch_camb_patch_check.py), de modo que estos chi2 miden el ajuste del
TERMINO DE PERTURBACION del CDM sobre un fondo LCDM, no la cerradura de
fondo del paper.
"""
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))


def read_minimum(path):
    """Extrae -log(Like) y chi-sq de un archivo .minimum de cobaya."""
    if not os.path.exists(path):
        return None
    text = open(path).read()
    logl = re.search(r"-log\(Like\)\s*=\s*([-\d.eE]+)", text)
    chi2 = re.search(r"chi-sq\s*=\s*([-\d.eE]+)", text)
    if not logl:
        return None
    return float(logl.group(1)), float(chi2.group(1)) if chi2 else None


def main():
    lcdm = read_minimum(os.path.join(_HERE, "dvch_lcdm_min_free.minimum"))
    dvch = read_minimum(os.path.join(_HERE, "dvch_dvch_free_min.minimum"))

    print("=== Comparacion CMB: minimizadores con nuisance oficial libre ===")
    for name, res in (("LCDM (n=0, beta=0)", lcdm), ("DVCH (n, beta libres)", dvch)):
        if res is None:
            print(f"  {name}: archivo .minimum no encontrado (pendiente)")
        else:
            print(f"  {name}: -log L = {res[0]:.2f}, chi2 = {2*res[0]:.2f}")

    if lcdm and dvch:
        dchi2 = 2 * dvch[0] - 2 * lcdm[0]
        print(f"\n  Delta chi2 (DVCH - LCDM) = {dchi2:+.2f}")
        if dchi2 > 0:
            print("  AVISO: en un modelo anidado el minimo de DVCH no puede")
            print("  ser mayor; esto indica un fallo del minimizador.")
        else:
            from scipy.stats import chi2 as chi2dist
            p = float(chi2dist.sf(-dchi2, 2))
            print(f"  LRT (2 dof, perturbacion sobre fondo LCDM): p = {p:.3g}")
    print("\n  Recuerda: estos chi2 miden el termino de perturbacion del CDM")
    print("  (el fondo del CAMB parcheado es LCDM), no la cerradura DVCH.")


if __name__ == "__main__":
    main()
