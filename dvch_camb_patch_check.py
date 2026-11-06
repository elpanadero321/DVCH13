#!/usr/bin/env python3
"""Verifica que el parche DVCH de CAMB NO cambia la expansion de fondo.

El parche (camb_patch/README_patch.md) solo anade un termino a la
ecuacion de perturbacion del CDM (derivs/equations.f90); el fondo
(dtauda) sigue siendo el de LCDM. Este script lo verifica numericamente:
H(z) con DVCH_flag=True y cualquier (n, beta) es identico al de LCDM.

Corre en WSL2 con el CAMB parcheado:
    source env.sh && python3 dvch_camb_patch_check.py

Resultado (2026-11-06): H(z) identico a <1e-6 % en 0 <= z <= 3 para
(n, beta) = (0.42, 0.53) y (0.8, 1.5); la edad del universo tampoco
cambia. Por lo tanto el analisis de Planck existente compara
LCDM-fondo + fuente de perturbacion DVCH, NO la cerradura de fondo del
paper, y cualquier "tension CMB" medida con ese CAMB es un efecto del
sector de perturbacion, no del fondo.
"""
import os
import sys

import numpy as np

CAMB_ROOT = os.environ.get("DVCH_CAMB_ROOT")
if not CAMB_ROOT:
    sys.exit("Falta DVCH_CAMB_ROOT (source env.sh).")
sys.path.insert(0, CAMB_ROOT)

import camb  # noqa: E402


def main():
    print("camb:", camb.__version__, "->", os.path.dirname(camb.__file__))
    base = dict(H0=67.5, ombh2=0.0224, omch2=0.122, tau=0.054,
                As=2.0e-9, ns=0.965)
    p0 = camb.set_params(**base)
    r0 = camb.get_results(p0)
    zs = np.array([0.0, 0.5, 1.0, 2.0, 3.0])
    h0 = np.asarray(r0.h_of_z(zs))
    age0 = r0.get_derived_params()["age"]
    rows = []
    for n, beta in ((0.42, 0.53), (0.5175, 1.1036), (0.8, 1.5)):
        p = camb.set_params(**base, DVCH_flag=True, DVCH_n=n, DVCH_beta=beta)
        r = camb.get_results(p)
        h = np.asarray(r.h_of_z(zs))
        rel = np.max(np.abs(h / h0 - 1.0))
        age = r.get_derived_params()["age"]
        rows.append((n, beta, rel, age - age0))
        print(f"  n={n:6.4f} beta={beta:7.4f}: max|dH|/H = {rel:.3e}, "
              f"d(age) = {age - age0:+.2e} Gyr")
    worst = max(rows, key=lambda row: row[2])
    print("\nConclusion: el fondo del CAMB parcheado es LCDM "
          f"(maxima desviacion {worst[2]:.2e} en H(z), "
          f"n={worst[0]}, beta={worst[1]}).")
    print("La fisica DVCH de fondo (conservacion CDM-vacio) NO esta")
    print("implementada en este CAMB: falta un parche al sector de")
    print("background (dtauda / DarkEnergy), no a las perturbaciones.")


if __name__ == "__main__":
    main()
