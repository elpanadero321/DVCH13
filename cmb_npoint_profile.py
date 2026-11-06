#!/usr/bin/env python3
"""Perfil de chi2 del CMB en el plano (n, beta) a cosmologia+nuisance fijos.

Evaluacion en el mejor ajuste de LCDM (dvch_lcdm_min.minimum): perfiles
del coste en chi2 del termino de perturbacion del CDM. Sirve para
cuantificar cuanto le cuesta al CMB la interaccion a los valores que
prefiere la expansion tardia (punto fijo) y a otros valores.

Corre en WSL2:  source env.sh && python3 cmb_npoint_profile.py

AVISO: el CAMB parcheado deja el fondo identico a LCDM, de modo que todo
el coste viene del termino de perturbacion (no de la cerradura de fondo).
"""
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(_ROOT)

_env = os.path.join(_ROOT, "env.sh")
if not os.path.exists(_env):
    sys.exit("Falta env.sh (copia env.sh.example y rellena las rutas).")
for line in open(_env):
    if line.strip().startswith("export"):
        parts = line.strip()[7:].split("=", 1)
        if len(parts) == 2:
            os.environ[parts[0]] = parts[1].strip().strip('"')

sys.path.insert(0, os.environ["DVCH_CLIK_EGG"])
import clik  # noqa: E402
import clik.hpy as hpy  # noqa: E402

from dvch_cobaya_planck import planck_loglike  # noqa: E402

LIKELIHOOD = os.environ["DVCH_PLANCK_LIKELIHOOD"]
likelihood = clik.clik(LIKELIHOOD)
names = tuple(likelihood.extra_parameter_names)
check = np.asarray(hpy.File(LIKELIHOOD, "r")["clik/check_param"][:])
nuisance = {n: float(check[7527 + i]) for i, n in enumerate(names)}

COSMO_LCDM = dict(H0=67.550217, ombh2=0.022426771, omch2=0.11952577,
                  tau=0.052422959, ln10As=3.0396044, ns=0.9672816)

GRID_N = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]
GRID_BETA = [0.0, 0.53, 1.1]


def chi2_cmb(n, beta):
    kwargs = dict(COSMO_LCDM)
    kwargs.update(nuisance)
    try:
        return -2.0 * planck_loglike(DVCH_n=n, DVCH_beta=beta, **kwargs)
    except Exception as exc:
        # En puntos extremos del cierre la integracion de fuentes de CAMB
        # puede no converger: el CMB no es interpretable ahi.
        print(f"  (sin convergencia a n={n}, beta={beta}: {type(exc).__name__})")
        return None


def main():
    print("Perfil chi2(n, beta) a cosmologia+nuisance del mejor ajuste LCDM")
    rows = []
    for beta in GRID_BETA:
        for n in GRID_N:
            c2 = chi2_cmb(n, beta)
            rows.append((n, beta, c2))
            print(f"  n={n:5.3f} beta={beta:4.2f}: chi2 = "
                  f"{c2 if c2 is None else round(c2, 2)}")
    import csv
    with open("dvch_cmb_npoint_profile.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["n", "beta", "chi2_fixed_cosmology"])
        for r in rows:
            w.writerow([f"{r[0]:.4f}", f"{r[1]:.4f}",
                        "" if r[2] is None else f"{r[2]:.3f}"])
    base = chi2_cmb(0.0, 0.0)
    print(f"\nchi2(LCDM, n=0 beta=0) = {base:.2f}")
    for n, beta, c2 in rows:
        if c2 is not None and c2 > base:
            print(f"  n={n} beta={beta}: delta chi2 = {c2-base:+.1f}")
    print("Wrote dvch_cmb_npoint_profile.csv")


if __name__ == "__main__":
    main()
