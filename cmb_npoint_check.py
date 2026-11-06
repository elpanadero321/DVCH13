#!/usr/bin/env python3
"""
El CMB excluye el (n, beta) que prefiere la expansion?

Estrategia (comparacion justa):
  1) evaluar chi2_CMB en el best-fit de LCDM (n=0, beta=0) ya encontrado;
  2) evaluar chi2_CMB en el (n, beta) del best-fit de la expansion (leido
     de dvch_late_time_fit_summary.csv, para que siempre este sincronizado
     con el analisis tardio) con la MISMA cosmologia -> chequeo rapido;
  3) reajustar la cosmologia + nuisance localmente con n, beta fijos ->
     chi2 minimo "justo" para DVCH en el CMB (launch_n47_minimize.sh).

Si la diferencia es pequena, el CMB no excluye el punto; si es grande
(mas de ~6 en chi2), hay tension real.

Necesita los datos .clik de Planck y el CAMB parcheado (env.sh); si no
estan disponibles, este script no puede correr (el analisis tardio si).
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(_HERE)


def load_late_time_bestfit():
    """Lee (n, beta) del best-fit tardio del CSV de resultados."""
    path = os.path.join(_HERE, "dvch_late_time_fit_summary.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Falta {path}. Corre dvch_vs_lcdm_real.py primero.")
    frame = pd.read_csv(path)
    out = {}
    for key in ("n", "beta"):
        col = f"{key}_median"
        series = frame[[col]].dropna()
        if series.empty:
            raise KeyError(f"Falta {col} en {path}")
        out[key] = float(series.iloc[0, 0])
    return out["n"], out["beta"]


# environment (env.sh no esta en el repo: rutas personales)
_env = os.path.join(_HERE, "env.sh")
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
print(f"nuisance cargados: {len(nuisance)}")

# cosmologia del best-fit LCDM encontrado por el minimizador
# (launch_lcdm_minimize.sh / dvch_lcdm_min.minimum)
COSMO_LCDM = dict(H0=67.550217, ombh2=0.022426771, omch2=0.11952577,
                  tau=0.052422959, ln10As=3.0396044, ns=0.9672816)


def chi2_cmb(cosmo, n, beta):
    kwargs = dict(cosmo)
    kwargs.update(nuisance)
    lp = planck_loglike(DVCH_n=n, DVCH_beta=beta, **kwargs)
    return -2.0 * lp


n_fix, beta_fix = load_late_time_bestfit()
print(f"\nBest-fit tardio: n = {n_fix:.4f}, beta = {beta_fix:.4f}")

print("\n=== Chequeo rapido (misma cosmologia, solo cambio n y beta) ===")
c0 = chi2_cmb(COSMO_LCDM, 0.0, 0.0)
print(f"  LCDM (n=0,     beta=0)    : chi2 = {c0:.2f}")
c1 = chi2_cmb(COSMO_LCDM, n_fix, beta_fix)
print(f"  DVCH (n={n_fix:.3f}, beta={beta_fix:.3f}): chi2 = {c1:.2f}")
print(f"  delta chi2 = {c1 - c0:+.2f}")
print("\nNota: este es un chequeo de punto fijo. La tension se debe medir")
print("contra el minimo de DVCH reajustando cosmologia+nuisance, no contra")
print("el de LCDM (ver launch_n47_minimize.sh).")
