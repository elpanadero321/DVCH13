#!/usr/bin/env python3
"""
El CMB excluye n=0.47, beta=1.04 (lo que prefiere la expansion)?

Estrategia (comparacion justa):
  1) evaluar chi2_CMB en el best-fit de LCDM (n=0, beta=0) ya encontrado;
  2) evaluar chi2_CMB en (n=0.47, beta=1.04) con la MISMA cosmologia
     -> chequeo rapido (2 minutos);
  3) reajustar la cosmologia + nuisance localmente con n, beta fijos en
     0.47, 1.04 -> chi2 minimo "justo" para DVCH en el CMB.

Si la diferencia es pequena, el CMB no excluye el punto; si es grande
(mas de ~6 en chi2), hay tension real.
"""
import os
import sys
import numpy as np

sys.path.insert(0, "/mnt/d/DVCH13-1")
os.chdir("/mnt/d/DVCH13-1")

# environment
from pathlib import Path
for line in open("env.sh"):
    if line.strip().startswith("export"):
        parts = line.strip()[7:].split("=", 1)
        if len(parts) == 2:
            os.environ[parts[0]] = parts[1].strip().strip('"')

sys.path.insert(0, os.environ["DVCH_CLIK_EGG"])
import clik
import clik.hpy as hpy

from dvch_cobaya_planck import planck_loglike

LIKELIHOOD = os.environ["DVCH_PLANCK_LIKELIHOOD"]
likelihood = clik.clik(LIKELIHOOD)
names = tuple(likelihood.extra_parameter_names)
check = np.asarray(hpy.File(LIKELIHOOD, "r")["clik/check_param"][:])
nuisance = {n: float(check[7527 + i]) for i, n in enumerate(names)}
print(f"nuisance cargados: {len(nuisance)}")

# cosmologia del best-fit LCDM encontrado por el minimizador
COSMO_LCDM = dict(H0=67.550217, ombh2=0.022426771, omch2=0.11952577,
                  tau=0.052422959, ln10As=3.0396044, ns=0.9672816)


def chi2_cmb(cosmo, n, beta):
    kwargs = dict(cosmo)
    kwargs.update(nuisance)
    lp = planck_loglike(DVCH_n=n, DVCH_beta=beta, **kwargs)
    return -2.0 * lp


print("\n=== Chequeo rapido (misma cosmologia, solo cambio n y beta) ===")
c0 = chi2_cmb(COSMO_LCDM, 0.0, 0.0)
print(f"  LCDM (n=0,    beta=0)    : chi2 = {c0:.2f}")
c1 = chi2_cmb(COSMO_LCDM, 0.47, 1.04)
print(f"  DVCH (n=0.47, beta=1.04): chi2 = {c1:.2f}")
print(f"  delta chi2 = {c1 - c0:+.2f}")
