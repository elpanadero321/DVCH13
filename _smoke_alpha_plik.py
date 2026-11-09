"""Smoke test: una sola evaluacion de la likelihood alpha (CAMB estandar
ppf) en la cosmologia del mejor ajuste LCDM del CMB, con alpha=0 y
alpha=0.597. El alpha=0 debe ser identico al LCDM normal; alpha=0.597
debe dar el desvio de theta* (+0.78%) y un chi2اذا sustancialmente peor
en el CMB."""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(_HERE)
for line in open(os.path.join(_HERE, "env.sh")):
    if line.strip().startswith("export"):
        parts = line.strip()[7:].split("=", 1)
        if len(parts) == 2:
            os.environ[parts[0]] = parts[1].strip().strip('"')

sys.path.insert(0, os.environ["DVCH_CLIK_EGG"])
sys.path.insert(0, _HERE)

import clik
import clik.hpy as hpy

from alpha_cobaya_planck import alpha_planck_loglike

LIKELIHOOD = os.environ["DVCH_PLANCK_LIKELIHOOD"]
likelihood = clik.clik(LIKELIHOOD)
names = tuple(likelihood.extra_parameter_names)
check = np.asarray(hpy.File(LIKELIHOOD, "r")["clik/check_param"][:])
nuisance = {n: float(check[7527 + i]) for i, n in enumerate(names)}

COSMO = dict(H0=67.550217, ombh2=0.022426771, omch2=0.11952577,
             tau=0.052422959, ln10As=3.0396044, ns=0.9672816)

for alpha in (0.0, 0.597):
    lp = alpha_planck_loglike(alpha=alpha, **COSMO, **nuisance)
    print(f"alpha={alpha}: chi2 = {-2.0 * lp:.2f}")
