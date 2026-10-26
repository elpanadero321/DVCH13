#!/usr/bin/env python3
"""
Reajuste local justo: con n=0.47, beta=1.04 FIJOS, optimizar la cosmologia
(6 params) + nuisance (18) para encontrar el minimo chi2 del CMB.

Si el chi2 justo de DVCH queda cerca del LCDM (delta <~ 6), el CMB no
excluye el punto: simplemente prefiere cosmologia distinta y hay que
ver el trade-off. Si delta es grande, hay tension real.
"""
import os
import sys
import numpy as np

sys.path.insert(0, "/mnt/d/DVCH13-1")
os.chdir("/mnt/d/DVCH13-1")
for line in open("env.sh"):
    if line.strip().startswith("export"):
        parts = line.strip()[7:].split("=", 1)
        if len(parts) == 2:
            os.environ[parts[0]] = parts[1].strip().strip('"')

# LD_LIBRARY_PATH ya queda seteado por env.sh antes de importar clik
sys.path.insert(0, os.environ["DVCH_CLIK_EGG"])
import clik
import clik.hpy as hpy
likelihood = clik.clik(os.environ["DVCH_PLANCK_LIKELIHOOD"])
names = tuple(likelihood.extra_parameter_names)
check = np.asarray(hpy.File(os.environ["DVCH_PLANCK_LIKELIHOOD"], "r")["clik/check_param"][:])
nuisance = {n: float(check[7527 + i]) for i, n in enumerate(names)}

from dvch_cobaya_planck import planck_loglike

# limites para la optimizacion (espacio reducido alrededor del best-fit)
COSMO_LO = np.array([60.0, 0.0215, 0.100, 0.03, 2.95, 0.92])
COSMO_HI = np.array([75.0, 0.0235, 0.140, 0.09, 3.15, 1.02])
NUI_KEYS = list(nuisance.keys())
# banda relativa +/-15% o absoluta para valores cercanos a cero
NUI_LO = np.array([
    max(nuisance[k] - max(abs(nuisance[k]) * 0.5, 1e-4), 0.0) if nuisance[k] > 0
    else min(nuisance[k] - max(abs(nuisance[k]) * 0.5, 1e-4), 0.0)
    for k in NUI_KEYS])
NUI_HI = np.array([
    nuisance[k] + max(abs(nuisance[k]) * 0.5, 1e-4)
    for k in NUI_KEYS])
NUI_LO = np.minimum(NUI_LO, NUI_HI - 1e-6)   # sanity: lo < hi siempre


def make_chi2(n, beta):
    def chi2(x):
        # x = [cosmo(6), nuisance(18)]
        try:
            cosmo = dict(zip(["H0", "ombh2", "omch2", "tau", "ln10As", "ns"], x[:6]))
            nu = {k: v for k, v in zip(NUI_KEYS, x[6:])}
            kwargs = dict(cosmo)
            kwargs.update(nu)
            lp = planck_loglike(DVCH_n=n, DVCH_beta=beta, **kwargs)
        except Exception:
            return 1e10
        if not np.isfinite(lp):
            return 1e10
        return -2.0 * lp
    return chi2


def run(n, beta, label):
    from scipy.optimize import minimize
    x0 = np.concatenate([np.array([67.55, 0.0224268, 0.1195258, 0.0524230, 3.0396044, 0.9672816]),
                         np.array([nuisance[k] for k in NUI_KEYS])])
    chi2 = make_chi2(n, beta)
    print(f"\n=== Optimizando {label} (n={n}, beta={beta}) ===", flush=True)
    # L-BFGS-B respeta los limites fisicos (evita recollapse y valores no validos)
    bounds = list(zip(np.concatenate([COSMO_LO, NUI_LO]),
                      np.concatenate([COSMO_HI, NUI_HI])))
    res = minimize(chi2, x0, method="L-BFGS-B", bounds=bounds,
                   options=dict(maxiter=300, ftol=1e-6, gtol=1e-5))
    print(f"  chi2 min = {res.fun:.2f}  (convergio: {res.success})", flush=True)
    print(f"  H0={res.x[0]:.3f} ombh2={res.x[1]:.5f} omch2={res.x[2]:.5f} "
          f"tau={res.x[3]:.4f} ln10As={res.x[4]:.4f} ns={res.x[5]:.4f}", flush=True)
    return res.fun


if __name__ == "__main__":
    c_lcdm = run(0.0, 0.0, "LCDM")
    c_dvch = run(0.47, 1.04, "DVCH (preferido por la expansion)")
    print(f"\n=== VEREDICTO TENSION ===")
    print(f"  chi2 LCDM (n=0)            = {c_lcdm:.2f}")
    print(f"  chi2 DVCH (n=0.47, b=1.04) = {c_dvch:.2f}")
    print(f"  delta chi2 = {c_dvch - c_lcdm:+.2f}")
    print("  (>~6: tension real; <~6: compatible)")
