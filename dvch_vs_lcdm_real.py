#!/usr/bin/env python3
"""
Comparacion UV-HDE + Planck Limit vs LCDM con datos REALES de expansion:
  - Pantheon+ SH0ES: 1701 SNe en el archivo, de las que se usan solo las
    z > 0.01 excluyendo las 77 calibradoras Cefeidas (analisis estandar
    con M libre), covarianza completa STAT+SYS.
  - DESI DR1 BAO: Tabla 1 oficial de arXiv:2404.03002 (DESI Collaboration
    2024, JCAP 02 (2025) 021): D_M/r_d y D_H/r_d correlacionados en 5
    trazadores, mas D_V/r_d (isotropico) en BGS y QSO. En total 12
    observables BAO.
  - Cosmic chronometers: compilacion de 31 puntos (Simon+2005, Stern+2010,
    Moresco+2012, Zhang+2014, Moresco+2016). Se excluyen z=0.44 (WiggleZ)
    y z=2.34 (BOSS Lyman-alpha) por no ser cronometros (cuentan BAO dos
    veces).

Fisica UV-HDE: La densidad de materia sigue Om*(1+z)^3 (conservacion
estandar, Q = 0). La energia oscura tiene una evolucion no trivial
dada por la solucion algebraica exacta del modelo UV-HDE.
El modelo tiene CERO parametros libres extra frente a LCDM (Dk = 0):
solo (H0, Omega_m0, Omega_r0) con Omega_DE0 = 1 - Omega_m0 - Omega_r0.
n=1/3 y beta=1 estan fijados internamente.

r_d es un parametro libre (prior plano [130, 180] Mpc) que se marginaliza.

Metodo: MCMC con emcee para ambos modelos + minimizacion (Nelder-Mead)
para el chi2_min libre de prior.

Politica del repo: solo se reportan numeros provenientes de datos reales.
"""
import math
import os
from functools import lru_cache

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Datos reales
# ------------------------------------------------------------------
Omega_r0 = 9.18e-5
c_speed = 299792.458
REALDATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "realdata")

# --- Cosmic chronometers (31 puntos, compilacion estandar) ---
cc_data = np.array([
    [0.07, 69.0, 19.6], [0.10, 69.0, 12.0], [0.12, 68.6, 26.2],
    [0.17, 83.0, 8.0], [0.179, 75.0, 4.0], [0.199, 75.0, 5.0],
    [0.20, 72.9, 29.6], [0.27, 77.0, 14.0], [0.28, 88.8, 36.6],
    [0.352, 83.0, 14.0], [0.3802, 83.0, 13.5], [0.40, 95.0, 17.0],
    [0.4004, 77.0, 10.2], [0.4247, 87.1, 11.2],
    [0.4497, 92.8, 12.9], [0.47, 89.0, 50.0], [0.4783, 80.9, 9.0],
    [0.48, 97.0, 62.0], [0.593, 103.8, 12.5], [0.6797, 92.0, 8.0],
    [0.7812, 105.0, 12.0], [0.8754, 125.0, 17.0], [0.88, 90.0, 40.0],
    [0.9, 117.0, 23.0], [1.037, 154.0, 20.0], [1.3, 168.0, 17.0],
    [1.363, 160.0, 33.6], [1.43, 177.0, 18.0], [1.53, 140.0, 14.0],
    [1.75, 202.0, 40.0], [1.965, 186.5, 50.4],
])

# --- DESI DR1 BAO: Tabla 1 oficial (arXiv:2404.03002) ---
desi_data = [
    ("BGS", 0.295, [("DV_rd", 7.93, 0.15)], None),
    ("LRG1", 0.510, [("DM_rd", 13.62, 0.25), ("DH_rd", 20.98, 0.61)], -0.445),
    ("LRG2", 0.706, [("DM_rd", 16.85, 0.32), ("DH_rd", 20.08, 0.60)], -0.420),
    ("LRG3+ELG1", 0.930, [("DM_rd", 21.71, 0.28), ("DH_rd", 17.88, 0.35)], -0.389),
    ("ELG2", 1.317, [("DM_rd", 27.79, 0.69), ("DH_rd", 13.82, 0.42)], -0.444),
    ("QSO", 1.491, [("DV_rd", 26.07, 0.67)], None),
    ("LyaQSO", 2.330, [("DM_rd", 39.71, 0.94), ("DH_rd", 8.52, 0.17)], -0.477),
]

RD_FID = 147.09
RD_BOUNDS = (130.0, 180.0)


def load_pantheon():
    """Pantheon+ SH0ES con el corte estandar para M libre."""
    path = os.path.join(REALDATA, "PantheonPlusSH0ES.dat")
    cov_path = os.path.join(REALDATA, "PantheonPlusSH0ES_STAT+SYS.cov")
    df = pd.read_csv(path, sep=r"\s+")
    z_hd = df["zHD"].to_numpy(dtype=float)
    z_hel = df["zHEL"].to_numpy(dtype=float)
    mb = df["m_b_corr"].to_numpy(dtype=float)
    dmb = df["m_b_corr_err_DIAG"].to_numpy(dtype=float)
    calib = df["IS_CALIBRATOR"].to_numpy(dtype=int)
    mask = (z_hd > 0.01) & (calib == 0)
    n_all = int(open(cov_path).readline().split()[0])
    cov_full = np.loadtxt(cov_path, skiprows=1).reshape(n_all, n_all)
    cov = cov_full[np.ix_(mask, mask)]
    cinv = np.linalg.inv(cov)
    return (z_hd[mask], z_hel[mask], mb[mask], dmb[mask], cov, cinv)


# ------------------------------------------------------------------
# Fisica: LCDM (analitico) y UV-HDE (solucion algebraica exacta)
# ------------------------------------------------------------------
_ZGRID = np.arange(0.0, 2.61, 0.01)


def E2_lcdm(z, Om):
    OL = 1.0 - Om - Omega_r0
    if OL <= 0:
        return np.nan
    opz = 1.0 + np.asarray(z, dtype=float)
    return Omega_r0 * opz**4 + Om * opz**3 + OL


def _cumulative_integral(e2_grid):
    """Cumulativa de dz'/E(z') sobre _ZGRID (trapecio)."""
    inv_e = 1.0 / np.sqrt(e2_grid)
    return np.concatenate([[0.0], np.cumsum((inv_e[1:] + inv_e[:-1]) * 0.5
                                            * np.diff(_ZGRID))])


# --- UV-HDE: solucion algebraica exacta para E^2(z) ---
from uvhde_cosmology import UVHDEPlanckCosmology

@lru_cache(maxsize=512)
def _uvhde_grid(Om, H0):
    """(E2, cum dz/E) sobre _ZGRID para (Om, H0) dados."""
    cosmo = UVHDEPlanckCosmology(H0=H0, Omega_m0=Om, Omega_r0=Omega_r0)
    e2 = cosmo.E2_classical(_ZGRID)
    if np.any(np.isnan(e2)) or np.any(e2 <= 0.0):
        return None
    return e2, _cumulative_integral(e2)


def E2_uvhde(z, Om, H0):
    """E^2(z) para UV-HDE (interpolada sobre el grid)."""
    if not (0.0 < Om < 1.0) or H0 <= 0:
        return np.nan
    grid = _uvhde_grid(Om, H0)
    if grid is None:
        return np.nan
    return np.interp(np.asarray(z, dtype=float), _ZGRID, grid[0])


# ------------------------------------------------------------------
# API comun de modelo: (E2_fn, cum_fn)
# ------------------------------------------------------------------
def model_functions(model, Om, H0):
    """Devuelve (E2(z), cum_dz_E(z)) para un modelo y parametros dados."""
    if model == "lcdm":
        e2_grid = E2_lcdm(_ZGRID, Om)
        if np.any(np.isnan(e2_grid)) or np.any(e2_grid <= 0):
            return None
        return (lambda z: E2_lcdm(z, Om),
                lambda z: np.interp(np.asarray(z, dtype=float), _ZGRID,
                                    _cumulative_integral(e2_grid)))
    grid = _uvhde_grid(Om, H0)
    if grid is None:
        return None
    e2_grid, cum_grid = grid
    return (lambda z: np.interp(np.asarray(z, dtype=float), _ZGRID, e2_grid),
            lambda z: np.interp(np.asarray(z, dtype=float), _ZGRID, cum_grid))


# ------------------------------------------------------------------
# Likelihoods
# ------------------------------------------------------------------
def chi2_cc(Om, H0, model):
    funcs = model_functions(model, Om, H0)
    if funcs is None:
        return 1e10
    e2f = funcs[0]
    chi2 = 0.0
    for z, H_obs, sigma in cc_data:
        e2 = e2f(z)
        if np.isnan(e2) or e2 <= 0:
            return 1e10
        chi2 += ((H0 * np.sqrt(e2) - H_obs) / sigma) ** 2
    return chi2


def chi2_bao(Om, H0, rd, model):
    """Chi2 BAO con la covarianza D_M/r_d - D_H/r_d de cada trazador."""
    funcs = model_functions(model, Om, H0)
    if funcs is None:
        return 1e10
    e2f, cumf = funcs
    if not (RD_BOUNDS[0] < rd < RD_BOUNDS[1]):
        return 1e10
    chi2 = 0.0
    for tracer, z, obs_list, corr in desi_data:
        z = float(z)
        chi_z = float(cumf(z))
        e2 = float(e2f(z))
        if np.isnan(e2) or e2 <= 0:
            return 1e10
        d_m = c_speed / H0 * chi_z
        d_h = c_speed / (H0 * np.sqrt(e2))
        if len(obs_list) == 2:
            th = np.array([d_m / rd, d_h / rd])
            obs = np.array([obs_list[0][1], obs_list[1][1]])
            s_m, s_h = obs_list[0][2], obs_list[1][2]
            det = s_m**2 * s_h**2 * (1.0 - corr**2)
            if det <= 0:
                return 1e10
            cinv = np.array([[s_h**2, -corr * s_m * s_h],
                             [-corr * s_m * s_h, s_m**2]]) / det
            d = th - obs
            chi2 += float(d @ cinv @ d)
        else:
            d_v = (z * d_m**2 * d_h) ** (1.0 / 3.0)
            _, val, sigma = obs_list[0]
            chi2 += ((d_v / rd - val) / sigma) ** 2
    return chi2


def chi2_sne(Om, H0, M, model, sne):
    z_hd, z_hel, mb, dmb, cov, cinv = sne
    funcs = model_functions(model, Om, H0)
    if funcs is None:
        return 1e10
    _, cumf = funcs
    integ = cumf(z_hd)
    dl = (1.0 + z_hel) * c_speed / H0 * integ
    if np.any(dl <= 0):
        return 1e10
    mu = 5.0 * np.log10(dl) + 25.0 + M
    r = mb - mu
    return float(r @ cinv @ r)


def total_chi2(params, model, sne):
    if model == "lcdm":
        Om, H0, M, rd = params
    else:
        Om, H0, M, rd = params  # UV-HDE has same params as LCDM (Dk=0)
    c2 = (chi2_cc(Om, H0, model)
          + chi2_bao(Om, H0, rd, model)
          + chi2_sne(Om, H0, M, model, sne))
    if not np.isfinite(c2):
        return 1e10
    return c2


# ------------------------------------------------------------------
# Priors y posterior
# ------------------------------------------------------------------
PARAM_NAMES = {
    "lcdm": ["Om", "H0", "M", "rd"],
    "uvhde": ["Om", "H0", "M", "rd"],  # Dk = 0
}
BOUNDS = {
    "lcdm": [(0.10, 0.50), (60.0, 80.0), (-25.0, -18.0), RD_BOUNDS],
    "uvhde": [(0.10, 0.50), (60.0, 80.0), (-25.0, -18.0), RD_BOUNDS],
}
NDIM = {"lcdm": 4, "uvhde": 4}


def log_prior(p, model):
    for (lo, hi), v in zip(BOUNDS[model], p):
        if not (lo < v < hi):
            return -np.inf
    return 0.0


def make_logpost(model, sne):
    def logpost(p):
        lp = log_prior(p, model)
        if not np.isfinite(lp):
            return -np.inf
        c2 = total_chi2(p, model, sne)
        if c2 > 1e9:
            return -np.inf
        return lp - 0.5 * c2
    return logpost


# ------------------------------------------------------------------
# Ajuste: minimo chi2 (Nelder-Mead) y MCMC (emcee)
# ------------------------------------------------------------------
def best_fit(model, sne, n_restarts=4):
    """Minimiza chi2 desde varios puntos de partida (libre de prior)."""
    from scipy.optimize import minimize
    ndim = NDIM[model]
    if model == "lcdm":
        starts = [[0.30, 70.0, -19.3, 147.0],
                  [0.25, 73.0, -19.5, 155.0],
                  [0.35, 67.0, -19.2, 140.0],
                  [0.20, 75.0, -19.4, 165.0]]
    else:
        starts = [[0.3153, 67.36, -19.3, 147.0],
                  [0.28, 70.0, -19.5, 155.0],
                  [0.34, 66.0, -19.2, 140.0],
                  [0.25, 72.0, -19.4, 165.0]]
    best = None
    for x0 in starts[:max(1, n_restarts)]:
        if len(x0) != ndim:
            continue
        res = minimize(total_chi2, np.array(x0), args=(model, sne),
                       method="Nelder-Mead",
                       options=dict(maxiter=6000, maxfev=6000,
                                    xatol=1e-5, fatol=1e-5))
        if best is None or res.fun < best.fun:
            best = res
    return best


def run_mcmc(model, sne, n_walkers=24, n_steps=3000, burn=800):
    import emcee
    np.random.seed(42)
    ndim = NDIM[model]
    if model == "lcdm":
        p0 = np.array([0.30, 70.0, -19.3, 147.0])
        scale = np.array([0.02, 0.5, 0.05, 2.0])
    else:
        p0 = np.array([0.3153, 67.36, -19.3, 147.0])
        scale = np.array([0.02, 0.5, 0.05, 2.0])
    start = p0 + scale * np.random.randn(n_walkers, ndim)
    sampler = emcee.EnsembleSampler(n_walkers, ndim, make_logpost(model, sne))
    sampler.run_mcmc(start, n_steps, progress=False)
    chain = sampler.get_chain(discard=burn, flat=True)
    return sampler, chain


def summarize_chain(chain, names):
    q16, q50, q84 = np.percentile(chain, [16, 50, 84], axis=0)
    rows = []
    for i, nm in enumerate(names):
        rows.append({"param": nm, "median": q50[i],
                     "lo68": q50[i] - q16[i], "hi68": q84[i] - q50[i]})
    return rows


def main():
    print("Cargando datos reales...")
    sne = load_pantheon()
    print(f"  Pantheon+: {len(sne[0])} SNe (z>0.01, sin calibradoras)")
    print(f"  DESI DR1 BAO: {sum(len(d[2]) for d in desi_data)} observables "
          f"en {len(desi_data)} trazadores")
    print(f"  Cosmic chronometers: {len(cc_data)} puntos")

    n_cc, n_bao, n_sne = len(cc_data), sum(len(d[2]) for d in desi_data), len(sne[0])
    n_data = n_cc + n_bao + n_sne
    results = {}
    for model in ["lcdm", "uvhde"]:
        names = PARAM_NAMES[model]
        print(f"\n=== {model.upper()} ===")
        print("  minimizacion (Nelder-Mead)...")
        bf = best_fit(model, sne)
        print(f"  chi2_min = {bf.fun:.2f}")
        for nm, v in zip(names, bf.x):
            print(f"    {nm:6s} = {v:.4f}")
        print("  MCMC (emcee)...")
        sampler, chain = run_mcmc(model, sne)
        print(f"  muestras post-burn: {chain.shape[0]}")
        for row in summarize_chain(chain, names):
            print(f"    {row['param']:6s} = {row['median']:.4f} "
                  f"+{row['hi68']:.4f} -{row['lo68']:.4f}")
        logp_max = float(np.nanmax(sampler.get_log_prob()))
        np.save(f"uvhde_real_chain_{model}.npy", chain)
        results[model] = {
            "chi2_min": bf.fun, "chi2_map": -2.0 * logp_max,
            "params": dict(zip(names, bf.x)),
            "posterior": summarize_chain(chain, names),
        }

    rows = []
    for model in ["lcdm", "uvhde"]:
        k = NDIM[model]
        r = results[model]
        rows.append({
            "model": model, "n_params": k, "n_data": n_data,
            "chi2_min": round(r["chi2_min"], 2),
            "chi2_map": round(r["chi2_map"], 2),
            "AIC": round(r["chi2_min"] + 2 * k, 2),
            "BIC": round(r["chi2_min"] + k * np.log(n_data), 2),
        })
        for row in r["posterior"]:
            rows.append({"model": model,
                         f"{row['param']}_median": row["median"],
                         f"{row['param']}_lo68": row["lo68"],
                         f"{row['param']}_hi68": row["hi68"]})
    df = pd.DataFrame(rows)
    df.to_csv("uvhde_late_time_fit_summary.csv", index=False)
    print("\n=== Comparacion (chi2_min libre de prior) ===")
    dchi2 = results["uvhde"]["chi2_min"] - results["lcdm"]["chi2_min"]
    dk = NDIM["uvhde"] - NDIM["lcdm"]  # = 0
    print(f"  Delta chi2 (UV-HDE - LCDM) = {dchi2:+.2f}  ({dk} params extra, Dk=0)")
    print(f"  Delta AIC = {dchi2 + 2 * dk:+.2f}")
    print(f"  Delta BIC = {dchi2 + dk * np.log(n_data):+.2f}")
    print("  (Delta AIC/BIC > 0 => LCDM preferido por parsimonia)")
    print("\nWrote uvhde_late_time_fit_summary.csv")


if __name__ == "__main__":
    main()