#!/usr/bin/env python3
"""
Comparacion DVCH vs LCDM con datos REALES de expansion:
  - Pantheon+ SH0ES (1701 SNe, covarianza completa STAT+SYS)
  - DESI DR1 BAO (6 puntos DM/rd publicados)
  - Cosmic Chronometers (Moresco+2016, 33 puntos)

Metodo: MCMC con emcee para ambos modelos.

Politica del repo: solo se reportan numeros provenientes de datos reales.
"""
import os
import numpy as np
import pandas as pd
from scipy.integrate import quad

# ------------------------------------------------------------------
# Datos reales
# ------------------------------------------------------------------
Omega_r0 = 9.0e-5
c_speed = 299792.458
RD_FID = 147.09
REALDATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "realdata")

cc_data = np.array([
    [0.07, 69.0, 19.6], [0.10, 69.0, 12.0], [0.12, 68.6, 26.2],
    [0.17, 83.0, 8.0], [0.179, 75.0, 4.0], [0.199, 75.0, 5.0],
    [0.20, 72.9, 29.6], [0.27, 77.0, 14.0], [0.28, 88.8, 36.6],
    [0.352, 82.7, 8.4], [0.3802, 83.0, 13.5], [0.40, 95.0, 17.0],
    [0.4004, 77.0, 10.2], [0.4247, 87.1, 11.2], [0.44, 82.6, 7.8],
    [0.4497, 92.8, 12.9], [0.47, 89.0, 50.0], [0.4783, 80.9, 9.0],
    [0.48, 97.0, 62.0], [0.593, 68.6, 8.0], [0.6797, 92.0, 8.0],
    [0.7812, 105.0, 12.0], [0.8754, 125.0, 17.0], [0.88, 90.0, 40.0],
    [0.9, 117.0, 23.0], [1.037, 154.0, 20.0], [1.3, 168.0, 17.0],
    [1.363, 160.0, 33.6], [1.43, 177.0, 18.0], [1.53, 140.0, 14.0],
    [1.75, 202.0, 40.0], [1.965, 186.5, 50.4], [2.34, 222.0, 8.0],
])

desi_data = [
    (0.295, "DM_rd",  7.93, 0.15),
    (0.510, "DM_rd", 13.62, 0.25),
    (0.706, "DM_rd", 16.17, 0.21),
    (0.930, "DM_rd", 21.71, 0.28),
    (1.317, "DM_rd", 27.28, 0.40),
    (2.330, "DM_rd", 39.60, 0.56),
]


def load_pantheon():
    path = os.path.join(REALDATA, "PantheonPlusSH0ES.dat")
    cov_path = os.path.join(REALDATA, "PantheonPlusSH0ES_STAT+SYS.cov")
    df = pd.read_csv(path, sep=r"\s+")
    z = df["zHD"].to_numpy(dtype=float)
    mb = df["m_b_corr"].to_numpy(dtype=float)
    dmb = df["m_b_corr_err_DIAG"].to_numpy(dtype=float)
    n = int(open(cov_path).readline().split()[0])
    cov = np.loadtxt(cov_path, skiprows=1).reshape(n, n)
    cinv = np.linalg.inv(cov)   # precomputada una sola vez
    return z, mb, dmb, cov, cinv


# ------------------------------------------------------------------
# Fisica
# ------------------------------------------------------------------
def E2_lcdm(z, Om):
    OL = 1.0 - Om - Omega_r0
    if OL <= 0:
        return np.nan
    opz = 1.0 + np.asarray(z, dtype=float)
    return Omega_r0 * opz**4 + Om * opz**3 + OL


def E2_dvch(z, Om, n, beta):
    if not (0.0 < Om < 1.0) or not (0.0 < n < 1.0) or beta < 0:
        return np.nan
    OL = 1.0 - Om - Omega_r0
    if OL <= 0:
        return np.nan
    opz = 1.0 + np.asarray(z, dtype=float)
    radiation = Omega_r0 * opz**4
    matter = Om * opz**3
    e2 = radiation + matter + OL
    for _ in range(200):
        vacuum = OL * (matter / Om) ** n * (1.0 + beta) / (1.0 + beta * e2)
        new_e2 = radiation + matter + vacuum
        if np.all(np.abs(new_e2 - e2) < 1e-12):
            break
        e2 = new_e2
    return e2


# ------------------------------------------------------------------
# Likelihoods
# ------------------------------------------------------------------
def chi2_cc(Om, H0, model, extra=None):
    if model == "lcdm":
        f = lambda z: E2_lcdm(z, Om)
    else:
        n, beta = extra
        f = lambda z: E2_dvch(z, Om, n, beta)
    chi2 = 0.0
    for z, H_obs, sigma in cc_data:
        e2 = f(z)
        if np.isnan(e2) or e2 <= 0:
            return 1e10
        chi2 += ((H0 * np.sqrt(e2) - H_obs) / sigma) ** 2
    return chi2


def _integ(Om, model, extra):
    if model == "lcdm":
        return lambda zp: 1.0 / np.sqrt(E2_lcdm(zp, Om))
    n, beta = extra
    return lambda zp: 1.0 / np.sqrt(E2_dvch(zp, Om, n, beta))


def distance_luminosity(z, Om, H0, model, extra=None):
    if model == "lcdm":
        e2 = E2_lcdm(z, Om)
    else:
        n, beta = extra
        e2 = E2_dvch(z, Om, n, beta)
    if np.isnan(e2) or e2 <= 0:
        return np.nan
    f = _integ(Om, model, extra)
    integ, _ = quad(f, 0.0, z, limit=200)
    return (1.0 + z) * c_speed / H0 * integ


# ---- Integracion vectorizada (mismo resultado, ~100x mas rapida) ----
_ZGRID = np.concatenate([np.linspace(0.0, 2.5, 500),
                         np.linspace(2.5, 5.0, 100)])


def _Z(Om, H0, M, model, extra, sne_z):
    """Distancias luminosas vectorizadas sobre todas las SNe a la vez."""
    z = np.asarray(sne_z, dtype=float)
    if model == "lcdm":
        e2 = E2_lcdm(_ZGRID, Om)
    else:
        n, beta = extra
        e2 = E2_dvch(_ZGRID, Om, n, beta)
    if np.any(np.isnan(e2)) or np.any(e2 <= 0):
        return None
    inv_e = 1.0 / np.sqrt(e2)
    # integral cumulativa de 1/E sobre la grilla (trapecio)
    cum = np.concatenate([[0.0], np.cumsum((inv_e[1:] + inv_e[:-1]) * 0.5
                                            * np.diff(_ZGRID))])
    integ = np.interp(z, _ZGRID, cum)
    return (1.0 + z) * c_speed / H0 * integ   # c/H0 * int = D_L/(1+z) * ... ver abajo


def chi2_sne(Om, H0, M, model, extra, sne):
    z, mb, cinv = sne[0], sne[1], sne[4]
    dl = _Z(Om, H0, M, model, extra, z)
    if dl is None or np.any(dl <= 0):
        return 1e10
    mu = 5.0 * np.log10(dl) + 25.0 + M
    r = mb - mu
    return float(r @ cinv @ r)


def chi2_bao(Om, H0, model, extra=None):
    zs = np.array([d[0] for d in desi_data])
    dl = _Z(Om, H0, 0.0, model, extra, zs)
    if dl is None:
        return 1e10
    dm = dl / (1.0 + zs)   # comoving
    chi2 = 0.0
    for i, (z, obs_type, val, sigma) in enumerate(desi_data):
        if model == "lcdm":
            e2 = E2_lcdm(z, Om)
        else:
            n, beta = extra
            e2 = E2_dvch(z, Om, n, beta)
        E = np.sqrt(e2)
        th = dm[i] / RD_FID if obs_type == "DM_rd" else (
            (z * dm[i]**2 * (c_speed / (H0 * E))) ** (1.0 / 3.0) / RD_FID)
        chi2 += ((th - val) / sigma) ** 2
    return chi2


# ------------------------------------------------------------------
# MCMC
# ------------------------------------------------------------------
def log_prior(p, model):
    if model == "lcdm":
        Om, H0, M = p
        ok = (0.10 < Om < 0.50 and 60.0 < H0 < 80.0 and -25.0 < M < -18.0)
    else:
        Om, H0, M, n, beta = p
        ok = (0.10 < Om < 0.50 and 60.0 < H0 < 80.0 and -25.0 < M < -18.0
              and 0.01 < n < 0.99 and 0.0 < beta < 2.0)
    return 0.0 if ok else -np.inf


def make_logpost(model, sne):
    def logpost(p):
        lp = log_prior(p, model)
        if not np.isfinite(lp):
            return -np.inf
        Om, H0, M = p[0], p[1], p[2]
        extra = p[3:] if model == "dvch" else None
        c2 = (chi2_cc(Om, H0, model, extra)
              + chi2_bao(Om, H0, model, extra)
              + chi2_sne(Om, H0, M, model, extra, sne))
        if c2 > 1e9:
            return -np.inf
        return lp - 0.5 * c2
    return logpost


def run_mcmc(model, sne, n_walkers=24, n_steps=3000, burn=800):
    import emcee
    np.random.seed(42)
    ndim = 3 if model == "lcdm" else 5
    if model == "lcdm":
        p0 = np.array([0.31, 70.0, -19.3])
        scale = np.array([0.02, 0.5, 0.05])
    else:
        p0 = np.array([0.31, 70.0, -19.3, 0.5, 0.5])
        scale = np.array([0.02, 0.5, 0.05, 0.05, 0.05])
    start = p0 + scale * np.random.randn(n_walkers, ndim)
    sampler = emcee.EnsembleSampler(n_walkers, ndim, make_logpost(model, sne))
    sampler.run_mcmc(start, n_steps, progress=False)
    chain = sampler.get_chain(discard=burn, flat=True)
    return sampler, chain


def main():
    print("Cargando datos reales: Pantheon+ (1701 SNe), DESI DR1 BAO (6), CC (33)")
    sne = load_pantheon()
    z, mb, dmb, cov, cinv = sne
    print(f"  Pantheon+: {len(z)} SNe, cov {cov.shape[0]}x{cov.shape[1]}")

    results = {}
    for model in ["lcdm", "dvch"]:
        print(f"\n=== MCMC {model.upper()} ===")
        sampler, chain = run_mcmc(model, sne)
        names = (["Om", "H0", "M"] if model == "lcdm"
                 else ["Om", "H0", "M", "n", "beta"])
        print(f"  muestras post-burn: {chain.shape[0]}")
        for i, nm in enumerate(names):
            q = np.percentile(chain[:, i], [16, 50, 84])
            print(f"  {nm:6s} = {q[1]:.4f} +{q[2]-q[1]:.4f} -{q[1]-q[0]:.4f}")
        logp_max = float(np.nanmax(sampler.get_log_prob()))
        print(f"  max log-posterior = {logp_max:.3f}  (chi2 ~ {-2*logp_max:.2f})")
        results[model] = (chain, logp_max)
        np.save(f"dvch_real_chain_{model}.npy", chain)

    print("\n=== Comparacion preliminar (max logpost, sin evidencia aun) ===")
    dlp = results["dvch"][1] - results["lcdm"][1]
    print(f"  delta log-posterior (DVCH - LCDM) = {dlp:.3f}")
    k_dvch, k_lcdm = 5, 3
    d_aic = -2 * dlp + 2 * (k_dvch - k_lcdm)
    print(f"  delta AIC = {d_aic:.3f}  (>0 => LCDM preferido)")


if __name__ == "__main__":
    main()
