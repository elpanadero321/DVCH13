#!/usr/bin/env python3
"""
DVCH full late-time MCMC pipeline (fast version).

Joint Cosmic Chronometers + DESI DR1 BAO likelihood, sampled with emcee
ensemble sampler. Uses the same fast CC+BAO dataset as
dvch_mcmc_convergence.py but with the full emcee diagnostic suite.

Reports:
  - full convergence diagnostics (Gelman-Rubin R-hat, ESS, autocorrelation
    time, acceptance fractions),
  - posterior summary (median, 16/84 and 2.5/97.5 quantiles),
  - information criteria (chi2, AIC, BIC),
  - corner plot and trace plots.

Outputs
-------
dvch_mcmc_chains_full.csv
dvch_mcmc_full_summary.csv
dvch_mcmc_full_convergence.csv
dvch_mcmc_full_evidence.csv
figures/dvch_full_mcmc_corner.png
figures/dvch_full_mcmc_traces.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import quad
import emcee
import corner
import os

FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)

Omega_r0 = 9.0e-5
c_speed = 299792.458
rd_fid = 147.09

# ---- Cosmic Chronometers (Moresco+2016) ----
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

# ---- DESI DR1 BAO: D_M/r_d and D_V/r_d ----
desi_data = [
    (0.122, 3.4663, "DV_rd", 0.08),
    (0.15,  4.2224, "DV_rd", 0.06),
    (0.38,  10.2149, "DM_rd", 0.15),
    (0.51,  13.2447, "DM_rd", 0.15),
    (0.61,  15.4220, "DM_rd", 0.17),
    (0.70,  17.2734, "DM_rd", 0.10),
    (0.845, 20.0545, "DM_rd", 0.16),
    (1.10,  24.4038, "DM_rd", 0.50),
    (1.48,  29.8250, "DM_rd", 0.40),
    (2.33,  38.7563, "DM_rd", 0.55),
]


def E2_dvch(z, Om, n, beta):
    """Implicit DVCH Friedmann solver."""
    OL = 1.0 - Om - Omega_r0
    if Om <= 0 or Om >= 1 or n <= 0 or n >= 1 or beta < 0 or OL <= 0:
        return np.nan
    opz = 1.0 + z
    rad = Omega_r0 * opz**4
    Om_z = Om * opz**3
    E2 = rad + Om_z + OL
    for _ in range(30):
        OL_z = OL * (Om_z / Om)**n * (1.0 + beta) / (1.0 + beta * E2)
        E2_new = rad + Om_z + OL_z
        if abs(E2_new - E2) < 1e-8:
            break
        E2 = E2_new
    if E2 <= 0 or np.isnan(E2):
        return np.nan
    return E2


def H_of_z(z, Om, n, beta, H0):
    E2 = E2_dvch(z, Om, n, beta)
    if np.isnan(E2) or E2 <= 0:
        return np.nan
    return H0 * np.sqrt(E2)


def comoving_distance_Mpc(z, Om, n, beta, H0):
    integral, _ = quad(lambda zp: 1.0 / np.sqrt(E2_dvch(zp, Om, n, beta)),
                       0, z, limit=100)
    return c_speed / H0 * integral


def DV_rd(z, Om, n, beta, H0):
    DM = comoving_distance_Mpc(z, Om, n, beta, H0)
    H = H_of_z(z, Om, n, beta, H0)
    if np.isnan(H) or H <= 0 or np.isnan(DM):
        return np.nan
    DH = c_speed / H
    DV = (DM**2 * DH * z)**(1.0 / 3.0)
    return DV / rd_fid


def log_likelihood(params):
    Om, n, beta, H0 = params

    if not (0.05 < Om < 0.95 and 0.001 < n < 0.999 and
            0.0 <= beta < 1.0 and 40.0 < H0 < 120.0):
        return -np.inf

    lnL = 0.0

    for row in cc_data:
        z_i, H_obs, sigma = row
        H_th = H_of_z(z_i, Om, n, beta, H0)
        if np.isnan(H_th):
            return -np.inf
        lnL -= 0.5 * ((H_th - H_obs) / sigma)**2

    for z_i, val, obs_type, sigma in desi_data:
        if obs_type == "DM_rd":
            th = comoving_distance_Mpc(z_i, Om, n, beta, H0) / rd_fid
        else:
            th = DV_rd(z_i, Om, n, beta, H0)
        if np.isnan(th):
            return -np.inf
        lnL -= 0.5 * ((th - val) / sigma)**2

    return lnL


def log_posterior(params):
    Om, n, beta, H0 = params
    if not (0.05 < Om < 0.95 and 0.001 < n < 0.999 and
            0.0 <= beta < 1.0 and 40.0 < H0 < 120.0):
        return -np.inf
    return log_likelihood(params)


if __name__ == "__main__":
    print("=== DVCH full late-time MCMC pipeline ===")

    ndim = 4
    param_names = ["Om", "n", "beta", "H0"]

    p0 = np.array([0.30, 0.09, 1.0e-4, 69.03])

    n_walkers = 24
    n_steps = 2000
    n_burn = 800

    rng = np.random.default_rng(20260827)
    pos = p0 + 1e-3 * rng.standard_normal((n_walkers, ndim))
    for i in range(n_walkers):
        pos[i, 0] = np.clip(pos[i, 0], 0.06, 0.94)
        pos[i, 1] = np.clip(pos[i, 1], 0.002, 0.998)
        pos[i, 2] = np.clip(pos[i, 2], 1e-8, 0.99)
        pos[i, 3] = np.clip(pos[i, 3], 41.0, 119.0)

    print(f"Running emcee: {n_walkers} walkers x {n_steps} steps (burn-in {n_burn})...")
    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_posterior)
    sampler.run_mcmc(pos, n_steps, progress=True)

    try:
        tau = sampler.get_autocorr_time(quiet=True)
        thin = max(1, int(0.5 * np.min(tau[np.isfinite(tau)]))) if np.any(np.isfinite(tau)) else 1
    except Exception:
        thin = 1

    flat = sampler.get_chain(discard=n_burn, thin=thin, flat=True)
    flat_logprob = sampler.get_log_prob(discard=n_burn, thin=thin, flat=True)

    print(f"Effective samples (after burn-in + thin={thin}): {flat.shape[0]}")

    # ---- Gelman-Rubin ----
    chains_for_rhat = sampler.get_chain(discard=n_burn, thin=thin)
    half = chains_for_rhat.shape[0] // 2
    m = n_walkers * 2
    split = np.zeros((m, half, ndim))
    for w in range(n_walkers):
        split[2*w] = chains_for_rhat[:half, w, :]
        split[2*w+1] = chains_for_rhat[half:, w, :]

    B = (half / (m - 1)) * np.sum(
        (split.mean(axis=1) - split.mean(axis=(0, 1)))**2, axis=0)
    W = np.mean([np.var(c, axis=0, ddof=1) for c in split], axis=0)
    var_hat = (1.0 - 1.0/half) * W + B / half
    R_hat = np.sqrt(var_hat / np.where(W > 0, W, 1e-30))

    # ---- ESS ----
    ess = np.zeros(ndim)
    try:
        tau_full = sampler.get_autocorr_time(quiet=True)
        for j in range(ndim):
            if np.isfinite(tau_full[j]):
                ess[j] = n_walkers * (n_steps - n_burn) / tau_full[j]
            else:
                ess[j] = flat.shape[0]
    except Exception:
        ess[:] = flat.shape[0]

    acc_frac = sampler.acceptance_fraction.mean()

    print(f"R-hat : {R_hat}")
    print(f"ESS   : {ess.astype(int)}")
    print(f"Acceptance fraction: {acc_frac:.3f}")

    # ---- Posterior summary ----
    medians = np.median(flat, axis=0)
    q16 = np.percentile(flat, 16, axis=0)
    q84 = np.percentile(flat, 84, axis=0)
    q025 = np.percentile(flat, 2.5, axis=0)
    q975 = np.percentile(flat, 97.5, axis=0)

    summary_rows = []
    for j, name in enumerate(param_names):
        summary_rows.append({
            "parameter": name,
            "median": medians[j],
            "mean": flat[:, j].mean(),
            "std": flat[:, j].std(),
            "q16": q16[j], "q84": q84[j],
            "q025": q025[j], "q975": q975[j],
        })

    # ---- Information criteria ----
    best_idx = np.argmax(flat_logprob)
    best_params = flat[best_idx]
    best_logL = log_likelihood(best_params)
    chi2_best = -2.0 * best_logL

    n_data = len(cc_data) + len(desi_data)
    k = ndim
    AIC = chi2_best + 2.0 * k
    BIC = chi2_best + k * np.log(n_data)

    print(f"\n=== Information criteria ===")
    print(f"chi2 best  : {chi2_best:.2f}  (n_data = {n_data})")
    print(f"AIC        : {AIC:.2f}")
    print(f"BIC        : {BIC:.2f}")

    # ---- Save outputs ----
    chain_df = pd.DataFrame(flat, columns=param_names)
    chain_df["log_posterior"] = flat_logprob
    chain_df.to_csv("dvch_mcmc_chains_full.csv", index=False)

    pd.DataFrame(summary_rows).to_csv("dvch_mcmc_full_summary.csv", index=False)

    conv_df = pd.DataFrame({
        "parameter": param_names,
        "R_hat": R_hat,
        "ESS": ess.astype(int),
    })
    conv_df.to_csv("dvch_mcmc_full_convergence.csv", index=False)

    ev_df = pd.DataFrame([{
        "n_data": n_data, "n_params": k,
        "chi2_best": chi2_best, "AIC": AIC, "BIC": BIC,
        "acceptance_fraction": acc_frac,
    }])
    ev_df.to_csv("dvch_mcmc_full_evidence.csv", index=False)

    # ---- Corner plot ----
    fig = corner.corner(flat, labels=param_names, truths=medians,
                        quantiles=[0.16, 0.5, 0.84], show_titles=True,
                        title_kwargs={"fontsize": 10})
    fig.savefig(os.path.join(FIGDIR, "dvch_full_mcmc_corner.png"), dpi=200)
    plt.close(fig)

    # ---- Trace plots ----
    chain_raw = sampler.get_chain()
    fig, axes = plt.subplots(ndim, 1, figsize=(10, 2.0 * ndim), sharex=True)
    for j in range(ndim):
        for w in range(min(n_walkers, 12)):
            axes[j].plot(chain_raw[:, w, j], alpha=0.3, lw=0.5)
        axes[j].axvline(n_burn, color="k", ls="--", lw=0.8, label="burn-in")
        axes[j].set_ylabel(param_names[j])
        axes[j].grid(alpha=0.2)
    axes[-1].set_xlabel("step")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("DVCH MCMC trace plots", y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "dvch_full_mcmc_traces.png"), dpi=180,
                bbox_inches="tight")
    plt.close(fig)

    print(f"\n=== Outputs written ===")
    print(f"  dvch_mcmc_chains_full.csv      ({flat.shape[0]} samples)")
    print(f"  dvch_mcmc_full_summary.csv")
    print(f"  dvch_mcmc_full_convergence.csv")
    print(f"  dvch_mcmc_full_evidence.csv")
    print(f"  figures/dvch_full_mcmc_corner.png")
    print(f"  figures/dvch_full_mcmc_traces.png")
    print("\nDone.")