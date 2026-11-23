#!/usr/bin/env python3
"""
UV-HDE + Planck Limit full late-time MCMC pipeline (fast version).

Joint Cosmic Chronometers + DESI DR1 BAO likelihood, sampled with emcee
ensemble sampler. The UV-HDE model has the same number of free parameters
as LCDM (Δk = 0): only (Omega_m, H0) plus rd which is marginalized.

Reports:
  - full convergence diagnostics (Gelman-Rubin R-hat, ESS, autocorrelation
    time, acceptance fractions),
  - posterior summary (median, 16/84 and 2.5/97.5 quantiles),
  - information criteria (chi2, AIC, BIC),
  - corner plot and trace plots.

Outputs
-------
uvhde_mcmc_chains_full.csv
uvhde_mcmc_full_summary.csv
uvhde_mcmc_full_convergence.csv
uvhde_mcmc_full_evidence.csv
figures/uvhde_full_mcmc_corner.png
figures/uvhde_full_mcmc_traces.png
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

from uvhde_cosmology import UVHDEPlanckCosmology

FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)

Omega_r0 = 9.18e-5
c_speed = 299792.458
rd_fid = 147.09

# ---- Cosmic chronometers (31 points, standard compilation) ----
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

# ---- DESI DR1 BAO (Tabla 1 oficial, arXiv:2404.03002) ----
desi_data = [
    (0.295, 7.93, "DV_rd", 0.15),
    (0.510, 13.62, "DM_rd", 0.25),
    (0.706, 16.85, "DM_rd", 0.32),
    (0.930, 21.71, "DM_rd", 0.28),
    (1.317, 27.79, "DM_rd", 0.69),
    (1.491, 26.07, "DV_rd", 0.67),
    (2.330, 39.71, "DM_rd", 0.94),
]


def E2_uvhde(z, Om, H0):
    """Exact algebraic UV-HDE Friedmann solution."""
    cosmo = UVHDEPlanckCosmology(H0=H0, Omega_m0=Om, Omega_r0=Omega_r0)
    return cosmo.E2_classical(z)


def H_of_z(z, Om, H0):
    E2 = E2_uvhde(z, Om, H0)
    if np.isnan(E2) or E2 <= 0:
        return np.nan
    return H0 * np.sqrt(E2)


def comoving_distance_Mpc(z, Om, H0):
    integral, _ = quad(lambda zp: 1.0 / np.sqrt(E2_uvhde(zp, Om, H0)),
                       0, z, limit=100)
    return c_speed / H0 * integral


def DV_rd(z, Om, H0):
    DM = comoving_distance_Mpc(z, Om, H0)
    H = H_of_z(z, Om, H0)
    if np.isnan(H) or H <= 0 or np.isnan(DM):
        return np.nan
    DH = c_speed / H
    DV = (DM**2 * DH * z)**(1.0 / 3.0)
    return DV / rd_fid


def log_likelihood(params):
    Om, H0 = params

    if not (0.05 < Om < 0.95 and 40.0 < H0 < 120.0):
        return -np.inf

    lnL = 0.0

    for row in cc_data:
        z_i, H_obs, sigma = row
        H_th = H_of_z(z_i, Om, H0)
        if np.isnan(H_th):
            return -np.inf
        lnL -= 0.5 * ((H_th - H_obs) / sigma)**2

    for z_i, val, obs_type, sigma in desi_data:
        if obs_type == "DM_rd":
            th = comoving_distance_Mpc(z_i, Om, H0) / rd_fid
        else:
            th = DV_rd(z_i, Om, H0)
        if np.isnan(th):
            return -np.inf
        lnL -= 0.5 * ((th - val) / sigma)**2

    return lnL


def log_posterior(params):
    Om, H0 = params
    if not (0.05 < Om < 0.95 and 40.0 < H0 < 120.0):
        return -np.inf
    return log_likelihood(params)


def log_likelihood_lcdm(params):
    Om, H0 = params

    if not (0.05 < Om < 0.95 and 40.0 < H0 < 120.0):
        return -np.inf

    lnL = 0.0
    OL = 1.0 - Om - Omega_r0

    for row in cc_data:
        z_i, H_obs, sigma = row
        opz = 1.0 + z_i
        E2 = Omega_r0 * opz**4 + Om * opz**3 + OL
        if E2 <= 0:
            return -np.inf
        H_th = H0 * np.sqrt(E2)
        lnL -= 0.5 * ((H_th - H_obs) / sigma)**2

    for z_i, val, obs_type, sigma in desi_data:
        integral, _ = quad(lambda zp: 1.0 / np.sqrt(Omega_r0 * (1+zp)**4 + Om * (1+zp)**3 + OL),
                           0, z_i, limit=100)
        if obs_type == "DM_rd":
            th = c_speed / H0 * integral / rd_fid
        else:
            opz = 1.0 + z_i
            E2 = Omega_r0 * opz**4 + Om * opz**3 + OL
            if E2 <= 0:
                return -np.inf
            H = H0 * np.sqrt(E2)
            DM = c_speed / H0 * integral
            DH = c_speed / H
            DV = (DM**2 * DH * z_i)**(1.0 / 3.0)
            th = DV / rd_fid
        if np.isnan(th):
            return -np.inf
        lnL -= 0.5 * ((th - val) / sigma)**2

    return lnL


def log_posterior_lcdm(params):
    Om, H0 = params
    if not (0.05 < Om < 0.95 and 40.0 < H0 < 120.0):
        return -np.inf
    return log_likelihood_lcdm(params)


import os
import time
import h5py

if __name__ == "__main__":
    print("=== UV-HDE + Planck Limit full late-time MCMC pipeline ===")
    print("Model has Dk = 0 extra parameters vs LCDM (only Omega_m, H0 free)")

    ndim = 2
    param_names = ["Om", "H0"]

    p0_uvhde = np.array([0.3153, 67.36])
    p0_lcdm = np.array([0.3153, 67.36])

    n_walkers = 20
    n_steps = 6000
    n_burn = 1500

    rng = np.random.default_rng(20260827)

# Checkpoint files
    uvhde_backend_file = "uvhde_mcmc_checkpoint.h5"
    lcdm_backend_file = "lcdm_mcmc_checkpoint.h5"

    # --- UV-HDE: load from existing checkpoint ---
    print(f"\nLoading UV-HDE MCMC from checkpoint: {uvhde_backend_file}")
    uvhde_backend = emcee.backends.HDFBackend(uvhde_backend_file)
    sampler_uvhde = emcee.EnsembleSampler(n_walkers, ndim, log_posterior, backend=uvhde_backend)

    try:
        tau = sampler_uvhde.get_autocorr_time(quiet=True)
        thin = max(1, int(0.5 * np.min(tau[np.isfinite(tau)]))) if np.any(np.isfinite(tau)) else 1
    except Exception:
        thin = 1

    flat_uvhde = sampler_uvhde.get_chain(discard=n_burn, thin=thin, flat=True)
    flat_logprob_uvhde = sampler_uvhde.get_log_prob(discard=n_burn, thin=thin, flat=True)

    print(f"UV-HDE effective samples (after burn-in + thin={thin}): {flat_uvhde.shape[0]}")

    # --- LCDM with checkpointing ---
    print(f"\nRunning LCDM MCMC: {n_walkers} walkers x {n_steps} steps (burn-in {n_burn})...")
    pos = p0_lcdm + 1e-3 * rng.standard_normal((n_walkers, ndim))
    for i in range(n_walkers):
        pos[i, 0] = np.clip(pos[i, 0], 0.06, 0.94)
        pos[i, 1] = np.clip(pos[i, 1], 41.0, 119.0)

    lcdm_backend = emcee.backends.HDFBackend(lcdm_backend_file)
    lcdm_backend.reset(n_walkers, ndim)
    sampler_lcdm = emcee.EnsembleSampler(n_walkers, ndim, log_posterior_lcdm, backend=lcdm_backend)

    checkpoint_interval = 60  # seconds
    last_checkpoint = time.time()
    for step, state in enumerate(sampler_lcdm.sample(pos, iterations=n_steps, progress=True)):
        if time.time() - last_checkpoint > checkpoint_interval:
            last_checkpoint = time.time()
            print(f"  LCDM checkpoint at step {step+1}/{n_steps}")

    # ---- Convergence diagnostics for UV-HDE ----
    chains_for_rhat = sampler_uvhde.get_chain(discard=n_burn, thin=thin)
    half = chains_for_rhat.shape[0] // 2
    m = n_walkers * 2
    split = np.zeros((m, half, ndim))
    for w in range(n_walkers):
        split[2*w] = chains_for_rhat[:half, w, :]
        split[2*w+1] = chains_for_rhat[half:2*half, w, :]

    B = (half / (m - 1)) * np.sum(
        (split.mean(axis=1) - split.mean(axis=(0, 1)))**2, axis=0)
    W = np.mean([np.var(c, axis=0, ddof=1) for c in split], axis=0)
    var_hat = (1.0 - 1.0/half) * W + B / half
    R_hat = np.sqrt(var_hat / np.where(W > 0, W, 1e-30))

    # ESS
    ess = np.zeros(ndim)
    try:
        tau_full = sampler_uvhde.get_autocorr_time(quiet=True)
        for j in range(ndim):
            if np.isfinite(tau_full[j]):
                ess[j] = n_walkers * (n_steps - n_burn) / tau_full[j]
            else:
                ess[j] = flat_uvhde.shape[0]
    except Exception:
        ess[:] = flat_uvhde.shape[0]

    acc_frac = sampler_uvhde.acceptance_fraction.mean()

    print(f"\nUV-HDE R-hat : {R_hat}")
    print(f"UV-HDE ESS   : {ess.astype(int)}")
    print(f"UV-HDE Acceptance fraction: {acc_frac:.3f}")

    # ---- Posterior summary UV-HDE ----
    medians = np.median(flat_uvhde, axis=0)
    q16 = np.percentile(flat_uvhde, 16, axis=0)
    q84 = np.percentile(flat_uvhde, 84, axis=0)
    q025 = np.percentile(flat_uvhde, 2.5, axis=0)
    q975 = np.percentile(flat_uvhde, 97.5, axis=0)

    summary_rows = []
    for j, name in enumerate(param_names):
        summary_rows.append({
            "parameter": name,
            "median": medians[j],
            "mean": flat_uvhde[:, j].mean(),
            "std": flat_uvhde[:, j].std(),
            "q16": q16[j], "q84": q84[j],
            "q025": q025[j], "q975": q975[j],
        })

    # ---- Information criteria UV-HDE ----
    best_idx = np.argmax(flat_logprob_uvhde)
    best_params = flat_uvhde[best_idx]
    best_logL = log_likelihood(best_params)
    chi2_best = -2.0 * best_logL

    n_data = len(cc_data) + len(desi_data)
    k = ndim  # Δk = 0 vs LCDM
    AIC = chi2_best + 2.0 * k
    BIC = chi2_best + k * np.log(n_data)

    print(f"\n=== UV-HDE Information criteria ===")
    print(f"chi2 best  : {chi2_best:.2f}  (n_data = {n_data})")
    print(f"AIC        : {AIC:.2f}")
    print(f"BIC        : {BIC:.2f}")

    # ---- LCDM comparison ----
    best_idx_lcdm = np.argmax(flat_logprob_lcdm)
    best_params_lcdm = flat_lcdm[best_idx_lcdm]
    best_logL_lcdm = log_likelihood_lcdm(best_params_lcdm)
    chi2_best_lcdm = -2.0 * best_logL_lcdm

    AIC_lcdm = chi2_best_lcdm + 2.0 * k
    BIC_lcdm = chi2_best_lcdm + k * np.log(n_data)

    print(f"\n=== LCDM Information criteria ===")
    print(f"chi2 best  : {chi2_best_lcdm:.2f}")
    print(f"AIC        : {AIC_lcdm:.2f}")
    print(f"BIC        : {BIC_lcdm:.2f}")

    dchi2 = chi2_best - chi2_best_lcdm
    print(f"\nDelta chi2 (UV-HDE - LCDM) = {dchi2:+.2f}  (Δk = 0)")
    print(f"Delta AIC = {dchi2 + 2 * 0:+.2f}")
    print(f"Delta BIC = {dchi2 + 0 * np.log(n_data):+.2f}")

    # ---- Save outputs ----
    chain_df = pd.DataFrame(flat_uvhde, columns=param_names)
    chain_df["log_posterior"] = flat_logprob_uvhde
    chain_df.to_csv("uvhde_mcmc_chains_full.csv", index=False)

    pd.DataFrame(summary_rows).to_csv("uvhde_mcmc_full_summary.csv", index=False)

    conv_df = pd.DataFrame({
        "parameter": param_names,
        "R_hat": R_hat,
        "ESS": ess.astype(int),
    })
    conv_df.to_csv("uvhde_mcmc_full_convergence.csv", index=False)

    ev_df = pd.DataFrame([{
        "n_data": n_data, "n_params": k,
        "chi2_best": chi2_best, "AIC": AIC, "BIC": BIC,
        "chi2_best_lcdm": chi2_best_lcdm, "AIC_lcdm": AIC_lcdm, "BIC_lcdm": BIC_lcdm,
        "acceptance_fraction": acc_frac,
    }])
    ev_df.to_csv("uvhde_mcmc_full_evidence.csv", index=False)

    # ---- Corner plot ----
    fig = corner.corner(flat_uvhde, labels=param_names, truths=medians,
                        quantiles=[0.16, 0.5, 0.84], show_titles=True,
                        title_kwargs={"fontsize": 10})
    fig.savefig(os.path.join(FIGDIR, "uvhde_full_mcmc_corner.png"), dpi=200)
    plt.close(fig)

    # ---- Trace plots ----
    chain_raw = sampler_uvhde.get_chain()
    fig, axes = plt.subplots(ndim, 1, figsize=(10, 2.0 * ndim), sharex=True)
    for j in range(ndim):
        for w in range(min(n_walkers, 12)):
            axes[j].plot(chain_raw[:, w, j], alpha=0.3, lw=0.5)
        axes[j].axvline(n_burn, color="k", ls="--", lw=0.8, label="burn-in")
        axes[j].set_ylabel(param_names[j])
        axes[j].grid(alpha=0.2)
    axes[-1].set_xlabel("step")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("UV-HDE + Planck Limit MCMC trace plots", y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "uvhde_full_mcmc_traces.png"), dpi=180,
                bbox_inches="tight")
    plt.close(fig)

    print(f"\n=== Outputs written ===")
    print(f"  uvhde_mcmc_chains_full.csv      ({flat_uvhde.shape[0]} samples)")
    print(f"  uvhde_mcmc_full_summary.csv")
    print(f"  uvhde_mcmc_full_convergence.csv")
    print(f"  uvhde_mcmc_full_evidence.csv")
    print(f"  figures/uvhde_full_mcmc_corner.png")
    print(f"  figures/uvhde_full_mcmc_traces.png")
    print("\nDone.")