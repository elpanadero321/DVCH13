#!/usr/bin/env python3
"""Corrida MCMC extendida para demostrar convergencia (R-hat < 1.01).

La configuración por defecto de ``dvch_full_mcmc_pipeline.py`` (24 walkers x
2000 pasos) produce R-hat > 1.01 en algunos parámetros.  Este script NO
modifica ese pipeline: importa su likelihood/posterior y corre una cadena más
larga con salida separada, para verificar si el criterio de convergencia del
repo (R-hat < 1.01, ESS > 100) se alcanza con más pasos.

Salidas:
  dvch_mcmc_extended_summary.csv
  dvch_mcmc_extended_convergence.csv
  figures/dvch_mcmc_extended_traces.png
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import emcee

import dvch_full_mcmc_pipeline as P

FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)

N_WALKERS = 32
N_STEPS = 8000
N_BURN = 2000
SEED = 20260909


def split_rhat(chains):
    """chains: (nsteps, nwalkers, ndim) -> split-R-hat por parámetro."""
    nsteps, nwalkers, ndim = chains.shape
    half = nsteps // 2
    m = nwalkers * 2
    split = np.zeros((m, half, ndim))
    for w in range(nwalkers):
        split[2 * w] = chains[:half, w, :]
        split[2 * w + 1] = chains[half:, w, :]
    B = (half / (m - 1)) * np.sum(
        (split.mean(axis=1) - split.mean(axis=(0, 1))) ** 2, axis=0)
    W = np.mean([np.var(c, axis=0, ddof=1) for c in split], axis=0)
    var_hat = (1.0 - 1.0 / half) * W + B / half
    return np.sqrt(var_hat / np.where(W > 0, W, 1e-30))


def main():
    ndim = 4
    names = ["Om", "n", "beta", "H0"]
    p0 = np.array([0.30, 0.09, 1.0e-4, 69.03])
    rng = np.random.default_rng(SEED)
    pos = p0 + 1e-3 * rng.standard_normal((N_WALKERS, ndim))
    for i in range(N_WALKERS):
        pos[i, 0] = np.clip(pos[i, 0], 0.06, 0.94)
        pos[i, 1] = np.clip(pos[i, 1], 0.002, 0.998)
        pos[i, 2] = np.clip(pos[i, 2], 1e-8, 0.99)
        pos[i, 3] = np.clip(pos[i, 3], 41.0, 119.0)

    print(f"Extended emcee run: {N_WALKERS} walkers x {N_STEPS} steps "
          f"(burn {N_BURN})")
    sampler = emcee.EnsembleSampler(N_WALKERS, ndim, P.log_posterior)
    sampler.run_mcmc(pos, N_STEPS, progress=False)

    chains = sampler.get_chain(discard=N_BURN)
    rhat = split_rhat(chains)

    try:
        tau = sampler.get_autocorr_time(quiet=True)
    except Exception:
        tau = np.full(ndim, np.nan)
    ess = np.where(np.isfinite(tau),
                   N_WALKERS * (N_STEPS - N_BURN) / np.where(np.isfinite(tau), tau, 1),
                   chains.shape[0] * N_WALKERS)

    flat = sampler.get_chain(discard=N_BURN, flat=True)
    med = np.median(flat, axis=0)
    q16 = np.percentile(flat, 16, axis=0)
    q84 = np.percentile(flat, 84, axis=0)
    lo = np.percentile(flat, 2.5, axis=0)
    hi = np.percentile(flat, 97.5, axis=0)

    print("\n=== Extended posterior ===")
    for j, nm in enumerate(names):
        print(f"  {nm:5s} = {med[j]:.4f} +{q84[j]-med[j]:.4f} -{med[j]-q16[j]:.4f}")
    print("\n=== Convergence ===")
    for j, nm in enumerate(names):
        print(f"  {nm:5s} R-hat={rhat[j]:.4f}  ESS={ess[j]:.0f}  tau={tau[j]:.1f}")
    print(f"  acceptance = {sampler.acceptance_fraction.mean():.3f}")

    pd.DataFrame({
        "parameter": names, "median": med,
        "q16": q16, "q84": q84,
    }).to_csv("dvch_mcmc_extended_summary.csv", index=False)
    pd.DataFrame({
        "parameter": names, "R_hat": rhat, "ESS": ess.astype(int), "tau": tau,
    }).to_csv("dvch_mcmc_extended_convergence.csv", index=False)
    print("Wrote dvch_mcmc_extended_summary.csv / dvch_mcmc_extended_convergence.csv")

    raw = sampler.get_chain()
    fig, axes = plt.subplots(ndim, 1, figsize=(10, 2.0 * ndim), sharex=True)
    for j in range(ndim):
        for w in range(min(N_WALKERS, 12)):
            axes[j].plot(raw[:, w, j], alpha=0.3, lw=0.5)
        axes[j].axvline(N_BURN, color="k", ls="--", lw=0.8)
        axes[j].axhline(med[j], color="C0", lw=1.0)
        axes[j].axhspan(lo[j], hi[j], color="C0", alpha=0.15)
        axes[j].set_ylabel(names[j])
        axes[j].grid(alpha=0.2)
    axes[-1].set_xlabel("step")
    fig.suptitle("DVCH extended MCMC traces", y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "dvch_mcmc_extended_traces.png"),
                dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {FIGDIR}/dvch_mcmc_extended_traces.png")

    ok = np.all(rhat < 1.01) and np.all(ess > 100)
    print(f"CONVERGENCE CRITERION (R-hat<1.01 & ESS>100): "
          f"{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
