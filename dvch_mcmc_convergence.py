#!/usr/bin/env python3
"""
DVCH MCMC convergence diagnostic: runs 4 independent Metropolis chains for the
local DVCH BAO+chronometer likelihood and reports convergence diagnostics.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)

Omega_r0 = 9.0e-5
beta_fixed = 1.0e-4

# Cosmic chronometers
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

# DESI BAO (H*r_d and D_M/r_d)
desi_data = [
    (0.38, 81.5, "H_rd", 2.6), (0.38, 18.92, "DM_rd", 0.51),
    (0.51, 90.3, "H_rd", 3.4), (0.51, 20.73, "DM_rd", 0.50),
    (0.61, 97.3, "H_rd", 3.7), (0.61, 21.93, "DM_rd", 0.56),
    (0.70, 99.9, "H_rd", 4.2), (0.70, 16.85, "DM_rd", 0.32),
]
rd_fid = 147.09
c_speed = 299792.458


def E2_dvch(z, Om, H0, n):
    OL = 1.0 - Om - Omega_r0
    opz = 1.0 + z
    rad = Omega_r0 * opz**4
    Om_z = Om * opz**3
    E2 = rad + Om_z + OL
    for _ in range(30):
        OL_z = OL * (Om_z / Om)**n * (1.0 + beta_fixed) / (1.0 + beta_fixed * E2)
        E2_new = rad + Om_z + OL_z
        if abs(E2_new - E2) < 1e-8:
            break
        E2 = E2_new
    return E2


def chi2(params):
    Om, H0, n = params
    if Om <= 0 or Om >= 1 or H0 <= 0 or n <= 0 or n >= 1:
        return 1e10
    chi2_val = 0.0
    for z, H_obs, sigma in cc_data:
        H_th = H0 * np.sqrt(E2_dvch(z, Om, H0, n))
        chi2_val += ((H_th - H_obs) / sigma)**2
    from scipy.integrate import quad
    for z, val, obs_type, sigma in desi_data:
        E = np.sqrt(E2_dvch(z, Om, H0, n))
        H = H0 * E
        if obs_type == "H_rd":
            th = H * rd_fid / 1000.0
        else:
            integral, _ = quad(lambda zp: 1.0/np.sqrt(E2_dvch(zp, Om, H0, n)), 0, z, limit=100)
            th = c_speed / H0 * integral / rd_fid
        chi2_val += ((th - val) / sigma)**2
    return chi2_val


def metropolis_chain(x0, n_steps, seed):
    rng = np.random.default_rng(seed)
    x = np.array(x0, dtype=float)
    chi2_current = chi2(x)
    chain = [x.copy()]
    chi2_vals = [chi2_current]
    n_accept = 0
    step_sizes = np.array([0.01, 0.5, 0.02])

    for i in range(n_steps):
        x_prop = x + rng.normal(0, step_sizes)
        chi2_prop = chi2(x_prop)
        if chi2_prop < chi2_current:
            accept = True
        else:
            accept = rng.random() < np.exp(-0.5 * (chi2_prop - chi2_current))
        if accept:
            x = x_prop
            chi2_current = chi2_prop
            n_accept += 1
        chain.append(x.copy())
        chi2_vals.append(chi2_current)

    return np.array(chain), np.array(chi2_vals), n_accept / n_steps


def gelman_rubin(chains):
    n = chains[0].shape[0]
    m = len(chains)
    means = np.array([c.mean(axis=0) for c in chains])
    overall_mean = means.mean(axis=0)
    B = n / (m - 1) * np.sum((means - overall_mean)**2, axis=0)
    W = np.mean([np.var(c, axis=0, ddof=1) for c in chains], axis=0)
    var_hat = (1 - 1/n) * W + B / n
    R_hat = np.sqrt(var_hat / np.where(W > 0, W, 1e-10))
    return R_hat


def effective_sample_size(chains):
    ess_list = []
    for dim in range(chains[0].shape[1]):
        combined = np.concatenate([c[:, dim] for c in chains])
        acf = np.correlate(combined - combined.mean(), combined - combined.mean(), mode='full')
        acf = acf[len(acf)//2:]
        acf = acf / acf[0] if acf[0] > 0 else acf
        tau = 1.0
        for i in range(1, min(len(acf), 100)):
            if acf[i] < 0:
                break
            tau += 2 * acf[i]
        ess = len(combined) / tau
        ess_list.append(ess)
    return np.array(ess_list)


def main():
    print("=" * 60)
    print("DVCH MCMC Convergence Diagnostic")
    print("=" * 60)

    n_steps = 3000
    n_burn = 500
    n_chains = 4
    x0_list = [
        [0.30, 69.0, 0.20],
        [0.28, 71.0, 0.15],
        [0.32, 67.0, 0.25],
        [0.29, 70.0, 0.18],
    ]

    chains = []
    chi2_chains = []
    accept_rates = []

    for i in range(n_chains):
        print(f"Running chain {i+1}/{n_chains}...")
        chain, chi2_vals, acc = metropolis_chain(x0_list[i], n_steps, seed=i*100+42)
        chains.append(chain[n_burn:])
        chi2_chains.append(chi2_vals[n_burn:])
        accept_rates.append(acc)
        print(f"  acceptance: {acc:.3f}")

    R_hat = gelman_rubin(chains)
    ess = effective_sample_size(chains)
    print(f"max R_hat = {np.max(R_hat):.3f}")
    print(f"min N_eff = {np.min(ess):.0f}")

    # Write outputs
    all_chains = []
    for i, c in enumerate(chains):
        df_c = pd.DataFrame(c, columns=["Om", "H0", "n"])
        df_c["chain"] = i
        all_chains.append(df_c)
    pd.concat(all_chains).to_csv("dvch_mcmc_chains.csv", index=False)
    print("Wrote dvch_mcmc_chains.csv")

    summary = pd.DataFrame({
        "metric": ["max_R_hat", "min_N_eff", "n_chains", "n_steps", "burn_in",
                   "mean_Om", "mean_H0", "mean_n",
                   "std_Om", "std_H0", "std_n"],
        "value": [f"{np.max(R_hat):.3f}", f"{np.min(ess):.0f}",
                  str(n_chains), str(n_steps), str(n_burn),
                  f"{np.mean([c[:,0].mean() for c in chains]):.4f}",
                  f"{np.mean([c[:,1].mean() for c in chains]):.3f}",
                  f"{np.mean([c[:,2].mean() for c in chains]):.4f}",
                  f"{np.mean([c[:,0].std() for c in chains]):.4f}",
                  f"{np.mean([c[:,1].std() for c in chains]):.3f}",
                  f"{np.mean([c[:,2].std() for c in chains]):.4f}"],
    })
    summary.to_csv("dvch_mcmc_convergence_summary.csv", index=False)
    print("Wrote dvch_mcmc_convergence_summary.csv")

    pd.DataFrame({"chain": range(n_chains), "acceptance": accept_rates}).to_csv(
        "dvch_mcmc_acceptance.csv", index=False)
    print("Wrote dvch_mcmc_acceptance.csv")

    make_figure(chains, chi2_chains, R_hat, ess, n_burn)


def make_figure(chains, chi2_chains, R_hat, ess, n_burn):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = ['b', 'g', 'r', 'm']

    # Panel 1: Chain traces (H0)
    ax1 = axes[0]
    for i, c in enumerate(chains):
        ax1.plot(c[:, 1], color=colors[i], alpha=0.7, lw=0.5, label=f'Chain {i+1}')
    ax1.axvline(n_burn, color='k', ls='--', lw=1, label='Burn-in')
    ax1.set_xlabel('Step')
    ax1.set_ylabel(r'$H_0$')
    ax1.set_title(f'MCMC Traces (max $\\hat{{R}}$={np.max(R_hat):.3f})')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: chi2 traces
    ax2 = axes[1]
    for i, c in enumerate(chi2_chains):
        ax2.plot(c, color=colors[i], alpha=0.7, lw=0.5, label=f'Chain {i+1}')
    ax2.axvline(n_burn, color='k', ls='--', lw=1, label='Burn-in')
    ax2.set_xlabel('Step')
    ax2.set_ylabel(r'$\chi^2$')
    ax2.set_title(r'$\chi^2$ Traces')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Posterior scatter (Om vs n)
    ax3 = axes[2]
    for i, c in enumerate(chains):
        ax3.scatter(c[:, 0], c[:, 2], s=1, alpha=0.3, color=colors[i], label=f'Chain {i+1}')
    ax3.set_xlabel(r'$\Omega_{m0}$')
    ax3.set_ylabel('n')
    ax3.set_title(f'Posterior (min $N_{{\\rm eff}}$={np.min(ess):.0f})')
    ax3.legend(fontsize=8, markerscale=5)
    ax3.grid(True, alpha=0.3)

    fig.suptitle('DVCH MCMC Convergence Diagnostic (BAO + Chronometers)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f"{FIGDIR}/dvch_mcmc_convergence.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved: {FIGDIR}/dvch_mcmc_convergence.png")


if __name__ == "__main__":
    main()
