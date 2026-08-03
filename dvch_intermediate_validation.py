#!/usr/bin/env python3
"""
Strong intermediate late-time validation for DVCH.

This script extends the efficient BAO+CC+H0 proof with the real Pantheon+
supernova compilation (without the z < 0.01 calibrator subset). It keeps the
runtime moderate by:

1. Fixing beta_DVCH during the MCMC stage.
2. Sampling only {Omega_m0, H0, n_DVCH}.
3. Evaluating beta dependence through a profile-likelihood scan.

Datasets used in the default run
--------------------------------
- DESI 2024 BAO compressed measurements
- SDSS DR12 BAO-only consensus measurements
- 6dF BAO
- Cosmic chronometers
- Pantheon+ (zcmb > 0.01 only, absolute magnitude marginalized analytically)
- Gaussian local H0 prior
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from dvch_mcmc_convergence import (
    BAOData,
    CC_DATA,
    C_KM_S,
    FitResult,
    chi2_bao_breakdown,
    chi2_bao_from_background,
    chi2_cc_from_background,
    chi2_h0_prior,
    effective_sample_size,
    fast_dvch_background,
    gelman_rubin,
    load_bao_data,
    make_redshift_grid,
    sound_horizon_eisenstein_hu,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d


FIGDIR = Path("figures")
FIGDIR.mkdir(exist_ok=True)

DEFAULT_SN_DIR = Path(os.environ.get("DVCH_PANTHEONPLUS_DIR", ""))
if str(DEFAULT_SN_DIR) == ".":
    DEFAULT_SN_DIR = Path(
        r"C:\Users\danie\.copilot\session-state\ae4c478e-bea7-43f4-86e8-e89badbfeff3\files\sn_data\PantheonPlus"
    )


@dataclass(frozen=True)
class PantheonPlusData:
    zcmb: np.ndarray
    zhel: np.ndarray
    mag: np.ndarray
    inv_cov: np.ndarray
    amarg_E: float
    n_sn: int
    source_dir: str


def load_pantheonplus_data(data_dir: Path) -> PantheonPlusData:
    data_file = data_dir / "Pantheon+SH0ES.dat"
    cov_file = data_dir / "Pantheon+SH0ES_STAT+SYS.cov"
    if not data_file.exists() or not cov_file.exists():
        raise FileNotFoundError(
            "Pantheon+ data not found. Expected "
            f"'{data_file}' and '{cov_file}'."
        )

    table = pd.read_csv(data_file, sep=r"\s+")
    mask = table["zCMB"].to_numpy() > 0.01

    zcmb = table.loc[mask, "zCMB"].to_numpy()
    zhel = table.loc[mask, "zHEL"].to_numpy()
    mag = table.loc[mask, "m_b_corr"].to_numpy()

    with open(cov_file, "r", encoding="utf-8") as handle:
        n_total = int(handle.readline().strip())
        values = np.fromiter(
            (float(line.strip()) for line in handle if line.strip()),
            dtype=float,
            count=n_total * n_total,
        )
    covariance = values.reshape((n_total, n_total))[np.ix_(mask, mask)]
    inv_cov = np.linalg.inv(covariance)
    amarg_E = float(np.sum(inv_cov))

    return PantheonPlusData(
        zcmb=zcmb,
        zhel=zhel,
        mag=mag,
        inv_cov=inv_cov,
        amarg_E=amarg_E,
        n_sn=len(zcmb),
        source_dir=str(data_dir),
    )


def make_full_redshift_grid(
    bao_data: BAOData,
    sn_data: PantheonPlusData,
    grid_points: int,
) -> np.ndarray:
    z_max = max(float(sn_data.zcmb.max()), float(bao_data.table["z"].max()), float(CC_DATA[:, 0].max()))
    return np.unique(
        np.concatenate(
            (
                [0.0],
                np.linspace(0.0, z_max, grid_points),
                bao_data.table["z"].values,
                CC_DATA[:, 0],
            )
        )
    )


def build_dvch_interpolators(
    omega_m0: float,
    H0: float,
    n_dvch: float,
    beta_fixed: float,
    z_grid: np.ndarray,
) -> tuple[interp1d, interp1d, float] | None:
    background = fast_dvch_background(omega_m0, H0, n_dvch, z_grid, beta_fixed=beta_fixed)
    if background is None:
        return None
    z_eval, E_eval = background
    Hz = H0 * E_eval
    Dc = cumulative_trapezoid(C_KM_S / Hz, z_eval, initial=0.0)
    rs_drag = sound_horizon_eisenstein_hu(omega_m0 * (H0 / 100.0) ** 2)
    if not np.isfinite(rs_drag) or rs_drag <= 0.0:
        return None
    return (
        interp1d(z_eval, Hz, kind="linear", fill_value="extrapolate"),
        interp1d(z_eval, Dc, kind="linear", fill_value="extrapolate"),
        rs_drag,
    )


def chi2_pantheonplus(
    D_interp,
    sn_data: PantheonPlusData,
) -> float:
    distances = np.asarray(D_interp(sn_data.zcmb), dtype=float)
    if np.any(distances <= 0.0) or np.any(~np.isfinite(distances)):
        return 1.0e30
    lumdists = 5.0 * np.log10((1.0 + sn_data.zhel) * distances)
    diff = sn_data.mag - lumdists
    weighted = sn_data.inv_cov.dot(diff)
    return float(diff.dot(weighted) - (np.sum(weighted) ** 2) / sn_data.amarg_E)


def evaluate_dvch_components(
    params: np.ndarray,
    beta_fixed: float,
    bao_data: BAOData,
    sn_data: PantheonPlusData,
    z_grid: np.ndarray,
) -> dict[str, object] | None:
    omega_m0, H0, n_dvch = params
    if not (0.15 < omega_m0 < 0.45 and 60.0 < H0 < 80.0 and 0.01 < n_dvch < 0.45):
        return None
    interpolators = build_dvch_interpolators(omega_m0, H0, n_dvch, beta_fixed, z_grid)
    if interpolators is None:
        return None
    H_interp, D_interp, rs_drag = interpolators
    chi2_bao = chi2_bao_from_background(H_interp, D_interp, rs_drag, bao_data)
    chi2_cc = chi2_cc_from_background(H_interp)
    chi2_h0 = chi2_h0_prior(H0)
    chi2_sn = chi2_pantheonplus(D_interp, sn_data)
    return {
        "chi2_bao": chi2_bao,
        "chi2_cc": chi2_cc,
        "chi2_h0": chi2_h0,
        "chi2_sn": chi2_sn,
        "chi2_total": chi2_bao + chi2_cc + chi2_h0 + chi2_sn,
        "H_interp": H_interp,
        "D_interp": D_interp,
        "rs_drag": rs_drag,
    }


def chi2_dvch_full(
    params: np.ndarray,
    beta_fixed: float,
    bao_data: BAOData,
    sn_data: PantheonPlusData,
    z_grid: np.ndarray,
) -> float:
    components = evaluate_dvch_components(params, beta_fixed, bao_data, sn_data, z_grid)
    return components["chi2_total"] if components is not None else 1.0e30


def lcdm_interpolators(
    omega_m0: float,
    H0: float,
    z_grid: np.ndarray,
) -> tuple[interp1d, interp1d, float] | None:
    h = H0 / 100.0
    omega_r0 = 2.47282e-5 * (1.0 + 0.22710731766 * 3.044) / h**2
    omega_lambda0 = 1.0 - omega_m0 - omega_r0
    if omega_lambda0 <= 0.0:
        return None
    E2 = omega_r0 * (1.0 + z_grid) ** 4 + omega_m0 * (1.0 + z_grid) ** 3 + omega_lambda0
    if np.any(E2 <= 0.0):
        return None
    Hz = H0 * np.sqrt(E2)
    Dc = cumulative_trapezoid(C_KM_S / Hz, z_grid, initial=0.0)
    rs_drag = sound_horizon_eisenstein_hu(omega_m0 * h * h)
    if not np.isfinite(rs_drag) or rs_drag <= 0.0:
        return None
    return (
        interp1d(z_grid, Hz, kind="linear", fill_value="extrapolate"),
        interp1d(z_grid, Dc, kind="linear", fill_value="extrapolate"),
        rs_drag,
    )


def evaluate_lcdm_components(
    params: np.ndarray,
    bao_data: BAOData,
    sn_data: PantheonPlusData,
    z_grid: np.ndarray,
) -> dict[str, object] | None:
    omega_m0, H0 = params
    if not (0.15 < omega_m0 < 0.45 and 60.0 < H0 < 80.0):
        return None
    interpolators = lcdm_interpolators(omega_m0, H0, z_grid)
    if interpolators is None:
        return None
    H_interp, D_interp, rs_drag = interpolators
    chi2_bao = chi2_bao_from_background(H_interp, D_interp, rs_drag, bao_data)
    chi2_cc = chi2_cc_from_background(H_interp)
    chi2_h0 = chi2_h0_prior(H0)
    chi2_sn = chi2_pantheonplus(D_interp, sn_data)
    return {
        "chi2_bao": chi2_bao,
        "chi2_cc": chi2_cc,
        "chi2_h0": chi2_h0,
        "chi2_sn": chi2_sn,
        "chi2_total": chi2_bao + chi2_cc + chi2_h0 + chi2_sn,
        "H_interp": H_interp,
        "D_interp": D_interp,
        "rs_drag": rs_drag,
    }


def chi2_lcdm_full(
    params: np.ndarray,
    bao_data: BAOData,
    sn_data: PantheonPlusData,
    z_grid: np.ndarray,
) -> float:
    components = evaluate_lcdm_components(params, bao_data, sn_data, z_grid)
    return components["chi2_total"] if components is not None else 1.0e30


def fit_dvch_full(
    beta_fixed: float,
    bao_data: BAOData,
    sn_data: PantheonPlusData,
    z_grid: np.ndarray,
) -> FitResult:
    starts = [
        np.array([0.34, 70.0, 0.10]),
        np.array([0.30, 69.0, 0.16]),
        np.array([0.37, 71.0, 0.08]),
    ]
    best = None
    for start in starts:
        result = minimize(
            lambda x: chi2_dvch_full(x, beta_fixed, bao_data, sn_data, z_grid),
            start,
            method="Nelder-Mead",
            options={"maxiter": 220, "xatol": 1.0e-3, "fatol": 1.0e-3},
        )
        if best is None or result.fun < best.fun:
            best = result
    return FitResult(chi2=float(best.fun), params=np.asarray(best.x, dtype=float))


def fit_lcdm_full(
    bao_data: BAOData,
    sn_data: PantheonPlusData,
    z_grid: np.ndarray,
) -> FitResult:
    starts = [
        np.array([0.31, 67.0]),
        np.array([0.29, 68.0]),
        np.array([0.33, 66.5]),
    ]
    best = None
    for start in starts:
        result = minimize(
            lambda x: chi2_lcdm_full(x, bao_data, sn_data, z_grid),
            start,
            method="Nelder-Mead",
            options={"maxiter": 220, "xatol": 1.0e-4, "fatol": 1.0e-4},
        )
        if best is None or result.fun < best.fun:
            best = result
    return FitResult(chi2=float(best.fun), params=np.asarray(best.x, dtype=float))


def metropolis_chain_generic(
    start: np.ndarray,
    proposal_cov: np.ndarray,
    n_steps: int,
    seed: int,
    chi2_fn,
) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    current = np.asarray(start, dtype=float)
    current_chi2 = chi2_fn(current)
    chain = []
    chi2_values = []
    accepted = 0

    for _ in range(n_steps):
        proposal = rng.multivariate_normal(current, proposal_cov)
        proposal_chi2 = chi2_fn(proposal)
        if np.isfinite(proposal_chi2) and (
            proposal_chi2 < current_chi2
            or rng.random() < np.exp(-0.5 * (proposal_chi2 - current_chi2))
        ):
            current = proposal
            current_chi2 = proposal_chi2
            accepted += 1
        chain.append(current.copy())
        chi2_values.append(current_chi2)

    return np.asarray(chain), np.asarray(chi2_values), accepted / n_steps


def run_adaptive_mcmc(
    center: np.ndarray,
    chi2_fn,
    pilot_cov: np.ndarray,
    offsets: np.ndarray,
    pilot_steps: int,
    pilot_burn: int,
    final_steps: int,
    final_burn: int,
    rminus1_target: float,
    max_extensions: int,
    extension_steps: int,
) -> dict[str, object]:
    pilot_chains = []
    acceptance_rows = []

    for index, offset in enumerate(offsets):
        chain, _, acceptance = metropolis_chain_generic(
            center + offset,
            pilot_cov,
            pilot_steps,
            seed=500 + index,
            chi2_fn=chi2_fn,
        )
        pilot_chains.append(chain)
        acceptance_rows.append({"chain": index, "phase": "pilot", "acceptance": acceptance})

    pooled = np.vstack([chain[pilot_burn:] for chain in pilot_chains])
    empirical_cov = np.cov(pooled.T)
    final_cov = (2.4**2 / pooled.shape[1]) * (empirical_cov + 1.0e-8 * np.eye(pooled.shape[1]))

    final_full = []
    final_trimmed = []
    final_chi2 = []
    current_states = center + 0.4 * offsets

    for index, state in enumerate(current_states):
        chain, chi2_values, acceptance = metropolis_chain_generic(
            state,
            final_cov,
            final_steps,
            seed=700 + index,
            chi2_fn=chi2_fn,
        )
        final_full.append(chain)
        final_trimmed.append(chain[final_burn:])
        final_chi2.append(chi2_values[final_burn:])
        acceptance_rows.append({"chain": index, "phase": "final", "acceptance": acceptance})

    max_rminus1 = float(np.max(gelman_rubin(final_trimmed) - 1.0))
    extension_count = 0
    while max_rminus1 > rminus1_target and extension_count < max_extensions:
        extension_count += 1
        new_trimmed = []
        new_chi2 = []
        for index, chain in enumerate(final_full):
            extra_chain, extra_chi2, acceptance = metropolis_chain_generic(
                chain[-1],
                final_cov,
                extension_steps,
                seed=900 + 10 * extension_count + index,
                chi2_fn=chi2_fn,
            )
            final_full[index] = np.vstack([final_full[index], extra_chain])
            final_chi2[index] = np.concatenate([final_chi2[index], extra_chi2])
            acceptance_rows.append(
                {"chain": index, "phase": f"extension_{extension_count}", "acceptance": acceptance}
            )
            new_trimmed.append(final_full[index][final_burn:])
            new_chi2.append(final_chi2[index])
        final_trimmed = new_trimmed
        final_chi2 = new_chi2
        max_rminus1 = float(np.max(gelman_rubin(final_trimmed) - 1.0))

    return {
        "chains": final_trimmed,
        "chi2_chains": final_chi2,
        "acceptance": acceptance_rows,
        "pilot_cov": pilot_cov,
        "final_cov": final_cov,
        "extensions": extension_count,
    }


def beta_profile_scan(
    beta_values: list[float],
    bao_data: BAOData,
    sn_data: PantheonPlusData,
    z_grid: np.ndarray,
) -> pd.DataFrame:
    rows = []
    previous = None
    for beta in beta_values:
        starts = [np.array([0.34, 70.0, 0.10])]
        if previous is not None:
            starts.insert(0, previous)
        best = None
        for start in starts:
            result = minimize(
                lambda x: chi2_dvch_full(x, beta, bao_data, sn_data, z_grid),
                start,
                method="Nelder-Mead",
                options={"maxiter": 180, "xatol": 1.0e-3, "fatol": 1.0e-3},
            )
            if best is None or result.fun < best.fun:
                best = result
        previous = np.asarray(best.x, dtype=float)
        rows.append(
            {
                "beta_fixed": beta,
                "log10_beta": np.log10(beta),
                "chi2": float(best.fun),
                "Omega_m0": previous[0],
                "H0": previous[1],
                "n_dvch": previous[2],
            }
        )
    return pd.DataFrame(rows)


def write_chain_outputs(chains: list[np.ndarray], chi2_chains: list[np.ndarray]) -> None:
    frames = []
    for chain_index, (chain, chi2_values) in enumerate(zip(chains, chi2_chains)):
        frame = pd.DataFrame(chain, columns=["Omega_m0", "H0", "n_dvch"])
        frame["chi2"] = chi2_values[: len(frame)]
        frame["chain"] = chain_index
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_csv("dvch_intermediate_chains.csv", index=False)


def make_figure(
    chains: list[np.ndarray],
    chi2_chains: list[np.ndarray],
    R_hat: np.ndarray,
    dvch_fit: FitResult,
    lcdm_fit: FitResult,
    beta_scan: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]

    ax1 = axes[0]
    for index, chain in enumerate(chains):
        ax1.plot(chain[:, 1], color=colors[index], alpha=0.75, lw=0.6, label=f"Chain {index + 1}")
    ax1.axhline(dvch_fit.params[1], color="black", ls="--", lw=1, label="DVCH best fit")
    ax1.axhline(lcdm_fit.params[1], color="gray", ls=":", lw=1, label=r"$\Lambda$CDM best fit")
    ax1.set_xlabel("Sample")
    ax1.set_ylabel(r"$H_0$ [km/s/Mpc]")
    ax1.set_title(f"Pantheon+ run traces (max R-1 = {np.max(R_hat - 1):.4f})")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    for index, values in enumerate(chi2_chains):
        ax2.plot(values, color=colors[index], alpha=0.75, lw=0.6, label=f"Chain {index + 1}")
    ax2.axhline(dvch_fit.chi2, color="black", ls="--", lw=1, label="DVCH best fit")
    ax2.axhline(lcdm_fit.chi2, color="gray", ls=":", lw=1, label=r"$\Lambda$CDM best fit")
    ax2.set_xlabel("Sample")
    ax2.set_ylabel(r"$\chi^2$")
    ax2.set_title(r"Late-time full $\chi^2$ traces")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    delta = beta_scan["chi2"] - beta_scan["chi2"].min()
    ax3.plot(beta_scan["log10_beta"], delta, "o-", lw=2, color="#4c72b0")
    ax3.axvline(np.log10(beta_scan.loc[beta_scan["chi2"].idxmin(), "beta_fixed"]), color="black", ls="--", lw=1)
    ax3.set_xlabel(r"$\log_{10}\beta_{\rm DVCH}$")
    ax3.set_ylabel(r"$\Delta\chi^2$")
    ax3.set_title(r"Profile scan in fixed $\beta_{\rm DVCH}$")
    ax3.grid(True, alpha=0.3)

    fig.suptitle(
        "DVCH strong intermediate validation: BAO + chronometers + Pantheon+ + local H0 prior",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(FIGDIR / "dvch_intermediate_validation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the strong intermediate DVCH late-time validation.")
    parser.add_argument("--pantheonplus-dir", default=str(DEFAULT_SN_DIR))
    parser.add_argument("--grid-points", type=int, default=260)
    parser.add_argument("--beta-fixed", type=float, default=1.0e-4)
    parser.add_argument("--pilot-steps", type=int, default=350)
    parser.add_argument("--pilot-burn", type=int, default=100)
    parser.add_argument("--final-steps", type=int, default=1100)
    parser.add_argument("--final-burn", type=int, default=200)
    parser.add_argument("--extension-steps", type=int, default=300)
    parser.add_argument("--max-extensions", type=int, default=2)
    parser.add_argument("--rminus1-target", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()

    bao_data = load_bao_data()
    sn_data = load_pantheonplus_data(Path(args.pantheonplus_dir))
    z_grid = make_full_redshift_grid(bao_data, sn_data, args.grid_points)

    print("=" * 76)
    print("DVCH strong intermediate validation")
    print("=" * 76)
    print("Data: DESI 2024 BAO + SDSS DR12 BAO + 6dF BAO + chronometers + Pantheon+ + local H0 prior")
    print(f"Pantheon+ source: {sn_data.source_dir}")
    print(f"Pantheon+ usable supernovae: {sn_data.n_sn}")
    print(f"Fixed beta for MCMC = {args.beta_fixed:.1e}")

    dvch_fit = fit_dvch_full(args.beta_fixed, bao_data, sn_data, z_grid)
    lcdm_fit = fit_lcdm_full(bao_data, sn_data, z_grid)
    print(
        f"DVCH best fit: Omega_m0={dvch_fit.params[0]:.4f}, "
        f"H0={dvch_fit.params[1]:.3f}, n={dvch_fit.params[2]:.4f}, chi2={dvch_fit.chi2:.4f}"
    )
    print(
        f"LCDM best fit: Omega_m0={lcdm_fit.params[0]:.4f}, "
        f"H0={lcdm_fit.params[1]:.3f}, chi2={lcdm_fit.chi2:.4f}"
    )

    chi2_fn = lambda x: chi2_dvch_full(x, args.beta_fixed, bao_data, sn_data, z_grid)
    offsets = np.asarray(
        [
            [0.006, 0.5, -0.020],
            [-0.008, -0.5, 0.018],
            [0.005, -0.4, 0.022],
            [-0.006, 0.4, -0.016],
        ]
    )
    pilot_cov = np.diag([0.0035**2, 0.30**2, 0.012**2])
    mcmc = run_adaptive_mcmc(
        center=dvch_fit.params,
        chi2_fn=chi2_fn,
        pilot_cov=pilot_cov,
        offsets=offsets,
        pilot_steps=args.pilot_steps,
        pilot_burn=args.pilot_burn,
        final_steps=args.final_steps,
        final_burn=args.final_burn,
        rminus1_target=args.rminus1_target,
        max_extensions=args.max_extensions,
        extension_steps=args.extension_steps,
    )

    chains = mcmc["chains"]
    chi2_chains = mcmc["chi2_chains"]
    R_hat = gelman_rubin(chains)
    ess = effective_sample_size(chains)

    beta_scan = beta_profile_scan(
        beta_values=[1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2],
        bao_data=bao_data,
        sn_data=sn_data,
        z_grid=z_grid,
    )

    samples = np.vstack(chains)
    dvch_components = evaluate_dvch_components(dvch_fit.params, args.beta_fixed, bao_data, sn_data, z_grid)
    lcdm_components = evaluate_lcdm_components(lcdm_fit.params, bao_data, sn_data, z_grid)
    dvch_bao = chi2_bao_breakdown(
        dvch_components["H_interp"],
        dvch_components["D_interp"],
        dvch_components["rs_drag"],
        bao_data,
    )
    lcdm_bao = chi2_bao_breakdown(
        lcdm_components["H_interp"],
        lcdm_components["D_interp"],
        lcdm_components["rs_drag"],
        bao_data,
    )

    elapsed = time.perf_counter() - t0
    delta_chi2 = dvch_fit.chi2 - lcdm_fit.chi2
    n_data = len(bao_data.table) + len(CC_DATA) + sn_data.n_sn + 1
    delta_aic = delta_chi2 + 2.0 * (3 - 2)
    delta_bic = delta_chi2 + np.log(n_data) * (3 - 2)

    print(f"max R-1 = {np.max(R_hat - 1.0):.5f}")
    print(f"min N_eff = {np.min(ess):.0f}")
    print(f"runtime = {elapsed:.1f} s")

    summary = pd.DataFrame(
        {
            "metric": [
                "chi2_DVCH",
                "chi2_LCDM",
                "delta_chi2",
                "delta_AIC",
                "delta_BIC",
                "chi2_DVCH_DESI",
                "chi2_DVCH_SDSS_DR12",
                "chi2_DVCH_6dF",
                "chi2_DVCH_CC",
                "chi2_DVCH_H0_prior",
                "chi2_DVCH_PantheonPlus",
                "chi2_LCDM_DESI",
                "chi2_LCDM_SDSS_DR12",
                "chi2_LCDM_6dF",
                "chi2_LCDM_CC",
                "chi2_LCDM_H0_prior",
                "chi2_LCDM_PantheonPlus",
                "bestfit_Omega_m0_DVCH",
                "bestfit_H0_DVCH",
                "bestfit_n_DVCH",
                "bestfit_Omega_m0_LCDM",
                "bestfit_H0_LCDM",
                "mean_Omega_m0",
                "mean_H0",
                "mean_n_DVCH",
                "std_Omega_m0",
                "std_H0",
                "std_n_DVCH",
                "max_R_minus_1",
                "min_N_eff",
                "n_chains",
                "beta_fixed_for_mcmc",
                "beta_scan_best",
                "beta_scan_best_log10",
                "runtime_seconds",
                "pantheonplus_nsn",
                "n_data",
            ],
            "value": [
                f"{dvch_fit.chi2:.6f}",
                f"{lcdm_fit.chi2:.6f}",
                f"{delta_chi2:.6f}",
                f"{delta_aic:.6f}",
                f"{delta_bic:.6f}",
                f"{dvch_bao['DESI_DR1']:.6f}",
                f"{dvch_bao['SDSS_DR12_BAO']:.6f}",
                f"{dvch_bao['6dF_BAO']:.6f}",
                f"{dvch_components['chi2_cc']:.6f}",
                f"{dvch_components['chi2_h0']:.6f}",
                f"{dvch_components['chi2_sn']:.6f}",
                f"{lcdm_bao['DESI_DR1']:.6f}",
                f"{lcdm_bao['SDSS_DR12_BAO']:.6f}",
                f"{lcdm_bao['6dF_BAO']:.6f}",
                f"{lcdm_components['chi2_cc']:.6f}",
                f"{lcdm_components['chi2_h0']:.6f}",
                f"{lcdm_components['chi2_sn']:.6f}",
                f"{dvch_fit.params[0]:.6f}",
                f"{dvch_fit.params[1]:.6f}",
                f"{dvch_fit.params[2]:.6f}",
                f"{lcdm_fit.params[0]:.6f}",
                f"{lcdm_fit.params[1]:.6f}",
                f"{np.mean(samples[:, 0]):.6f}",
                f"{np.mean(samples[:, 1]):.6f}",
                f"{np.mean(samples[:, 2]):.6f}",
                f"{np.std(samples[:, 0], ddof=1):.6f}",
                f"{np.std(samples[:, 1], ddof=1):.6f}",
                f"{np.std(samples[:, 2], ddof=1):.6f}",
                f"{np.max(R_hat - 1.0):.6f}",
                f"{np.min(ess):.2f}",
                str(len(chains)),
                f"{args.beta_fixed:.1e}",
                f"{beta_scan.loc[beta_scan['chi2'].idxmin(), 'beta_fixed']:.1e}",
                f"{beta_scan.loc[beta_scan['chi2'].idxmin(), 'log10_beta']:.6f}",
                f"{elapsed:.2f}",
                str(sn_data.n_sn),
                str(n_data),
            ],
        }
    )

    summary.to_csv("dvch_intermediate_validation_summary.csv", index=False)
    beta_scan.to_csv("dvch_intermediate_beta_scan.csv", index=False)
    pd.DataFrame(mcmc["acceptance"]).to_csv("dvch_intermediate_acceptance.csv", index=False)
    write_chain_outputs(chains, chi2_chains)
    make_figure(chains, chi2_chains, R_hat, dvch_fit, lcdm_fit, beta_scan)

    print("Wrote dvch_intermediate_validation_summary.csv")
    print("Wrote dvch_intermediate_beta_scan.csv")
    print("Wrote dvch_intermediate_acceptance.csv")
    print("Wrote dvch_intermediate_chains.csv")
    print(f"Figure saved: {FIGDIR / 'dvch_intermediate_validation.png'}")


if __name__ == "__main__":
    main()
