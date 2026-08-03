#!/usr/bin/env python3
"""
Efficient late-time DVCH proof with real data.

This script uses:
1. DESI 2024 BAO compressed measurements.
2. SDSS DR12 BAO-only consensus measurements.
3. The 6dF BAO point.
4. Cosmic chronometer H(z) measurements.
5. A Gaussian local H0 prior (SH0ES-style late-time anchor).

To keep the test cheap enough for a local workstation, the low-redshift
curvature-suppression parameter is fixed to beta = 1e-4 and the MCMC samples
only the parameters that the late-time data constrain meaningfully:

    theta_DVCH = {Omega_m0, H0, n_DVCH}.

The code first finds the best-fit DVCH and LCDM points, then runs an adaptive
four-chain Metropolis sampler for DVCH and reports Gelman-Rubin convergence.
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
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.linalg import block_diag

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FIGDIR = Path("figures")
FIGDIR.mkdir(exist_ok=True)

BAO_DATA_DIR = Path("data") / "bao"
DESI_MEAN_FILE = BAO_DATA_DIR / "desi_2024_gaussian_bao_ALL_GCcomb_mean.txt"
DESI_COV_FILE = BAO_DATA_DIR / "desi_2024_gaussian_bao_ALL_GCcomb_cov.txt"
SDSS_DR12_MEAN_FILE = BAO_DATA_DIR / "sdss_DR12Consensus_bao.dat"
SDSS_DR12_COV_FILE = BAO_DATA_DIR / "BAO_consensus_covtot_dM_Hz.txt"

OMEGA_B_H2_FIXED = 0.02237
TCMB = 2.7255
THETA_27 = TCMB / 2.7
N_EFF = 3.044
C_KM_S = 299792.458
H0_PRIOR_MEAN = 73.04
H0_PRIOR_SIGMA = 1.04
BETA_FIXED = 1.0e-4

CC_DATA = np.array(
    [
        [0.07, 69.0, 19.6],
        [0.10, 69.0, 12.0],
        [0.12, 68.6, 26.2],
        [0.17, 83.0, 8.0],
        [0.179, 75.0, 4.0],
        [0.199, 75.0, 5.0],
        [0.20, 72.9, 29.6],
        [0.27, 77.0, 14.0],
        [0.28, 88.8, 36.6],
        [0.352, 82.7, 8.4],
        [0.3802, 83.0, 13.5],
        [0.40, 95.0, 17.0],
        [0.4004, 77.0, 10.2],
        [0.4247, 87.1, 11.2],
        [0.44, 82.6, 7.8],
        [0.4497, 92.8, 12.9],
        [0.47, 89.0, 50.0],
        [0.4783, 80.9, 9.0],
        [0.48, 97.0, 62.0],
        [0.593, 68.6, 8.0],
        [0.6797, 92.0, 8.0],
        [0.7812, 105.0, 12.0],
        [0.8754, 125.0, 17.0],
        [0.88, 90.0, 40.0],
        [0.9, 117.0, 23.0],
        [1.037, 154.0, 20.0],
        [1.3, 168.0, 17.0],
        [1.363, 160.0, 33.6],
        [1.43, 177.0, 18.0],
        [1.53, 140.0, 14.0],
        [1.75, 202.0, 40.0],
        [1.965, 186.5, 50.4],
        [2.34, 222.0, 8.0],
    ],
    dtype=float,
)


@dataclass(frozen=True)
class BAOData:
    table: pd.DataFrame
    inv_cov: np.ndarray
    block_slices: dict[str, tuple[int, int]]


@dataclass(frozen=True)
class FitResult:
    chi2: float
    params: np.ndarray


def load_bao_data() -> BAOData:
    if (
        not DESI_MEAN_FILE.exists()
        or not DESI_COV_FILE.exists()
        or not SDSS_DR12_MEAN_FILE.exists()
        or not SDSS_DR12_COV_FILE.exists()
    ):
        raise FileNotFoundError(
            "BAO data files are missing. Expected DESI and SDSS DR12 inputs under "
            f"'{BAO_DATA_DIR}'."
        )

    desi_table = pd.read_csv(
        DESI_MEAN_FILE,
        sep=r"\s+",
        comment="#",
        header=None,
        names=["z", "value", "observable"],
    )
    desi_table["block"] = "DESI_DR1"
    desi_table["rs_scale_factor"] = 1.0
    desi_cov = np.loadtxt(DESI_COV_FILE)

    dr12_table = pd.read_csv(
        SDSS_DR12_MEAN_FILE,
        sep=r"\s+",
        comment="#",
        header=None,
        names=["z", "value", "observable"],
    )
    dr12_table["observable"] = dr12_table["observable"].str.replace("bao_", "", regex=False)
    dr12_table["block"] = "SDSS_DR12_BAO"
    dr12_table["rs_scale_factor"] = 147.78
    dr12_cov = np.loadtxt(SDSS_DR12_COV_FILE)

    sixdf_table = pd.DataFrame(
        {
            "z": [0.106],
            "value": [0.336],
            "observable": ["rs_over_DV"],
            "block": ["6dF_BAO"],
            "rs_scale_factor": [153.9 / 149.8],
        }
    )
    sixdf_cov = np.array([[0.015**2]])

    table = pd.concat([desi_table, dr12_table, sixdf_table], ignore_index=True)
    cov = block_diag(desi_cov, dr12_cov, sixdf_cov)
    inv_cov = np.linalg.inv(cov)

    block_slices = {}
    start = 0
    for name, size in (
        ("DESI_DR1", len(desi_table)),
        ("SDSS_DR12_BAO", len(dr12_table)),
        ("6dF_BAO", len(sixdf_table)),
    ):
        block_slices[name] = (start, start + size)
        start += size
    return BAOData(table=table, inv_cov=inv_cov, block_slices=block_slices)


def sound_horizon_eisenstein_hu(
    omega_m_h2: float,
    omega_b_h2: float = OMEGA_B_H2_FIXED,
) -> float:
    if omega_m_h2 <= omega_b_h2 or omega_b_h2 <= 0.0:
        return np.nan
    z_eq = 2.50e4 * omega_m_h2 / THETA_27**4
    k_eq = 7.46e-2 * omega_m_h2 / THETA_27**2
    b1 = 0.313 * omega_m_h2 ** (-0.419) * (1.0 + 0.607 * omega_m_h2**0.674)
    b2 = 0.238 * omega_m_h2**0.223
    z_drag = (
        1291.0
        * omega_m_h2**0.251
        / (1.0 + 0.659 * omega_m_h2**0.828)
        * (1.0 + b1 * omega_b_h2**b2)
    )
    R_eq = 31.5 * omega_b_h2 / THETA_27**4 * (1000.0 / z_eq)
    R_drag = 31.5 * omega_b_h2 / THETA_27**4 * (1000.0 / z_drag)
    return (2.0 / (3.0 * k_eq)) * np.sqrt(6.0 / R_eq) * np.log(
        (np.sqrt(1.0 + R_drag) + np.sqrt(R_drag + R_eq)) / (1.0 + np.sqrt(R_eq))
    )


def radiation_density_today(h: float) -> float:
    return 2.47282e-5 * (1.0 + 0.22710731766 * N_EFF) / h**2


def make_redshift_grid(bao_data: BAOData, grid_points: int) -> np.ndarray:
    z_obs = np.unique(np.concatenate((CC_DATA[:, 0], bao_data.table["z"].values)))
    return np.unique(np.concatenate(([0.0], np.linspace(0.0, z_obs.max(), grid_points), z_obs)))


def fast_dvch_background(
    omega_m0: float,
    H0: float,
    n_dvch: float,
    z_grid: np.ndarray,
    beta_fixed: float = BETA_FIXED,
) -> tuple[np.ndarray, np.ndarray] | None:
    h = H0 / 100.0
    omega_m_h2 = omega_m0 * h * h
    omega_cdm_h2 = omega_m_h2 - OMEGA_B_H2_FIXED
    if omega_cdm_h2 <= 0.0:
        return None

    omega_r0 = radiation_density_today(h)
    omega_lambda0 = 1.0 - omega_m0 - omega_r0
    if omega_lambda0 <= 0.0:
        return None

    omega_m = np.empty_like(z_grid)
    E = np.empty_like(z_grid)
    omega_m[0] = omega_m0
    E[0] = 1.0
    previous_E2 = 1.0

    def q_and_e2(z: float, omega_m_z: float, guess_E2: float) -> tuple[float | None, float | None]:
        if omega_m_z <= 0.0:
            return None, None
        one_plus_z = 1.0 + z
        omega_r_z = omega_r0 * one_plus_z**4
        ratio = (omega_m_z / omega_m0) ** n_dvch
        E2 = max(guess_E2, omega_r_z + omega_m_z + 1.0e-8)
        for _ in range(15):
            omega_lambda_z = (
                omega_lambda0
                * ratio
                * (1.0 + beta_fixed)
                / (1.0 + beta_fixed * E2)
            )
            new_E2 = omega_r_z + omega_m_z + omega_lambda_z
            if abs(new_E2 - E2) < 1.0e-11:
                break
            E2 = 0.65 * E2 + 0.35 * new_E2
        omega_lambda_z = (
            omega_lambda0
            * ratio
            * (1.0 + beta_fixed)
            / (1.0 + beta_fixed * E2)
        )
        E2 = omega_r_z + omega_m_z + omega_lambda_z
        if E2 <= 0.0 or omega_lambda_z <= 0.0:
            return None, None
        E_value = np.sqrt(E2)
        qtilde = -E_value * omega_lambda_z / (1.0 + n_dvch * omega_lambda_z / omega_m_z) * (
            n_dvch
            - beta_fixed * (4.0 * omega_r_z + 3.0 * omega_m_z) / (3.0 * (1.0 + beta_fixed * E2))
        )
        return qtilde, E2

    def derivative(z: float, omega_m_z: float, guess_E2: float) -> tuple[float | None, float | None]:
        qtilde, E2 = q_and_e2(z, omega_m_z, guess_E2)
        if qtilde is None or E2 is None:
            return None, None
        return 3.0 * (omega_m_z + qtilde / np.sqrt(E2)) / (1.0 + z), E2

    for idx in range(1, len(z_grid)):
        z0 = z_grid[idx - 1]
        z1 = z_grid[idx]
        step = z1 - z0

        k1, E2_1 = derivative(z0, omega_m[idx - 1], previous_E2)
        if k1 is None or E2_1 is None:
            return None
        k2, E2_2 = derivative(z0 + 0.5 * step, omega_m[idx - 1] + 0.5 * step * k1, E2_1)
        if k2 is None or E2_2 is None:
            return None
        k3, E2_3 = derivative(z0 + 0.5 * step, omega_m[idx - 1] + 0.5 * step * k2, E2_2)
        if k3 is None or E2_3 is None:
            return None
        k4, E2_4 = derivative(z1, omega_m[idx - 1] + step * k3, E2_3)
        if k4 is None or E2_4 is None:
            return None

        omega_m[idx] = omega_m[idx - 1] + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        _, previous_E2 = q_and_e2(z1, omega_m[idx], E2_4)
        if previous_E2 is None:
            return None
        E[idx] = np.sqrt(previous_E2)

    return z_grid, E


def lcdm_background(
    omega_m0: float,
    H0: float,
    z_grid: np.ndarray,
) -> np.ndarray | None:
    h = H0 / 100.0
    omega_m_h2 = omega_m0 * h * h
    if omega_m_h2 <= OMEGA_B_H2_FIXED:
        return None
    omega_r0 = radiation_density_today(h)
    omega_lambda0 = 1.0 - omega_m0 - omega_r0
    if omega_lambda0 <= 0.0:
        return None
    E2 = omega_r0 * (1.0 + z_grid) ** 4 + omega_m0 * (1.0 + z_grid) ** 3 + omega_lambda0
    if np.any(E2 <= 0.0):
        return None
    return np.sqrt(E2)


def chi2_bao_from_background(
    H_interp,
    D_interp,
    rs_drag: float,
    bao_data: BAOData,
) -> float:
    theory = []
    for row in bao_data.table.itertuples(index=False):
        z = float(row.z)
        DM = float(D_interp(z))
        H = float(H_interp(z))
        DH = C_KM_S / H
        rs_scale = float(row.rs_scale_factor)
        if row.observable == "DV_over_rs":
            theory.append((z * DM * DM * DH) ** (1.0 / 3.0) * rs_scale / rs_drag)
        elif row.observable == "DM_over_rs":
            theory.append(DM * rs_scale / rs_drag)
        elif row.observable == "DH_over_rs":
            theory.append(DH * rs_scale / rs_drag)
        elif row.observable == "Hz_rs":
            theory.append(H * rs_drag / rs_scale)
        elif row.observable == "rs_over_DV":
            theory.append(rs_drag * rs_scale / ((z * DM * DM * DH) ** (1.0 / 3.0)))
        else:
            raise ValueError(f"Unsupported BAO observable: {row.observable}")
    delta = np.asarray(theory) - bao_data.table["value"].values
    return float(delta @ bao_data.inv_cov @ delta)


def chi2_bao_breakdown(
    H_interp,
    D_interp,
    rs_drag: float,
    bao_data: BAOData,
) -> dict[str, float]:
    theory = []
    for row in bao_data.table.itertuples(index=False):
        z = float(row.z)
        DM = float(D_interp(z))
        H = float(H_interp(z))
        DH = C_KM_S / H
        rs_scale = float(row.rs_scale_factor)
        if row.observable == "DV_over_rs":
            theory.append((z * DM * DM * DH) ** (1.0 / 3.0) * rs_scale / rs_drag)
        elif row.observable == "DM_over_rs":
            theory.append(DM * rs_scale / rs_drag)
        elif row.observable == "DH_over_rs":
            theory.append(DH * rs_scale / rs_drag)
        elif row.observable == "Hz_rs":
            theory.append(H * rs_drag / rs_scale)
        elif row.observable == "rs_over_DV":
            theory.append(rs_drag * rs_scale / ((z * DM * DM * DH) ** (1.0 / 3.0)))
        else:
            raise ValueError(f"Unsupported BAO observable: {row.observable}")

    theory = np.asarray(theory)
    observed = bao_data.table["value"].values
    breakdown = {}
    for block_name, (start, stop) in bao_data.block_slices.items():
        inv_cov_block = bao_data.inv_cov[start:stop, start:stop]
        delta = theory[start:stop] - observed[start:stop]
        breakdown[block_name] = float(delta @ inv_cov_block @ delta)
    return breakdown


def chi2_cc_from_background(H_interp) -> float:
    residuals = (np.array([float(H_interp(z)) for z in CC_DATA[:, 0]]) - CC_DATA[:, 1]) / CC_DATA[:, 2]
    return float(np.sum(residuals**2))


def chi2_h0_prior(H0: float) -> float:
    return ((H0 - H0_PRIOR_MEAN) / H0_PRIOR_SIGMA) ** 2


def chi2_dvch(params: np.ndarray, bao_data: BAOData, z_grid: np.ndarray) -> float:
    omega_m0, H0, n_dvch = params
    if not (0.15 < omega_m0 < 0.45 and 60.0 < H0 < 80.0 and 0.01 < n_dvch < 0.45):
        return 1.0e30
    background = fast_dvch_background(omega_m0, H0, n_dvch, z_grid)
    if background is None:
        return 1.0e30
    z_eval, E_eval = background
    Hz = H0 * E_eval
    Dc = cumulative_trapezoid(C_KM_S / Hz, z_eval, initial=0.0)
    H_interp = interp1d(z_eval, Hz, kind="linear", fill_value="extrapolate")
    D_interp = interp1d(z_eval, Dc, kind="linear", fill_value="extrapolate")
    rs_drag = sound_horizon_eisenstein_hu(omega_m0 * (H0 / 100.0) ** 2)
    if not np.isfinite(rs_drag) or rs_drag <= 0.0:
        return 1.0e30
    return (
        chi2_bao_from_background(H_interp, D_interp, rs_drag, bao_data)
        + chi2_cc_from_background(H_interp)
        + chi2_h0_prior(H0)
    )


def chi2_lcdm(params: np.ndarray, bao_data: BAOData, z_grid: np.ndarray) -> float:
    omega_m0, H0 = params
    if not (0.15 < omega_m0 < 0.45 and 60.0 < H0 < 80.0):
        return 1.0e30
    E_eval = lcdm_background(omega_m0, H0, z_grid)
    if E_eval is None:
        return 1.0e30
    Hz = H0 * E_eval
    Dc = cumulative_trapezoid(C_KM_S / Hz, z_grid, initial=0.0)
    H_interp = interp1d(z_grid, Hz, kind="linear", fill_value="extrapolate")
    D_interp = interp1d(z_grid, Dc, kind="linear", fill_value="extrapolate")
    rs_drag = sound_horizon_eisenstein_hu(omega_m0 * (H0 / 100.0) ** 2)
    if not np.isfinite(rs_drag) or rs_drag <= 0.0:
        return 1.0e30
    return (
        chi2_bao_from_background(H_interp, D_interp, rs_drag, bao_data)
        + chi2_cc_from_background(H_interp)
        + chi2_h0_prior(H0)
    )


def build_background_interpolators_dvch(
    params: np.ndarray,
    z_grid: np.ndarray,
) -> tuple[interp1d, interp1d, float]:
    omega_m0, H0, n_dvch = params
    background = fast_dvch_background(omega_m0, H0, n_dvch, z_grid)
    if background is None:
        raise RuntimeError("DVCH background construction failed for the requested parameters.")
    z_eval, E_eval = background
    Hz = H0 * E_eval
    Dc = cumulative_trapezoid(C_KM_S / Hz, z_eval, initial=0.0)
    rs_drag = sound_horizon_eisenstein_hu(omega_m0 * (H0 / 100.0) ** 2)
    return (
        interp1d(z_eval, Hz, kind="linear", fill_value="extrapolate"),
        interp1d(z_eval, Dc, kind="linear", fill_value="extrapolate"),
        rs_drag,
    )


def build_background_interpolators_lcdm(
    params: np.ndarray,
    z_grid: np.ndarray,
) -> tuple[interp1d, interp1d, float]:
    omega_m0, H0 = params
    E_eval = lcdm_background(omega_m0, H0, z_grid)
    if E_eval is None:
        raise RuntimeError("LCDM background construction failed for the requested parameters.")
    Hz = H0 * E_eval
    Dc = cumulative_trapezoid(C_KM_S / Hz, z_grid, initial=0.0)
    rs_drag = sound_horizon_eisenstein_hu(omega_m0 * (H0 / 100.0) ** 2)
    return (
        interp1d(z_grid, Hz, kind="linear", fill_value="extrapolate"),
        interp1d(z_grid, Dc, kind="linear", fill_value="extrapolate"),
        rs_drag,
    )


def fit_dvch(bao_data: BAOData, z_grid: np.ndarray) -> FitResult:
    starts = [
        np.array([0.30, 69.0, 0.18]),
        np.array([0.26, 72.0, 0.10]),
        np.array([0.33, 67.5, 0.25]),
    ]
    best = None
    for start in starts:
        result = minimize(
            lambda x: chi2_dvch(x, bao_data, z_grid),
            start,
            method="Nelder-Mead",
            options={"maxiter": 250, "xatol": 1.0e-3, "fatol": 1.0e-3},
        )
        if best is None or result.fun < best.fun:
            best = result
    return FitResult(chi2=float(best.fun), params=np.asarray(best.x, dtype=float))


def fit_lcdm(bao_data: BAOData, z_grid: np.ndarray) -> FitResult:
    starts = [
        np.array([0.30, 69.0]),
        np.array([0.33, 70.0]),
        np.array([0.28, 72.0]),
    ]
    best = None
    for start in starts:
        result = minimize(
            lambda x: chi2_lcdm(x, bao_data, z_grid),
            start,
            method="Nelder-Mead",
            options={"maxiter": 220, "xatol": 1.0e-4, "fatol": 1.0e-4},
        )
        if best is None or result.fun < best.fun:
            best = result
    return FitResult(chi2=float(best.fun), params=np.asarray(best.x, dtype=float))


def metropolis_chain(
    start: np.ndarray,
    proposal_cov: np.ndarray,
    n_steps: int,
    seed: int,
    bao_data: BAOData,
    z_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    current = np.asarray(start, dtype=float)
    current_chi2 = chi2_dvch(current, bao_data, z_grid)
    chain = []
    chi2_values = []
    accepted = 0

    for _ in range(n_steps):
        proposal = rng.multivariate_normal(current, proposal_cov)
        proposal_chi2 = chi2_dvch(proposal, bao_data, z_grid)
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


def gelman_rubin(chains: list[np.ndarray]) -> np.ndarray:
    n = min(chain.shape[0] for chain in chains)
    array = np.asarray([chain[:n] for chain in chains])
    m = array.shape[0]
    means = array.mean(axis=1)
    overall_mean = means.mean(axis=0)
    B = n * np.sum((means - overall_mean) ** 2, axis=0) / (m - 1)
    W = np.mean(np.var(array, axis=1, ddof=1), axis=0)
    var_hat = ((n - 1.0) / n) * W + B / n
    return np.sqrt(var_hat / W)


def effective_sample_size(chains: list[np.ndarray]) -> np.ndarray:
    ess_values = []
    for dim in range(chains[0].shape[1]):
        series = np.concatenate([chain[:, dim] for chain in chains])
        centered = series - np.mean(series)
        acf = np.correlate(centered, centered, mode="full")
        acf = acf[len(acf) // 2 :]
        acf = acf / acf[0]
        tau = 1.0
        for idx in range(1, min(len(acf), 200)):
            if acf[idx] < 0:
                break
            tau += 2.0 * acf[idx]
        ess_values.append(len(series) / tau)
    return np.asarray(ess_values)


def run_adaptive_dvch_mcmc(
    center: np.ndarray,
    bao_data: BAOData,
    z_grid: np.ndarray,
    pilot_steps: int,
    pilot_burn: int,
    final_steps: int,
    final_burn: int,
    rminus1_target: float,
    max_extensions: int,
    extension_steps: int,
) -> dict[str, object]:
    offsets = np.asarray(
        [
            [0.008, 0.7, -0.025],
            [-0.010, -0.7, 0.020],
            [0.006, -0.5, 0.028],
            [-0.007, 0.5, -0.018],
        ]
    )
    pilot_cov = np.diag([0.004**2, 0.35**2, 0.015**2])
    pilot_chains = []
    acceptance_rows = []

    for index, offset in enumerate(offsets):
        chain, chi2_values, acceptance = metropolis_chain(
            center + offset,
            pilot_cov,
            pilot_steps,
            seed=100 + index,
            bao_data=bao_data,
            z_grid=z_grid,
        )
        pilot_chains.append(chain)
        acceptance_rows.append(
            {"chain": index, "phase": "pilot", "acceptance": acceptance}
        )

    pooled = np.vstack([chain[pilot_burn:] for chain in pilot_chains])
    empirical_cov = np.cov(pooled.T)
    final_cov = (2.4**2 / pooled.shape[1]) * (empirical_cov + 1.0e-8 * np.eye(pooled.shape[1]))

    final_full = []
    final_trimmed = []
    final_chi2 = []
    current_states = center + 0.4 * offsets

    for index, state in enumerate(current_states):
        chain, chi2_values, acceptance = metropolis_chain(
            state,
            final_cov,
            final_steps,
            seed=200 + index,
            bao_data=bao_data,
            z_grid=z_grid,
        )
        final_full.append(chain)
        final_trimmed.append(chain[final_burn:])
        final_chi2.append(chi2_values[final_burn:])
        acceptance_rows.append(
            {"chain": index, "phase": "final", "acceptance": acceptance}
        )

    max_rminus1 = float(np.max(gelman_rubin(final_trimmed) - 1.0))
    extension_count = 0

    while max_rminus1 > rminus1_target and extension_count < max_extensions:
        extension_count += 1
        new_trimmed = []
        new_chi2 = []
        for index, chain in enumerate(final_full):
            last_state = chain[-1]
            extra_chain, extra_chi2, acceptance = metropolis_chain(
                last_state,
                final_cov,
                extension_steps,
                seed=300 + 10 * extension_count + index,
                bao_data=bao_data,
                z_grid=z_grid,
            )
            final_full[index] = np.vstack([final_full[index], extra_chain])
            final_chi2[index] = np.concatenate([final_chi2[index], extra_chi2])
            acceptance_rows.append(
                {
                    "chain": index,
                    "phase": f"extension_{extension_count}",
                    "acceptance": acceptance,
                }
            )
            new_trimmed.append(final_full[index][final_burn:])
            new_chi2.append(final_chi2[index])
        final_trimmed = new_trimmed
        max_rminus1 = float(np.max(gelman_rubin(final_trimmed) - 1.0))

    return {
        "pilot_cov": pilot_cov,
        "final_cov": final_cov,
        "pilot_chains": pilot_chains,
        "chains": final_trimmed,
        "full_chains": final_full,
        "chi2_chains": final_chi2,
        "acceptance": acceptance_rows,
        "extensions": extension_count,
    }


def write_chain_outputs(chains: list[np.ndarray], chi2_chains: list[np.ndarray]) -> None:
    frames = []
    for chain_index, (chain, chi2_values) in enumerate(zip(chains, chi2_chains)):
        frame = pd.DataFrame(chain, columns=["Omega_m0", "H0", "n_dvch"])
        frame["chi2"] = chi2_values[: len(frame)]
        frame["chain"] = chain_index
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_csv("dvch_mcmc_chains.csv", index=False)


def make_figure(
    chains: list[np.ndarray],
    chi2_chains: list[np.ndarray],
    R_hat: np.ndarray,
    ess: np.ndarray,
    dvch_fit: FitResult,
    lcdm_fit: FitResult,
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
    ax1.set_title(f"Real-data DVCH traces (max R-1 = {np.max(R_hat - 1):.4f})")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    for index, values in enumerate(chi2_chains):
        ax2.plot(values, color=colors[index], alpha=0.75, lw=0.6, label=f"Chain {index + 1}")
    ax2.axhline(dvch_fit.chi2, color="black", ls="--", lw=1, label="DVCH best fit")
    ax2.axhline(lcdm_fit.chi2, color="gray", ls=":", lw=1, label=r"$\Lambda$CDM best fit")
    ax2.set_xlabel("Sample")
    ax2.set_ylabel(r"$\chi^2$")
    ax2.set_title(r"Late-time $\chi^2$ traces")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    for index, chain in enumerate(chains):
        ax3.scatter(chain[:, 1], chain[:, 2], s=3, alpha=0.20, color=colors[index], label=f"Chain {index + 1}")
    ax3.plot(dvch_fit.params[1], dvch_fit.params[2], "k*", ms=12, label="DVCH best fit")
    ax3.axvline(lcdm_fit.params[1], color="gray", ls=":", lw=1, label=r"$\Lambda$CDM $H_0$")
    ax3.set_xlabel(r"$H_0$ [km/s/Mpc]")
    ax3.set_ylabel(r"$n_{\rm DVCH}$")
    ax3.set_title(f"Posterior cloud (min $N_{{\\rm eff}}$={np.min(ess):.0f})")
    ax3.legend(fontsize=8, markerscale=3)
    ax3.grid(True, alpha=0.3)

    fig.suptitle(
        "DVCH efficient real-data proof: DESI + SDSS DR12 + 6dF BAO + chronometers + local H0 prior",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(FIGDIR / "dvch_mcmc_convergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the efficient real-data DVCH late-time proof.")
    parser.add_argument("--grid-points", type=int, default=120)
    parser.add_argument("--pilot-steps", type=int, default=700)
    parser.add_argument("--pilot-burn", type=int, default=200)
    parser.add_argument("--final-steps", type=int, default=1700)
    parser.add_argument("--final-burn", type=int, default=300)
    parser.add_argument("--extension-steps", type=int, default=400)
    parser.add_argument("--max-extensions", type=int, default=2)
    parser.add_argument("--rminus1-target", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()

    bao_data = load_bao_data()
    z_grid = make_redshift_grid(bao_data, args.grid_points)

    print("=" * 68)
    print("DVCH efficient real-data proof")
    print("=" * 68)
    print("Data: DESI 2024 BAO + SDSS DR12 BAO + 6dF BAO + cosmic chronometers + local H0 prior")
    print(f"Fixed beta = {BETA_FIXED:.1e}")

    dvch_fit = fit_dvch(bao_data, z_grid)
    lcdm_fit = fit_lcdm(bao_data, z_grid)
    print(
        f"DVCH best fit: Omega_m0={dvch_fit.params[0]:.4f}, "
        f"H0={dvch_fit.params[1]:.3f}, n={dvch_fit.params[2]:.4f}, "
        f"chi2={dvch_fit.chi2:.4f}"
    )
    print(
        f"LCDM best fit: Omega_m0={lcdm_fit.params[0]:.4f}, "
        f"H0={lcdm_fit.params[1]:.3f}, chi2={lcdm_fit.chi2:.4f}"
    )

    mcmc = run_adaptive_dvch_mcmc(
        center=dvch_fit.params,
        bao_data=bao_data,
        z_grid=z_grid,
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
    elapsed = time.perf_counter() - t0

    print(f"max R-1 = {np.max(R_hat - 1.0):.5f}")
    print(f"min N_eff = {np.min(ess):.0f}")
    print(f"runtime = {elapsed:.1f} s")

    write_chain_outputs(chains, chi2_chains)
    pd.DataFrame(mcmc["acceptance"]).to_csv("dvch_mcmc_acceptance.csv", index=False)

    samples = np.vstack(chains)
    delta_chi2 = dvch_fit.chi2 - lcdm_fit.chi2
    dvch_H_interp, dvch_D_interp, dvch_rs = build_background_interpolators_dvch(dvch_fit.params, z_grid)
    lcdm_H_interp, lcdm_D_interp, lcdm_rs = build_background_interpolators_lcdm(lcdm_fit.params, z_grid)
    dvch_bao = chi2_bao_breakdown(dvch_H_interp, dvch_D_interp, dvch_rs, bao_data)
    lcdm_bao = chi2_bao_breakdown(lcdm_H_interp, lcdm_D_interp, lcdm_rs, bao_data)
    dvch_cc = chi2_cc_from_background(dvch_H_interp)
    lcdm_cc = chi2_cc_from_background(lcdm_H_interp)
    dvch_h0 = chi2_h0_prior(dvch_fit.params[1])
    lcdm_h0 = chi2_h0_prior(lcdm_fit.params[1])
    n_data = len(CC_DATA) + len(bao_data.table) + 1
    delta_aic = delta_chi2 + 2.0 * (3 - 2)
    delta_bic = delta_chi2 + np.log(n_data) * (3 - 2)

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
                "chi2_LCDM_DESI",
                "chi2_LCDM_SDSS_DR12",
                "chi2_LCDM_6dF",
                "chi2_LCDM_CC",
                "chi2_LCDM_H0_prior",
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
                "pilot_steps",
                "final_steps",
                "extensions_used",
                "beta_fixed",
                "runtime_seconds",
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
                f"{dvch_cc:.6f}",
                f"{dvch_h0:.6f}",
                f"{lcdm_bao['DESI_DR1']:.6f}",
                f"{lcdm_bao['SDSS_DR12_BAO']:.6f}",
                f"{lcdm_bao['6dF_BAO']:.6f}",
                f"{lcdm_cc:.6f}",
                f"{lcdm_h0:.6f}",
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
                str(args.pilot_steps),
                str(args.final_steps),
                str(mcmc["extensions"]),
                f"{BETA_FIXED:.1e}",
                f"{elapsed:.2f}",
                str(n_data),
            ],
        }
    )
    summary.to_csv("dvch_mcmc_convergence_summary.csv", index=False)

    make_figure(chains, chi2_chains, R_hat, ess, dvch_fit, lcdm_fit)

    print("Wrote dvch_mcmc_chains.csv")
    print("Wrote dvch_mcmc_acceptance.csv")
    print("Wrote dvch_mcmc_convergence_summary.csv")
    print(f"Figure saved: {FIGDIR / 'dvch_mcmc_convergence.png'}")


if __name__ == "__main__":
    main()
