#!/usr/bin/env python3
"""
DVCH background utilities and Cobaya/CLASS bridge helpers.

The model implemented here follows the effective closure

    rho_Lambda = rho_Lambda0 * (rho_m / rho_m0)**dvch_n * (1 + dvch_beta) / (1 + dvch_beta * E**2),

with E(z) = H(z) / H0. Once the closure is specified, the coupled conservation
equations fix the interaction term uniquely. The exact background transfer used
by the helper routines is

    Qtilde = Q / (3 H0 rho_crit,0)
           = - E Omega_Lambda / (1 + dvch_n Omega_Lambda / Omega_m)
             * [dvch_n - dvch_beta (4 Omega_r + 3 Omega_m) / (3 (1 + dvch_beta E**2))].

This module does not patch CLASS by itself. Instead, it provides:
1. A mathematically consistent background solver for DVCH fiducial checks.
2. The exact decay diagnostic d Omega_Lambda / d ln a = 3 Qtilde / E.
3. A helper that assembles the CLASS extra_args expected by a patched
   classy/CLASS backend used from Cobaya.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp


OMEGA_GAMMA_H2 = 2.47282e-5
NEFF_DEFAULT = 3.044


@dataclass(frozen=True)
class DVCHParameters:
    omega_b: float = 0.02237
    omega_cdm: float = 0.1199
    h: float = 0.689
    dvch_n: float = 0.18
    dvch_beta: float = 1.0e-4
    n_eff: float = NEFF_DEFAULT
    omega_k: float = 0.0

    @property
    def H0(self) -> float:
        return 100.0 * self.h

    @property
    def Omega_b0(self) -> float:
        return self.omega_b / self.h**2

    @property
    def Omega_cdm0(self) -> float:
        return self.omega_cdm / self.h**2

    @property
    def Omega_m0(self) -> float:
        return (self.omega_b + self.omega_cdm) / self.h**2

    @property
    def Omega_r0(self) -> float:
        return OMEGA_GAMMA_H2 * (1.0 + 0.22710731766 * self.n_eff) / self.h**2

    @property
    def Omega_lambda0(self) -> float:
        value = 1.0 - self.omega_k - self.Omega_m0 - self.Omega_r0
        if value <= 0.0:
            raise ValueError(
                "The inferred present vacuum fraction is non-positive; adjust "
                "omega_b, omega_cdm, h, n_eff, or omega_k."
            )
        return value

    def validate(self) -> None:
        if not (0.0 < self.h < 2.0):
            raise ValueError("h must satisfy 0 < h < 2.")
        if not (0.0 < self.dvch_n < 1.0):
            raise ValueError("dvch_n must satisfy 0 < dvch_n < 1.")
        if self.dvch_beta < 0.0:
            raise ValueError("dvch_beta must be non-negative.")
        if self.Omega_m0 <= 0.0:
            raise ValueError("The present matter fraction must be positive.")
        _ = self.Omega_lambda0


def radiation_density(z: float, params: DVCHParameters) -> float:
    return params.Omega_r0 * (1.0 + z) ** 4


def vacuum_density(
    z: float,
    omega_m_z: float,
    E2: float,
    params: DVCHParameters,
) -> float:
    return (
        params.Omega_lambda0
        * (omega_m_z / params.Omega_m0) ** params.dvch_n
        * (1.0 + params.dvch_beta)
        / (1.0 + params.dvch_beta * E2)
    )


def solve_e2(
    z: float,
    omega_m_z: float,
    params: DVCHParameters,
    *,
    tol: float = 1.0e-12,
    max_iter: int = 400,
) -> tuple[float, float]:
    omega_r_z = radiation_density(z, params)
    E2 = omega_r_z + omega_m_z + params.Omega_lambda0
    for _ in range(max_iter):
        omega_lambda_z = vacuum_density(z, omega_m_z, E2, params)
        new_E2 = omega_r_z + omega_m_z + omega_lambda_z
        if abs(new_E2 - E2) < tol:
            return new_E2, omega_lambda_z
        E2 = 0.5 * (E2 + new_E2)
    raise RuntimeError(f"DVCH fixed-point iteration did not converge at z={z:.6g}.")


def q_tilde(
    z: float,
    omega_m_z: float,
    params: DVCHParameters,
) -> tuple[float, float, float]:
    if omega_m_z <= 0.0:
        raise ValueError("omega_m_z must be positive to evaluate the DVCH source term.")
    E2, omega_lambda_z = solve_e2(z, omega_m_z, params)
    E = np.sqrt(E2)
    bracket = params.dvch_n - params.dvch_beta * (
        4.0 * radiation_density(z, params) + 3.0 * omega_m_z
    ) / (3.0 * (1.0 + params.dvch_beta * E2))
    qt = -E * omega_lambda_z / (
        1.0 + params.dvch_n * omega_lambda_z / omega_m_z
    ) * bracket
    return qt, E2, omega_lambda_z


def domega_m_dz(z: float, y: np.ndarray, params: DVCHParameters) -> list[float]:
    omega_m_z = max(float(y[0]), 1.0e-15)
    qt, E2, _ = q_tilde(z, omega_m_z, params)
    E = np.sqrt(E2)
    return [3.0 * (omega_m_z + qt / E) / (1.0 + z)]


def solve_background(
    params: DVCHParameters,
    z_samples: np.ndarray,
) -> dict[str, np.ndarray]:
    params.validate()
    if len(z_samples) < 2:
        raise ValueError("z_samples must contain at least two points.")
    if not np.isclose(z_samples[0], 0.0):
        raise ValueError("z_samples must start at z=0 for the present-day normalization.")
    if np.any(np.diff(z_samples) <= 0.0):
        raise ValueError("z_samples must be strictly increasing.")

    z_max = float(z_samples[-1])
    solution = solve_ivp(
        lambda z, y: domega_m_dz(z, y, params),
        (0.0, z_max),
        [params.Omega_m0],
        t_eval=z_samples,
        rtol=1.0e-9,
        atol=1.0e-12,
        max_step=max(0.02, z_max / 250.0),
    )
    if not solution.success:
        raise RuntimeError(f"Background integration failed: {solution.message}")

    omega_m = solution.y[0]
    omega_r = radiation_density(z_samples, params)
    E2 = np.empty_like(z_samples)
    omega_lambda = np.empty_like(z_samples)
    qtilde = np.empty_like(z_samples)

    for idx, (z, omega_m_z) in enumerate(zip(z_samples, omega_m, strict=False)):
        qtilde[idx], E2[idx], omega_lambda[idx] = q_tilde(float(z), float(omega_m_z), params)

    E = np.sqrt(E2)
    domega_lambda_dln_a = 3.0 * qtilde / E
    w_lambda_eff = -1.0 - qtilde / (E * omega_lambda)

    return {
        "z": z_samples,
        "E": E,
        "E2": E2,
        "Omega_m": omega_m,
        "Omega_r": omega_r,
        "Omega_lambda": omega_lambda,
        "Qtilde": qtilde,
        "dOmegaLambda_dln_a": domega_lambda_dln_a,
        "w_lambda_eff": w_lambda_eff,
    }


def fiducial_decay_summary(
    params: DVCHParameters | None = None,
    *,
    z_max: float = 5.0,
    n_points: int = 300,
) -> dict[str, float | bool]:
    params = params or DVCHParameters()
    z_samples = np.linspace(0.0, z_max, n_points)
    bg = solve_background(params, z_samples)
    return {
        "z_max": z_max,
        "min_E2": float(np.min(bg["E2"])),
        "max_Qtilde": float(np.max(bg["Qtilde"])),
        "min_Qtilde": float(np.min(bg["Qtilde"])),
        "max_dOmegaLambda_dln_a": float(np.max(bg["dOmegaLambda_dln_a"])),
        "min_dOmegaLambda_dln_a": float(np.min(bg["dOmegaLambda_dln_a"])),
        "vacuum_decays_monotonically": bool(np.all(bg["dOmegaLambda_dln_a"] < 1.0e-12)),
        "positive_energy_densities": bool(
            np.all(bg["Omega_m"] > 0.0)
            and np.all(bg["Omega_lambda"] > 0.0)
            and np.all(bg["Omega_r"] > 0.0)
        ),
    }


def build_classy_extra_args(
    *,
    z_pk: str = "0 0.5 1 2",
    l_max_scalars: int = 3500,
    pk_max_h_mpc: float = 5.0,
) -> dict[str, Any]:
    return {
        "output": "tCl,pCl,lCl,mPk",
        "lensing": "yes",
        "non linear": "hmcode",
        "l_max_scalars": l_max_scalars,
        "P_k_max_h/Mpc": pk_max_h_mpc,
        "z_pk": z_pk,
        "N_ncdm": 1,
        "m_ncdm": 0.06,
        "N_ur": 2.0328,
        # The following keys are consumed by the patched classy/CLASS backend.
        "dvch_model": "tracking_curvature_suppressed_vacuum",
        "dvch_use_exact_q": "yes",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a fiducial DVCH decay self-check.")
    defaults = DVCHParameters()
    parser.add_argument("--omega-b", type=float, default=defaults.omega_b)
    parser.add_argument("--omega-cdm", type=float, default=defaults.omega_cdm)
    parser.add_argument("--h", type=float, default=defaults.h)
    parser.add_argument("--dvch-n", type=float, default=defaults.dvch_n)
    parser.add_argument("--dvch-beta", type=float, default=defaults.dvch_beta)
    parser.add_argument("--z-max", type=float, default=5.0)
    parser.add_argument("--n-points", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = DVCHParameters(
        omega_b=args.omega_b,
        omega_cdm=args.omega_cdm,
        h=args.h,
        dvch_n=args.dvch_n,
        dvch_beta=args.dvch_beta,
    )
    summary = fiducial_decay_summary(params, z_max=args.z_max, n_points=args.n_points)
    print("DVCH fiducial self-check")
    print("-" * 32)
    for key, value in asdict(params).items():
        print(f"{key}: {value}")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
