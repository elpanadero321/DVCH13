#!/usr/bin/env python3
"""UV-HDE + Planck Limit linear perturbation closure.

The Generalized Cuscuton field does not propagate scalar modes on sub-horizon
scales (c_s^2 -> infinity, delta_rho_DE -> 0), so G_eff/G = 1.
The linear growth equation uses standard gravity:

    delta_m'' + (2 + dlnE/dln a) delta_m' - (3/2) Omega_m(a) delta_m = 0

No dark interaction (Q = 0), standard matter and radiation conservation.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PerturbationState:
    """Dimensionless synchronous-gauge state for the CDM sector."""

    delta_m: float
    theta_m: float
    h_n: float


def growth_ode_rhs(
    z: float,
    state: PerturbationState,
    e2: float,
    de2_dz: float,
    omega_m: float,
) -> tuple[float, float]:
    """Return (delta_m', theta_m') for standard gravity (mu=1, Q=0).

    Independent variable is redshift z.
    """
    if e2 <= 0.0 or omega_m <= 0.0:
        raise ValueError("background expansion and matter density must be positive")

    zp1 = 1.0 + z
    # dlnE/dln a = -(1+z)/E * dE/dz = -(1+z)/(2*E^2) * dE^2/dz
    dlnE_dN = -0.5 * zp1 / e2 * de2_dz

    # Standard growth equation: delta'' + (2 + dlnE/dN) delta' - 1.5 * Om * delta = 0
    delta_n = state.theta_m / e2**0.5  # derivative wrt N = ln a
    delta_pp = -(2.0 + dlnE_dN) * delta_n + 1.5 * omega_m * state.delta_m

    # theta_m' = -theta_m (no interaction force, standard Euler eq)
    theta_n = -state.theta_m

    return delta_pp, theta_n


def synchronous_cdm_rhs(
    z: float,
    state: PerturbationState,
    e2: float,
    de2_dz: float,
    omega_m: float,
) -> tuple[float, float]:
    """Return (delta_m', theta_m') in synchronous gauge for standard gravity.

    Independent variable is conformal time (or N = ln a).
    """
    if e2 <= 0.0 or omega_m <= 0.0:
        raise ValueError("background expansion and matter density must be positive")

    zp1 = 1.0 + z
    dlnE_dN = -0.5 * zp1 / e2 * de2_dz

    # Standard growth in synchronous gauge
    # delta_m' = -theta_m / (aH) - 0.5 h'  (but we return d/dN)
    delta_n = -state.theta_m - 0.5 * state.h_n
    theta_n = -state.theta_m

    return delta_n, theta_n


def adiabatic_initial_conditions(delta_gamma: float) -> dict[str, float]:
    """Return the standard early-time adiabatic CDM/DE initial conditions."""
    return {
        "delta_m": 0.75 * float(delta_gamma),
        "theta_m": 0.0,
        "delta_de": 0.0,  # No DE perturbations (c_s^2 -> inf)
        "theta_de": 0.0,
    }


def growth_factor_integral(
    z_eval: float | np.ndarray,
    cosmo,
    sigma8_0: float = 0.8190,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute growth factor D(z) and f*sigma8(z) using standard gravity.

    Solves: d^2D/dN^2 + (2 + dlnE/dN) dD/dN - 1.5 * Omega_m(a) * D = 0
    from z=1000 to z=0.
    """
    from scipy.integrate import solve_ivp

    N_ini = np.log(1.0 / 1001.0)

    def ode(N, y):
        z = np.exp(-N) - 1.0
        zp1 = 1.0 + z
        e2 = float(cosmo.E2_classical(z))
        de2 = float(cosmo.dE2_dz(z))
        dlnE_dN = -0.5 * (zp1 / e2) * de2
        Om_a = cosmo.Omega_m0 * (zp1**3) / e2
        return [y[1], -(2.0 + dlnE_dN) * y[1] + 1.5 * Om_a * y[0]]

    sol = solve_ivp(
        ode,
        [N_ini, 0.0],
        [np.exp(N_ini), np.exp(N_ini)],
        dense_output=True,
        rtol=1e-11,
        atol=1e-13,
    )
    D_0 = sol.sol(0.0)[0]
    N_eval = -np.log(1.0 + np.asarray(z_eval, dtype=np.float64))
    D_val, dD_dN = sol.sol(N_eval)
    f_val = dD_dN / D_val
    fs8 = f_val * sigma8_0 * (D_val / D_0)
    return D_val / D_0, fs8


def mu_sigma(z: float, cosmo) -> tuple[float, float]:
    """Return (mu, Sigma) for UV-HDE + Planck Limit.

    On sub-horizon scales, Generalized Cuscuton does not propagate:
    c_s^2 -> inf, delta_rho_DE -> 0 => G_eff/G = 1, lensing = 1.
    """
    return 1.0, 1.0