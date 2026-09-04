#!/usr/bin/env python3
"""DVCH linear perturbation closure from the report's Sec. VII equations.

The closure couples CDM to the vacuum through ``Q^mu = Q u_m^mu``.  Radiation
and baryons remain uncoupled, the CDM Euler equation has no interaction force,
and the vacuum perturbation is set to zero in its rest frame.
"""

from __future__ import annotations

from dataclasses import dataclass

from dvch_camb_background import DVCHParameters


@dataclass(frozen=True)
class PerturbationState:
    """Dimensionless synchronous-gauge state for the CDM sector."""

    delta_m: float
    theta_m: float
    h_n: float


def interaction_coefficients(
    e: float,
    omega_m: float,
    omega_lambda: float,
    omega_r: float,
    parameters: DVCHParameters,
) -> tuple[float, float, float]:
    """Return the analytic ``(c_m, c_lambda, c_E)`` coefficients."""
    p = parameters
    p.validate()
    if min(e, omega_m, omega_lambda) <= 0.0:
        raise ValueError("background densities and E must be positive")
    denominator = 1.0 + p.n * omega_lambda / omega_m
    bracket = p.n - p.beta * (4.0 * omega_r + 3.0 * omega_m) / (
        3.0 * (1.0 + p.beta * e * e)
    )
    c_m = (
        e * omega_lambda * p.beta / ((1.0 + p.beta * e * e) * denominator)
        - p.n * e * omega_lambda**2 * bracket
        / (omega_m**2 * denominator**2)
    )
    c_lambda = -e * bracket / denominator**2
    c_e = -omega_lambda / denominator * (
        bracket
        + 2.0 * p.beta**2 * e * e * (4.0 * omega_r + 3.0 * omega_m)
        / (3.0 * (1.0 + p.beta * e * e) ** 2)
    )
    return c_m, c_lambda, c_e


def delta_q_tilde(
    e: float,
    omega_m: float,
    omega_lambda: float,
    omega_r: float,
    delta_m: float,
    delta_e: float = 0.0,
    parameters: DVCHParameters = DVCHParameters(),
) -> float:
    """Return ``delta Q/(3 H0 rho_crit,0)`` using the report's Taylor closure."""
    c_m, c_lambda, c_e = interaction_coefficients(
        e, omega_m, omega_lambda, omega_r, parameters
    )
    delta_omega_m = omega_m * delta_m
    return c_m * delta_omega_m + c_lambda * 0.0 + c_e * delta_e


def synchronous_cdm_rhs(
    e: float,
    omega_m: float,
    omega_lambda: float,
    omega_r: float,
    state: PerturbationState,
    delta_e: float = 0.0,
    parameters: DVCHParameters = DVCHParameters(),
) -> tuple[float, float]:
    """Return ``(delta_m,N, theta_m,N)`` for ``Q^mu = Q u_m^mu``.

    ``theta_m`` is the usual conformal-time velocity divergence and therefore
    enters as ``theta_m / (a H)`` when the independent variable is ``N``.
    """
    p = parameters
    p.validate()
    if e <= 0.0 or omega_m <= 0.0:
        raise ValueError("background expansion and matter density must be positive")
    q_tilde = -e * omega_lambda / (1.0 + p.n * omega_lambda / omega_m) * (
        p.n - p.beta * (4.0 * omega_r + 3.0 * omega_m)
        / (3.0 * (1.0 + p.beta * e * e))
    )
    delta_q = delta_q_tilde(
        e, omega_m, omega_lambda, omega_r, state.delta_m, delta_e, p
    )
    interaction = 3.0 * q_tilde / (e * omega_m)
    delta_interaction = 3.0 * delta_q / (e * omega_m)
    delta_n = -state.theta_m / e - 0.5 * state.h_n + (
        interaction * state.delta_m - delta_interaction
    )
    theta_n = -state.theta_m
    return delta_n, theta_n


def adiabatic_initial_conditions(delta_gamma: float) -> dict[str, float]:
    """Return the documented early-time adiabatic CDM/vacuum conditions."""
    return {
        "delta_m": 0.75 * float(delta_gamma),
        "theta_m": 0.0,
        "delta_lambda": 0.0,
        "theta_lambda": 0.0,
    }
