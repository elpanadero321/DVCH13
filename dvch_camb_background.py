#!/usr/bin/env python3
"""Numerical UV-HDE + Planck Limit background provider for a CAMB/CLASS integration.

This module implements the homogeneous Friedmann closure for UV-HDE with
exact algebraic solution for E^2(z). No dark interaction (Q=0), standard
matter and radiation conservation.

Model: UV-HDE + Límite de Planck (versión definitiva, Δk = 0).
- Sin interacción oscura (Q = 0).
- Cero parámetros libres extra frente a ΛCDM (n = 1/3, β = 1 fijados).
- Solución algebraica exacta y cerrada para E^2(z).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from uvhde_cosmology import UVHDEPlanckCosmology


@dataclass(frozen=True)
class UVHDEParameters:
    """Parameters required by the homogeneous UV-HDE closure.

    Only Omega_m0 and H0 are free (same as LCDM).
    Omega_r0 is fixed to Planck 2018 value.
    n=1/3, beta=1 are fixed internally (not free parameters).
    """

    omega_m0: float = 0.3153
    omega_r0: float = 9.18e-5
    H0: float = 67.36

    @property
    def omega_de0(self) -> float:
        return 1.0 - self.omega_m0 - self.omega_r0

    def validate(self) -> None:
        if not 0.0 < self.omega_m0 < 1.0:
            raise ValueError("omega_m0 must be between zero and one")
        if not 0.0 < self.omega_r0 < 1.0:
            raise ValueError("omega_r0 must be between zero and one")
        if self.omega_de0 <= 0.0:
            raise ValueError("omega_de0 must be positive")
        if self.H0 <= 0.0:
            raise ValueError("H0 must be positive")


def background(z_values: Iterable[float],
               parameters: UVHDEParameters = UVHDEParameters()) -> np.ndarray:
    """Build z, E2, E, H, Omega_m, Omega_DE, Omega_r rows.

    Uses the exact algebraic solution for E^2(z).
    Matter density follows standard conservation: Omega_m(z) = Omega_m0 * (1+z)^3 / E^2(z)
    """
    p = parameters
    p.validate()

    cosmo = UVHDEPlanckCosmology(H0=p.H0, Omega_m0=p.omega_m0, Omega_r0=p.omega_r0)

    z = np.asarray(tuple(z_values), dtype=float)
    if z.ndim != 1 or z.size == 0 or np.any(~np.isfinite(z)) or np.any(z < 0):
        raise ValueError("z_values must be a non-empty finite one-dimensional grid")
    if np.any(np.diff(z) < 0):
        raise ValueError("z_values must be sorted in ascending order")
    if z[0] != 0.0:
        raise ValueError("z_values must start at z=0")

    E2 = cosmo.E2_classical(z)
    E = np.sqrt(E2)
    H = p.H0 * E

    opz = 1.0 + z
    Om = p.omega_m0 * opz**3 / E2
    OL = cosmo.Omega_DE0 * opz * 2.0 / (1.0 + E2) / E2
    Or = p.omega_r0 * opz**4 / E2

    return np.column_stack((z, E2, E, H, Om, OL, Or))


def write_background_table(path: str,
                           z_values: Iterable[float],
                           parameters: UVHDEParameters = UVHDEParameters()) -> None:
    """Write a CSV table suitable for a compiled-backend adapter."""
    table = background(z_values, parameters)
    np.savetxt(
        path,
        table,
        delimiter=",",
        header="z,E2,E,H,Omega_m,Omega_DE,Omega_r",
        comments="",
    )


def E2_uvhde(z: float | np.ndarray, parameters: UVHDEParameters = UVHDEParameters()) -> np.ndarray:
    """Return E^2(z) for UV-HDE model."""
    cosmo = UVHDEPlanckCosmology(H0=parameters.H0, Omega_m0=parameters.omega_m0, Omega_r0=parameters.omega_r0)
    return cosmo.E2_classical(z)


def w_eff_uvhde(z: float | np.ndarray, parameters: UVHDEParameters = UVHDEParameters()) -> np.ndarray:
    """Return w_eff(z) for UV-HDE model."""
    cosmo = UVHDEPlanckCosmology(H0=parameters.H0, Omega_m0=parameters.omega_m0, Omega_r0=parameters.omega_r0)
    return cosmo.w_eff(z)


def q_uvhde(z: float | np.ndarray, parameters: UVHDEParameters = UVHDEParameters()) -> np.ndarray:
    """Return q(z) for UV-HDE model."""
    cosmo = UVHDEPlanckCosmology(H0=parameters.H0, Omega_m0=parameters.omega_m0, Omega_r0=parameters.omega_r0)
    return cosmo.q(z)


def Om_diagnostic_uvhde(z: float | np.ndarray, parameters: UVHDEParameters = UVHDEParameters()) -> np.ndarray:
    """Return Om_diagnostic(z) for UV-HDE model."""
    cosmo = UVHDEPlanckCosmology(H0=parameters.H0, Omega_m0=parameters.omega_m0, Omega_r0=parameters.omega_r0)
    return cosmo.Om_diagnostic(z)


if __name__ == "__main__":
    # Quick test
    params = UVHDEParameters()
    z = np.linspace(0, 5, 11)
    table = background(z, params)
    print("z, E2, E, H, Omega_m, Omega_DE, Omega_r")
    for row in table:
        print(", ".join(f"{v:.6f}" for v in row))