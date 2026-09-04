#!/usr/bin/env python3
"""Numerical DVCH background provider for a CAMB/CLASS integration.

This module implements the part of DVCH that is unambiguously defined by the
repository: the homogeneous Friedmann closure and its interaction source.
It is deliberately separate from the compiled CAMB hierarchy because a
relativistic prescription for ``delta Q`` and momentum transfer is not defined
in the current model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class DVCHParameters:
    """Parameters required by the homogeneous DVCH closure."""

    omega_m0: float = 0.30
    omega_r0: float = 9.0e-5
    n: float = 0.20
    beta: float = 1.0e-4
    H0: float = 69.03

    @property
    def omega_l0(self) -> float:
        return 1.0 - self.omega_m0 - self.omega_r0

    def validate(self) -> None:
        if not 0.0 < self.omega_m0 < 1.0:
            raise ValueError("omega_m0 must be between zero and one")
        if not 0.0 < self.omega_r0 < 1.0:
            raise ValueError("omega_r0 must be between zero and one")
        if self.omega_l0 <= 0.0:
            raise ValueError("omega_l0 must be positive")
        if not 0.0 < self.n < 1.0:
            raise ValueError("n must be between zero and one")
        if self.beta < 0.0:
            raise ValueError("beta must be non-negative")
        if self.H0 <= 0.0:
            raise ValueError("H0 must be positive")


def vacuum_density(z: float | np.ndarray, omega_m: float | np.ndarray,
                   parameters: DVCHParameters) -> np.ndarray:
    """Return dimensionless vacuum density for a fixed matter density."""
    p = parameters
    p.validate()
    z_arr = np.asarray(z, dtype=float)
    matter = np.asarray(omega_m, dtype=float)
    radiation = p.omega_r0 * (1.0 + z_arr) ** 4
    e2 = radiation + matter + p.omega_l0
    vacuum = p.omega_l0 * (matter / p.omega_m0) ** p.n
    for _ in range(200):
        vacuum = (
            p.omega_l0 * (matter / p.omega_m0) ** p.n
            * (1.0 + p.beta) / (1.0 + p.beta * e2)
        )
        new_e2 = radiation + matter + vacuum
        if np.all(np.abs(new_e2 - e2) < 1.0e-12):
            break
        e2 = new_e2
    return vacuum


def background(z_values: Iterable[float],
               parameters: DVCHParameters = DVCHParameters()) -> np.ndarray:
    """Build ``z, E2, E, H, Omega_m, Omega_Lambda, Qtilde`` rows.

    The matter density is evolved from ``z=0`` using the conservation
    equation implied by the DVCH closure.  This is the homogeneous background
    that a compiled Boltzmann solver can consume.
    """
    from scipy.integrate import solve_ivp

    p = parameters
    p.validate()
    z = np.asarray(tuple(z_values), dtype=float)
    if z.ndim != 1 or z.size == 0 or np.any(~np.isfinite(z)) or np.any(z < 0):
        raise ValueError("z_values must be a non-empty finite one-dimensional grid")
    if np.any(np.diff(z) < 0):
        raise ValueError("z_values must be sorted in ascending order")
    if z[0] != 0.0:
        raise ValueError("z_values must start at z=0")

    def closure(redshift: float, matter: float) -> tuple[float, float]:
        radiation = p.omega_r0 * (1.0 + redshift) ** 4
        e2 = radiation + matter + p.omega_l0
        vacuum = p.omega_l0 * (matter / p.omega_m0) ** p.n
        for _ in range(200):
            vacuum = (
                p.omega_l0 * (matter / p.omega_m0) ** p.n
                * (1.0 + p.beta) / (1.0 + p.beta * e2)
            )
            new_e2 = radiation + matter + vacuum
            if abs(new_e2 - e2) < 1.0e-12:
                e2 = new_e2
                break
            e2 = new_e2
        return e2, vacuum

    def rhs(redshift: float, state: np.ndarray) -> list[float]:
        matter = float(state[0])
        e2, vacuum = closure(redshift, matter)
        d_vacuum_d_matter = vacuum * p.n / matter
        d_vacuum_d_e2 = (
            -p.omega_l0 * (matter / p.omega_m0) ** p.n
            * (1.0 + p.beta) * p.beta / (1.0 + p.beta * e2) ** 2
        )
        denominator = 1.0 - d_vacuum_d_e2
        coefficient = (1.0 + d_vacuum_d_matter) * (
            1.0 + d_vacuum_d_e2 / denominator
        )
        explicit = 4.0 * p.omega_r0 * (1.0 + redshift) ** 3
        derivative = (
            3.0 * matter / (1.0 + redshift)
            - (d_vacuum_d_e2 / denominator) * explicit
        ) / coefficient
        return [derivative]

    solution = solve_ivp(
        rhs, (0.0, float(z[-1])), [p.omega_m0], t_eval=z,
        rtol=1.0e-10, atol=1.0e-12, method="RK45",
    )
    if not solution.success:
        raise RuntimeError(f"DVCH background integration failed: {solution.message}")

    matter = solution.y[0]
    e2 = np.empty_like(z)
    vacuum = np.empty_like(z)
    radiation = p.omega_r0 * (1.0 + z) ** 4
    for index, redshift in enumerate(z):
        e2[index], vacuum[index] = closure(float(redshift), float(matter[index]))
    if np.any(~np.isfinite(e2)) or np.any(e2 <= 0):
        raise RuntimeError("DVCH background produced a non-positive expansion rate")

    e = np.sqrt(e2)
    qtilde = -e * vacuum / (1.0 + p.n * vacuum / matter) * (
        p.n - p.beta * (4.0 * radiation + 3.0 * matter)
        / (3.0 * (1.0 + p.beta * e2))
    )
    return np.column_stack((z, e2, e, p.H0 * e, matter, vacuum, qtilde))


def write_background_table(path: str,
                           z_values: Iterable[float],
                           parameters: DVCHParameters = DVCHParameters()) -> None:
    """Write a CSV table suitable for a compiled-backend adapter."""
    table = background(z_values, parameters)
    np.savetxt(
        path,
        table,
        delimiter=",",
        header="z,E2,E,H,Omega_m,Omega_Lambda,Qtilde",
        comments="",
    )
