"""DVCH background cosmology.

Implements the manuscript conventions exactly:

  Q = rho_Lambda_dot                                     (Eq. 8)
  rho_m_dot + 3 H rho_m = -Q                             (Eq. 7)
  rho_r_dot + 4 H rho_r = 0                              (Eq. 6)
  Omega_L(z) = Omega_L0 (Om/Om0)^n (1+beta)/(1+beta E^2) (Eq. 16)
  E^2(z) = Or0 (1+z)^4 + Om(z) + Omega_L(z)              (Eq. 19)
  Qtilde(z) = Q/(3 H0 rho_crit0)
            = -(E Omega_L)/(1 + n Omega_L/Om)
              * [ n - beta (4 Or0 (1+z)^4 + 3 Om) / (3 (1+beta E^2)) ]   (Eq. 25)
  w_eff(z) = -1 - Q/(3 H rho_L) = -1 - Qtilde/(E Omega_L)               (Eq. 30)

Omega_m(z) obeys the exact conservation ODE (Eq. 21) with dOmega_L/dz given
by the chain rule on the closure; E(z) is obtained at each step by the
Picard fixed-point iteration of Appendix A.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp

C_KM_S = 299792.458  # speed of light [km/s]


class ConvergenceError(Exception):
    """Raised when the implicit Friedmann fixed point fails to converge."""


@dataclass(frozen=True)
class DVCHParams:
    H0: float = 70.0          # km/s/Mpc
    Omega_m0: float = 0.30
    Omega_r0: float = 9.0e-5
    n: float = 0.20
    beta: float = 0.0

    @property
    def Omega_L0(self) -> float:
        return 1.0 - self.Omega_m0 - self.Omega_r0


def solve_E(z, Omega_m, p: DVCHParams, E_seed=1.0, tol=1e-10, max_iter=500):
    """Picard fixed-point solver of Appendix A. Returns (E, Omega_L)."""
    E_prev = max(E_seed, 1e-12)
    Omega_r = p.Omega_r0 * (1.0 + z) ** 4
    pref = p.Omega_L0 * (Omega_m / p.Omega_m0) ** p.n
    for _ in range(max_iter):
        Omega_L = pref * (1.0 + p.beta) / (1.0 + p.beta * E_prev**2)
        E2_new = Omega_r + Omega_m + Omega_L
        if E2_new <= 0.0:
            raise ConvergenceError("non-positive E^2")
        E_new = np.sqrt(E2_new)
        if abs(E_new - E_prev) < tol * E_new:
            return E_new, Omega_L
        E_prev = E_new
    raise ConvergenceError("DVCH fixed point did not converge")


def _dOmega_m_dz(z, Omega_m, p: DVCHParams):
    """Exact dOmega_m/dz from conservation (Eq. 21) + closure chain rule.

    dOL = (n OL/Om) dOm - OL * beta dE2/(1+beta E2), with
    dE2/dz = 4 Or0 (1+z)^3 + 3 Om/(1+z) (Eq. 22).
    Solving dOm + dOL = 3 Om/(1+z) for dOm:
      dOm (1 + n OL/Om) = 3 Om/(1+z) + OL beta dE2/(1+beta E2)
    """
    E, OL = solve_E(z, Omega_m, p, E_seed=np.sqrt(max(Omega_m, 1e-12)))
    dE2 = 4.0 * p.Omega_r0 * (1.0 + z) ** 3 + 3.0 * Omega_m / (1.0 + z)
    rhs = 3.0 * Omega_m / (1.0 + z) + OL * p.beta * dE2 / (1.0 + p.beta * E**2)
    return rhs / (1.0 + p.n * OL / Omega_m)


class BackgroundSolver:
    """Tabulated DVCH background on a redshift grid."""

    def __init__(self, params: DVCHParams | None = None, z_max: float = 1200.0,
                 n_grid: int = 4000):
        self.p = params or DVCHParams()
        # log-spaced grid in (1+z) for accuracy at both ends
        self.z = np.concatenate([[0.0], np.geomspace(1e-4, z_max, n_grid)])
        self._solve()

    def _solve(self):
        p = self.p
        sol = solve_ivp(
            _dOmega_m_dz, (0.0, self.z[-1]), [p.Omega_m0], t_eval=self.z,
            args=(p,), method="LSODA", rtol=1e-10, atol=1e-14,
        )
        if not sol.success:
            raise ConvergenceError(f"Omega_m ODE failed: {sol.message}")
        self.Omega_m = sol.y[0]
        self.E = np.empty_like(self.z)
        self.Omega_L = np.empty_like(self.z)
        for i, (zi, Omi) in enumerate(zip(self.z, self.Omega_m)):
            self.E[i], self.Omega_L[i] = solve_E(zi, Omi, p)
        self.Omega_r = p.Omega_r0 * (1.0 + self.z) ** 4
        self.H = p.H0 * self.E

    # --- exact kernel and derived quantities -------------------------------
    def Qtilde(self):
        """Q/(3 H0 rho_crit0), Eq. (25)."""
        p, z = self.p, self.z
        bracket = p.n - p.beta * (4 * p.Omega_r0 * (1 + z) ** 4 + 3 * self.Omega_m) / (
            3 * (1 + p.beta * self.E**2)
        )
        return -self.E * self.Omega_L / (1 + p.n * self.Omega_L / self.Omega_m) * bracket

    def w_eff(self):
        """Effective vacuum EoS, Eq. (30): w = -1 - Qtilde/(E Omega_L)."""
        return -1.0 - self.Qtilde() / (self.E * self.Omega_L)

    def dE_dz(self):
        """Eq. (22)."""
        return (4 * self.p.Omega_r0 * (1 + self.z) ** 3
                + 3 * self.Omega_m / (1 + self.z)) / (2 * self.E)

    def deceleration(self):
        """q(z) = -1 + (1+z) E'/E."""
        return -1.0 + (1.0 + self.z) * self.dE_dz() / self.E

    def transition_redshift(self):
        """Smallest z with q(z)=0 (sign change), linear interpolation."""
        q = self.deceleration()
        s = np.where(np.diff(np.sign(q)) != 0)[0]
        if len(s) == 0:
            return None
        i = s[0]
        z0, z1, q0, q1 = self.z[i], self.z[i + 1], q[i], q[i + 1]
        return z0 - q0 * (z1 - z0) / (q1 - q0)

    # --- distances ----------------------------------------------------------
    def comoving_distance(self):
        """D_C(z) [Mpc] on the grid."""
        integrand = 1.0 / self.E
        chi = cumulative_trapezoid(integrand, self.z, initial=0.0)
        return C_KM_S / self.p.H0 * chi

    def luminosity_distance(self):
        return (1.0 + self.z) * self.comoving_distance()

    def distance_modulus(self):
        dl = np.maximum(self.luminosity_distance(), 1e-12)
        return 5.0 * np.log10(dl) + 25.0

    # --- consistency gates ---------------------------------------------------
    def closure_residual(self):
        """|E^2 - (Or+Om+OL)| — must be ~ 0 by construction."""
        return np.abs(self.E**2 - (self.Omega_r + self.Omega_m + self.Omega_L))

    def conservation_residual(self):
        """Residual of Eq. (21) evaluated with numerical derivatives."""
        dOm = np.gradient(self.Omega_m, self.z)
        dOL = np.gradient(self.Omega_L, self.z)
        return dOm + dOL - 3 * self.Omega_m / (1 + self.z)

    def nec_satisfied(self):
        """rho_tot + p_tot = rho_m + 4 rho_r/3 >= 0 (Eq. 67) => GSL Sh_dot >= 0."""
        return np.all(self.Omega_m + 4.0 * self.Omega_r / 3.0 >= 0.0)

    def bbn_gate(self, threshold=0.04):
        """rho_L / rho_rad < threshold at the highest tabulated redshifts."""
        mask = self.z > 0.5 * self.z[-1]
        return np.all(self.Omega_L[mask] / self.Omega_r[mask] < threshold)

    def fEDE(self, z_star=1100.0):
        """Omega_L/Omega_m at z_star (EDE-type diagnostic, Eq. 28)."""
        i = np.argmin(np.abs(self.z - z_star))
        return self.Omega_L[i] / self.Omega_m[i]

    def is_viable(self):
        """Background viability gate used in the (n,beta) scans (Eq. 81)."""
        finite = np.all(np.isfinite(self.E)) and np.all(np.isfinite(self.Qtilde()))
        positive = np.all(self.E**2 > 0) and np.all(self.Omega_m > 0) and np.all(self.Omega_L >= 0)
        zt = self.transition_redshift()
        return bool(finite and positive and self.nec_satisfied()
                    and zt is not None and 0.0 < zt < 2.0)
