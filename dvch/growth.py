"""Sub-horizon growth for DVCH (manuscript Eq. 62).

delta'' + [2 + H'/H + Gamma] delta' - (3/2) Omega_m^(H)(a) mu(a) delta = 0,
with ' = d/dN, N = ln a, Gamma = Q/(H rho_m) (Eq. 63), mu = 1 (GR, no fifth
force). Omega_m^(H) = rho_m/(3 Mpl^2 H^2) = Omega_m/E^2.

This is the controlled sub-horizon approximation of Sec. VII; the full
relativistic treatment lives in the patched Boltzmann code.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from .background import BackgroundSolver


class GrowthSolver:
    def __init__(self, bg: BackgroundSolver, z_ini: float = 100.0):
        self.bg = bg
        self.z_ini = z_ini
        zg = bg.z[bg.z <= z_ini * 1.05]
        N = -np.log(1.0 + bg.z)
        self._E = interp1d(N, bg.E, kind="cubic")
        self._Om = interp1d(N, bg.Omega_m, kind="cubic")
        Qt = bg.Qtilde()
        # Gamma = Q/(H rho_m) = 3 Qtilde / (E Omega_m)
        self._Gamma = interp1d(N, 3.0 * Qt / (bg.E * bg.Omega_m), kind="cubic")
        dlnE_dN = np.gradient(np.log(bg.E), N)
        self._dlnE = interp1d(N, dlnE_dN, kind="cubic")
        self._solve()

    def _rhs(self, N, y):
        d, dp = y
        E = self._E(N)
        Om = self._Om(N)
        fric = 2.0 + self._dlnE(N) + self._Gamma(N)
        src = 1.5 * (Om / E**2) * d
        return [dp, -fric * dp + src]

    def _solve(self):
        N_ini = -np.log(1.0 + self.z_ini)
        # matter-domination initial conditions: D ~ a
        sol = solve_ivp(self._rhs, (N_ini, 0.0), [np.exp(N_ini), np.exp(N_ini)],
                        dense_output=True, rtol=1e-9, atol=1e-12, method="LSODA")
        if not sol.success:
            raise RuntimeError(f"growth ODE failed: {sol.message}")
        self._sol = sol
        self.D0 = sol.y[0, -1]

    def D(self, z):
        """Linear growth factor normalized to D(z=0)=1."""
        N = -np.log(1.0 + np.asarray(z, dtype=float))
        return self._sol.sol(N)[0] / self.D0

    def f(self, z):
        """Growth rate f = dlnD/dlnN."""
        N = -np.log(1.0 + np.asarray(z, dtype=float))
        d, dp = self._sol.sol(N)
        return dp / d

    def fsigma8(self, z, sigma8_0):
        return self.f(z) * sigma8_0 * self.D(z)

    def sigma8_rescaled(self, sigma8_lcdm_ref: float, D_lcdm_ratio: float = None,
                        bg_lcdm: BackgroundSolver = None):
        """Low-redshift diagnostic normalization of Sec. VI: rescale a reference
        LCDM sigma8 by the ratio of growth factors integrated from z_ini."""
        if bg_lcdm is None:
            from .background import DVCHParams
            p = self.bg.p
            bg_lcdm = BackgroundSolver(DVCHParams(H0=p.H0, Omega_m0=p.Omega_m0,
                                                  Omega_r0=p.Omega_r0, n=0.0, beta=0.0))
        g_ref = GrowthSolver(bg_lcdm, z_ini=self.z_ini)
        return sigma8_lcdm_ref * self.D0 / g_ref.D0
