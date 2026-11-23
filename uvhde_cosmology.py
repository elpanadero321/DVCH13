import numpy as np
from scipy.integrate import quad, solve_ivp


class UVHDEPlanckCosmology:
    """
    Formulación cosmológica definitiva UV-HDE + Límite de Planck:
    - Sin interacción oscura (Q = 0) y cero parámetros extra (Delta k = 0).
    - Solución algebraica exacta y rebote cuántico primordial acotado por rho_Pl.
    """

    def __init__(self, H0=67.36, Omega_m0=0.3153, Omega_r0=9.18e-5, rd=147.09):
        self.H0 = float(H0)
        self.Omega_m0 = float(Omega_m0)
        self.Omega_r0 = float(Omega_r0)
        self.Omega_DE0 = 1.0 - self.Omega_m0 - self.Omega_r0
        self.rd = float(rd)

        # Constantes físicas SI (CODATA)
        self.G = 6.67430e-11
        self.c = 299792458.0
        self.c_kms = self.c / 1000.0
        self.hbar = 1.054571817e-34
        self.k_B = 1.380649e-23

        # Escala de Planck, saturación y rebote cuántico primordial
        self.rho_Pl = self.c**5 / (self.hbar * self.G**2)
        H0_si = self.H0 * 1000.0 / 3.085677581491367e22
        self.rho_crit0 = 3.0 * (H0_si**2) / (8.0 * np.pi * self.G)
        self.eps_Pl = self.rho_crit0 / self.rho_Pl
        self.a_min = (self.Omega_r0 * self.eps_Pl) ** 0.25
        self.z_bounce = 1.0 / self.a_min - 1.0

    def E2_classical(self, z):
        """Solución algebraica exacta de E^2(z) libre de cancelación numérica."""
        zp1 = 1.0 + np.asarray(z, dtype=np.float64)
        A = self.Omega_r0 * zp1**4 + self.Omega_m0 * zp1**3
        B = 2.0 * self.Omega_DE0 * zp1
        return A + (2.0 * B) / (1.0 + A + np.sqrt((1.0 + A) ** 2 + 4.0 * B))

    def H2(self, z):
        """Ecuación de Friedmann completa con límite cuántico de Planck."""
        e2 = self.E2_classical(z)
        return (self.H0**2) * e2 * (1.0 - self.eps_Pl * e2)

    def H(self, z):
        """Parámetro de Hubble H(z) en km/s/Mpc."""
        return np.sqrt(np.maximum(self.H2(z), 0.0))

    def dE2_dz(self, z):
        """Derivada analítica exacta d(E^2)/dz."""
        zp1 = 1.0 + np.asarray(z, dtype=np.float64)
        e2 = self.E2_classical(z)
        B = 2.0 * self.Omega_DE0 * zp1
        num = (
            4.0 * self.Omega_r0 * zp1**3
            + 3.0 * self.Omega_m0 * zp1**2
            + (2.0 * self.Omega_DE0) / (1.0 + e2)
        )
        den = 1.0 + B / (1.0 + e2) ** 2
        return num / den

    def rho_DE_ratio(self, z):
        """Evolución normalizada de la densidad oscura rho_DE(z) / rho_DE(0)."""
        zp1 = 1.0 + np.asarray(z, dtype=np.float64)
        return zp1 * 2.0 / (1.0 + self.E2_classical(z))

    def w_eff(self, z):
        """Ecuación de estado efectiva de la energía oscura."""
        zp1 = 1.0 + np.asarray(z, dtype=np.float64)
        e2 = self.E2_classical(z)
        return -1.0 + (zp1 / 3.0) * (1.0 / zp1 - self.dE2_dz(z) / (1.0 + e2))

    def q(self, z):
        """Parámetro de desaceleración cósmica q(z)."""
        zp1 = 1.0 + np.asarray(z, dtype=np.float64)
        e2 = self.E2_classical(z)
        return -1.0 + (zp1 / (2.0 * e2)) * self.dE2_dz(z)

    def Om_diagnostic(self, z):
        """Diagnóstico Om(z) de Sahni-Shafieloo-Starobinsky (en LCDM es constante)."""
        z_arr = np.asarray(z, dtype=np.float64)
        scalar_input = z_arr.ndim == 0
        z_1d = np.atleast_1d(z_arr)
        out = np.empty_like(z_1d)
        zero_mask = np.isclose(z_1d, 0.0, atol=1e-12)
        if np.any(zero_mask):
            out[zero_mask] = float(self.dE2_dz(0.0)) / 3.0
        if np.any(~zero_mask):
            zn = z_1d[~zero_mask]
            out[~zero_mask] = (self.E2_classical(zn) - 1.0) / ((1.0 + zn) ** 3 - 1.0)
        return float(out[0]) if scalar_input else out

    def D_H(self, z):
        """Distancia radial de Hubble c / H(z) en Mpc."""
        return self.c_kms / self.H(z)

    def D_M(self, z):
        """Distancia comóvil transversal en Mpc (soporta escalar o array)."""
        z_arr = np.asarray(z, dtype=np.float64)
        if z_arr.ndim == 0:
            val, _ = quad(lambda zp: self.c_kms / float(self.H(zp)), 0.0, float(z_arr))
            return float(val)
        return np.array(
            [
                quad(lambda zp: self.c_kms / float(self.H(zp)), 0.0, float(zi))[0]
                for zi in z_arr
            ]
        )

    def D_V(self, z):
        """Distancia BAO promedio volumétrica en Mpc."""
        z_arr = np.asarray(z, dtype=np.float64)
        dm = self.D_M(z_arr)
        dh = self.D_H(z_arr)
        return (z_arr * (dm**2) * dh) ** (1.0 / 3.0)

    def growth_factor_and_fsigma8(self, z_eval, sigma8_0=0.8190):
        """Crecimiento lineal estándar (G_eff = G, Q = 0) desde z=1000 hasta z=0."""
        N_ini = np.log(1.0 / 1001.0)

        def ode(N, y):
            z = np.exp(-N) - 1.0
            zp1 = 1.0 + z
            e2 = float(self.E2_classical(z))
            de2 = float(self.dE2_dz(z))
            dlnE_dN = -0.5 * (zp1 / e2) * de2
            Om_a = self.Omega_m0 * (zp1**3) / e2
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
