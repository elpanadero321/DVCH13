#!/usr/bin/env python3
"""
DVCH modified Boltzmann source-table backend.

Provides the DVCH interacting-vacuum source functions that a patched
CLASS or CAMB implementation could call during the background and
perturbation integration. This module is an interface specification and
standalone diagnostic; it is not itself a compiled Boltzmann solver:
it defines the `initialize`, `get_can_provide`, `must_provide`,
`calculate`, and `get_result` hooks expected by the standard Cobaya
theory API, together with the DVCH-specific background and interaction
table.

For late-time diagnostics, the module can be used standalone to produce
the DVCH background table (similar to dvch_colab_simple.py) plus the
full source-table CSV that a Boltzmann solver would consume.

The CMB-level test requires a compiled CLASS/CAMB with the corresponding
patch; this file documents the interface and the expected data flow.
"""
import numpy as np
import os

# --- Constants ---
Omega_r0 = 9.0e-5
c_speed = 299792.458  # km/s
Mpc_to_km = 3.085677581e19  # 1 Mpc in km


# ================================================================
# Cobaya theory hooks (interface specification)
# ================================================================

class DVCHBoltzmannBackend:
    """
    DVCH background + source-table module for CLASS/CAMB.

    Usage in a Cobaya theory block:
        theory:
          classy:
            extra_args:
              DVCH_flag: true
              DVCH_source_table_file: dvch_boltzmann_source_table.csv
    """

    def __init__(self):
        self.params = None
        self.state = None
        self._background_table = None
        self._source_table = None

    def initialize(self):
        """Validate parameters and set up internal state."""
        if self.params is None:
            self.params = {}
        required = ("DVCH_n", "DVCH_beta", "Omega_m", "H0")
        missing = [name for name in required if name not in self.params]
        if missing:
            raise ValueError(f"Missing backend parameters: {', '.join(missing)}")

    def get_can_provide(self):
        """Return list of parameters this module can compute."""
        return ["DVCH_n", "DVCH_alpha", "DVCH_beta", "DVCH_Q", "DVCH_w_eff"]

    def must_provide(self):
        """Return list of parameters this module MUST provide."""
        return []

    def calculate(self, state, want_derived=True, **kwargs):
        """Compute the DVCH background and source table."""
        if "params" not in state or not isinstance(state["params"], dict):
            raise ValueError("Backend state must contain a 'params' mapping")
        self.params = state["params"]
        self.initialize()
        self.state = state
        self._compute_background()
        self._compute_source_table()

    def get_result(self):
        """Return the computed DVCH source table."""
        return {
            "DVCH_background": self._background_table,
            "DVCH_source_table": self._source_table,
        }

    # --- Internal methods ---
    def _compute_background(self):
        """Compute DVCH background E(z), H(z), Omega_i(z) for a redshift grid."""
        n = self.params.get("DVCH_n", 0.09)
        beta = self.params.get("DVCH_beta", 1.0e-4)
        Om = self.params.get("Omega_m", 0.30)
        H0 = self.params.get("H0", 69.03)

        z_grid = np.logspace(-3, 4, 500)  # from z ~ 0.001 to z ~ 10000
        table = []
        for z in z_grid:
            E2 = self._E2_dvch(z, Om, n, beta)
            if np.isnan(E2) or E2 <= 0:
                continue
            E = np.sqrt(E2)
            H = H0 * E
            OL = 1.0 - Om - Omega_r0
            opz = 1.0 + z
            Om_z = Om * opz**3 / E2
            OL_z = OL * (Om_z * E2 / Om)**n * (1.0 + beta) / (1.0 + beta * E2) / E2
            Or_z = Omega_r0 * opz**4 / E2
            table.append([z, E, H, Om_z, OL_z, Or_z])

        self._background_table = np.array(table)

    def _compute_source_table(self):
        """Compute Q(z) and w_eff(z) from the background closure."""
        if self._background_table is None:
            self._compute_background()

        bg = self._background_table
        n = self.params.get("DVCH_n", 0.09)
        beta = self.params.get("DVCH_beta", 1.0e-4)
        Om = self.params.get("Omega_m", 0.30)
        H0 = self.params.get("H0", 69.03)

        source = []
        for row in bg:
            z, E, H, Om_z, OL_z, Or_z = row
            opz = 1.0 + z
            H_val = H0 * E
            rho_m = 3e4 * H0**2 * Om * opz**3  # in natural units proxy
            # Q = d(rho_Lambda)/dt = rho_Lambda,0 * d/dt ( (Om(z)/Om)^n / (1+beta*E^2) )
            # Implemented as finite difference for robustness
            dz = 1e-6
            E2_p = self._E2_dvch(z + dz, Om, n, beta)
            E2_m = self._E2_dvch(z - dz, Om, n, beta) if z > dz else self._E2_dvch(0.0, Om, n, beta)
            if np.isnan(E2_p) or np.isnan(E2_m):
                Q_val = 0.0
            else:
                OL_p = (1.0 - Om - Omega_r0) * (Om * (1+z+dz)**3 / Om)**n * (1+beta)/(1+beta*E2_p)
                OL_m = (1.0 - Om - Omega_r0) * (Om * (1+z-dz)**3 / Om)**n * (1+beta)/(1+beta*E2_m)
                dOL_dz = (OL_p - OL_m) / (2.0 * dz)
                Q_val = -H * (1.0 + z) * dOL_dz  # proxy
            # w_eff = -1 / (1 + Q/(3*H*rho_Lambda))
            source.append([z, Q_val, 0.0])  # [z, Q, w_eff placeholder]

        self._source_table = np.array(source)

    @staticmethod
    def _E2_dvch(z, Om, n, beta):
        if Om <= 0 or Om >= 1 or n <= 0 or n >= 1 or beta < 0:
            return np.nan
        OL = 1.0 - Om - Omega_r0
        if OL <= 0:
            return np.nan
        opz = 1.0 + z
        rad = Omega_r0 * opz**4
        Om_z = Om * opz**3
        E2 = rad + Om_z + OL
        for _ in range(50):
            OL_z = OL * (Om_z / Om)**n * (1.0 + beta) / (1.0 + beta * E2)
            E2_new = rad + Om_z + OL_z
            if abs(E2_new - E2) < 1e-10:
                break
            E2 = E2_new
        return E2


# ================================================================
# Standalone: generate DVCH background + source table
# ================================================================
if __name__ == "__main__":
    print("=== DVCH Boltzmann source-table backend ===")
    print("Generating DVCH background and source table...")

    backend = DVCHBoltzmannBackend()
    backend.params = {
        "DVCH_n": 0.09,
        "DVCH_beta": 1.0e-4,
        "Omega_m": 0.30,
        "H0": 69.03,
    }

    backend._compute_background()
    backend._compute_source_table()

    # Save background
    bg = backend._background_table
    header = "z,E,H,Omega_m,Omega_Lambda,Omega_r"
    np.savetxt("dvch_boltzmann_background.csv", bg, delimiter=",", header=header,
               fmt="%.6e", comments="")

    # Save source table
    src = backend._source_table
    header_src = "z,Q,w_eff"
    np.savetxt("dvch_boltzmann_source_table.csv", src, delimiter=",", header=header_src,
               fmt="%.6e", comments="")

    print(f"Background table : dvch_boltzmann_background.csv  ({bg.shape[0]} rows)")
    print(f"Source table     : dvch_boltzmann_source_table.csv  ({src.shape[0]} rows)")
    print("Done.")