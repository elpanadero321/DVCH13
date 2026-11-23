#!/usr/bin/env python3
"""
UV-HDE + Planck Limit modified Boltzmann source-table backend.

Provides the UV-HDE background and source functions that a patched
CLASS or CAMB implementation could call during the background and
perturbation integration. This module is an interface specification and
standalone diagnostic; it is not itself a compiled Boltzmann solver:
it defines the `initialize`, `get_can_provide`, `must_provide`,
`calculate`, and `get_result` hooks expected by the standard Cobaya
theory API, together with the UV-HDE-specific background and source table.

For late-time diagnostics, the module can be used standalone to produce
the UV-HDE background table plus the full source-table CSV that a
Boltzmann solver would consume.

The CMB-level test requires a compiled CLASS/CAMB with the corresponding
patch; this file documents the interface and the expected data flow.

Model: UV-HDE + Límite de Planck (versión definitiva, Δk = 0).
- Sin interacción oscura (Q = 0).
- Cero parámetros libres extra frente a ΛCDM (n = 1/3, β = 1 fijados).
- Gravedad estándar en perturbaciones (μ = 1, Σ = 1).
"""
import numpy as np
from uvhde_cosmology import UVHDEPlanckCosmology

# --- Constants ---
Omega_r0 = 9.18e-5
c_speed = 299792.458  # km/s


# ================================================================
# Cobaya theory hooks (interface specification)
# ================================================================

class UVHDEBoltzmannBackend:
    """
    UV-HDE background + source-table module for CLASS/CAMB.

    Usage in a Cobaya theory block:
        theory:
          classy:
            extra_args:
              UVHDE_flag: true
              UVHDE_source_table_file: uvhde_boltzmann_source_table.csv
    """

    def __init__(self):
        self.params: dict[str, float] | None = None
        self.state = None
        self._background_table: np.ndarray | None = None
        self._source_table: np.ndarray | None = None
        self._cosmo: UVHDEPlanckCosmology | None = None

    def initialize(self):
        """Validate parameters and set up internal state."""
        if self.params is None:
            self.params = {}
        required = ("Omega_m", "H0")
        missing = [name for name in required if name not in self.params]
        if missing:
            raise ValueError(f"Missing backend parameters: {', '.join(missing)}")

    def get_can_provide(self):
        """Return list of parameters this module can compute."""
        return ["UVHDE_w_eff", "UVHDE_q", "UVHDE_Om_diagnostic"]

    def must_provide(self):
        """Return list of parameters this module MUST provide."""
        return []

    def calculate(self, state, want_derived=True, **kwargs):
        """Compute the UV-HDE background and source table."""
        del want_derived, kwargs
        if "params" not in state or not isinstance(state["params"], dict):
            raise ValueError("Backend state must contain a 'params' mapping")
        self.params = state["params"]
        self.initialize()
        self.state = state
        self._compute_background()
        self._compute_source_table()

    def get_result(self):
        """Return the computed UV-HDE source table."""
        return {
            "UVHDE_background": self._background_table,
            "UVHDE_source_table": self._source_table,
        }

    # --- Internal methods ---
    def _compute_background(self):
        """Compute UV-HDE background E(z), H(z), Omega_i(z) for a redshift grid."""
        if self.params is None:
            raise RuntimeError("Backend parameters must be initialized before computing")

        Om = self.params.get("Omega_m", 0.3153)
        H0 = self.params.get("H0", 67.36)

        self._cosmo = UVHDEPlanckCosmology(H0=H0, Omega_m0=Om, Omega_r0=Omega_r0)

        # Redshift grid: from z ~ 0.001 to z ~ 10000 (log spacing)
        z_grid = np.logspace(-3, 4, 500)
        table = []
        for z in z_grid:
            e2 = self._cosmo.E2_classical(z)
            if np.isnan(e2) or e2 <= 0:
                continue
            E = np.sqrt(e2)
            H = H0 * E
            opz = 1.0 + z
            Om_z = Om * opz**3 / e2
            OL_z = self._cosmo.Omega_DE0 * opz * 2.0 / (1.0 + e2) / e2
            Or_z = Omega_r0 * opz**4 / e2
            table.append([z, E, H, Om_z, OL_z, Or_z])

        self._background_table = np.array(table)

    def _compute_source_table(self):
        """Compute w_eff(z), q(z), Om_diagnostic(z) from the background."""
        if self.params is None:
            raise RuntimeError("Backend parameters must be initialized before computing")
        if self._background_table is None:
            self._compute_background()
        if self._cosmo is None:
            raise RuntimeError("Cosmology object not initialized")

        bg = self._background_table
        if bg is None:
            raise RuntimeError("Background table was not computed")

        source = []
        for row in bg:
            z, E, H, Om_z, OL_z, Or_z = row
            w_eff = self._cosmo.w_eff(z)
            q_val = self._cosmo.q(z)
            Om_diag = self._cosmo.Om_diagnostic(z)
            source.append([z, w_eff, q_val, Om_diag])

        self._source_table = np.array(source)


# ================================================================
# Standalone: generate UV-HDE background + source table
# ================================================================
if __name__ == "__main__":
    print("=== UV-HDE + Planck Limit Boltzmann source-table backend ===")
    print("Generating UV-HDE background and source table...")

    backend = UVHDEBoltzmannBackend()
    backend.params = {
        "Omega_m": 0.3153,
        "H0": 67.36,
    }

    backend._compute_background()
    backend._compute_source_table()

    # Save background
    bg = backend._background_table
    if bg is None:
        raise RuntimeError("Background table was not generated")
    header = "z,E,H,Omega_m,Omega_DE,Omega_r"
    np.savetxt("uvhde_boltzmann_background.csv", bg, delimiter=",", header=header,
               fmt="%.6e", comments="")

    # Save source table
    src = backend._source_table
    if src is None:
        raise RuntimeError("Source table was not generated")
    header_src = "z,w_eff,q,Om_diagnostic"
    np.savetxt("uvhde_boltzmann_source_table.csv", src, delimiter=",", header=header_src,
               fmt="%.6e", comments="")

    print(f"Background table : uvhde_boltzmann_background.csv  ({bg.shape[0]} rows)")
    print(f"Source table     : uvhde_boltzmann_source_table.csv  ({src.shape[0]} rows)")
    print("Done.")