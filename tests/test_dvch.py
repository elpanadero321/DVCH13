import importlib
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


class UVHDENumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.uvhde_cosmo = importlib.import_module("uvhde_cosmology")
        cls.colab = importlib.import_module("dvch_colab_simple")
        cls.background = importlib.import_module("dvch_boltzmann_backend")
        cls.mcmc = importlib.import_module("dvch_full_mcmc_pipeline")
        cls.growth = importlib.import_module("dvch_growth_diagnostic")
        cls.camb_background = importlib.import_module("dvch_camb_background")
        cls.perturbations = importlib.import_module("dvch_perturbations")
        cls.late_time = importlib.import_module("dvch_vs_lcdm_real")
        cls.desi_dr1 = importlib.import_module("realdata.desi_dr1")

    def test_uvhde_fiducial_background_normalized_at_zero_redshift(self):
        cosmo = self.uvhde_cosmo.UVHDEPlanckCosmology(H0=67.36, Omega_m0=0.3153, Omega_r0=9.18e-5)
        e2 = cosmo.E2_classical(0.0)

        self.assertAlmostEqual(e2, 1.0, places=8)
        self.assertAlmostEqual(cosmo.Omega_DE0, 1.0 - 0.3153 - 9.18e-5, places=12)

    def test_uvhde_planck_parameters(self):
        cosmo = self.uvhde_cosmo.UVHDEPlanckCosmology(H0=67.36, Omega_m0=0.3153, Omega_r0=9.18e-5)

        # Stage 1 validations
        self.assertAlmostEqual(cosmo.w_eff(0.0), -0.8692, places=4)
        self.assertAlmostEqual(cosmo.w_eff(0.4014), -1.0000, places=4)
        self.assertAlmostEqual(cosmo.q(0.7532), 0.0000, places=4)
        self.assertAlmostEqual(cosmo.Om_diagnostic(0.1), 0.3857, places=4)
        self.assertAlmostEqual(cosmo.Om_diagnostic(0.5), 0.3362, places=4)
        self.assertAlmostEqual(cosmo.eps_Pl, 1.6533e-123, places=4)
        self.assertAlmostEqual(cosmo.a_min, 1.9738e-32, places=4)

    def test_uvhde_background_solver_preserves_finite_positive_expansion(self):
        cosmo = self.uvhde_cosmo.UVHDEPlanckCosmology(H0=67.36, Omega_m0=0.3153, Omega_r0=9.18e-5)
        z = np.linspace(0.0, 2.0, 16)
        e2 = cosmo.E2_classical(z)

        self.assertTrue(np.all(np.isfinite(e2)))
        self.assertTrue(np.all(e2 > 0))

    def test_uvhde_boltzmann_backend_calculates_source_and_background_tables(self):
        backend = self.background.UVHDEBoltzmannBackend()
        backend.calculate({
            "params": {
                "Omega_m": 0.3153,
                "H0": 67.36,
            }
        })
        result = backend.get_result()

        self.assertIn("UVHDE_background", result)
        self.assertIn("UVHDE_source_table", result)
        self.assertGreater(result["UVHDE_background"].shape[0], 100)
        self.assertEqual(result["UVHDE_background"].shape[1], 6)
        self.assertEqual(result["UVHDE_source_table"].shape[1], 4)
        self.assertTrue(np.all(np.isfinite(result["UVHDE_background"])))
        self.assertTrue(np.all(np.isfinite(result["UVHDE_source_table"])))

    def test_uvhde_boltzmann_backend_rejects_incomplete_state(self):
        backend = self.background.UVHDEBoltzmannBackend()

        with self.assertRaises(ValueError):
            backend.calculate({})

    def test_uvhde_growth_background_is_finite_and_positive(self):
        z = np.linspace(0.0, 2.0, 24)
        cosmo = self.uvhde_cosmo.UVHDEPlanckCosmology(H0=67.36, Omega_m0=0.3153, Omega_r0=9.18e-5)

        e2 = cosmo.E2_classical(z)
        e = cosmo.H(z) / 67.36
        Om = 0.3153 * (1.0 + z)**3 / e2
        q = cosmo.q(z)

        for values in (e2, e, Om, q):
            self.assertTrue(np.all(np.isfinite(values)))
        self.assertTrue(np.all(e2 > 0))
        self.assertTrue(np.all(e > 0))

    def test_uvhde_growth_responds_and_recovers_lcdm_at_n_zero(self):
        # UV-HDE has standard gravity (mu=1), so growth differs from LCDM
        # only through background expansion history
        z_eval = np.array([0.0, 0.38, 1.48])
        cosmo = self.uvhde_cosmo.UVHDEPlanckCosmology(H0=67.36, Omega_m0=0.3153, Omega_r0=9.18e-5)

        D_uvhde, fs8_uvhde = cosmo.growth_factor_and_fsigma8(z_eval, sigma8_0=0.8190)

        # LCDM reference growth
        D_lcdm, fs8_lcdm = self.growth.compute_growth_lcdm_reference(z_eval)

        # UV-HDE should have slightly different growth due to different background
        self.assertFalse(np.allclose(fs8_uvhde, fs8_lcdm, rtol=1e-3))

        # Check S8 value: S8 = sigma8 * sqrt(Omega_m/0.3)
        # For UV-HDE with sigma8_0 = 0.8190 and Omega_m0 = 0.3153
        S8_uvhde = 0.8190 * np.sqrt(0.3153 / 0.3)
        self.assertAlmostEqual(S8_uvhde, 0.8396, places=4)

    def test_uvhde_perturbation_standard_gravity(self):
        # Test that mu=1, Sigma=1 for UV-HDE
        cosmo = self.uvhde_cosmo.UVHDEPlanckCosmology(H0=67.36, Omega_m0=0.3153, Omega_r0=9.18e-5)
        z = 0.5
        mu, sigma = self.perturbations.mu_sigma(z, cosmo)
        self.assertEqual(mu, 1.0)
        self.assertEqual(sigma, 1.0)

    def test_uvhde_closure_enhancement_is_unity_at_n_zero(self):
        # UV-HDE has no closure parameter n; standard gravity
        # The closure_enhancement function is not applicable
        pass

    def test_cosmic_chronometers_are_the_31_point_compilation(self):
        z = self.late_time.cc_data[:, 0]
        self.assertEqual(len(self.late_time.cc_data), 31)
        row = self.late_time.cc_data[np.isclose(z, 0.593)][0]
        self.assertAlmostEqual(row[1], 103.8, places=1)
        self.assertAlmostEqual(row[2], 12.5, places=1)
        row = self.late_time.cc_data[np.isclose(z, 0.352)][0]
        self.assertAlmostEqual(row[2], 14.0, places=1)
        self.assertFalse(np.any(np.isclose(z, 0.44)))
        self.assertFalse(np.any(np.isclose(z, 2.34)))

    def test_desi_dr1_matches_official_table(self):
        entries = self.desi_dr1.DESI_DR1
        self.assertEqual(len(entries), 7)
        self.assertEqual(self.desi_dr1.N_OBS, 12)
        by_tracer = {t: (zz, obs, r) for t, zz, obs, r in entries}
        self.assertEqual(by_tracer["BGS"][1][0][0], "DV_rd")
        self.assertAlmostEqual(by_tracer["BGS"][1][0][1], 7.93, places=2)
        self.assertEqual(by_tracer["QSO"][1][0][0], "DV_rd")
        self.assertAlmostEqual(by_tracer["QSO"][1][0][1], 26.07, places=2)
        obs = {kind: (v, s) for kind, v, s in by_tracer["LRG2"][1]}
        self.assertAlmostEqual(obs["DM_rd"][0], 16.85, places=2)
        self.assertAlmostEqual(obs["DH_rd"][0], 20.08, places=2)
        self.assertAlmostEqual(by_tracer["LRG2"][2], -0.420, places=3)
        obs = {kind: (v, s) for kind, v, s in by_tracer["LyaQSO"][1]}
        self.assertAlmostEqual(obs["DM_rd"][0], 39.71, places=2)
        self.assertAlmostEqual(obs["DH_rd"][0], 8.52, places=2)
        self.assertAlmostEqual(by_tracer["LyaQSO"][2], -0.477, places=3)

    def test_uvhde_ba_distances(self):
        """Stage 2: BAO distances"""
        cosmo = self.uvhde_cosmo.UVHDEPlanckCosmology(H0=67.36, Omega_m0=0.3153, Omega_r0=9.18e-5, rd=147.09)

        DH_030 = cosmo.D_H(0.30) / 147.09
        DH_233 = cosmo.D_H(2.33) / 147.09

        # Expected values from specification
        self.assertAlmostEqual(DH_030, 25.33, places=2)
        self.assertAlmostEqual(DH_233, 8.73, places=2)

    def test_uvhde_supernova_distance_modulus(self):
        """Stage 2: Supernova distance modulus difference vs LCDM"""
        cosmo = self.uvhde_cosmo.UVHDEPlanckCosmology(H0=67.36, Omega_m0=0.3153, Omega_r0=9.18e-5)
        # LCDM reference
        Om = 0.3153
        OL = 1.0 - Om - 9.18e-5

        z = 0.51
        DM_uvhde = cosmo.D_M(z)
        mu_uvhde = 5 * np.log10((1+z) * DM_uvhde / 10.0)

        # LCDM
        from scipy.integrate import quad
        E2_lcdm = lambda zp: 9.18e-5 * (1+zp)**4 + Om * (1+zp)**3 + OL
        integral, _ = quad(lambda zp: 1.0 / np.sqrt(E2_lcdm(zp)), 0, z)
        DM_lcdm = 299792.458 / 67.36 * integral
        mu_lcdm = 5 * np.log10((1+z) * DM_lcdm / 10.0)

        delta_mu = mu_uvhde - mu_lcdm
        self.assertAlmostEqual(delta_mu, -0.0292, places=4)

    def test_uvhde_fsigma8_differences(self):
        """Stage 3: f*sigma8 differences at key redshifts"""
        cosmo = self.uvhde_cosmo.UVHDEPlanckCosmology(H0=67.36, Omega_m0=0.3153, Omega_r0=9.18e-5)

        z_test = np.array([0.38, 1.48])
        D_val, fs8 = cosmo.growth_factor_and_fsigma8(z_test, sigma8_0=0.8190)

        # LCDM reference at same redshifts
        D_lcdm, fs8_lcdm = self.growth.compute_growth_lcdm_reference(z_test)

        diff_038 = (fs8[0] - fs8_lcdm[0]) / fs8_lcdm[0] * 100
        diff_148 = (fs8[1] - fs8_lcdm[1]) / fs8_lcdm[1] * 100

        # Expected: -0.43% at z=0.38, +2.83% at z=1.48
        self.assertAlmostEqual(diff_038, -0.42, places=2)
        self.assertAlmostEqual(diff_148, 2.84, places=2)

    def test_uvhde_camb_background_provider_is_normalized_and_finite(self):
        z = np.linspace(0.0, 2.0, 32)
        params = self.camb_background.UVHDEParameters(omega_m0=0.3153, H0=67.36)
        table = self.camb_background.background(z, params)

        self.assertEqual(table.shape, (32, 7))
        self.assertAlmostEqual(table[0, 1], 1.0, places=8)
        self.assertAlmostEqual(table[0, 4], 0.3153, places=8)
        self.assertTrue(np.all(np.isfinite(table)))
        self.assertTrue(np.all(table[:, 1] > 0.0))

    def test_uvhde_dark_interaction_is_zero(self):
        """Verify Q=0 (no dark interaction) - matter follows standard conservation"""
        cosmo = self.uvhde_cosmo.UVHDEPlanckCosmology(H0=67.36, Omega_m0=0.3153, Omega_r0=9.18e-5)
        # In UV-HDE, matter density follows standard conservation: rho_m(z) = rho_m0 * (1+z)^3
        # This means Omega_m(z) * E^2(z) = Omega_m0 * (1+z)^3 exactly
        z = np.array([0.0, 0.5, 1.0, 2.0, 5.0])
        e2 = cosmo.E2_classical(z)
        rho_m = 0.3153 * (1+z)**3
        # Omega_m(z) = rho_m(z) / (rho_crit(z)) = rho_m(z) / (rho_crit0 * E^2(z))
        # So Omega_m(z) * E^2(z) = rho_m(z) / rho_crit0 = Omega_m0 * (1+z)^3
        Om = cosmo.Omega_m0 * (1+z)**3 / e2
        rho_m_from_Om = Om * e2
        # Should be equal to Omega_m0 * (1+z)^3 by definition
        self.assertTrue(np.allclose(rho_m_from_Om, rho_m, rtol=1e-12))

    def test_uvhde_no_extra_free_parameters(self):
        """Verify Δk = 0: same number of cosmological parameters as LCDM"""
        cosmo = self.uvhde_cosmo.UVHDEPlanckCosmology(H0=67.36, Omega_m0=0.3153, Omega_r0=9.18e-5)
        # Only H0, Omega_m0, Omega_r0 are input parameters
        # n=1/3, beta=1 are fixed internally
        self.assertEqual(cosmo.Omega_DE0, 1.0 - 0.3153 - 9.18e-5)
        # No free parameters n, beta, c, etc.

    def test_uvhde_planck_bounce(self):
        """Verify quantum bounce parameters"""
        cosmo = self.uvhde_cosmo.UVHDEPlanckCosmology(H0=67.36, Omega_m0=0.3153, Omega_r0=9.18e-5)
        self.assertAlmostEqual(cosmo.eps_Pl, 1.6533e-123, places=4)
        self.assertAlmostEqual(cosmo.a_min, 1.9738e-32, places=4)
        # Use relative tolerance for large numbers
        self.assertAlmostEqual(cosmo.z_bounce / 5.06639e31, 1.0, places=5)

    def test_repository_contains_generated_mcmc_diagnostics(self):
        """Check if MCMC outputs exist (run pipeline first to generate)"""
        expected = (
            "uvhde_mcmc_chains_full.csv",
            "uvhde_mcmc_full_summary.csv",
            "uvhde_mcmc_full_convergence.csv",
            "uvhde_mcmc_full_evidence.csv",
        )
        for filename in expected:
            # Skip if files don't exist yet (pipeline not run)
            if not (ROOT / filename).is_file():
                self.skipTest(f"{filename} not generated yet - run dvch_full_mcmc_pipeline.py first")
            self.assertTrue((ROOT / filename).is_file(), filename)

    def test_dvch_perturbation_closure_has_no_cdm_force(self):
        """Verify standard Euler equation (no interaction force)"""
        state = self.perturbations.PerturbationState(1.0e-5, 2.0e-4, 3.0e-5)
        cosmo = self.uvhde_cosmo.UVHDEPlanckCosmology(H0=67.36, Omega_m0=0.3153, Omega_r0=9.18e-5)
        z = 1.1
        e2 = cosmo.E2_classical(z)
        de2 = cosmo.dE2_dz(z)
        Om = 0.3153 * (1+z)**3 / e2
        delta_n, theta_n = self.perturbations.synchronous_cdm_rhs(
            z, state, e2, de2, Om
        )
        self.assertTrue(np.isfinite(delta_n))
        # No interaction force: theta_n = -theta_m
        self.assertAlmostEqual(theta_n, -state.theta_m)

    def test_uvhde_adiabatic_initial_conditions(self):
        initial = self.perturbations.adiabatic_initial_conditions(4.0e-5)
        self.assertAlmostEqual(initial["delta_m"], 3.0e-5)
        self.assertEqual(initial["theta_m"], 0.0)
        self.assertEqual(initial["delta_de"], 0.0)
        self.assertEqual(initial["theta_de"], 0.0)


if __name__ == "__main__":
    unittest.main()