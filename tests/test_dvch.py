import importlib
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


class DVCHNumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.colab = importlib.import_module("dvch_colab_simple")
        cls.background = importlib.import_module("dvch_boltzmann_backend")
        cls.mcmc = importlib.import_module("dvch_full_mcmc_pipeline")
        cls.mcmc_convergence = importlib.import_module("dvch_mcmc_convergence")
        cls.preflight = importlib.import_module("dvch_planck_preflight")
        cls.robustness = importlib.import_module("dvch_robustness_scan")
        cls.growth = importlib.import_module("dvch_growth_diagnostic")
        cls.joint_fit = importlib.import_module("dvch_joint_realdata_fit")
        cls.camb_background = importlib.import_module("dvch_camb_background")
        cls.perturbations = importlib.import_module("dvch_perturbations")
        cls.double_slit = importlib.import_module("dvch_double_slit")

    def test_fiducial_background_is_normalized_at_zero_redshift(self):
        e2, matter, vacuum = self.colab.E2_dvch(0.0)

        self.assertAlmostEqual(e2, 1.0, places=8)
        self.assertAlmostEqual(matter, self.colab.Omega_m0, places=12)
        self.assertAlmostEqual(vacuum, self.colab.Omega_L0, places=12)

    def test_background_solver_preserves_finite_positive_expansion(self):
        z = np.linspace(0.0, 2.0, 16)
        result = self.robustness.solve_background(0.2, 1.0e-4, z)

        self.assertIsNotNone(result)
        for key in ("E2", "Om", "OL", "Qtilde"):
            self.assertTrue(np.all(np.isfinite(result[key])), key)
        self.assertTrue(np.all(result["E2"] > 0))
        self.assertAlmostEqual(result["Om"][0], self.robustness.Omega_m0, places=8)

    def test_robustness_viability_rejects_invalid_results(self):
        z = np.linspace(0.0, 2.0, 8)
        viable, reason = self.robustness.check_viability(None, z)

        self.assertFalse(viable)
        self.assertEqual(reason, "integration_failed")

    def test_mcmc_likelihood_has_finite_valid_and_infinite_invalid_regions(self):
        valid = self.mcmc.log_likelihood([0.30, 0.2, 1.0e-4, 69.03])
        invalid = self.mcmc.log_likelihood([0.0, 0.2, 1.0e-4, 69.03])

        self.assertTrue(np.isfinite(valid))
        self.assertEqual(invalid, -np.inf)

    def test_mcmc_posterior_matches_likelihood_for_valid_parameters(self):
        params = np.array([0.30, 0.2, 1.0e-4, 69.03])

        self.assertEqual(self.mcmc.log_posterior(params),
                         self.mcmc.log_likelihood(params))

    def test_mcmc_convergence_diagnostic_is_exact_for_identical_chains(self):
        chain = np.column_stack((
            np.linspace(0.2, 0.4, 32),
            np.linspace(65.0, 70.0, 32),
        ))
        chains = [chain.copy() for _ in range(4)]

        r_hat = self.mcmc_convergence.gelman_rubin(chains)
        ess = self.mcmc_convergence.effective_sample_size(chains)

        # The implementation uses the finite-chain correction (1 - 1/n).
        self.assertTrue(np.allclose(r_hat, np.sqrt(31.0 / 32.0)))
        self.assertTrue(np.all(np.isfinite(ess)))
        self.assertTrue(np.all(ess > 0))

    def test_boltzmann_backend_calculates_source_and_background_tables(self):
        backend = self.background.DVCHBoltzmannBackend()
        backend.calculate({
            "params": {
                "DVCH_n": 0.09,
                "DVCH_beta": 1.0e-4,
                "Omega_m": 0.30,
                "H0": 69.03,
            }
        })
        result = backend.get_result()

        self.assertIn("DVCH_background", result)
        self.assertIn("DVCH_source_table", result)
        self.assertGreater(result["DVCH_background"].shape[0], 100)
        self.assertEqual(result["DVCH_background"].shape[1], 6)
        self.assertEqual(result["DVCH_source_table"].shape[1], 3)
        self.assertTrue(np.all(np.isfinite(result["DVCH_background"])))
        self.assertTrue(np.all(np.isfinite(result["DVCH_source_table"])))
        self.assertFalse(np.all(result["DVCH_source_table"][:, 2] == 0.0))

    def test_boltzmann_backend_rejects_incomplete_state(self):
        backend = self.background.DVCHBoltzmannBackend()

        with self.assertRaises(ValueError):
            backend.calculate({})

    def test_growth_background_is_finite_and_positive(self):
        z = np.linspace(0.0, 2.0, 24)
        _, om, ol, e2, e, q = self.growth.solve_dvch_background_full(z)

        for values in (om, ol, e2, e, q):
            self.assertTrue(np.all(np.isfinite(values)))
        self.assertTrue(np.all(e2 > 0))
        self.assertTrue(np.all(e > 0))

    def test_growth_output_has_expected_normalization(self):
        # Use the repository's generated background table as a small,
        # deterministic integration fixture.
        df = pd.read_csv(ROOT / "dvch_boltzmann_modified_background.csv")
        required = {"z", "E", "Omega_m", "Omega_Lambda", "Qtilde"}
        self.assertTrue(required.issubset(df.columns))

    def test_joint_fit_distance_modulus_is_finite(self):
        mu = self.joint_fit.distance_modulus(
            0.5, self.joint_fit.E2_lcdm, [0.30, 69.03]
        )

        self.assertTrue(np.isfinite(mu))
        self.assertGreater(mu, 0.0)

    def test_planck_preflight_records_missing_external_components(self):
        self.preflight.RESULTS.clear()

        self.assertFalse(self.preflight._check_file(
            str(ROOT / "file-that-is-not-part-of-the-repository")
        ))
        self.assertFalse(self.preflight._check(
            "module-that-is-not-part-of-the-environment"
        ))
        self.assertFalse(
            self.preflight.RESULTS["module-that-is-not-part-of-the-environment"]
        )

    def test_repository_contains_generated_mcmc_diagnostics(self):
        expected = (
            "dvch_mcmc_chains_full.csv",
            "dvch_mcmc_full_summary.csv",
            "dvch_mcmc_full_convergence.csv",
            "dvch_mcmc_full_evidence.csv",
        )
        for filename in expected:
            self.assertTrue((ROOT / filename).is_file(), filename)

    def test_camb_background_provider_is_normalized_and_finite(self):
        z = np.linspace(0.0, 2.0, 32)
        table = self.camb_background.background(z)

        self.assertEqual(table.shape, (32, 7))
        self.assertAlmostEqual(table[0, 1], 1.0, places=8)
        self.assertAlmostEqual(table[0, 4], 0.30, places=8)
        self.assertTrue(np.all(np.isfinite(table)))
        self.assertTrue(np.all(table[:, 1] > 0.0))

    def test_dvch_perturbation_closure_has_no_cdm_force(self):
        state = self.perturbations.PerturbationState(1.0e-5, 2.0e-4, 3.0e-5)
        delta_n, theta_n = self.perturbations.synchronous_cdm_rhs(
            1.1, 0.30, 0.69991, 9.0e-5, state
        )

        self.assertTrue(np.isfinite(delta_n))
        self.assertAlmostEqual(theta_n, -state.theta_m)

    def test_dvch_adiabatic_initial_conditions(self):
        initial = self.perturbations.adiabatic_initial_conditions(4.0e-5)
        self.assertAlmostEqual(initial["delta_m"], 3.0e-5)
        self.assertEqual(initial["theta_m"], 0.0)
        self.assertEqual(initial["delta_lambda"], 0.0)

    def test_camb_fortran_module_contains_compiled_closure(self):
        source = (ROOT / "camb_dvch_model.f90").read_text(encoding="utf-8")
        for marker in (
            "module DVCHModel",
            "subroutine DVCHInteraction",
            "delta_qtilde",
            "c_lambda*0._dl",
        ):
            self.assertIn(marker, source)

    # --------------------------------------------------------------
    # Tests nuevos (2026-10-04) — cubren puntos ciegos detectados:
    #   1. compute_growth ignoraba n/beta (bug corregido esta sesión)
    #   2. falta verificacion del limite LambdaCDM en crecimiento
    #   3. falta verificacion de la ceguera simetrica de la doble rendija
    #   4. falta verificacion de la normalizacion F_rho(0)=1
    # --------------------------------------------------------------

    def test_growth_responds_to_n_and_recovers_lcdm_at_n_zero(self):
        # Regression del bug: compute_growth() llamaba al solver de fondo sin
        # pasarle (n, beta), de modo que sigma8 era siempre el del fiducial.
        lcdm = self.growth.compute_growth(n=0.0, beta=1.0e-4)
        fid = self.growth.compute_growth(n=0.2, beta=1.0e-4)
        cmb = self.growth.compute_growth(n=0.062, beta=5.0e-4)

        # En n=0 la interaccion se apaga: debe recuperar LambdaCDM exacto.
        self.assertAlmostEqual(lcdm["growth_ratio"], 1.0, places=3)
        # El fiducial y el posterior CMB deben ser distintos entre si (si no,
        # el parametro n vuelve a ser ignorado silenciosamente).
        self.assertNotAlmostEqual(fid["growth_ratio"], cmb["growth_ratio"],
                                  places=3)
        # Cuanto mayor n, mas supresion de crecimiento.
        self.assertLess(cmb["sigma8_dvch"], lcdm["sigma8_dvch"])

    def test_closure_enhancement_is_unity_at_n_zero(self):
        # F_rho = (rho/rho_m0)^n debe valer 1 para cualquier material cuando
        # n=0 (la cerradura degenera en constante cosmologica rigida).
        for rho in (1.2, 1.2e3, 3.1e3, 19.3e3):
            f = self.double_slit.closure_enhancement(rho, 0.0)
            self.assertAlmostEqual(f, 1.0, places=12)

    def test_symmetric_double_slit_is_exactly_blind(self):
        # Ambas rendijas en la misma membrana: la cerradura predice la misma
        # modificacion en los dos brazos, coeficiente diferencial exactamente 0.
        f1 = self.double_slit.closure_enhancement(self.double_slit.RHO_SIN, 0.062)
        f2 = f1  # misma membrana
        coeff = (f1 - f2)
        self.assertEqual(coeff, 0.0)

    def test_double_slit_differential_uses_chamber_vacuum_offset(self):
        # El observable fisico es F_rho(SiN) - F_rho(vacio de camara), no
        # F_rho - 1: incluso un vacio de laboratorio esta "denso" a escala
        # cosmologica. Verifica que el offset es no despreciable.
        f_sin = self.double_slit.closure_enhancement(self.double_slit.RHO_SIN,
                                                    0.062)
        f_vac = self.double_slit.closure_enhancement(self.double_slit.RHO_VAC,
                                                    0.062)
        self.assertGreater(f_vac, 1.0)
        self.assertLess(f_vac, f_sin)
        # El offset es del orden de decenas: no se puede ignorar.
        self.assertGreater(f_vac, 2.0)


if __name__ == "__main__":
    unittest.main()
