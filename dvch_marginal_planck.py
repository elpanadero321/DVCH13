"""Exact analytic marginalisation over the Gaussian nuisance of Planck plik.

Verified empirically (see _quadratic_test.py, _quadratic_test2.py): the plik
TTTEEE chi2 is *exactly quadratic* in the 18 sampled nuisance parameters when
the theory C_l's are held fixed, because the model vector is affine in them:

    model = R(q_hat) + A_nuis . nuisance

With a Gaussian prior N(mu, P) on the nuisance, the marginal likelihood is

    log L_marg(theory) = max_nu [ log L(theory, nu) + log prior(nu) ]
                        + 0.5 log det(2 pi Sigma_post)
    Sigma_post^{-1} = H + P^{-1},  H = d^2 chi2/d nu^2 (constant)

This is the standard "plik_lite" technique: sample only the cosmology while the
nuisance is handled analytically and exactly. Sampling 8 parameters instead of
26 cuts the random-walk autocorrelation time by a large factor, which is what
is needed to reach R-1 < 0.01.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from dvch_cobaya_planck import DVCHPlanckLikelihood

ROOT = Path(__file__).resolve().parent

SAMPLED_NUISANCE = (
    "gal545_A_100", "gal545_A_143", "gal545_A_143_217", "gal545_A_217",
    "galf_EE_A_100", "galf_EE_A_100_143", "galf_EE_A_100_217", "galf_EE_A_143",
    "galf_EE_A_143_217", "galf_EE_A_217",
    "galf_TE_A_100", "galf_TE_A_100_143", "galf_TE_A_100_217", "galf_TE_A_143",
    "galf_TE_A_143_217", "galf_TE_A_217",
    "calib_100T", "calib_217T",
)

BASE_PARAMS = (
    "H0", "ombh2", "omch2", "tau", "ln10As", "ns", "DVCH_n", "DVCH_beta",
)


def _spectra_blocks(spectra: np.ndarray, lens: np.ndarray, lmax: tuple):
    """Blocks of the theory vector for a clik component with the given lmax."""
    z = np.zeros(spectra.shape[0])
    by_name = {
        "tt": spectra[:, 0], "ee": spectra[:, 1], "bb": spectra[:, 2],
        "te": spectra[:, 3], "tb": z, "eb": z, "pp": lens[:, 0],
    }
    order = ("pp", "tt", "ee", "bb", "te", "tb", "eb") if len(lmax) > 6 else (
        "tt", "ee", "bb", "te", "tb", "eb")
    return np.concatenate(
        [by_name[n][: m + 1] for n, m in zip(order, lmax) if m >= 0]
    )


class DVCHMarginalLikelihood:
    """plik TTTEEE with the 18 Gaussian nuisance marginalised analytically."""

    def initialize(self) -> None:
        from run_dvch_cobaya_full_highl import OFFICIAL_FIXED, OFFICIAL_PRIORS

        self._base = DVCHPlanckLikelihood()
        self._base.initialize()
        self._camb = self._base._camb
        names = tuple(self._base._likelihood.extra_parameter_names)
        idx = {n: i for i, n in enumerate(names)}
        self._s_idx = np.array([idx[n] for n in SAMPLED_NUISANCE])

        check = self._base._check_vectors[0]
        self._nu_fixed = np.array(
            [float(check[7527 + i]) for i in range(len(names))]
        )
        for n, v in OFFICIAL_FIXED.items():
            if n in idx:
                self._nu_fixed[idx[n]] = v

        self._prior_mu = np.array([OFFICIAL_PRIORS[n][0] for n in SAMPLED_NUISANCE])
        self._prior_sd = np.array([OFFICIAL_PRIORS[n][1] for n in SAMPLED_NUISANCE])
        self._P_inv = np.diag(1.0 / self._prior_sd**2)
        self._k = len(SAMPLED_NUISANCE)
        self._mu = self._prior_mu
        self._h = 1e-4 * np.abs(self._mu) + 1e-6
        self._components = self._base._components

        # Hessian of the plik chi2 in nuisance-space. Because the model is
        # affine in the nuisance, H is a CONSTANT (independent of cosmology);
        # we verify this at initialisation with two different cosmologies.
        self._H = self._hessian(np.array(
            [67.32, 0.0224, 0.0120, 0.0546, 3.046, 0.965, 0.076, 0.00047]
        ))
        H2 = self._hessian(np.array(
            [66.0, 0.0230, 0.0135, 0.07, 3.02, 0.955, 0.12, 0.0009]
        ))
        rel = float(np.max(np.abs(H2 - self._H) / np.maximum(np.abs(self._H), 1e-30)))
        if rel > 1e-6:
            raise RuntimeError(
                f"nuisance Hessian is cosmology-dependent (rel diff {rel:.2e}); "
                "the analytic marginalisation would not be exact."
            )
        self._Q_inv = np.linalg.inv(self._H + self._P_inv)
        self._log_term = 0.5 * (
            self._k * np.log(2 * np.pi) + np.linalg.slogdet(
                self._H + self._P_inv
            )[1]
        )

    # ------------------------------------------------------------------
    def _theory(self, theta: np.ndarray):
        As = float(np.exp(theta[4]) / 1.0e10)
        params = self._camb.set_params(
            H0=theta[0], ombh2=theta[1], omch2=theta[2], tau=theta[3],
            As=As, ns=theta[5], lmax=2508,
            DVCH_flag=True, DVCH_n=theta[6], DVCH_beta=theta[7],
        )
        results = self._camb.get_results(params)
        spectra = results.get_cmb_power_spectra(
            params, lmax=2508, CMB_unit="muK", raw_cl=True
        )["lensed_scalar"]
        lens = results.get_lens_potential_cls(lmax=2500, raw_cl=True)
        return spectra, lens

    def _plik_chi2(self, theory_block: np.ndarray, sampled: np.ndarray) -> float:
        nu = self._nu_fixed.copy()
        nu[self._s_idx] = sampled
        return float(
            np.asarray(self._components[0](
                np.concatenate([theory_block, nu])
            ))[0]
        )

    def _theory_block(self, spectra, lens) -> np.ndarray:
        lmax = tuple(int(x) for x in self._components[0].lmax)
        return _spectra_blocks(spectra, lens, lmax)

    def _hessian(self, theta: np.ndarray) -> np.ndarray:
        spectra, lens = self._theory(theta)
        block = self._theory_block(spectra, lens)
        mu, h, k = self._mu, self._h, self._k
        c0 = self._plik_chi2(block, mu)
        H = np.zeros((k, k))
        # diagonal via second differences
        for j in range(k):
            ep = mu.copy(); ep[j] += h[j]
            em = mu.copy(); em[j] -= h[j]
            H[j, j] = (
                self._plik_chi2(block, ep) - 2 * c0 + self._plik_chi2(block, em)
            ) / h[j] ** 2
        # off-diagonal via the standard cross-difference of quadratics
        for j in range(k):
            for l in range(j + 1, k):
                vpp = mu.copy(); vpp[j] += h[j]; vpp[l] += h[l]
                vpm = mu.copy(); vpm[j] += h[j]; vpm[l] -= h[l]
                vmp = mu.copy(); vmp[j] -= h[j]; vmp[l] += h[l]
                vmm = mu.copy(); vmm[j] -= h[j]; vmm[l] -= h[l]
                H[j, l] = H[l, j] = (
                    self._plik_chi2(block, vpp)
                    - self._plik_chi2(block, vpm)
                    - self._plik_chi2(block, vmp)
                    + self._plik_chi2(block, vmm)
                ) / (4 * h[j] * h[l])
        return H

    # ------------------------------------------------------------------
    def logp(self, H0, ombh2, omch2, tau, ln10As, ns, DVCH_n, DVCH_beta) -> float:
        theta = np.array(
            [H0, ombh2, omch2, tau, ln10As, ns, DVCH_n, DVCH_beta], dtype=float
        )
        spectra, lens = self._theory(theta)
        block = self._theory_block(spectra, lens)

        mu, h, k = self._mu, self._h, self._k
        chi2_0 = self._plik_chi2(block, mu)
        # gradient of the chi2 in nuisance-space at the current theory
        g = np.empty(k)
        for j in range(k):
            ep = mu.copy(); ep[j] += h[j]
            em = mu.copy(); em[j] -= h[j]
            g[j] = (self._plik_chi2(block, ep) - self._plik_chi2(block, em)) / (
                2 * h[j]
            )
        # Gaussian integral of exp(-0.5 chi2) over nu ~ N(mu, P):
        #   chi2(nu) = chi2_0 + g.(nu-mu) + 0.5 (nu-mu)^T H (nu-mu)
        #   exponent = -0.5 chi2_0 - g.d - 0.5 d^T (0.5 H + P^{-1}) d
        # (prior contributes -0.5 d^T P^{-1} d, and -0.5 * 0.5 d^T H d = -0.25...)
        # Careful: -0.5*chi2 = -0.5 chi2_0 - 0.5 g.d - 0.25 d^T H d.
        # Combined with prior -0.5 d^T P^{-1} d, the quadratic matrix is
        #   Q = 0.5 H + P^{-1},  linear term  -0.5 g.
        # max value = -0.5 chi2_0 + 0.125 g^T Q^{-1} g, integral factor
        #   det(2 pi Q^{-1})^{1/2}.
        Q = 0.5 * self._H + self._P_inv
        Q_inv = np.linalg.inv(Q)
        quad = 0.125 * float(g @ Q_inv @ g)
        logdet = 0.5 * (k * np.log(2 * np.pi) + np.linalg.slogdet(Q)[1])
        loglike = -0.5 * chi2_0 + quad + logdet

        # Remaining components (low-l TT/EE and lensing): their single nuisance
        # parameter is held fixed at the official value, as in the original run.
        for ci, comp in enumerate(self._components[1:], start=1):
            lmax = tuple(int(x) for x in comp.lmax)
            blocks = _spectra_blocks(spectra, lens, lmax)
            expected = tuple(comp.extra_parameter_names)
            if not expected:
                vec = blocks
            else:
                check = self._base._check_vectors[ci]
                off = len(check) - len(expected)
                nu = np.array([float(check[off + i]) for i in range(len(expected))])
                vec = np.concatenate([blocks, nu])
            if np.isfinite(vec).all():
                loglike += float(np.asarray(comp(vec))[0])
        return loglike


_FUNCTION_LIKELIHOOD: "DVCHMarginalLikelihood | None" = None


def planck_loglike_marg(
    H0, ombh2, omch2, tau, ln10As, ns, DVCH_n, DVCH_beta
) -> float:
    global _FUNCTION_LIKELIHOOD
    if _FUNCTION_LIKELIHOOD is None:
        _FUNCTION_LIKELIHOOD = DVCHMarginalLikelihood()
        _FUNCTION_LIKELIHOOD.initialize()
    return _FUNCTION_LIKELIHOOD.logp(
        H0, ombh2, omch2, tau, ln10As, ns, DVCH_n, DVCH_beta
    )
