"""Cobaya adapter for the alpha vacuum model with STANDARD CAMB (ppf).

The alpha background is exactly a decoupled dark-energy fluid with

    rho_DE(a) = OL0 + alpha*(1-a)^2

so we can use CAMB's standard PPF dark energy with

    w_eff(a) = -1 + 2*alpha*a*(1-a) / (3*(OL0 + alpha*(1-a)^2))

No DVCH CAMB patch needed (DVCH_flag defaults to False).  alpha=0 must be
bit-comparable with plain LCDM through the same code path (control run).

If the env DVCH_PLANCK_LOWL / DVCH_PLANCK_LOWE / DVCH_PLANCK_LENSING are
defined, those likelihoods are added to the total chi2, exactly like
dvch_cobaya_planck.py.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


class AlphaPlanckLikelihood:
    def initialize(self) -> None:
        camb_root = Path(os.environ["DVCH_CAMB_ROOT"])
        clik_egg = Path(os.environ["DVCH_CLIK_EGG"])
        likelihood_path = Path(os.environ["DVCH_PLANCK_LIKELIHOOD"])
        sys.path.insert(0, str(camb_root))
        sys.path.insert(0, str(clik_egg))
        import camb  # type: ignore
        import clik  # type: ignore
        import clik.hpy as hpy  # type: ignore

        self._camb = camb
        self._likelihood = clik.clik(str(likelihood_path))
        self._components = [self._likelihood]
        low_l_path = os.environ.get("DVCH_PLANCK_LOWL")
        low_e_path = os.environ.get("DVCH_PLANCK_LOWE")
        lensing_path = os.environ.get("DVCH_PLANCK_LENSING")
        if low_l_path:
            self._components.append(clik.clik(low_l_path))
        if low_e_path:
            self._components.append(clik.clik(low_e_path))
        if lensing_path:
            self._components.append(clik.clik_lensing(lensing_path))
        self._check_vectors = [
            np.asarray(
                hpy.File(str(path), "r")["clik/check_param"][:]
            )
            for path in (
                [likelihood_path]
                + ([Path(low_l_path)] if low_l_path else [])
                + ([Path(low_e_path)] if low_e_path else [])
            )
        ]
        self._vector = self._check_vectors[0]
        if self._vector.shape != (7574,):
            raise ValueError(
                f"Unexpected Planck vector shape: {self._vector.shape}")

    def _params(self, H0, ombh2, omch2, tau, ln10As, ns, alpha):
        camb = self._camb
        As = float(np.exp(ln10As) / 1.0e10)
        params = camb.set_params(
            H0=H0, ombh2=ombh2, omch2=omch2, tau=tau, As=As, ns=ns,
            lmax=2508, lens_potential_accuracy=1,
            WantTensors=False, DoLensing=True,
        )
        if alpha != 0.0:
            h = H0 / 100.0
            OL = 1.0 - ombh2 / h**2 - omch2 / h**2
            a = np.geomspace(1e-5, 1.0, 2000)
            w = -1.0 + 2.0 * alpha * a * (1.0 - a) / (
                3.0 * (OL + alpha * (1.0 - a) ** 2))
            params.set_dark_energy_w_a(a, w, dark_energy_model="ppf")
        return params

    def logp(
        self,
        H0: float,
        ombh2: float,
        omch2: float,
        tau: float,
        ln10As: float,
        ns: float,
        alpha: float,
        **nuisance: float,
    ) -> float:
        params = self._params(H0, ombh2, omch2, tau, ln10As, ns, alpha)
        results = self._camb.get_results(params)
        spectra = results.get_cmb_power_spectra(
            params, lmax=2508, CMB_unit="muK", raw_cl=True
        )["lensed_scalar"]
        lens = results.get_lens_potential_cls(lmax=2500, raw_cl=True)
        total = 0.0
        for component_index, component in enumerate(self._components):
            lmax = tuple(int(x) for x in component.lmax)
            spectra_by_name = {
                "tt": spectra[:, 0],
                "ee": spectra[:, 1],
                "bb": spectra[:, 2],
                "te": spectra[:, 3],
                "tb": np.zeros(spectra.shape[0]),
                "eb": np.zeros(spectra.shape[0]),
                "pp": lens[:, 0],
            }
            names = ("pp", "tt", "ee", "bb", "te", "tb", "eb") if len(lmax) > 6 else (
                "tt", "ee", "bb", "te", "tb", "eb"
            )
            blocks = [
                spectra_by_name[name][: max_l + 1]
                for name, max_l in zip(names, lmax)
                if max_l >= 0
            ]
            expected_names = tuple(component.extra_parameter_names)
            component_vector = np.concatenate(
                blocks
                + [
                    np.asarray(
                        [
                            nuisance.get(
                                name,
                                (
                                    self._check_vectors[component_index][
                                        -len(expected_names) + index
                                    ]
                                    if component_index
                                    < len(self._check_vectors)
                                    else 1.0
                                ),
                            )
                            for index, name in enumerate(expected_names)
                        ]
                    )
                ]
                if expected_names
                else blocks
            )
            if not np.isfinite(component_vector).all():
                return -np.inf
            total += float(np.asarray(component(component_vector))[0])
        return total


_FUNCTION_LIKELIHOOD: AlphaPlanckLikelihood | None = None


def alpha_planck_loglike(
    H0: float,
    ombh2: float,
    omch2: float,
    tau: float,
    ln10As: float,
    ns: float,
    alpha: float,
    **nuisance: float,
) -> float:
    global _FUNCTION_LIKELIHOOD
    if _FUNCTION_LIKELIHOOD is None:
        _FUNCTION_LIKELIHOOD = AlphaPlanckLikelihood()
        _FUNCTION_LIKELIHOOD.initialize()
    return _FUNCTION_LIKELIHOOD.logp(
        H0, ombh2, omch2, tau, ln10As, ns, alpha, **nuisance
    )
