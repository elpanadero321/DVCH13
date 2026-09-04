"""Cobaya adapter for the compiled DVCH CAMB and Planck plik likelihood.

The nuisance vector is initialized from Planck's official ``check_param``
vector. Cosmological parameters are sampled; nuisance optimization is a
separate required production step and is intentionally not claimed here.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


class DVCHPlanckLikelihood:
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
            raise ValueError(f"Unexpected Planck vector shape: {self._vector.shape}")

    def logp(
        self,
        H0: float,
        ombh2: float,
        omch2: float,
        tau: float,
        ln10As: float,
        ns: float,
        DVCH_n: float,
        DVCH_beta: float,
        **nuisance: float,
    ) -> float:
        As = float(np.exp(ln10As) / 1.0e10)
        params = self._camb.set_params(
            H0=H0,
            ombh2=ombh2,
            omch2=omch2,
            tau=tau,
            As=As,
            ns=ns,
            lmax=2508,
            DVCH_flag=True,
            DVCH_n=DVCH_n,
            DVCH_beta=DVCH_beta,
        )
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
                                    if component_index < len(self._check_vectors)
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


_FUNCTION_LIKELIHOOD: DVCHPlanckLikelihood | None = None


def planck_loglike(
    H0: float,
    ombh2: float,
    omch2: float,
    tau: float,
    ln10As: float,
    ns: float,
    DVCH_n: float,
    DVCH_beta: float,
    **nuisance: float,
) -> float:
    global _FUNCTION_LIKELIHOOD
    if _FUNCTION_LIKELIHOOD is None:
        _FUNCTION_LIKELIHOOD = DVCHPlanckLikelihood()
        _FUNCTION_LIKELIHOOD.initialize()
    return _FUNCTION_LIKELIHOOD.logp(
        H0, ombh2, omch2, tau, ln10As, ns, DVCH_n, DVCH_beta, **nuisance
    )
