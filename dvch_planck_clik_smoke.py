"""Run one physical CAMB-DVCH point through the official Planck clik.

This runner is intentionally separate from the Windows test suite because the
compiled CAMB and clik libraries are Linux/WSL artifacts. It preserves the
official clik nuisance vector and replaces its TT, EE, and TE blocks with the
CAMB spectra, which makes the point evaluation reproducible without inventing
nuisance values.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _configure_paths(camb_root: Path, clik_egg: Path) -> None:
    sys.path.insert(0, str(camb_root))
    sys.path.insert(0, str(clik_egg))


def evaluate(
    camb_root: Path,
    clik_egg: Path,
    likelihood_path: Path,
    *,
    H0: float,
    ombh2: float,
    omch2: float,
    tau: float,
    As: float,
    ns: float,
    dvch_n: float,
    dvch_beta: float,
) -> float:
    _configure_paths(camb_root, clik_egg)
    import camb  # type: ignore
    import clik  # type: ignore
    import clik.hpy as hpy  # type: ignore

    likelihood = clik.clik(str(likelihood_path))
    vector = np.asarray(hpy.File(str(likelihood_path), "r")["clik/check_param"][:])
    if vector.ndim != 1:
        raise ValueError("Planck check vector must be one-dimensional")

    params = camb.set_params(
        H0=H0,
        ombh2=ombh2,
        omch2=omch2,
        tau=tau,
        As=As,
        ns=ns,
        lmax=2508,
        DVCH_flag=True,
        DVCH_n=dvch_n,
        DVCH_beta=dvch_beta,
    )
    spectra = camb.get_results(params).get_cmb_power_spectra(
        params,
        lmax=2508,
        CMB_unit="muK",
        raw_cl=True,
    )["lensed_scalar"]

    # Planck clik order is TT EE BB TE TB EB. This likelihood has no BB/TB/EB.
    vector[0:2509] = spectra[:, 0]
    vector[2509:5018] = spectra[:, 1]
    vector[5018:7527] = spectra[:, 3]
    if not np.isfinite(vector).all():
        raise ValueError("CAMB/clik vector contains non-finite values")
    return float(np.asarray(likelihood(vector))[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camb-root", type=Path, required=True)
    parser.add_argument("--clik-egg", type=Path, required=True)
    parser.add_argument("--likelihood", type=Path, required=True)
    args = parser.parse_args()
    value = evaluate(
        args.camb_root,
        args.clik_egg,
        args.likelihood,
        H0=67.0,
        ombh2=0.0224,
        omch2=0.12,
        tau=0.054,
        As=2.1e-9,
        ns=0.965,
        dvch_n=0.09,
        dvch_beta=1.0e-4,
    )
    print(f"Planck plik TTTEEE log-likelihood: {value:.8f}")


if __name__ == "__main__":
    main()
