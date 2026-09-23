"""Validate exact quadraticity of the plik chi2 in the 18 SAMPLED nuisance params.

The main plik likelihood is linear in the sampled nuisance amplitudes
(gal545_*, galf_*, calib_100T, calib_217T) when the theory Cl's are held fixed.
If chi2(C_l, nu) is exactly quadratic in nu, then the nuisance can be
marginalised analytically (Gaussian integral), giving an exact marginal
likelihood that depends only on the 8 cosmological parameters.

Test: hold the CAMB theory spectra fixed, perturb nu along random directions,
and check that the third finite difference vanishes to machine precision.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, "/home/danieproyect/plc-3.1/clik-egg")
sys.path.insert(0, os.environ["DVCH_CAMB_ROOT"])

import clik  # noqa: E402
import clik.hpy as hpy  # noqa: E402
import camb  # noqa: E402

from run_dvch_cobaya_full_highl import OFFICIAL_PRIORS, OFFICIAL_FIXED  # noqa: E402

LIK = Path(os.environ["DVCH_PLANCK_LIKELIHOOD"])


def build_vector_factory():
    """Return (vec0, sampled_names, sampled_mu, sampled_scale, set_nuisance)."""
    lk = clik.clik(str(LIK))
    names = tuple(lk.extra_parameter_names)
    check = np.asarray(hpy.File(str(LIK), "r")["clik/check_param"][:])
    nu_default = np.array([float(check[7527 + i]) for i in range(len(names))])

    sampled = [n for n in names if n in OFFICIAL_PRIORS]
    idx = {n: i for i, n in enumerate(names)}
    mu = np.array([OFFICIAL_PRIORS[n][0] for n in sampled])
    scale = np.array([OFFICIAL_PRIORS[n][1] for n in sampled])
    # put sampled nuisance at their prior means, others at official values
    nu = nu_default.copy()
    for n in sampled:
        nu[idx[n]] = OFFICIAL_PRIORS[n][0]
    for n, v in OFFICIAL_FIXED.items():
        if n in idx:
            nu[idx[n]] = v
    return lk, names, idx, sampled, mu, scale, nu


def theory_vector(lk, names) -> np.ndarray:
    """Build the theory Cl part (first 7527 entries) from a real CAMB run."""
    p = camb.set_params(
        H0=67.32, ombh2=0.0224, omch2=0.0120, tau=0.0546,
        As=np.exp(3.046) / 1e10, ns=0.965, lmax=2508,
        DVCH_flag=True, DVCH_n=0.076, DVCH_beta=0.00047,
    )
    r = camb.get_results(p)
    spectra = r.get_cmb_power_spectra(p, lmax=2508, CMB_unit="muK", raw_cl=True)[
        "lensed_scalar"
    ]
    lmax = tuple(int(x) for x in lk.lmax)
    blocks = {
        "tt": spectra[:, 0], "ee": spectra[:, 1], "bb": spectra[:, 2],
        "te": spectra[:, 3], "tb": np.zeros(spectra.shape[0]),
        "eb": np.zeros(spectra.shape[0]),
    }
    order = ("tt", "ee", "bb", "te", "tb", "eb")
    return np.concatenate([blocks[k][: m + 1] for k, m in zip(order, lmax)])


def main() -> None:
    lk, names, idx, sampled, mu, scale, nu = build_vector_factory()
    print(f"sampled nuisance ({len(sampled)}): {sampled}")
    theory = theory_vector(lk, names)
    print(f"theory vector: {theory.shape} (expect (7527,))")

    s_idx = np.array([idx[n] for n in sampled])

    def chi2_of(nu_sampled: np.ndarray) -> float:
        nu_vec = nu.copy()
        nu_vec[s_idx] = nu_sampled
        return chi2_full(nu_vec)

    def chi2_full(nu_vec: np.ndarray) -> float:
        v = np.concatenate([theory, nu_vec])
        return float(np.asarray(lk(v))[0])

    base = chi2_of(nu[s_idx].copy())
    rng = np.random.default_rng(11)

    print(f"\nchi2 at prior means: {base:.6f}")
    print(f"{'direction':>10} {'eps':>10} {'d3 (rel)':>12}")
    worst = 0.0
    for trial in range(4):
        u = rng.standard_normal(len(sampled)) * scale
        u /= np.linalg.norm(u) / np.linalg.norm(scale)
        for eps in (1e-3, 1e-2):
            f = lambda t: chi2_of(nu[s_idx] + t * u)  # noqa: E731
            d3 = (f(3 * eps) - 3 * f(2 * eps) + 3 * f(eps) - f(0.0)) / eps**3
            # normalise by the scale of the quadratic term
            d2 = (f(eps) - 2 * f(0.0) + f(-eps)) / eps**2
            rel = abs(d3) / max(abs(d2) / eps, 1e-30)
            print(f"{trial:>10} {eps:>10.0e} {rel:>12.3e}")
            worst = max(worst, rel)
    print(f"\nworst relative third difference: {worst:.3e}")
    print("=> EXACTLY QUADRATIC" if worst < 1e-6 else "=> NOT quadratic (abandon)")


if __name__ == "__main__":
    main()
