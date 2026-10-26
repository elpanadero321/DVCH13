"""Robust linearity/quadraticity test for plik in the sampled nuisance params.

Signature of an exactly quadratic chi2:
  1. The second difference d2(t) = [chi2(x+t) - 2 chi2(x) + chi2(x-t)] / t^2
     is CONSTANT in t (independent of step size), and
  2. the residual of a 3-point parabola fit vanishes to ~1e-12 relative.
We test each nuisance separately along its prior scale, and also test
bilinearity: chi2(a+b) + chi2(0) - chi2(a) - chi2(b) = 0 for two directions.
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
from _quadratic_test import build_vector_factory, theory_vector  # noqa: E402

LIK = Path(os.environ["DVCH_PLANCK_LIKELIHOOD"])


def main() -> None:
    lk, names, idx, sampled, mu, scale, nu = build_vector_factory()
    theory = theory_vector(lk, names)
    s_idx = np.array([idx[n] for n in sampled])

    def chi2_of(sampled_vec: np.ndarray) -> float:
        nu_vec = nu.copy()
        nu_vec[s_idx] = sampled_vec
        return float(np.asarray(lk(np.concatenate([theory, nu_vec])))[0])

    x0 = mu.copy()
    base = chi2_of(x0)
    print(f"chi2 at prior means: {base:.6f}")
    rng = np.random.default_rng(3)

    print(f"\n{'param':>18} {'d2(1s)/d2(0.1s)-1':>20} {'parabola resid':>16}")
    worst_scale = 0.0
    for j, name in enumerate(sampled):
        e = np.zeros(len(sampled))
        e[j] = scale[j]
        f = lambda t: chi2_of(x0 + t * e)  # noqa: E731
        ts = (0.25, 0.5, 1.0, 2.0)
        d2 = [(f(t) - 2 * f(0.0) + f(-t)) / t**2 for t in ts]
        rel_d2 = (max(d2) - min(d2)) / max(abs(np.mean(d2)), 1e-30)
        # parabola residual through t=-1,0,1 evaluated at t=0.5
        pred = 0.5 * (f(1.0) + f(-1.0))
        resid = abs(f(0.5) - 0.5 * (f(1.0) + f(-1.0))) / max(abs(f(1.0) - f(-1.0)), 1e-30)
        print(f"{name:>18} {rel_d2:>20.3e} {resid:>16.3e}")
        worst_scale = max(worst_scale, rel_d2, resid)

    # bilinearity between two random directions
    print("\nbilinearity: chi2(a+b)+chi2(0)-chi2(a)-chi2(b)")
    for trial in range(3):
        a = rng.standard_normal(len(sampled)) * scale
        b = rng.standard_normal(len(sampled)) * scale
        lhs = chi2_of(x0 + a + b) + base - chi2_of(x0 + a) - chi2_of(x0 + b)
        denom = max(abs(chi2_of(x0 + a) - base), abs(chi2_of(x0 + b) - base), 1e-30)
        print(f"  trial {trial}: {lhs:+.6e}  (rel {abs(lhs)/denom:.2e})")

    print(f"\nworst scale violation: {worst_scale:.3e}")
    ok = worst_scale < 1e-8
    print("=> EXACTLY QUADRATIC" if ok else "=> nonlinear component present")


if __name__ == "__main__":
    main()
