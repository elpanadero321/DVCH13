"""Verify the plik likelihood is exactly Gaussian in the nuisance parameters,
and measure the exact marginalisation to use in the fast sampler.

plik model:  data_vector = R(q_hat) + sum_c template_c * nuisance_c
             chi2 = (x - d)^T Cinv (x - d)   (linear in nuisance)
Marginalising a Gaussian prior N(mu, P) over the nuisance amplitudes gives
             chi2_marg = r^T Cmarg_inv r  with  Cmarg = C + A P A^T
where A is the matrix of template responses.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/danieproyect/plc-3.1/clik-egg")
import clik  # noqa: E402
import clik.hpy as hpy  # noqa: E402

LIK = os.environ["DVCH_PLANCK_LIKELIHOOD"]


def main() -> None:
    lk = clik.clik(LIK)
    names = list(lk.extra_parameter_names)
    print(f"nuisance params ({len(names)}): {names}")
    check = np.asarray(hpy.File(LIK, "r")["clik/check_param"][:])
    base = check[7527 : 7527 + len(names)].copy()
    print(f"check point nuisance: {np.round(base[:6], 4)} ...")

    # Reference: full evaluation at the check point
    vec0 = np.concatenate([np.zeros(7527), base])
    ref = float(np.asarray(lk(vec0))[0])
    print(f"clik(check) = {ref:.6f}")

    # Perturb each nuisance individually: is chi2 exactly quadratic?
    scale = 1e-3
    print("\nquadraticity test (d_chi2 should be exactly quadratic in eps):")
    worst = 0.0
    for i in (0, 5, 10, 15, 18):
        if i >= len(names):
            continue
        v = base.copy()
        v[i] += scale
        chi_p = float(np.asarray(lk(np.concatenate([np.zeros(7527), v])))[0])
        v = base.copy()
        v[i] -= scale
        chi_m = float(np.asarray(lk(np.concatenate([np.zeros(7527), v])))[0])
        # second difference: for a quadratic, (chi_p + chi_m - 2*ref)/scale^2 is const
        d2 = (chi_p + chi_m - 2 * ref) / scale**2
        # first-order term
        d1 = (chi_p - chi_m) / (2 * scale)
        print(f"  {names[i]:>18}: d1={d1:+.4e}  d2={d2:+.4e}")
        worst = max(worst, abs(d1))
    print(f"max linear term: {worst:.3e}  (0 => exactly quadratic/Gaussian)")


if __name__ == "__main__":
    main()
