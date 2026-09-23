"""Time the components of the likelihood to size the marginalisation overhead."""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, "/home/danieproyect/plc-3.1/clik-egg")
import clik  # noqa: E402
import clik.hpy as hpy  # noqa: E402

LIK = os.environ["DVCH_PLANCK_LIKELIHOOD"]


def main() -> None:
    lk = clik.clik(LIK)
    names = list(lk.extra_parameter_names)
    check = np.asarray(hpy.File(LIK, "r")["clik/check_param"][:])
    base = check[7527 : 7527 + len(names)].copy()
    vec = np.concatenate([np.zeros(7527), base])

    # warm-up
    for _ in range(3):
        lk(vec)
    n = 200
    t0 = time.perf_counter()
    for _ in range(n):
        lk(vec)
    dt = (time.perf_counter() - t0) / n
    print(f"clik eval: {dt * 1e3:.2f} ms  -> 191 evals = {dt * 191:.2f} s")
    print(f"clik eval: {dt * 1e3:.2f} ms  -> 60 evals  = {dt * 60:.2f} s")

    # CAMB timing (the real bottleneck)
    sys.path.insert(0, os.environ["DVCH_CAMB_ROOT"])
    import camb  # noqa: E402

    p = camb.set_params(
        H0=67.32, ombh2=0.0224, omch2=0.0120, tau=0.0546,
        As=np.exp(3.046) / 1e10, ns=0.965, lmax=2508,
        DVCH_flag=True, DVCH_n=0.076, DVCH_beta=0.00047,
    )
    for _ in range(2):
        r = camb.get_results(p)
        r.get_cmb_power_spectra(p, lmax=2508, CMB_unit="muK", raw_cl=True)
        r.get_lens_potential_cls(lmax=2500, raw_cl=True)
    n = 5
    t0 = time.perf_counter()
    for _ in range(n):
        r = camb.get_results(p)
        r.get_cmb_power_spectra(p, lmax=2508, CMB_unit="muK", raw_cl=True)
        r.get_lens_potential_cls(lmax=2500, raw_cl=True)
    dt = (time.perf_counter() - t0) / n
    print(f"CAMB full eval (DVCH, lmax=2508+lens): {dt:.2f} s")


if __name__ == "__main__":
    main()
