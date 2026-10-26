"""Break down CAMB cost: is get_results being called twice per likelihood eval?

camb.CAMBdata.get_cmb_power_spectra(params, ...) re-runs the full Boltzmann solve
when params is not None. The production likelihood passes params again, so the
transfer functions are computed twice per evaluation.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.environ["DVCH_CAMB_ROOT"])
import camb  # noqa: E402

p = camb.set_params(
    H0=67.32, ombh2=0.0224, omch2=0.0120, tau=0.0546,
    As=float(np.exp(3.046) / 1e10), ns=0.965, lmax=2508,
    DVCH_flag=True, DVCH_n=0.076, DVCH_beta=0.00047,
)


def bench(fn, n=3, label=""):
    fn()  # warm up
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    dt = (time.perf_counter() - t0) / n
    print(f"{label}: {dt:.2f}s")
    return dt


# A: current production pattern (params re-passed -> double Boltzmann solve)
def current():
    r = camb.get_results(p)
    r.get_cmb_power_spectra(p, lmax=2508, CMB_unit="muK", raw_cl=True)
    r.get_lens_potential_cls(lmax=2500, raw_cl=True)


# B: reuse results, no re-solve
def reuse():
    r = camb.get_results(p)
    r.get_cmb_power_spectra(lmax=2508, CMB_unit="muK", raw_cl=True)
    r.get_lens_potential_cls(lmax=2500, raw_cl=True)


# C: how expensive is each piece?
def only_results():
    camb.get_results(p)


def main():
    import os
    print(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS','unset')}")
    t_cur = bench(current, label="A current (double solve)")
    t_re = bench(reuse, label="B reuse results  (single solve)")
    t_res = bench(only_results, label="C get_results only")
    print(f"\nspeedup factor: {t_cur / t_re:.2f}x")
    print(f"get_results is {100 * t_res / t_cur:.0f}% of the current cost")
    # also: cost of the spectra given results
    r = camb.get_results(p)
    t_spec = bench(
        lambda: r.get_cmb_power_spectra(lmax=2508, CMB_unit="muK", raw_cl=True),
        label="spectra given results")
    t_lens = bench(
        lambda: r.get_lens_potential_cls(lmax=2500, raw_cl=True),
        label="lensing cls given results")


if __name__ == "__main__":
    main()
