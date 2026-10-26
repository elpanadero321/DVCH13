#!/usr/bin/env python3
"""Aggregate MCMC throughput on this box.

Likelihood cost = CAMB (OpenMP) + clik (serial). Measures both, then measures
real aggregate evals/sec for several (procs x threads) layouts on 12 cores.

Usage: run inside WSL after `source env.sh`:
    OMP_NUM_THREADS=T python3 _bench_throughput.py camb      # CAMB timing
    OMP_NUM_THREADS=T python3 _bench_throughput.py clik      # clik timing
    python3 _bench_throughput.py parallel PROCS THREADS      # contention
"""
import os
import subprocess
import sys
import time

import numpy as np


def time_camb(n: int = 3) -> float:
    sys.path.insert(0, os.environ["DVCH_CAMB_ROOT"])
    import camb

    p = camb.set_params(
        H0=67.32, ombh2=0.0224, omch2=0.0120, tau=0.0546,
        As=float(np.exp(3.046) / 1e10), ns=0.965, lmax=2508,
        DVCH_flag=True, DVCH_n=0.076, DVCH_beta=0.00047,
    )

    def ev():
        r = camb.get_results(p)
        r.get_cmb_power_spectra(p, lmax=2508, CMB_unit="muK", raw_cl=True)
        r.get_lens_potential_cls(lmax=2500, raw_cl=True)

    ev(); ev()
    t0 = time.perf_counter()
    for _ in range(n):
        ev()
    return (time.perf_counter() - t0) / n


def time_clik(n: int = 60) -> float:
    sys.path.insert(0, os.environ["DVCH_CLIK_EGG"])
    import clik

    lk = clik.clik(os.environ["DVCH_PLANCK_LIKELIHOOD"])
    npar = len(list(lk.extra_parameter_names))
    vec = np.concatenate([np.zeros(7527), np.ones(npar)])
    for _ in range(3):
        lk(vec)
    t0 = time.perf_counter()
    for _ in range(n):
        lk(vec)
    return (time.perf_counter() - t0) / n


def parallel(procs: int, threads: int, n: int = 3) -> float:
    """Wall time for `procs` independent workers each doing n CAMB evals."""
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(threads)
    cmd = [sys.executable, os.path.abspath(__file__), "camb"]
    t0 = time.perf_counter()
    jobs = [subprocess.Popen(cmd + [str(n)], env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            for _ in range(procs)]
    outs = [j.communicate()[0].decode().strip().splitlines() for j in jobs]
    wall = time.perf_counter() - t0
    per = [float(o[-1].split()[-1].rstrip("s")) for o in outs if o]
    return wall, per


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "camb":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        print(f"mode=camb threads={os.environ.get('OMP_NUM_THREADS','def')} "
              f"t={time_camb(n):.2f}s")
    elif mode == "clik":
        print(f"mode=clik t={time_clik():.4f}s")
    elif mode == "parallel":
        procs, threads = int(sys.argv[2]), int(sys.argv[3])
        n = int(sys.argv[4]) if len(sys.argv) > 4 else 3
        wall, per = parallel(procs, threads, n)
        print(f"mode=parallel procs={procs} threads={threads} wall={wall:.1f}s "
              f"per_eval_avg={np.mean(per):.2f}s "
              f"aggregate={procs*n/wall:.2f} evals/s "
              f"accepted_per_hr={procs*n/wall*0.273*3600:.0f}")
