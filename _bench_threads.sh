#!/bin/bash
# Aggregate-throughput benchmark: how many MCMC likelihood evals/sec can this
# 12-core box actually deliver, as a function of (mpi procs x OMP threads)?
# Measures CAMB-only scaling, clik-only cost, then real parallel contention.
set -uo pipefail
cd /mnt/d/DVCH13-1 || exit 1
source env.sh >/dev/null 2>&1

python3 - "$@" <<'PY'
import os, sys, time, subprocess, concurrent.futures as cf
import numpy as np

sys.path.insert(0, os.environ["DVCH_CAMB_ROOT"])
import camb

p = camb.set_params(H0=67.32, ombh2=0.0224, omch2=0.0120, tau=0.0546,
                    As=float(np.exp(3.046)/1e10), ns=0.965, lmax=2508,
                    DVCH_flag=True, DVCH_n=0.076, DVCH_beta=0.00047)

def camb_eval():
    r = camb.get_results(p)
    r.get_cmb_power_spectra(p, lmax=2508, CMB_unit="muK", raw_cl=True)
    r.get_lens_potential_cls(lmax=2500, raw_cl=True)

camb_eval(); camb_eval()
n = 3
t0 = time.perf_counter()
for _ in range(n):
    camb_eval()
dt_camb = (time.perf_counter()-t0)/n
print(f"CAMB/eval single-proc threads={os.environ.get('OMP_NUM_THREADS','default')}: {dt_camb:.2f}s")
PY
