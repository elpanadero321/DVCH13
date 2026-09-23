import os
import sys
import time

sys.path.insert(0, os.environ["DVCH_CAMB_ROOT"])
import camb  # noqa: E402

p = camb.set_params(
    H0=67.32, ombh2=0.0224, omch2=0.0120, tau=0.0546,
    As=float(__import__("numpy").exp(3.046) / 1e10), ns=0.965, lmax=2508,
    DVCH_flag=True, DVCH_n=0.076, DVCH_beta=0.00047,
)


def bench():
    r = camb.get_results(p)
    r.get_cmb_power_spectra(p, lmax=2508, CMB_unit="muK", raw_cl=True)
    r.get_lens_potential_cls(lmax=2500, raw_cl=True)


bench()
n = 4
t0 = time.perf_counter()
for _ in range(n):
    bench()
print(f"threads={os.environ.get('OMP_NUM_THREADS','?')} CAMB={ (time.perf_counter()-t0)/n:.2f}s")
