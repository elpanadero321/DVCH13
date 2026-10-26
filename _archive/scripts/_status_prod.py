#!/usr/bin/env python3
"""Health/status report for the definitive dvch_prod run.

Reports sample counts, acceptance rate, Cobaya's R-1, classic Gelman-Rubin,
ESS per parameter, measured throughput and ETA to the R-1 = 0.03 target.

Usage: bash _status_prod.sh [window_minutes]
"""
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from _r1_diagnostic import load, sampled_cols, header_names, cobaya_rminus1, classic_gr, ess

ROOT = Path(__file__).resolve().parent
TARGET_R1 = 0.03
TAU = 45.0  # measured autocorrelation time at the RWM floor


def acceptance(files):
    rates = []
    for f in files:
        d = np.atleast_2d(np.loadtxt(f, comments="#"))
        w = d[:, 0]
        k = min(300, len(w))
        rates.append(k / w[-k:].sum())
    return float(np.mean(rates))


def main(window_min=15):
    files = sorted(ROOT.glob("dvch_prod.*.txt"))
    names = header_names(files[0])
    chains = [sampled_cols(load(f), names)[0] for f in files]
    n = [len(c) for c in chains]
    total = sum(n)

    r1, _, _ = cobaya_rminus1(chains, 0.5)
    gr = classic_gr([c[int(len(c) * 0.5):] for c in chains])
    e = ess(chains)
    acc = acceptance(files)

    # throughput from the log over the last `window_min` minutes
    try:
        log = subprocess.run(["tail", "-n", "400", "dvch_prod_run.log"],
                             capture_output=True, text=True, cwd=ROOT).stdout
        pts = []
        for line in log.splitlines():
            if "] Progress @" in line:
                toks = line.split()
                # ['[2',':','mcmc]','Progress','@','DATE','TIME',':','N','steps','taken,','and','M','accepted.']
                pts.append((toks[-9] + " " + toks[-8], int(toks[-6]), int(toks[-2])))
        # earliest vs latest per chain, converted to epoch
        import datetime as dt
        per = {}
        for ts, st, ac in pts:
            t = dt.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=dt.timezone.utc).timestamp()
            per.setdefault(0, [])
        # simpler: global min/max accepted over the window
        tss = [dt.datetime.strptime(p[0], "%Y-%m-%d %H:%M:%S").timestamp() for p in pts]
        accs = [p[2] for p in pts]
        i0, i1 = int(np.argmin(tss)), int(np.argmax(tss))
        dt_s = max(tss[i1] - tss[i0], 1.0)
        rate = (max(accs) - min(accs)) * 3600 / dt_s if len(set(accs)) > 1 else 0.0
    except Exception:
        rate = 0.0

    n_tail = total / len(chains) / 2
    print(f"dvch_prod definitive run  ({len(chains)} chains)")
    print(f"  samples/chain : {n}   total = {total}")
    print(f"  acceptance    : {acc:.3f}  (optimal ~0.23-0.27)")
    print(f"  Cobaya R-1    : {r1:.4f}   (target {TARGET_R1})")
    print(f"  classic GR    : max {np.nanmax(gr):.4f} / median {np.nanmedian(gr):.4f} (last 50%)")
    print(f"  ESS/param     : min {np.nanmin(e):.0f} / median {np.nanmedian(e):.0f}")
    print(f"  throughput    : {rate:.0f} accepted samples/hr (last {window_min} min)")
    # model: R-1 = 11.9 * tau / n_tail  (empirically calibrated on this run)
    pred_r1 = 11.9 * TAU / n_tail
    need = len(chains) * 2 * 11.9 * TAU / TARGET_R1
    if rate > 0:
        eta_h = (need - total) / rate
        print(f"  model R-1     : {pred_r1:.4f} (tau={TAU:.0f}); "
              f"need ~{need:.0f} total for R-1={TARGET_R1}")
        print(f"  ETA           : {eta_h:.0f} h = {eta_h/24:.1f} days at current rate")
    print(f"  progress file R-1 (last checkpoint): "
          f"{ROOT.joinpath('dvch_prod.progress').read_text().strip().splitlines()[-1]}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 15)
