"""Decisive test: what is Cobaya's Rminus1 *noise floor* for m chains of n samples?

If iid draws from the TRUE posterior still give R-1 >> 0.01, then the current
plateau at ~0.14 is the test's floor for m=4, not real non-convergence -- and the
legal fix is simply more chains (also faster in aggregate).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from _r1_diagnostic import cobaya_rminus1, load, sampled_cols, header_names


def empirical_posterior(prefix: str):
    root = Path(__file__).resolve().parent
    files = sorted(root.glob(f"{prefix}.*.txt"))
    names = header_names(files[0])
    chains = [sampled_cols(load(f), names)[0] for f in files]
    keep = [c[int(len(c) * 0.75) :] for c in chains]  # stationary part only
    allsamples = np.concatenate(keep, axis=0)
    return np.cov(allsamples.T), np.mean(allsamples, axis=0)


def main():
    prefix = "dvch_prod"
    W, mu = empirical_posterior(prefix)
    p = W.shape[0]
    print(f"empirical posterior: p={p}  (cov from last 25% of current chains)")
    # Whiten: sample x = mu + L z,  z ~ N(0,I)
    L = np.linalg.cholesky(W)
    rng = np.random.default_rng(20260922)
    print(f"\n{'m':>3} {'n/chain':>8} {'R-1 mean':>9} {'R-1 median':>11} {'R-1 p90':>8}")
    for m in (4, 6, 8, 12):
        for n in (2000, 4000, 8000, 16000):
            vals = []
            reps = 12 if n <= 8000 else 6
            for _ in range(reps):
                z = rng.standard_normal((m, n, p))
                chains = [mu + (L @ z[i].T).T for i in range(m)]
                r1, _, _ = cobaya_rminus1(chains, 0.5)
                vals.append(r1)
            vals = np.array(vals)
            print(
                f"{m:>3} {n:>8} {vals.mean():>9.4f} {np.median(vals):>11.4f} "
                f"{np.percentile(vals, 90):>8.4f}"
            )


if __name__ == "__main__":
    main()
