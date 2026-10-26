"""Calibrate Cobaya's Rminus1 against the chain autocorrelation time tau.

Synthetic AR(1) chains with the empirical posterior covariance and lag-1
correlation rho (tau = (1+rho)/(1-rho)). Inverting the measured R-1 gives the
effective tau of the real chains, which sets how many samples the definitive
fresh run needs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from _r1_diagnostic import cobaya_rminus1, load, sampled_cols, header_names

TARGET = 0.1445  # measured Cobaya R-1 on the current chains (last 50%)


def empirical_posterior(prefix: str):
    root = Path(__file__).resolve().parent
    files = sorted(root.glob(f"{prefix}.*.txt"))
    names = header_names(files[0])
    chains = [sampled_cols(load(f), names) for f in files]
    keep = [c[int(len(c) * 0.5) :] for c in chains]
    allsamples = np.concatenate(keep, axis=0)
    return np.cov(allsamples.T), np.mean(allsamples, axis=0), len(keep[0])


def ar1_chains(mu, L, m, n, rho, rng):
    p = len(mu)
    z = rng.standard_normal((m, n, p))
    x = np.empty_like(z)
    x[:, 0] = z[:, 0]
    for t in range(1, n):
        x[:, t] = rho * x[:, t - 1] + np.sqrt(1 - rho * rho) * z[:, t]
    return [mu + (L @ x[i].T).T for i in range(m)]


def main():
    prefix = "dvch_prod"
    W, mu, n_real = empirical_posterior(prefix)
    L = np.linalg.cholesky(W)
    m_real = 4
    rng = np.random.default_rng(7)
    print(f"real chains: m={m_real} n/chain~{n_real}  target R-1={TARGET}")
    print(f"\n{'tau':>6} {'m':>3} {'n/chain':>8} {'R-1':>8}")
    found = None
    for tau in (5, 10, 15, 20, 25, 30, 40, 60):
        rho = (tau - 1) / (tau + 1)
        vals = []
        reps = 8
        for _ in range(reps):
            chains = ar1_chains(mu, L, m_real, n_real, rho, rng)
            r1, _, _ = cobaya_rminus1(chains, 0.5)
            vals.append(r1)
        r1 = float(np.mean(vals))
        print(f"{tau:>6} {m_real:>3} {n_real:>8} {r1:>8.4f}")
        if found is None and r1 > TARGET:
            found = tau
    print(f"\n=> effective tau of the real chains ~ {found} samples")

    # Now: samples needed for R-1 < 0.01 as a function of m, at that tau
    if found:
        rho = (found - 1) / (found + 1)
        print(f"\nSamples needed for R-1 < 0.01 at tau={found}:")
        print(f"{'m':>3} {'n/chain':>9} {'R-1':>8}")
        for m in (4, 6, 8):
            for n in (5000, 10000, 15000, 20000, 30000, 40000):
                vals = []
                reps = 6 if n <= 15000 else 3
                for _ in range(reps):
                    chains = ar1_chains(mu, L, m, n, rho, rng)
                    r1, _, _ = cobaya_rminus1(chains, 0.5)
                    vals.append(r1)
                r1 = float(np.mean(vals))
                flag = "  <-- OK" if r1 < 0.01 else ""
                print(f"{m:>3} {n:>9} {r1:>8.4f}{flag}")
                if r1 < 0.01:
                    break


if __name__ == "__main__":
    main()
