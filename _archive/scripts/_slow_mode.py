"""Identify the slow mode that dominates Cobaya's eigenvalue R-1.

Cobaya's R-1 = max|eig| of  Linv corr_B Linv.T  with Linv = inv-chol(corr_W),
corr_B = B/outer(d,d), corr_W = W/outer(d,d), d = sqrt(diag(B)), where B is the
covariance of the m chain means and W the within-chain covariance.

The leading eigenvector picks out the direction in which the chain means disagree
most (relative to within-chain spread) -- i.e. the slowest-mixing direction.
We (a) express it in parameter space, (b) measure its autocorrelation time, and
(c) print block means to distinguish slow mixing from residual drift/burn-in.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from _r1_diagnostic import load, sampled_cols, header_names


def main(prefix: str, frac: float = 0.5, nblocks: int = 20):
    root = Path(__file__).resolve().parent
    files = sorted(root.glob(f"{prefix}.*.txt"))
    names = header_names(files[0])
    chains = [sampled_cols(load(f), names)[0] for f in files]
    p = chains[0].shape[1]
    keep = [i for i, nm in enumerate(names)
            if nm not in ("weight", "minuslogpost")
            and not nm.startswith("minuslogprior") and not nm.startswith("chi2")]
    pnames = [names[i] for i in keep]

    tails = [c[int(len(c) * frac):] for c in chains]
    n = min(len(t) for t in tails)
    tails = np.array([t[:n] for t in tails])
    means = tails.mean(1)
    covs = np.array([np.cov(t.T) for t in tails])
    W = covs.mean(0)
    B = np.atleast_2d(np.cov(means.T))
    d = np.sqrt(np.diag(B))
    corr_B = (B / d).T / d
    corr_W = (W / d).T / d
    Linv = np.linalg.inv(np.linalg.cholesky(corr_W))
    M = Linv @ corr_B @ Linv.T
    eig, vec = np.linalg.eigh(M)
    order = np.argsort(-eig)
    print(f"m={len(chains)} n_tail={n} p={p}")
    print(f"top-3 eigenvalues of Cobaya's M: {eig[order[:3]]}")
    print(f"(R-1 = {eig[order[0]]:.4f})\n")

    for k in range(2):
        w = vec[:, order[k]]
        u = Linv.T @ w                # direction in corr-normalized space
        direction = d * u             # parameter-space direction (B = outer(d,d)*corr_B)
        direction = direction / np.linalg.norm(direction)
        print(f"--- mode {k} (eig={eig[order[k]]:.4f}) ---")
        comp = np.abs(direction) / np.abs(direction).sum()
        top = np.argsort(-comp)[:8]
        for j in top:
            print(f"  {pnames[j]:>18}  dir={direction[j]:+9.4f}  |w|={comp[j]:6.3f}")
        # project chains onto the direction (in units of the posterior std along it)
        scale = np.sqrt(max(direction @ W @ direction, 1e-300))
        proj = np.array([t @ direction / scale for t in tails])
        # autocorrelation time of the projection (mean over chains)
        taus = []
        for x in proj:
            xc = x - x.mean()
            f = np.fft.rfft(xc, n=2 * len(xc))
            acf = np.fft.irfft(f * np.conj(f), n=2 * len(xc))[: len(xc)]
            acf /= acf[0]
            s = 0.0
            for kk in range(1, len(xc)):
                if acf[kk] < 0:
                    break
                s += acf[kk]
            taus.append(1 + 2 * s)
        print(f"  tau along mode (per chain): {[f'{t:.0f}' for t in taus]}")
        print(f"  spread of chain means along mode (in posterior sigma): "
              f"{proj.mean(1).std(ddof=1):.3f}  (expected ~{1/np.sqrt(n):.3f})")
        blk = np.array([np.split(x, nblocks) for x in proj])  # (m, nblocks, len)
        bm = blk.mean(2)
        print(f"  block means (chain 0): {np.round(bm[0], 2)}")
        print(f"  block means (chain 1): {np.round(bm[1], 2)}")
        print(f"  overall chain means:   {np.round(proj.mean(1), 2)}\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dvch_prod",
         float(sys.argv[2]) if len(sys.argv) > 2 else 0.5)
