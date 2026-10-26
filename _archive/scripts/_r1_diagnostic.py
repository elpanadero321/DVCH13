"""Reproduce Cobaya's exact Rminus1 plus classic Gelman-Rubin and ESS.

Cobaya's check_convergence_and_learn_proposal uses, per chain:
    use_first = int(n/2); mean = mean(first=use_first); cov = cov(first=use_first)
then, over the m chain means/covs:
    W = weighted mean of covs; B = np.cov(means.T)   (unweighted!)
    Rminus1 = max|eigvals| of  Linv * corr(B) * Linv.T ,  Linv=inv-chol(corr(W))
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PARAM_COLS = None  # all sampled columns (exclude weight/logpost/derived)


def load(path: Path) -> np.ndarray:
    data = np.atleast_2d(np.loadtxt(path, comments="#"))
    return data


def sampled_cols(chain: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[int]]:
    keep = [
        i
        for i, name in enumerate(names)
        if name not in ("weight", "minuslogpost", "minuslogprior", "chi2")
        and not name.startswith("minuslogprior")
        and not name.startswith("chi2")
    ]
    return chain[:, keep], keep


def header_names(path: Path) -> list[str]:
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                return line[1:].split()
    raise RuntimeError("no header")


def cobaya_rminus1(chains: list[np.ndarray], first_frac: float = 0.5):
    """Per-chain mean/cov over samples [first_frac*n, n), then Cobaya's eigen-test."""
    per_chain = []
    for ch in chains:
        n = len(ch)
        use_first = int(n * first_frac)
        per_chain.append((n - use_first, ch[use_first:].mean(0), np.cov(ch[use_first:].T)))
    Ns = np.array([p[0] for p in per_chain], dtype=float)
    means = np.array([p[1] for p in per_chain])
    covs = np.array([p[2] for p in per_chain])
    W = np.tensordot(Ns / Ns.sum(), covs, axes=1)
    B = np.atleast_2d(np.cov(means.T))
    d = np.sqrt(np.diag(B))
    corr_B = (B / d).T / d
    corr_W = (W / d).T / d
    Linv = np.linalg.inv(np.linalg.cholesky(corr_W))
    eig = np.linalg.eigvalsh(Linv @ corr_B @ Linv.T)
    return float(np.max(np.abs(eig))), means, W


def classic_gr(chains: list[np.ndarray]):
    """Standard Gelman-Rubin R-hat - 1 per parameter (between/within variance)."""
    m = len(chains)
    n = min(len(c) for c in chains)
    tr = [c[:n] for c in chains]
    means = np.array([c.mean(0) for c in tr])
    variances = np.array([c.var(0, ddof=1) for c in tr])
    W = variances.mean(0)
    grand = means.mean(0)
    B = n / (m - 1) * ((means - grand) ** 2).sum(0)
    var_hat = (n - 1) / n * W + B / n
    rhat = np.sqrt(var_hat / W)
    return rhat - 1.0


def ess(chains: list[np.ndarray]) -> np.ndarray:
    """Effective sample size per parameter via integrated autocorrelation time."""
    m = len(chains)
    n = min(len(c) for c in chains)
    tr = np.array([c[:n] for c in chains])
    out = np.zeros(tr.shape[2])
    for j in range(tr.shape[2]):
        x = tr[:, :, j]
        mu = x.mean()
        var = x.var()
        if var == 0:
            out[j] = np.nan
            continue
        # averaged autocorrelation over chains
        rho = np.zeros(n)
        for lag in range(1, min(n // 2, 2000)):
            r = np.mean(
                [(x[i, :-lag] - mu) * (x[i, lag:] - mu) for i in range(m)], axis=0
            ).mean()
            rho[lag] = r / var
            if rho[lag] < 0:
                break
        tau = 1 + 2 * rho[rho > 0].sum() if (rho > 0).any() else 1.0
        tau = max(tau, 1.0)
        out[j] = m * n / tau
    return out


def main(prefix: str) -> None:
    root = Path(__file__).resolve().parent
    files = sorted(root.glob(f"{prefix}.*.txt"))
    if not files:
        raise SystemExit(f"no chains for prefix {prefix}")
    names = header_names(files[0])
    chains = [sampled_cols(load(f), names)[0] for f in files]
    keep_idx = sampled_cols(load(files[0]), names)[1]
    n = [len(c) for c in chains]
    print(f"prefix={prefix}  chains={len(chains)}  n_per_chain={n}  params={chains[0].shape[1]}")
    for frac in (0.5, 0.6, 0.7, 0.75, 0.8, 0.9):
        r1, means, W = cobaya_rminus1(chains, frac)
        print(f"  Cobaya-R1 (from {frac:.0%} of chain) = {r1:.4f}")
    # classic GR on the full chain and on last 50%
    r_full = classic_gr(chains)
    half = [c[int(len(c) * 0.5) :] for c in chains]
    r_half = classic_gr(half)
    e = ess(chains)
    print(f"  classic GR R-1 (full chain): max={np.nanmax(r_full):.4f} median={np.nanmedian(r_full):.4f}")
    print(f"  classic GR R-1 (last 50%):   max={np.nanmax(r_half):.4f} median={np.nanmedian(r_half):.4f}")
    print(f"  ESS: min={np.nanmin(e):.0f} median={np.nanmedian(e):.0f} (of {sum(n)} raw samples)")
    worst = np.argsort(-np.nan_to_num(r_full))[:6]
    nm = [names[keep_idx[i]] for i in worst if i < len(keep_idx)]
    print(f"  worst params by classic GR: {list(zip(nm, np.round(r_full[worst], 4)))}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dvch_prod")
