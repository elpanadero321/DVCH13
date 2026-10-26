"""Per-parameter autocorrelation time and ESS on the real chains."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from _r1_diagnostic import load, sampled_cols, header_names


def autocorr_time(x: np.ndarray) -> float:
    """Integrated autocorrelation time via Sokal's windowed estimator."""
    n = len(x)
    x = x - x.mean()
    var = np.dot(x, x) / n
    if var == 0:
        return np.nan
    # FFT-based autocorrelation
    f = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(f * np.conj(f), n=2 * n)[:n]
    acf /= acf[0]
    # Sokal window: sum until first negative
    s = 0.0
    for k in range(1, n):
        if acf[k] < 0:
            break
        s += acf[k]
    return 1.0 + 2.0 * s


def main(prefix: str, frac: float = 0.5):
    root = Path(__file__).resolve().parent
    files = sorted(root.glob(f"{prefix}.*.txt"))
    names = header_names(files[0])
    chains = [sampled_cols(load(f), names)[0] for f in files]
    p = chains[0].shape[1]
    # keep only the stationary tail
    chains = [c[int(len(c) * frac):] for c in chains]
    n = min(len(c) for c in chains)
    chains = np.array([c[:n] for c in chains])
    keep = [i for i, nm in enumerate(names)
            if nm not in ("weight", "minuslogpost")
            and not nm.startswith("minuslogprior") and not nm.startswith("chi2")]
    sampled_names = [names[i] for i in keep]
    print(f"prefix={prefix}  m={len(chains)}  n={n} (after {frac:.0%} cut)  p={p}")
    print(f"\n{'param':>18} {'tau':>7} {'ESS':>8}")
    taus = []
    for j in range(p):
        t = np.mean([autocorr_time(chains[i][:, j]) for i in range(len(chains))])
        taus.append(t)
    order = np.argsort(-np.nan_to_num(taus))
    for rank, j in enumerate(order):
        if rank >= 14:
            break
        print(f"{sampled_names[j]:>18} {taus[j]:>7.1f} {n*len(chains)/max(taus[j],1):>8.0f}")
    tau_med = float(np.nanmedian(taus))
    tau_max = float(np.nanmax(taus))
    print(f"\nmedian tau = {tau_med:.1f}   max tau = {tau_max:.1f}")
    print(f"ESS (m*n/tau_max) = {n*len(chains)/tau_max:.0f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dvch_prod",
         float(sys.argv[2]) if len(sys.argv) > 2 else 0.5)
